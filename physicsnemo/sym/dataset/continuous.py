# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" PhysicsNeMo Dataset constructors for continuous type data
"""

from typing import Dict, List, Callable

import numpy as np

from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from .dataset import Dataset, IterableDataset, _DictDatasetMixin


class _DictPointwiseDatasetMixin(_DictDatasetMixin):
    "Special mixin class for dealing with dictionaries as input"

    def save_dataset(self, filename):

        named_lambda_weighting = {
            "lambda_" + key: value for key, value in self.lambda_weighting.items()
        }
        save_var = {**self.invar, **self.outvar, **named_lambda_weighting}
        var_to_polyvtk(filename, save_var)


class DictPointwiseDataset(_DictPointwiseDatasetMixin, Dataset):
    """A map-style dataset for a finite set of pointwise training examples."""

    auto_collation = True

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        lambda_weighting: Dict[str, np.array] = None,
    ):
        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        outvar = _DictDatasetMixin._idx_var(self.outvar, idx)
        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx)
        return (invar, outvar, lambda_weighting)

    def __len__(self):
        return self.length


class DictInferencePointwiseDataset(Dataset):
    """
    A map-style dataset for inferencing the model, only contains inputs
    """

    auto_collation = True

    def __init__(
        self,
        invar: Dict[str, np.array],
        output_names: List[str],  # Just names of output vars
    ):

        self.invar = Dataset._to_tensor_dict(invar)
        self.output_names = output_names
        self.length = len(next(iter(invar.values())))

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        return (invar,)

    def __len__(self):
        return self.length

    @property
    def invar_keys(self):
        return list(self.invar.keys())

    @property
    def outvar_keys(self):
        return list(self.output_names)


class ContinuousPointwiseIterableDataset(IterableDataset):
    """
    An infinitely iterable dataset for a continuous set of pointwise training examples.
    This will resample training examples (create new ones) every iteration.
    """

    def __init__(
        self,
        invar_fn: Callable,
        outvar_fn: Callable,
        lambda_weighting_fn: Callable = None,
    ):

        self.invar_fn = invar_fn
        self.outvar_fn = outvar_fn
        self.lambda_weighting_fn = lambda_weighting_fn
        if lambda_weighting_fn is None:
            lambda_weighting_fn = lambda _, outvar: {
                key: np.ones_like(x) for key, x in outvar.items()
            }

        def iterable_function():
            while True:
                invar = Dataset._to_tensor_dict(self.invar_fn())
                outvar = Dataset._to_tensor_dict(self.outvar_fn(invar))
                lambda_weighting = Dataset._to_tensor_dict(
                    self.lambda_weighting_fn(invar, outvar)
                )
                yield (invar, outvar, lambda_weighting)

        self.iterable_function = iterable_function

    def __iter__(self):
        yield from self.iterable_function()

    @property
    def invar_keys(self):
        invar = self.invar_fn()
        return list(invar.keys())

    @property
    def outvar_keys(self):
        invar = self.invar_fn()
        outvar = self.outvar_fn(invar)
        return list(outvar.keys())

    def save_dataset(self, filename):
        # Cannot save continuous data-set
        pass


class DictImportanceSampledPointwiseIterableDataset(
    _DictPointwiseDatasetMixin, IterableDataset
):
    """
    An infinitely iterable dataset that applies importance sampling for faster more accurate monte carlo integration
    """

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        importance_measure: Callable,
        lambda_weighting: Dict[str, np.array] = None,
        shuffle: bool = True,
        resample_freq: int = 1000,
    ):
        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

        self.batch_size = min(batch_size, self.length)
        self.shuffle = shuffle
        self.resample_freq = resample_freq
        self.importance_measure = importance_measure

        def iterable_function():

            # TODO: re-write idx calculation using pytorch sampling - to improve performance

            counter = 0
            while True:
                # resample all points when needed
                if counter % self.resample_freq == 0:
                    list_importance = []
                    list_invar = {
                        key: np.split(value, value.shape[0] // self.batch_size)
                        for key, value in self.invar.items()
                    }
                    for i in range(len(next(iter(list_invar.values())))):
                        importance = self.importance_measure(
                            {key: value[i] for key, value in list_invar.items()}
                        )
                        list_importance.append(importance)
                    importance = np.concatenate(list_importance, axis=0)
                    prob = importance / np.sum(self.invar["area"].numpy() * importance)

                # sample points from probability distribution and store idx
                idx = np.array([])
                while True:
                    r = np.random.uniform(0, np.max(prob), size=self.batch_size)
                    try_idx = np.random.choice(self.length, self.batch_size)
                    if_sample = np.less(r, prob[try_idx, :][:, 0])
                    idx = np.concatenate([idx, try_idx[if_sample]])
                    if idx.shape[0] >= batch_size:
                        idx = idx[:batch_size]
                        break
                idx = idx.astype(np.int64)

                # gather invar, outvar, and lambda weighting
                invar = _DictDatasetMixin._idx_var(self.invar, idx)
                outvar = _DictDatasetMixin._idx_var(self.outvar, idx)
                lambda_weighting = _DictDatasetMixin._idx_var(
                    self.lambda_weighting, idx
                )

                # set area value from importance sampling
                invar["area"] = 1.0 / (prob[idx] * batch_size)

                # return and count up
                counter += 1
                yield (invar, outvar, lambda_weighting)

        self.iterable_function = iterable_function

    def __iter__(self):
        yield from self.iterable_function()


class ListIntegralDataset(_DictDatasetMixin, Dataset):
    """
    A map-style dataset for a finite set of integral training examples.
    """

    auto_collation = True

    def __init__(
        self,
        list_invar: List[Dict[str, np.array]],
        list_outvar: List[Dict[str, np.array]],
        list_lambda_weighting: List[Dict[str, np.array]] = None,
    ):
        if list_lambda_weighting is None:
            list_lambda_weighting = []
            for outvar in list_outvar:
                list_lambda_weighting.append(
                    {key: np.ones_like(x) for key, x in outvar.items()}
                )

        invar = _stack_list_numpy_dict(list_invar)
        outvar = _stack_list_numpy_dict(list_outvar)
        lambda_weighting = _stack_list_numpy_dict(list_lambda_weighting)

        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        outvar = _DictDatasetMixin._idx_var(self.outvar, idx)
        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx)
        return (invar, outvar, lambda_weighting)

    def __len__(self):
        return self.length

    def save_dataset(self, filename):
        for idx in range(self.length):
            var_to_polyvtk(
                filename + "_" + str(idx).zfill(5),
                _DictDatasetMixin._idx_var(self.invar, idx),
            )


class ContinuousIntegralIterableDataset(IterableDataset):
    """
    An infinitely iterable dataset for a continuous set of integral training examples.
    This will resample training examples (create new ones) every iteration.
    """

    def __init__(
        self,
        invar_fn: Callable,
        outvar_fn: Callable,
        batch_size: int,
        lambda_weighting_fn: Callable = None,
        param_ranges_fn: Callable = None,
    ):

        self.invar_fn = invar_fn
        self.outvar_fn = outvar_fn
        self.lambda_weighting_fn = lambda_weighting_fn
        if lambda_weighting_fn is None:
            lambda_weighting_fn = lambda _, outvar: {
                key: np.ones_like(x) for key, x in outvar.items()
            }
        if param_ranges_fn is None:
            param_ranges_fn = lambda: {}  # Potentially unsafe?
        self.param_ranges_fn = param_ranges_fn

        self.batch_size = batch_size

        # TODO: re-write iterable function so that for loop not needed - to improve performance

        def iterable_function():
            while True:
                list_invar = []
                list_outvar = []
                list_lambda_weighting = []
                for _ in range(self.batch_size):
                    param_range = self.param_ranges_fn()
                    list_invar.append(self.invar_fn(param_range))
                    if (
                        not param_range
                    ):  # TODO this can be removed after a np_lambdify rewrite
                        param_range = {"_": next(iter(list_invar[-1].values()))[0:1]}

                    list_outvar.append(self.outvar_fn(param_range))
                    list_lambda_weighting.append(
                        self.lambda_weighting_fn(param_range, list_outvar[-1])
                    )
                invar = Dataset._to_tensor_dict(_stack_list_numpy_dict(list_invar))
                outvar = Dataset._to_tensor_dict(_stack_list_numpy_dict(list_outvar))
                lambda_weighting = Dataset._to_tensor_dict(
                    _stack_list_numpy_dict(list_lambda_weighting)
                )
                yield (invar, outvar, lambda_weighting)

        self.iterable_function = iterable_function

    def __iter__(self):
        yield from self.iterable_function()

    @property
    def invar_keys(self):
        param_range = self.param_ranges_fn()
        invar = self.invar_fn(param_range)
        return list(invar.keys())

    @property
    def outvar_keys(self):
        param_range = self.param_ranges_fn()
        invar = self.invar_fn(param_range)
        outvar = self.outvar_fn(invar)
        return list(outvar.keys())

    def save_dataset(self, filename):
        # Cannot save continuous data-set
        pass


class DictVariationalDataset(Dataset):
    """
    A map-style dataset for a finite set of variational training examples.
    """

    auto_collation = True

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar_names: List[str],  # Just names of output vars
    ):

        self.invar = Dataset._to_tensor_dict(invar)
        self.outvar_names = outvar_names
        self.length = len(next(iter(invar.values())))

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        return invar

    def __len__(self):
        return self.length

    @property
    def invar_keys(self):
        return list(self.invar.keys())

    @property
    def outvar_keys(self):
        return list(self.outvar_names)

    def save_dataset(self, filename):
        for i, invar in self.invar.items():
            var_to_polyvtk(invar, filename + "_" + str(i))


def _stack_list_numpy_dict(list_var):
    var = {}
    for key in list_var[0].keys():
        var[key] = np.stack([v[key] for v in list_var], axis=0)
    return var
