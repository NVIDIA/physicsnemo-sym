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

""" PhysicsNeMo Dataset constructors for discrete type data
"""

from pathlib import Path
from typing import Union, Dict, List

import numpy as np
import h5py

from physicsnemo.sym.utils.io.vtk import grid_to_vtk
from physicsnemo.sym.dataset.dataset import Dataset, _DictDatasetMixin


class _DictGridDatasetMixin(_DictDatasetMixin):
    "Special mixin class for dealing with dictionaries as input"

    def save_dataset(self, filename):

        named_lambda_weighting = {
            "lambda_" + key: value for key, value in self.lambda_weighting.items()
        }
        save_var = {**self.invar, **self.outvar, **named_lambda_weighting}
        grid_to_vtk(filename, save_var)  # Structured grid output in future


class DictGridDataset(_DictGridDatasetMixin, Dataset):
    """Default map-style grid dataset

    Parameters
    ----------
    invar : Dict[str, np.array]
        Dictionary of numpy arrays as input. Input arrays should be of form [B, cin, xdim, ...]
    outvar : Dict[str, np.array]
        Dictionary of numpy arrays as target outputs. Target arrays should be of form [B, cin, xdim, ...]
    lambda_weighting : Dict[str, np.array], optional
        The weighting of the each example, by default None
    """

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


class HDF5GridDataset(Dataset):
    """lazy-loading HDF5 map-style grid dataset"""

    def __init__(
        self,
        filename: Union[str, Path],
        invar_keys: List[str],
        outvar_keys: List[str],
        n_examples: int = None,
    ):

        self._invar_keys = invar_keys
        self._outvar_keys = outvar_keys
        self.path = Path(filename)

        # check path
        assert self.path.is_file(), f"Could not find file {self.path}"
        assert self.path.suffix in [
            ".h5",
            ".hdf5",
        ], f"File type should be HDF5, got {self.path.suffix}"

        # check dataset/ get length
        with h5py.File(self.path, "r") as f:

            # check keys exist
            for k in invar_keys + outvar_keys:
                if not k in f.keys():
                    raise KeyError(f"Variable {k} not found in HDF5 file")

            length = len(f[k])

        if n_examples is not None:
            assert (
                n_examples <= length
            ), "error, n_examples greater than length of file data"
            length = min(n_examples, length)

        self.length = length

    def __getitem__(self, idx):
        invar = Dataset._to_tensor_dict(
            {k: self.f[k][idx, ...] for k in self.invar_keys}
        )
        outvar = Dataset._to_tensor_dict(
            {k: self.f[k][idx, ...] for k in self.outvar_keys}
        )
        lambda_weighting = Dataset._to_tensor_dict(
            {k: np.ones_like(v) for k, v in outvar.items()}
        )
        return invar, outvar, lambda_weighting

    def __len__(self):
        return self.length

    def worker_init_fn(self, iworker):
        super().worker_init_fn(iworker)
        # open file on worker thread
        # note each torch DataLoader worker process should open file individually when reading
        # do not share open file descriptors across separate workers!
        # note files are closed when worker process is destroyed so no need to explicitly close
        self.f = h5py.File(self.path, "r")

    @property
    def invar_keys(self):
        return list(self._invar_keys)

    @property
    def outvar_keys(self):
        return list(self._outvar_keys)
