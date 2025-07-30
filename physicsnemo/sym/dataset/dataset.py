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

"""Dataset classes"""

from typing import Dict

import numpy as np
import torch.utils.data

from physicsnemo.sym.constants import tf_dt
from physicsnemo.sym.distributed import DistributedManager


class _BaseDataset:
    "Defines common requirements across map- and iterable- style datasets"

    def worker_init_fn(self, iworker):
        "Called by each worker in torch dataloader when it initialises"

        # get the distributed manager object
        manager = DistributedManager()
        worker_rank = manager.group_rank("data_parallel") if manager.distributed else 0
        worker_size = manager.group_size("data_parallel") if manager.distributed else 1

        # set different numpy seed per worker
        # set seed so first worker id's seed matches single-process case
        np.random.seed(seed=(worker_rank + iworker * worker_size))

    @property
    def invar_keys(self):
        "Return list of invar keys"

        raise NotImplementedError("subclass must implement this")

    @property
    def outvar_keys(self):
        "Return list of outvar keys"

        raise NotImplementedError("subclass must implement this")

    def save_dataset(self, filename):
        "Save dataset to file"

        raise NotImplementedError("subclass must implement this")

    @staticmethod
    def _to_tensor_dict(var_dict, device=None):
        # convert to torch
        tensor_dict = {
            key: torch.as_tensor(value, dtype=tf_dt, device=device)
            for key, value in var_dict.items()
        }

        return tensor_dict


class Dataset(_BaseDataset, torch.utils.data.Dataset):
    "For defining map-style datasets, can be subclassed by user"

    auto_collation = False

    def __getitem__(self, idx):
        """Must return a single example tuple e.g. (invar, outvar, lambda_weighting)
        if Dataset.auto_collation is False, or a batched example tuple if
        Dataset.auto_collation is True. For the latter case idx is a batch of indices.
        """

        raise NotImplementedError("subclass must implement this")

    def __len__(self):
        raise NotImplementedError("subclass must implement this")


class IterableDataset(_BaseDataset, torch.utils.data.IterableDataset):
    "For defining iterable-style datasets, can be subclassed by user"

    def __iter__(self):
        "Must yield batched example tuple e.g. (invar, outvar, lambda_weighting)"

        raise NotImplementedError("subclass must implement this")


class _DictDatasetMixin:
    "Special mixin class for dealing with dictionary-based datasets"

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        lambda_weighting: Dict[str, np.array] = None,
    ):
        # get default lambda weighting
        if lambda_weighting is None:
            lambda_weighting = {key: np.ones_like(x) for key, x in outvar.items()}

        # convert dataset arrays to tensors
        self.invar = Dataset._to_tensor_dict(invar)
        self.outvar = Dataset._to_tensor_dict(outvar)
        self.lambda_weighting = Dataset._to_tensor_dict(lambda_weighting)

        # get length
        self.length = len(next(iter(self.invar.values())))

    @property
    def invar_keys(self):
        return list(self.invar.keys())

    @property
    def outvar_keys(self):
        return list(self.outvar.keys())

    @staticmethod
    def _idx_var(var, idx):
        # index, idx can be an int or an array
        idx_var = {}
        for key, value in var.items():
            idx_var[key] = value[idx]
        return idx_var
