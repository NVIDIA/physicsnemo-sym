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

from typing import Iterable, List

import h5py
import numpy as np
import torch

from physicsnemo.sym.dataset import IterableDataset
from physicsnemo.sym.distributed import DistributedManager

from src.dataset import ERA5HDF5GridBaseDataset

try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as dali_pth
except ImportError:
    print(
        """DALI dataset requires NVIDIA DALI package to be installed.
The package can be installed by running:

pip install nvidia-dali-cuda110
"""
    )
    raise SystemExit(1)


class ERA5HDF5GridDaliIterableDataset(ERA5HDF5GridBaseDataset, IterableDataset):
    """ERA5 DALI iterable-style dataset."""

    def __init__(
        self,
        data_dir: str,
        chans: List[int],
        tstep: int = 1,
        n_tsteps: int = 1,
        patch_size: int = None,
        n_samples_per_year: int = None,
        stats_dir: str = None,
        batch_size: int = 1,
        num_workers: int = 1,
        shuffle: bool = False,
    ):
        super().__init__(
            data_dir, chans, tstep, n_tsteps, patch_size, n_samples_per_year, stats_dir
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.pipe = self._create_pipeline()

    def worker_init_fn(self, iworker):
        super().worker_init_fn(iworker)

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        dali_iter = dali_pth.DALIGenericIterator([self.pipe], ["invar", "outvar"])

        # Read batches.
        for batch_data in dali_iter:
            # Only one source is expected.
            assert len(batch_data) == 1
            batch = batch_data[0]

            invar = {self.invar_keys[0]: batch["invar"]}
            invar = self._to_tensor_dict(invar)

            outvar = batch["outvar"]
            # Should be [N,T,C,H,W] shape.
            assert outvar.ndim == 5
            outvar = {self.outvar_keys[t]: outvar[:, t] for t in range(self.n_tsteps)}
            outvar = self._to_tensor_dict(outvar)

            lambda_weighting = {k: torch.ones_like(v) for k, v in outvar.items()}

            yield invar, outvar, lambda_weighting

    def _create_pipeline(self) -> dali.Pipeline:
        # TODO: make num_threads and prefetch queue configurable?
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=DistributedManager().device.index,
            py_start_method="spawn",
        )

        with pipe:
            source = ERA5DaliExternalSource(
                self.data_paths,
                self.length,
                self.chans,
                self.n_tsteps,
                self.tstep,
                self.n_samples_per_year,
                self.batch_size,
                self.shuffle,
            )
            # Read current batch.
            invar, outvar = dali.fn.external_source(
                source,
                num_outputs=2,
                parallel=True,
                batch=False,
            )
            # Move tensors to GPU as external_source won't do that.
            invar = invar.gpu()
            outvar = outvar.gpu()
            # Crop.
            h, w = self.img_shape
            invar = invar[:, :h, :w]
            outvar = outvar[:, :, :h, :w]
            # Standardize.
            invar = dali.fn.normalize(invar, mean=self.mu[0], stddev=self.sd[0])
            outvar = dali.fn.normalize(outvar, mean=self.mu, stddev=self.sd)

            # Set outputs.
            pipe.set_outputs(invar, outvar)

        return pipe


class ERA5DaliExternalSource:
    """ERA5 DALI external callable source.

    For more information about DALI external source operator:
    https://docs.nvidia.com/deeplearning/dali/archives/dali_1_13_0/user-guide/docs/examples/general/data_loading/parallel_external_source.html
    """

    def __init__(
        self,
        data_paths: Iterable[str],
        num_samples: int,
        channels: Iterable[int],
        n_tsteps: int,
        tstep: int,
        n_samples_per_year: int,
        batch_size: int,
        shuffle: bool,
    ):
        self.data_paths = list(data_paths)
        # Will be populated later once each worker starts running in its own process.
        self.data_files = None
        self.num_samples = num_samples
        self.chans = list(channels)
        self.n_tsteps = n_tsteps
        self.tstep = tstep
        self.n_samples_per_year = n_samples_per_year
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.last_epoch = None

        self.indices = np.arange(num_samples)
        # If running in distributed mode, select appropriate shard from indices.
        m = DistributedManager()
        if m.distributed:
            # Each shard will get its own subset of indices (possibly empty).
            self.indices = np.array_split(self.indices, m.world_size)[m.rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        # Also, DALI external source does not support incomplete batches in parallel mode.
        self.num_batches = len(self.indices) // self.batch_size

    def __call__(self, sample_info: dali.types.SampleInfo):
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        if self.data_files is None:
            # This will be called once per worker. Workers are persistent,
            # so there is no need to explicitly close the files - this will be done
            # when corresponding pipeline/dataset is destroyed.
            self.data_files = [h5py.File(path, "r") for path in self.data_paths]

        # Shuffle before the next epoch starts.
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            # All workers use the same rng seed so the resulting
            # indices are the same across workers.
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index.
        idx = self.indices[sample_info.idx_in_epoch]
        year_idx = idx // self.n_samples_per_year
        in_idx = idx % self.n_samples_per_year
        #
        data = self.data_files[year_idx]["fields"]
        # Has [C,H,W] shape.
        invar = data[in_idx, self.chans]

        # Has [T,C,H,W] shape.
        outvar = np.empty((self.n_tsteps,) + invar.shape, dtype=invar.dtype)
        for i in range(self.n_tsteps):
            out_idx = in_idx + (i + 1) * self.tstep
            # If at end of dataset, just learn identity instead.
            if out_idx >= data.shape[0]:
                out_idx = in_idx
            outvar[i] = data[out_idx, self.chans]

        return invar, outvar
