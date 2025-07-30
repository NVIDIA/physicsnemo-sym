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

import h5py
import logging
import numpy as np

from typing import List
from pathlib import Path

from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.dataset import Dataset


class ERA5HDF5GridBaseDataset:
    """Lazy-loading ERA5 dataset.
    Provides common implementation that is used in map- or iterable-style datasets.

    Parameters
    ----------
    data_dir : str
        Directory where ERA5 data is stored
    chans : List[int]
        Defines which ERA5 variables to load
    tstep : int
        Defines the size of the timestep between the input and output variables
    n_tsteps : int, optional
        Defines how many timesteps are included in the output variables
        Default is 1
    patch_size : int, optional
        If specified, crops input and output variables so image dimensions are
        divisible by patch_size
        Default is None
    n_samples_per_year : int, optional
        If specified, randomly selects n_samples_per_year samples from each year
        rather than all of the samples per year
        Default is None
    stats_dir : str, optional
        Directory to test data statistic numpy files that have the global mean and variance
    """

    def __init__(
        self,
        data_dir: str,
        chans: List[int],
        tstep: int = 1,
        n_tsteps: int = 1,
        patch_size: int = None,
        n_samples_per_year: int = None,
        stats_dir: str = None,
        **kwargs,
    ):
        self.data_dir = Path(to_absolute_path(data_dir))
        self.chans = chans
        self.nchans = len(self.chans)
        self.tstep = tstep
        self.n_tsteps = n_tsteps
        self.patch_size = patch_size
        self.n_samples_per_year = n_samples_per_year

        if stats_dir is None:
            self.stats_dir = self.data_dir.parent / "stats"

        # check root directory exists
        assert self.data_dir.is_dir(), (
            f"Error, data directory {self.data_dir} does not exist"
        )
        assert self.stats_dir.is_dir(), (
            f"Error, stats directory {self.stats_dir} does not exist"
        )

        # get all input data files
        self.data_paths = sorted(self.data_dir.glob("????.h5"))
        for data_path in self.data_paths:
            logging.info(f"ERA5 file found: {data_path}")
        self.n_years = len(self.data_paths)
        logging.info(f"Number of years: {self.n_years}")

        # get total number of examples and image shape from the first file,
        # assuming other files have exactly the same format.
        logging.info(f"Getting file stats from {self.data_paths[0]}")
        with h5py.File(self.data_paths[0], "r") as f:
            self.n_samples_per_year_all = f["fields"].shape[0]
            self.img_shape = f["fields"].shape[2:]
            logging.info(f"Number of channels available: {f['fields'].shape[1]}")

        # get example indices to use
        if self.n_samples_per_year is None:
            self.n_samples_per_year = self.n_samples_per_year_all
            self.samples = [
                np.arange(self.n_samples_per_year) for _ in range(self.n_years)
            ]
        else:
            if self.n_samples_per_year > self.n_samples_per_year_all:
                raise ValueError(
                    f"n_samples_per_year ({self.n_samples_per_year}) > number of samples available ({self.n_samples_per_year_all})!"
                )
            self.samples = [
                np.random.choice(
                    np.arange(self.n_samples_per_year_all),
                    self.n_samples_per_year,
                    replace=False,
                )
                for _ in range(self.n_years)
            ]
        logging.info(f"Number of samples/year: {self.n_samples_per_year}")

        # get total length
        self.length = self.n_years * self.n_samples_per_year

        # adjust image shape if patch_size defined
        if self.patch_size is not None:
            self.img_shape = [s - s % self.patch_size for s in self.img_shape]
        logging.info(f"Input image shape: {self.img_shape}")

        # load normalisation values
        # has shape [1, C, 1, 1]
        self.mu = np.load(self.stats_dir / "global_means.npy")[:, self.chans]
        # has shape [1, C, 1, 1]
        self.sd = np.load(self.stats_dir / "global_stds.npy")[:, self.chans]
        assert self.mu.shape == self.sd.shape == (1, self.nchans, 1, 1), (
            "Error, normalisation arrays have wrong shape"
        )

    @property
    def invar_keys(self):
        return ["x_t0"]

    @property
    def outvar_keys(self):
        return [f"x_t{(i + 1) * self.tstep}" for i in range(self.n_tsteps)]


class ERA5HDF5GridDataset(ERA5HDF5GridBaseDataset, Dataset):
    """Map-style ERA5 dataset."""

    def __getitem__(self, idx):
        # get local indices from global index
        year_idx = int(idx / self.n_samples_per_year)
        local_idx = int(idx % self.n_samples_per_year)
        in_idx = self.samples[year_idx][local_idx]

        # get output indices
        out_idxs = []
        for i in range(self.n_tsteps):
            out_idx = in_idx + (i + 1) * self.tstep
            # if at end of dataset, just learn identity instead
            if out_idx > (self.n_samples_per_year_all - 1):
                out_idx = in_idx
            out_idxs.append(out_idx)

        # get data
        xs = []
        for idx in [in_idx] + out_idxs:
            # get array
            # has shape [C, H, W]
            x = self.data_files[year_idx]["fields"][idx, self.chans]
            assert x.ndim == 3, f"Expected 3 dimensions, but got {x.shape}"

            # apply input / output normalisation (broadcasted operation)
            x = (x - self.mu[0]) / self.sd[0]

            # crop data if needed
            if self.patch_size is not None:
                x = x[..., : self.img_shape[0], : self.img_shape[1]]

            xs.append(x)

        # convert to tensor dicts
        assert len(self.invar_keys) == 1
        invar = {self.invar_keys[0]: xs[0]}

        assert len(self.outvar_keys) == len(xs) - 1
        outvar = {self.outvar_keys[i]: x for i, x in enumerate(xs[1:])}

        invar = Dataset._to_tensor_dict(invar)
        outvar = Dataset._to_tensor_dict(outvar)

        lambda_weighting = Dataset._to_tensor_dict(
            {k: np.ones_like(v) for k, v in outvar.items()}
        )

        return invar, outvar, lambda_weighting

    def __len__(self):
        return self.length

    def worker_init_fn(self, iworker):
        super().worker_init_fn(iworker)

        # open all year files at once on worker thread
        self.data_files = [h5py.File(path, "r") for path in self.data_paths]
