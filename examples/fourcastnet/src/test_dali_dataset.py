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

from functools import partial
from pathlib import Path
import pytest
import shutil
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.domain.constraint.constraint import Constraint

from src.dali_dataset import ERA5HDF5GridDaliIterableDataset
from src.dataset import ERA5HDF5GridDataset


# TODO: hardcoded for now. Parameterize in the future.
NUM_SAMPLES = 4
NUM_CHANNELS = 3
IMG_HEIGHT = 17
IMG_WIDTH = 32


@pytest.fixture(scope="module")
def test_data(tmp_path_factory):
    """Creates a small data sample in ERA5-like format."""

    data_dir = tmp_path_factory.mktemp("data")
    train_dir = create_test_data(data_dir)

    yield train_dir

    # Cleanup.
    shutil.rmtree(data_dir)


def create_test_data(data_dir: Path):
    """Creates a test data in ERA5 format."""

    train_dir = data_dir / "train"
    train_dir.mkdir()
    stats_dir = data_dir / "stats"
    stats_dir.mkdir()

    # Create and write data.
    data = (
        np.random.default_rng(seed=1)
        .normal(0.0, 1.0, (NUM_SAMPLES, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH))
        .astype(np.float32)
    )
    with h5py.File(train_dir / "1980.h5", mode="w") as h5file:
        h5file["fields"] = data

    # Write stats.
    np.save(
        stats_dir / "global_means.npy", np.mean(data, axis=(0, 2, 3), keepdims=True)
    )
    np.save(stats_dir / "global_stds.npy", np.std(data, axis=(0, 2, 3), keepdims=True))

    return train_dir


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("n_tsteps", [1, 2])
def test_dali_dataset_basic(
    test_data: Path, batch_size: int, num_workers: int, n_tsteps: int
):
    """Basic test to verify DALI dataset functionality."""

    data_path = test_data
    channels = list(range(NUM_CHANNELS))
    tstep = 1
    patch_size = 8

    base_loader, base_dset = _create_default_dataloader(
        data_path,
        channels,
        tstep,
        n_tsteps,
        patch_size,
        batch_size,
    )

    dali_loader, dali_dset = _create_dali_dataloader(
        data_path,
        channels,
        tstep,
        n_tsteps,
        patch_size,
        batch_size,
        num_workers,
    )

    assert dali_dset.invar_keys == base_dset.invar_keys
    assert dali_dset.outvar_keys == base_dset.outvar_keys

    num_epochs = 2
    for _ in range(num_epochs):
        num_iters = 0
        for batch_base, batch_dali in zip(base_loader, dali_loader):
            invar_b, outvar_b, lw_b = batch_base
            invar_d, outvar_d, lw_d = (
                Constraint._set_device(i, "cpu") for i in batch_dali
            )

            # Check invars.
            assert torch.allclose(invar_d["x_t0"], invar_b["x_t0"])

            # Check outvars.
            assert len(outvar_d) == len(outvar_b)
            assert len(lw_d) == len(lw_b)

            for k in outvar_d.keys():
                assert torch.allclose(outvar_d[k], outvar_b[k])
                # Weights are consts, so should be exactly the same.
                assert (lw_d[k] == lw_b[k]).all()
            num_iters += 1
        assert num_iters == NUM_SAMPLES // batch_size


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_workers", [1, 2])
def test_dali_shuffle(test_data: Path, batch_size: int, num_workers: int):
    """Checks sample random shuffling functionality."""

    data_path = test_data
    n_tsteps = 1
    channels = list(range(NUM_CHANNELS))
    tstep = 1
    patch_size = 8

    dali_loader = partial(
        _create_dali_dataloader,
        data_path,
        channels,
        tstep,
        n_tsteps,
        patch_size,
        batch_size,
        num_workers,
    )

    base_loader, _ = dali_loader(shuffle=False)
    shuf_loader, _ = dali_loader(shuffle=True)

    num_epochs = 3
    # Shuffled indices for each epoch.
    epoch_indices = [
        [2, 0, 1, 3],
        [2, 0, 1, 3],
        [3, 1, 2, 0],
    ]
    for epoch in range(num_epochs):
        base_batches = list(base_loader)
        shuf_batches = list(shuf_loader)
        # Check that shuf_batches is a permutation of the original.
        x_t0_base = torch.cat([b[0]["x_t0"] for b in base_batches], dim=0)
        assert x_t0_base.size(0) == NUM_SAMPLES
        x_t0_shuf = torch.cat([b[0]["x_t0"] for b in shuf_batches], dim=0)
        assert x_t0_shuf.size(0) == NUM_SAMPLES

        for i in range(NUM_SAMPLES):
            dst_idx = epoch_indices[epoch][i]
            assert (x_t0_shuf[i] == x_t0_base[dst_idx]).all(), (
                f"Mismatch at epoch {epoch}, sample {i}."
            )


@pytest.mark.skip(reason="The test should be run using mpirun, not pytest.")
def test_distributed_dali_loader(data_path: Path):
    n_tsteps = 1
    channels = list(range(NUM_CHANNELS))
    tstep = 1
    patch_size = 8
    batch_size = 1
    num_workers = 1

    m = DistributedManager()
    world_size = m.world_size
    # TODO: temporary restriction, remove.
    assert world_size == 2, (
        "Only 2-GPU configuration is supported for now. Please run with mpirun -np 2"
    )

    base_loader, _ = _create_default_dataloader(
        data_path,
        channels,
        tstep,
        n_tsteps,
        patch_size,
        batch_size,
    )
    base_batches = list(base_loader)
    x_t0_base = torch.cat([b[0]["x_t0"] for b in base_batches], dim=0)
    # Make sure baseline contains all samples.
    assert x_t0_base.size(0) == NUM_SAMPLES

    dali_loader, _ = _create_dali_dataloader(
        data_path,
        channels,
        tstep,
        n_tsteps,
        patch_size,
        batch_size,
        num_workers,
    )

    num_samples_per_rank = NUM_SAMPLES // world_size

    dali_batches = list(dali_loader)
    x_t0_dali = torch.cat([b[0]["x_t0"] for b in dali_batches], dim=0)
    assert x_t0_dali.size(0) == num_samples_per_rank

    # Check the samples are distributed across ranks properly.
    idx_start = num_samples_per_rank * m.rank
    assert torch.allclose(
        x_t0_base[idx_start : idx_start + num_samples_per_rank], x_t0_dali.cpu()
    )


def _create_default_dataloader(
    data_path: Path,
    channels: List[int],
    tstep: int,
    n_tsteps: int,
    patch_size: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
):
    dataset = ERA5HDF5GridDataset(
        data_path,
        chans=channels,
        tstep=tstep,
        n_tsteps=n_tsteps,
        patch_size=patch_size,
    )

    # Similar to Constraint.get_dataloader.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
    )
    if num_workers == 0:
        dataset.worker_init_fn(0)

    return loader, dataset


def _create_dali_dataloader(
    data_path: Path,
    channels: List[int],
    tstep: int,
    n_tsteps: int,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
):
    dataset = ERA5HDF5GridDaliIterableDataset(
        data_path,
        chans=channels,
        tstep=tstep,
        n_tsteps=n_tsteps,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return dataset, dataset
