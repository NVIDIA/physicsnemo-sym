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

# Script to train Fourcastnet on ERA5
# Ref: https://arxiv.org/abs/2202.11214

from warnings import warn

warn(
    f"This example will be depricated soon! Please switch to the FourCastNet recipe from PhysicsNeMo Launch repo.",
    DeprecationWarning,
)

import logging

import physicsnemo.sym

from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import Constraint, SupervisedGridConstraint
from physicsnemo.sym.domain.constraint.constraint import InfiniteDataLoader
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.utils.io import GridValidatorPlotter

from src.dali_dataset import ERA5HDF5GridDaliIterableDataset
from src.dataset import ERA5HDF5GridDataset
from src.fourcastnet import FourcastNetArch
from src.loss import LpLoss

logger = logging.getLogger(__name__)


@physicsnemo.sym.main(config_path="conf", config_name="config_FCN")
def run(cfg: PhysicsNeMoConfig) -> None:
    # load training/ test data
    channels = list(range(cfg.custom.n_channels))
    train_dataset = _create_dataset(
        cfg.custom.train_dataset.kind,
        data_dir=cfg.custom.train_dataset.data_path,
        chans=channels,
        tstep=cfg.custom.tstep,
        n_tsteps=cfg.custom.n_tsteps,
        patch_size=cfg.arch.afno.patch_size,
        batch_size=cfg.batch_size.grid,
        num_workers=cfg.custom.num_workers.grid,
        shuffle=True,
    )

    test_dataset = _create_dataset(
        cfg.custom.test_dataset.kind,
        data_dir=cfg.custom.test_dataset.data_path,
        chans=channels,
        tstep=cfg.custom.tstep,
        n_tsteps=cfg.custom.n_tsteps,
        patch_size=cfg.arch.afno.patch_size,
        n_samples_per_year=20,
        batch_size=cfg.batch_size.validation,
        num_workers=cfg.custom.num_workers.validation,
    )

    # Dataloader factory method needs to be updated before creating any constraints.
    update_get_dataloader()

    # define input/output keys
    input_keys = [Key(k, size=train_dataset.nchans) for k in train_dataset.invar_keys]
    output_keys = [Key(k, size=train_dataset.nchans) for k in train_dataset.outvar_keys]

    # make list of nodes to unroll graph on
    model = FourcastNetArch(
        input_keys=input_keys,
        output_keys=output_keys,
        img_shape=test_dataset.img_shape,
        patch_size=cfg.arch.afno.patch_size,
        embed_dim=cfg.arch.afno.embed_dim,
        depth=cfg.arch.afno.depth,
        num_blocks=cfg.arch.afno.num_blocks,
    )
    nodes = [model.make_node(name="FCN")]

    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
        loss=LpLoss(),
        num_workers=cfg.custom.num_workers.grid,
    )
    domain.add_constraint(supervised, "supervised")

    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        num_workers=cfg.custom.num_workers.validation,
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


def _create_dataset(dataset_kind: str, **kwargs):
    valid_dsets = {
        "default": ERA5HDF5GridDataset,
        "dali": ERA5HDF5GridDaliIterableDataset,
    }

    dset_cls = valid_dsets.get(dataset_kind, None)
    if dset_cls is None:
        raise ValueError(
            f"Expected one of {list(valid_dsets.keys())}, but got {dataset_kind}"
        )

    logger.info(f"Dataset: {dset_cls.__name__}")
    return dset_cls(**kwargs)


def update_get_dataloader():
    """Monkey-patch Constraint.get_dataloader method.

    DALI has its own multi-process worker functionality, similar to PyTorch DataLoader.
    This function patches Constraint.get_dataloader to avoid wrapping DALI dataset
    with another, redundant, layer of DataLoader.
    """
    default_get_dataloader = Constraint.get_dataloader

    def get_dataloader(
        dataset: "Union[Dataset, IterableDataset]",
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        distributed: bool = None,
        infinite: bool = True,
    ):
        if isinstance(dataset, ERA5HDF5GridDaliIterableDataset):
            if infinite:
                dataset = InfiniteDataLoader(dataset)
            return dataset

        return default_get_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            distributed=distributed,
            infinite=infinite,
        )

    Constraint.get_dataloader = get_dataloader


if __name__ == "__main__":
    run()
