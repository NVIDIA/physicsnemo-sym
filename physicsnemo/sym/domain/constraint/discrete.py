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

""" Continuous type constraints
"""

import logging
from typing import Dict, List, Union

import torch
from torch.nn.parallel import DistributedDataParallel
import numpy as np

from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.loss import Loss, PointwiseLossNorm
from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.utils.io.vtk import grid_to_vtk
from physicsnemo.sym.dataset import Dataset, IterableDataset, DictGridDataset

logger = logging.getLogger(__name__)


class SupervisedGridConstraint(Constraint):
    """Data-driven grid field constraint

    Parameters
    ----------
    nodes : List[Node]
            List of PhysicsNeMo Nodes to unroll graph with.
    dataset: Union[Dataset, IterableDataset]
        dataset which supplies invar and outvar examples
        Must be a subclass of Dataset or IterableDataset
    loss : Loss, optional
        PhysicsNeMo `Loss` function, by default PointwiseLossNorm()
    batch_size : int, optional
        Batch size used when running constraint, must be specified if Dataset used
        Not used if IterableDataset used
    shuffle : bool, optional
        Randomly shuffle examples in dataset every epoch, by default True
        Not used if IterableDataset used
    drop_last : bool, optional
        Drop last mini-batch if dataset not fully divisible but batch_size, by default False
        Not used if IterableDataset used
    num_workers : int, optional
        Number of dataloader workers, by default 0
    """

    def __init__(
        self,
        nodes: List[Node],
        dataset: Union[Dataset, IterableDataset],
        loss: Loss = PointwiseLossNorm(),
        batch_size: int = None,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):

        super().__init__(
            nodes=nodes,
            dataset=dataset,
            loss=loss,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    def save_batch(self, filename):
        # sample batch
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        invar0 = {key: value for key, value in invar.items()}
        invar = Constraint._set_device(invar, device=self.device, requires_grad=True)
        true_outvar = Constraint._set_device(true_outvar, device=self.device)
        lambda_weighting = Constraint._set_device(lambda_weighting, device=self.device)

        # If using DDP, strip out collective stuff to prevent deadlocks
        # This only works either when one process alone calls in to save_batch
        # or when multiple processes independently save data
        if hasattr(self.model, "module"):
            modl = self.model.module
        else:
            modl = self.model

        # compute pred outvar
        pred_outvar = modl(invar)

        # rename values and save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + key: value for key, value in true_outvar.items()}
        named_pred_outvar = {"pred_" + key: value for key, value in pred_outvar.items()}
        save_var = {
            **{key: value for key, value in invar0.items()},
            **named_true_outvar,
            **named_pred_outvar,
        }
        save_var = {
            key: value.cpu().detach().numpy() for key, value in save_var.items()
        }

        model_parallel_rank = (
            self.manager.group_rank("model_parallel") if self.manager.distributed else 0
        )

        # Using - as delimiter here since vtk ignores anything after .
        grid_to_vtk(save_var, filename + f"-{model_parallel_rank}")

    def load_data(self):
        # get train points from dataloader
        invar, true_outvar, lambda_weighting = next(self.dataloader)

        self._input_vars = Constraint._set_device(
            invar, device=self.device, requires_grad=True
        )
        self._target_vars = Constraint._set_device(true_outvar, device=self.device)
        self._lambda_weighting = Constraint._set_device(
            lambda_weighting, device=self.device
        )

    def load_data_static(self):
        if self._input_vars is None:
            # Default loading if vars not allocated
            self.load_data()
        else:
            # get train points from dataloader
            invar, true_outvar, lambda_weighting = next(self.dataloader)
            # Set grads to false here for inputs, static var has allocation already
            input_vars = Constraint._set_device(
                invar, device=self.device, requires_grad=False
            )
            target_vars = Constraint._set_device(true_outvar, device=self.device)
            lambda_weighting = Constraint._set_device(
                lambda_weighting, device=self.device
            )

            for key in input_vars.keys():
                self._input_vars[key].data.copy_(input_vars[key])
            for key in target_vars.keys():
                self._target_vars[key].copy_(target_vars[key])
            for key in lambda_weighting.keys():
                self._lambda_weighting[key].copy_(lambda_weighting[key])

    def forward(self):
        # compute pred outvar
        self._output_vars = self.model(self._input_vars)

    def loss(self, step: int) -> Dict[str, torch.Tensor]:
        if self._output_vars is None:
            logger.warn("Calling loss without forward call")
            return {}

        losses = self._loss(
            self._input_vars,
            self._output_vars,
            self._target_vars,
            self._lambda_weighting,
            step,
        )

        return losses


class _DeepONetConstraint(Constraint):
    def __init__(
        self,
        nodes: List[Node],
        invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        lambda_weighting: Dict[str, np.array],
        loss: Loss,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
    ):

        # TODO: add support for other datasets (like SupervisedGridConstraint)

        # get dataset and dataloader
        self.dataset = DictGridDataset(
            invar=invar_branch, outvar=outvar, lambda_weighting=lambda_weighting
        )
        # Get DDP manager
        self.manager = DistributedManager()
        self.device = self.manager.device
        if not drop_last and self.manager.cuda_graphs:
            logger.info("drop_last must be true when using cuda graphs")
            drop_last = True

        self.dataloader = iter(
            Constraint.get_dataloader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
            )
        )

        # construct model from nodes
        self.model = Graph(
            nodes,
            Key.convert_list(invar_branch.keys())
            + Key.convert_list(invar_trunk.keys()),
            Key.convert_list(outvar.keys()),
        )
        self.model.to(self.device)
        if self.manager.distributed:
            # https://pytorch.org/docs/master/notes/cuda.html#id5
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.manager.local_rank],
                    output_device=self.device,
                    broadcast_buffers=self.manager.broadcast_buffers,
                    find_unused_parameters=self.manager.find_unused_parameters,
                    process_group=self.manager.group(
                        "data_parallel"
                    ),  # None by default
                )
            torch.cuda.current_stream().wait_stream(s)

        self._input_names = Key.convert_list(self.dataset.invar_keys)
        self._output_names = Key.convert_list(self.dataset.outvar_keys)

        self._input_vars_branch = None
        self._target_vars = None
        self._lambda_weighting = None

        # put loss on device
        self._loss = loss.to(self.device)

    def save_batch(self, filename):
        # sample batch
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        invar0 = {key: value for key, value in invar.items()}
        invar = Constraint._set_device(invar, device=self.device, requires_grad=True)
        true_outvar = Constraint._set_device(true_outvar, device=self.device)
        lambda_weighting = Constraint._set_device(lambda_weighting, device=self.device)

        # If using DDP, strip out collective stuff to prevent deadlocks
        # This only works either when one process alone calls in to save_batch
        # or when multiple processes independently save data
        if hasattr(self.model, "module"):
            modl = self.model.module
        else:
            modl = self.model

        # compute pred outvar
        pred_outvar = modl({**invar, **self._input_vars_trunk})
        # rename values and save batch to vtk file TODO clean this up after graph unroll stuff
        named_lambda_weighting = {
            "lambda_" + key: value for key, value in lambda_weighting.items()
        }
        named_true_outvar = {"true_" + key: value for key, value in true_outvar.items()}
        named_pred_outvar = {"pred_" + key: value for key, value in pred_outvar.items()}
        save_var = {
            **{key: value for key, value in invar0.items()},
            **named_true_outvar,
            **named_pred_outvar,
            **named_lambda_weighting,
        }
        save_var = {
            key: value.cpu().detach().numpy() for key, value in save_var.items()
        }

        model_parallel_rank = (
            self.manager.group_rank("model_parallel") if self.manager.distributed else 0
        )

        np.savez_compressed(filename + f".{model_parallel_rank}.npz", **save_var)

    def load_data(self):
        # get train points from dataloader
        invar, true_outvar, lambda_weighting = next(self.dataloader)

        self._input_vars_branch = Constraint._set_device(
            invar, device=self.device, requires_grad=True
        )
        self._target_vars = Constraint._set_device(true_outvar, device=self.device)
        self._lambda_weighting = Constraint._set_device(
            lambda_weighting, device=self.device
        )

    def load_data_static(self):
        if self._input_vars_branch is None:
            # Default loading if vars not allocated
            self.load_data()
        else:
            # get train points from dataloader
            invar, true_outvar, lambda_weighting = next(self.dataloader)
            # Set grads to false here for inputs, static var has allocation already
            input_vars = Constraint._set_device(
                invar, device=self.device, requires_grad=False
            )
            target_vars = Constraint._set_device(true_outvar, device=self.device)
            lambda_weighting = Constraint._set_device(
                lambda_weighting, device=self.device
            )

            for key in input_vars.keys():
                self._input_vars_branch[key].data.copy_(input_vars[key])
            for key in target_vars.keys():
                self._target_vars[key].copy_(target_vars[key])
            for key in lambda_weighting.keys():
                self._lambda_weighting[key].copy_(lambda_weighting[key])

    def forward(self):
        # compute pred outvar
        self._output_vars = self.model(
            {**self._input_vars_branch, **self._input_vars_trunk}
        )


class DeepONetConstraint_Data(_DeepONetConstraint):
    def __init__(
        self,
        nodes: List[Node],
        invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        lambda_weighting: Dict[str, np.array] = None,
        loss: Loss = PointwiseLossNorm(),
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):

        super().__init__(
            nodes=nodes,
            invar_branch=invar_branch,
            invar_trunk=invar_trunk,
            outvar=outvar,
            batch_size=batch_size,
            lambda_weighting=lambda_weighting,
            loss=loss,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        self._input_vars_trunk = Constraint._set_device(
            invar_trunk, device=self.device, requires_grad=True
        )

    def loss(self, step: int):

        # compute loss
        losses = self._loss(
            self._input_vars_trunk,
            self._output_vars,
            self._target_vars,
            self._lambda_weighting,
            step,
        )
        return losses


class DeepONetConstraint_Physics(_DeepONetConstraint):
    def __init__(
        self,
        nodes: List[Node],
        invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        lambda_weighting: Dict[str, np.array] = None,
        loss: Loss = PointwiseLossNorm(),
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        tile_trunk_input: bool = True,
    ):

        super().__init__(
            nodes=nodes,
            invar_branch=invar_branch,
            invar_trunk=invar_trunk,
            outvar=outvar,
            batch_size=batch_size,
            lambda_weighting=lambda_weighting,
            loss=loss,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        if tile_trunk_input:
            for k, v in invar_trunk.items():
                invar_trunk[k] = np.tile(v, (batch_size, 1))

        self._input_vars_trunk = Constraint._set_device(
            invar_trunk, device=self.device, requires_grad=True
        )

    def loss(self, step: int):

        target_vars = {
            k: torch.reshape(v, (-1, 1)) for k, v in self._target_vars.items()
        }
        lambda_weighting = {
            k: torch.reshape(v, (-1, 1)) for k, v in self._lambda_weighting.items()
        }

        # compute loss
        losses = self._loss(
            self._input_vars_trunk,
            self._output_vars,
            target_vars,
            lambda_weighting,
            step,
        )
        return losses
