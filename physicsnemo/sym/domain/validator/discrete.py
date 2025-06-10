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

from typing import Dict, List

import torch
import numpy as np

from physicsnemo.sym.domain.validator import Validator
from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.utils.io.vtk import grid_to_vtk
from physicsnemo.sym.utils.io import GridValidatorPlotter, DeepONetValidatorPlotter
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.constants import TF_SUMMARY
from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.dataset import Dataset, DictGridDataset


class GridValidator(Validator):
    """Data-driven grid field validator

    Parameters
    ----------
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    dataset: Dataset
        dataset which contains invar and true outvar examples
    batch_size : int, optional
            Batch size used when running validation, by default 100
    plotter : GridValidatorPlotter
        PhysicsNeMo plotter for showing results in tensorboard.
    requires_grad : bool = False
        If automatic differentiation is needed for computing results.
    num_workers : int, optional
        Number of dataloader workers, by default 0
    """

    def __init__(
        self,
        nodes: List[Node],
        dataset: Dataset,
        batch_size: int = 100,
        plotter: GridValidatorPlotter = None,
        requires_grad: bool = False,
        num_workers: int = 0,
    ):

        # get dataset and dataloader
        self.dataset = dataset
        self.dataloader = Constraint.get_dataloader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            distributed=False,
            infinite=False,
        )

        # construct model from nodes
        self.model = Graph(
            nodes,
            Key.convert_list(self.dataset.invar_keys),
            Key.convert_list(self.dataset.outvar_keys),
        )
        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)

        # set foward method
        self.requires_grad = requires_grad
        self.forward = self.forward_grad if requires_grad else self.forward_nograd

        # set plotter
        self.plotter = plotter

    def save_results(self, name, results_dir, writer, save_filetypes, step):

        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.dataloader):
            # Move data to device (may need gradients in future, if so requires_grad=True)
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            true_outvar = Constraint._set_device(
                true_outvar0, device=self.device, requires_grad=self.requires_grad
            )
            pred_outvar = self.forward(invar)

            # Collect minibatch info into cpu dictionaries
            invar_cpu = {
                key: value + [invar[key].cpu().detach()]
                for key, value in invar_cpu.items()
            }
            true_outvar_cpu = {
                key: value + [true_outvar[key].cpu().detach()]
                for key, value in true_outvar_cpu.items()
            }
            pred_outvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach()]
                for key, value in pred_outvar_cpu.items()
            }

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}
        true_outvar_cpu = {
            key: torch.cat(value) for key, value in true_outvar_cpu.items()
        }
        pred_outvar_cpu = {
            key: torch.cat(value) for key, value in pred_outvar_cpu.items()
        }
        # compute losses on cpu
        losses = GridValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}

        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_true_outvar, **named_pred_outvar}
            )
        if "vtk" in save_filetypes:

            grid_to_vtk(
                {**invar, **named_true_outvar, **named_pred_outvar}, results_dir + name
            )

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Validators",
                name,
                results_dir,
                writer,
                step,
                invar,
                true_outvar,
                pred_outvar,
            )

        # add tensorboard scalars
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar(
                    "Validators/" + name + "/" + k, loss, step, new_style=True
                )
        return losses


class _DeepONet_Validator(Validator):
    def __init__(
        self,
        nodes: List[Node],
        invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array],
        true_outvar: Dict[str, np.array],
        batch_size: int,
        plotter: DeepONetValidatorPlotter,
        requires_grad: bool,
    ):

        # TODO: add support for other datasets?

        # get dataset and dataloader
        self.dataset = DictGridDataset(invar=invar_branch, outvar=true_outvar)
        self.dataloader = Constraint.get_dataloader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            distributed=False,
            infinite=False,
        )

        # construct model from nodes
        self.model = Graph(
            nodes,
            Key.convert_list(invar_branch.keys())
            + Key.convert_list(invar_trunk.keys()),
            Key.convert_list(true_outvar.keys()),
        )
        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)

        # set foward method
        self.requires_grad = requires_grad
        self.forward = self.forward_grad if requires_grad else self.forward_nograd

        # set plotter
        self.plotter = plotter


class DeepONet_Physics_Validator(_DeepONet_Validator):
    """
    DeepONet Validator
    """

    def __init__(
        self,
        nodes: List[Node],
        invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array],
        true_outvar: Dict[str, np.array],
        batch_size: int = 100,
        plotter: DeepONetValidatorPlotter = None,
        requires_grad: bool = False,
        tile_trunk_input: bool = True,
    ):

        super().__init__(
            nodes=nodes,
            invar_branch=invar_branch,
            invar_trunk=invar_trunk,
            true_outvar=true_outvar,
            batch_size=batch_size,
            plotter=plotter,
            requires_grad=requires_grad,
        )

        if tile_trunk_input:
            for k, v in invar_trunk.items():
                invar_trunk[k] = np.tile(v, (batch_size, 1))
        self.invar_trunk = invar_trunk
        self.batch_size = batch_size

    def save_results(self, name, results_dir, writer, save_filetypes, step):

        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        invar_trunk_gpu = Constraint._set_device(
            self.invar_trunk, device=self.device, requires_grad=self.requires_grad
        )
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.dataloader):
            # Move data to device (may need gradients in future, if so requires_grad=True)
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            true_outvar = Constraint._set_device(
                true_outvar0, device=self.device, requires_grad=self.requires_grad
            )
            pred_outvar = self.forward({**invar, **invar_trunk_gpu})

            # Collect minibatch info into cpu dictionaries
            invar_cpu = {
                key: value + [invar[key].cpu().detach()]
                for key, value in invar_cpu.items()
            }
            true_outvar_cpu = {
                key: value + [true_outvar[key].cpu().detach()]
                for key, value in true_outvar_cpu.items()
            }
            pred_outvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach()]
                for key, value in pred_outvar_cpu.items()
            }

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}
        true_outvar_cpu = {
            key: torch.cat(value) for key, value in true_outvar_cpu.items()
        }
        pred_outvar_cpu = {
            key: torch.cat(value) for key, value in pred_outvar_cpu.items()
        }
        # compute losses on cpu
        losses = DeepONet_Physics_Validator._l2_relative_error(
            true_outvar_cpu, pred_outvar_cpu
        )

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}

        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_true_outvar, **named_pred_outvar}
            )

        ndim = next(iter(self.invar_trunk.values())).shape[-1]
        invar_plotter = dict()
        true_outvar_plotter = dict()
        pred_outvar_plotter = dict()
        for k, v in self.invar_trunk.items():
            invar_plotter[k] = self.invar_trunk[k].reshape((self.batch_size, -1, ndim))

        for k, v in true_outvar.items():
            true_outvar_plotter[k] = true_outvar[k].reshape((self.batch_size, -1))

        for k, v in pred_outvar.items():
            pred_outvar_plotter[k] = pred_outvar[k].reshape((self.batch_size, -1))

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Validators",
                name,
                results_dir,
                writer,
                step,
                invar_plotter,
                true_outvar_plotter,
                pred_outvar_plotter,
            )

        # add tensorboard scalars
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar(
                    "Validators/" + name + "/" + k, loss, step, new_style=True
                )
        return losses

    @staticmethod
    def _l2_relative_error(true_var, pred_var):  # TODO replace with metric classes
        new_var = {}
        for key in true_var.keys():
            new_var["l2_relative_error_" + str(key)] = torch.sqrt(
                torch.mean(
                    torch.square(torch.reshape(true_var[key], (-1, 1)) - pred_var[key])
                )
                / torch.var(true_var[key])
            )
        return new_var


class DeepONet_Data_Validator(_DeepONet_Validator):
    """
    DeepONet Validator
    """

    def __init__(
        self,
        nodes: List[Node],
        invar_branch: Dict[str, np.array],
        invar_trunk: Dict[str, np.array],
        true_outvar: Dict[str, np.array],
        batch_size: int = 100,
        plotter: DeepONetValidatorPlotter = None,
        requires_grad: bool = False,
    ):

        super().__init__(
            nodes=nodes,
            invar_branch=invar_branch,
            invar_trunk=invar_trunk,
            true_outvar=true_outvar,
            batch_size=batch_size,
            plotter=plotter,
            requires_grad=requires_grad,
        )

        self.invar_trunk_plotter = dict()
        ndim = next(iter(invar_trunk.values())).shape[-1]
        for k, v in invar_trunk.items():
            self.invar_trunk_plotter[k] = np.tile(v, (batch_size, 1)).reshape(
                (batch_size, -1, ndim)
            )
        self.invar_trunk = invar_trunk

    def save_results(self, name, results_dir, writer, save_filetypes, step):

        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        invar_trunk_gpu = Constraint._set_device(
            self.invar_trunk, device=self.device, requires_grad=self.requires_grad
        )
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.dataloader):
            # Move data to device (may need gradients in future, if so requires_grad=True)
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            true_outvar = Constraint._set_device(
                true_outvar0, device=self.device, requires_grad=self.requires_grad
            )
            pred_outvar = self.forward({**invar, **invar_trunk_gpu})

            # Collect minibatch info into cpu dictionaries
            invar_cpu = {
                key: value + [invar[key].cpu().detach()]
                for key, value in invar_cpu.items()
            }
            true_outvar_cpu = {
                key: value + [true_outvar[key].cpu().detach()]
                for key, value in true_outvar_cpu.items()
            }
            pred_outvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach()]
                for key, value in pred_outvar_cpu.items()
            }

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}
        true_outvar_cpu = {
            key: torch.cat(value) for key, value in true_outvar_cpu.items()
        }
        pred_outvar_cpu = {
            key: torch.cat(value) for key, value in pred_outvar_cpu.items()
        }
        # compute losses on cpu
        losses = DeepONet_Data_Validator._l2_relative_error(
            true_outvar_cpu, pred_outvar_cpu
        )

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}

        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_true_outvar, **named_pred_outvar}
            )

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Validators",
                name,
                results_dir,
                writer,
                step,
                self.invar_trunk_plotter,
                true_outvar,
                pred_outvar,
            )

        # add tensorboard scalars
        for k, loss in losses.items():
            if TF_SUMMARY:
                writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)
            else:
                writer.add_scalar(
                    "Validators/" + name + "/" + k, loss, step, new_style=True
                )
        return losses
