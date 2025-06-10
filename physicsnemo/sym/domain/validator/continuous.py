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

import numpy as np
import torch

from typing import List, Dict
from pathlib import Path

from physicsnemo.sym.domain.validator import Validator
from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk, VTKBase
from physicsnemo.sym.utils.io import ValidatorPlotter
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.constants import TF_SUMMARY
from physicsnemo.sym.dataset import DictPointwiseDataset
from physicsnemo.sym.distributed import DistributedManager


class PointwiseValidator(Validator):
    """
    Pointwise Validator that allows walidating on pointwise data

    Parameters
    ----------
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    invar : Dict[str, np.ndarray (N, 1)]
        Dictionary of numpy arrays as input.
    true_outvar : Dict[str, np.ndarray (N, 1)]
        Dictionary of numpy arrays used to validate against validation.
    batch_size : int, optional
            Batch size used when running validation, by default 1024
    plotter : ValidatorPlotter
        PhysicsNeMo plotter for showing results in tensorboard.
    requires_grad : bool = False
        If automatic differentiation is needed for computing results.
    """

    def __init__(
        self,
        nodes: List[Node],
        invar: Dict[str, np.array],
        true_outvar: Dict[str, np.array],
        batch_size: int = 1024,
        plotter: ValidatorPlotter = None,
        requires_grad: bool = False,
    ):

        # TODO: add support for other datasets?

        # get dataset and dataloader
        self.dataset = DictPointwiseDataset(invar=invar, outvar=true_outvar)
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
        # TODO add metrics specific for validation
        # TODO: add potential support for lambda_weighting
        losses = PointwiseValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)

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
            var_to_polyvtk(
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


class PointVTKValidator(PointwiseValidator):
    """
    Pointwise validator using mesh points of VTK object

    Parameters
    ----------
    vtk_obj : VTKBase
        PhysicsNeMo VTK object to use point locations from
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    input_vtk_map : Dict[str, List[str]]
        Dictionary mapping from PhysicsNeMo input variables to VTK variable names {"physicsnemo.sym.name": ["vtk name"]}.
        Use colons to denote components of multi-dimensional VTK arrays ("name":# )
    true_vtk_map : Dict[str, List[str]]
        Dictionary mapping from PhysicsNeMo target variables to VTK variable names {"physicsnemo.sym.name": ["vtk name"]}.
    invar : Dict[str, np.array], optional
        Dictionary of additional numpy arrays as input, by default {}
    true_outvar : Dict[str, np.array], optional
        Dictionary of additional numpy arrays used to validate against validation, by default {}
    batch_size : int
        Batch size used when running validation.
    plotter : ValidatorPlotter
        PhysicsNeMo plotter for showing results in tensorboard.
    requires_grad : bool, optional
        If automatic differentiation is needed for computing results., by default True
    log_iter : bool, optional
        Save results to different file each call, by default False
    """

    def __init__(
        self,
        vtk_obj: VTKBase,
        nodes: List[Node],
        input_vtk_map: Dict[str, List[str]],
        true_vtk_map: Dict[str, List[str]],
        invar: Dict[str, np.array] = {},  # Additional inputs
        true_outvar: Dict[str, np.array] = {},  # Additional targets
        batch_size: int = 1024,
        plotter: ValidatorPlotter = None,
        requires_grad: bool = False,
        log_iter: bool = False,
    ):
        # Set VTK file save dir and file name
        self.vtk_obj = vtk_obj
        self.vtk_obj.file_dir = "./validators"
        self.vtk_obj.file_name = "validator"

        # Set up input/output names
        invar_vtk = self.vtk_obj.get_data_from_map(input_vtk_map)
        invar.update(invar_vtk)
        # Extract true vars from VTK
        true_vtk = self.vtk_obj.get_data_from_map(true_vtk_map)
        true_outvar.update(true_vtk)
        # set plotter
        self.plotter = plotter
        self.log_iter = log_iter

        # initialize inferencer
        super().__init__(
            nodes=nodes,
            invar=invar,
            true_outvar=true_outvar,
            batch_size=batch_size,
            plotter=plotter,
            requires_grad=requires_grad,
        )

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
        # TODO add metrics specific for validation
        # TODO: add potential support for lambda_weighting
        losses = PointwiseValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file TODO clean this up after graph unroll stuff
        named_true_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}

        # save batch to vtk/npz file TODO clean this up after graph unroll stuff
        self.vtk_obj.file_dir = Path(results_dir)
        self.vtk_obj.file_name = Path(name).stem
        if "np" in save_filetypes:
            np.savez(
                results_dir + name, {**invar, **named_true_outvar, **named_pred_outvar}
            )
        if "vtk" in save_filetypes:
            if self.log_iter:
                self.vtk_obj.var_to_vtk(data_vars={**pred_outvar}, step=step)
            else:
                self.vtk_obj.var_to_vtk(data_vars={**pred_outvar})

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
