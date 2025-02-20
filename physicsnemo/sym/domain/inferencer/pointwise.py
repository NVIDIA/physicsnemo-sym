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

from typing import Dict, List, Union, Callable
from pathlib import Path
import inspect

import torch
import numpy as np

from physicsnemo.sym.domain.inferencer import Inferencer
from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.utils.io import InferencerPlotter
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk, VTKBase, VTKUniformGrid
from physicsnemo.sym.dataset import DictInferencePointwiseDataset


class PointwiseInferencer(Inferencer):
    """
    Pointwise Inferencer that allows inferencing on pointwise data

    Parameters
    ----------
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    invar : Dict[str, np.ndarray (N, 1)]
        Dictionary of numpy arrays as input.
    output_names : List[str]
        List of desired outputs.
    batch_size : int, optional
            Batch size used when running validation, by default 1024
    plotter : InferencerPlotter
        PhysicsNeMo plotter for showing results in tensorboard.
    requires_grad : bool = False
        If automatic differentiation is needed for computing results.
    """

    def __init__(
        self,
        nodes: List[Node],
        invar: Dict[str, np.array],
        output_names: List[str],
        batch_size: int = 1024,
        plotter: InferencerPlotter = None,
        requires_grad: bool = False,
        model=None,
    ):

        # get dataset and dataloader
        self.dataset = DictInferencePointwiseDataset(
            invar=invar, output_names=output_names
        )
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
        if model is None:
            self.model = Graph(
                nodes,
                Key.convert_list(self.dataset.invar_keys),
                Key.convert_list(self.dataset.outvar_keys),
            )
        else:
            self.model = model
        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)

        # set foward method
        self.requires_grad = requires_grad
        self.forward = self.forward_grad if requires_grad else self.forward_nograd

        # set plotter
        self.plotter = plotter

    def eval_epoch(self):
        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        predvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0,) in enumerate(self.dataloader):
            # Move data to device
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            pred_outvar = self.forward(invar)

            invar_cpu = {key: value + [invar0[key]] for key, value in invar_cpu.items()}
            predvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach().numpy()]
                for key, value in predvar_cpu.items()
            }

        # Concat mini-batch arrays
        invar = {key: np.concatenate(value) for key, value in invar_cpu.items()}
        predvar = {key: np.concatenate(value) for key, value in predvar_cpu.items()}
        return invar, predvar

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        # evaluate on entire dataset
        invar, predvar = self.eval_epoch()

        # save batch to vtk/np files TODO clean this up after graph unroll stuff
        if "np" in save_filetypes:
            np.savez(results_dir + name, {**invar, **predvar})
        if "vtk" in save_filetypes:
            var_to_polyvtk({**invar, **predvar}, results_dir + name)

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Inferencers", name, results_dir, writer, step, invar, predvar
            )
