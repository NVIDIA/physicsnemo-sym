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

from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.utils.io import InferencerPlotter
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk, VTKBase, VTKUniformGrid
from physicsnemo.sym.dataset import DictInferencePointwiseDataset


class PointVTKInferencer(PointwiseInferencer):
    """
    Pointwise inferencer using mesh points of VTK object

    Parameters
    ----------
    vtk_obj : VTKBase
        PhysicsNeMo VTK object to use point locations from
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    input_vtk_map : Dict[str, List[str]]
        Dictionary mapping from PhysicsNeMo input variables to VTK variable names {"physicsnemo.sym.name": ["vtk name"]}.
        Use colons to denote components of multi-dimensional VTK arrays ("name":# )
    output_names : List[str]
        List of desired outputs.
    invar : Dict[str, np.array], optional
        Dictionary of additional numpy arrays as input, by default {}
    batch_size : int
        Batch size used when running inference
    mask_fn : Union[Callable, None], optional
        Masking function to remove points from inferencing, by default None
    mask_value : float, optional
        Value to assign masked points, by default Nan
    plotter : Plotter, optional
        PhysicsNeMo `Plotter` for showing results in tensorboard., by default None
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
        output_names: List[str],
        invar: Dict[str, np.array] = {},  # Additional inputs
        batch_size: int = 1024,
        mask_fn: Union[Callable, None] = None,
        mask_value: float = np.nan,
        plotter=None,
        requires_grad: bool = False,
        log_iter: bool = False,
        model=None,
    ):
        # Set VTK file save dir and file name
        self.vtk_obj = vtk_obj
        self.vtk_obj.file_dir = "./inferencers"
        self.vtk_obj.file_name = "inferencer"

        # Set up input dict
        invar_vtk = self.vtk_obj.get_data_from_map(input_vtk_map)
        invar.update(invar_vtk)

        # If mask set up mask indexes
        self.mask_value = mask_value
        self.mask_index = None
        if mask_fn is not None:
            args, _, _, _ = inspect.getargspec(mask_fn)
            # Fall back np_lambdify does not supply arguement names
            # Ideally np_lambdify should allow input names to be queried
            if len(args) == 0:
                args = list(invar.keys())  # Hope your inputs all go into the mask
            mask_input = {key: invar[key] for key in args if key in invar}
            mask = np.squeeze(mask_fn(**mask_input).astype(np.bool_))
            # True points get masked while False get kept, flip for index
            self.mask_index = np.logical_not(mask)
            # Mask out to only masked points (only inference here)
            for key, value in invar.items():
                invar[key] = value[self.mask_index]

        # set plotter
        self.plotter = plotter
        self.log_iter = log_iter

        # initialize inferencer
        super().__init__(
            nodes=nodes,
            invar=invar,
            output_names=output_names,
            batch_size=batch_size,
            plotter=plotter,
            requires_grad=requires_grad,
            model=model,
        )

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        # Compute results
        invar, predvar = self._compute_results()

        # Reconstruct full array if mask was applied
        if self.mask_index is not None:
            invar, predvar = self._mask_results(invar, predvar)

        # Write results to file
        self._write_results(
            invar, predvar, name, results_dir, writer, save_filetypes, step
        )

    def save_stream(
        self, name, results_dir, writer, step, save_results, save_filetypes, to_cpu
    ):
        if not to_cpu:
            raise NotImplementedError("to_cpu=False not supported.")

        # Compute results
        invar, predvar = self._compute_results()

        # Reconstruct full array if mask was applied
        if self.mask_index is not None:
            invar, predvar = self._mask_results(invar, predvar)

        # Write results to file
        if save_results:
            self._write_results(
                invar, predvar, name, results_dir, writer, save_filetypes, step
            )
        return {**invar, **predvar}

    def _compute_results(self):
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

    def _mask_results(self, invar, predvar):
        # Reconstruct full array if mask was applied
        for key, value in invar.items():
            full_array = np.full(
                (self.mask_index.shape[0], value.shape[1]),
                self.mask_value,
                dtype=value.dtype,
            )
            full_array[self.mask_index] = value
            invar[key] = full_array
        for key, value in predvar.items():
            full_array = np.full(
                (self.mask_index.shape[0], value.shape[1]),
                self.mask_value,
                dtype=value.dtype,
            )
            full_array[self.mask_index] = value
            predvar[key] = full_array
        return invar, predvar

    def _write_results(
        self, invar, predvar, name, results_dir, writer, save_filetypes, step
    ):
        # Save batch to vtk/np files
        if "np" in save_filetypes:
            np.savez(results_dir + name, {**invar, **predvar})
        if "vtk" in save_filetypes:
            self.vtk_obj.file_dir = Path(results_dir)
            self.vtk_obj.file_name = Path(name).stem
            if self.log_iter:
                self.vtk_obj.var_to_vtk(data_vars={**invar, **predvar}, step=step)
            else:
                self.vtk_obj.var_to_vtk(data_vars={**invar, **predvar})

        # Add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                "Inferencers", name, results_dir, writer, step, invar, predvar
            )
