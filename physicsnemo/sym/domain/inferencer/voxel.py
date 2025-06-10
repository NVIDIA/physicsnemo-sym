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

from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.utils.io import InferencerPlotter
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk, VTKBase, VTKUniformGrid
from physicsnemo.sym.dataset import DictInferencePointwiseDataset


class VoxelInferencer(PointVTKInferencer):
    """
    Inferencer for creating volex representations.
    This inferencer works bu creating a uniform mesh of voxels and masking out the ones defined by a callable function.
    The result is a voxel based representation of any complex geometery at any resolution.

    Parameters
    ----------
    bounds : List[List[int]]
        List of domain bounds to form uniform rectangular domain
    npoints : List[int]
        Resolution of voxels in each domain
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    output_names : List[str]
        List of desired outputs.
    export_map : Dict[str, List[str]], optional
        Export map dictionary with keys that are VTK variables names and values that are lists of output variables. Will use 1:1 mapping if none is provided, by default None
    invar : Dict[str, np.array], optional
        Dictionary of additional numpy arrays as input, by default {}
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
        bounds: List[List[int]],
        npoints: List[int],
        nodes: List[Node],
        output_names: List[str],
        export_map: Union[None, Dict[str, List[str]]] = None,
        invar: Dict[str, np.array] = {},  # Additional inputs
        batch_size: int = 1024,
        mask_fn: Union[Callable, None] = None,
        mask_value: float = np.nan,
        plotter=None,
        requires_grad: bool = False,
        log_iter: bool = False,
        model=None,
    ):
        # No export map means one to one with outvars
        self.npoints = npoints
        if export_map is None:
            export_map = {name: name for name in output_names}
        coords = ["x", "y", "z"]
        input_vtk_map = {coords[i]: coords[i] for i in range(len(bounds))}

        # Create uniform grid dataset
        vtk_obj = VTKUniformGrid(
            bounds=bounds,
            npoints=npoints,
            export_map=export_map,
        )

        super().__init__(
            vtk_obj,
            nodes,
            input_vtk_map=input_vtk_map,
            output_names=output_names,
            invar=invar,  # Additional inputs
            batch_size=batch_size,
            mask_fn=mask_fn,
            mask_value=mask_value,
            plotter=plotter,
            requires_grad=requires_grad,
            log_iter=log_iter,
            model=model,
        )

    def _write_results(
        self, invar, predvar, name, results_dir, writer, save_filetypes, step
    ):
        # Save batch to vtk/np files
        if "np" in save_filetypes:
            # Reshape into grid numpy arrays [cin, xdim, ydim, zdim]
            np_vars = {}
            for key, value in {**invar, **predvar}.items():
                shape = self.npoints + [value.shape[1]]
                np_vars[key] = np.moveaxis(np.reshape(value, (shape)), -1, 0)

            np.savez(results_dir + name, np_vars)

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
