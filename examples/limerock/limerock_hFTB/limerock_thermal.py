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

import torch
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import copy

import numpy as np
from sympy import Symbol, Eq, tanh, Or, And
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from typing import Dict

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import SequentialSolver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.utils.io import (
    VTKUniformGrid,
)
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.basic import NormalDotVec, GradNormal
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from physicsnemo.sym.distributed.manager import DistributedManager

from limerock_properties import *

from flux_diffusion import (
    FluxDiffusion,
    FluxIntegrateDiffusion,
    FluxGradNormal,
    FluxRobin,
    Dirichlet,
)


class hFTBArch(Arch):
    def __init__(
        self,
        arch: Arch,
    ) -> None:
        output_keys = arch.output_keys + [
            Key(x.name + "_prev_step") for x in arch.output_keys
        ]
        super().__init__(
            input_keys=arch.input_keys,
            output_keys=output_keys,
            periodicity=arch.periodicity,
        )

        # set networks for current and prev time window
        self.arch_prev_step = arch
        self.arch = copy.deepcopy(arch)
        for param, param_prev_step in zip(
            self.arch.parameters(), self.arch_prev_step.parameters()
        ):
            param_prev_step.requires_grad = False

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        y_prev_step = self.arch_prev_step.forward(in_vars)
        y = self.arch.forward(in_vars)
        for key, b in y_prev_step.items():
            y[key + "_prev_step"] = b
        return y

    def move_network(self):
        for param, param_prev_step in zip(
            self.arch.parameters(), self.arch_prev_step.parameters()
        ):
            param_prev_step.data = param.detach().clone().data
            param_prev_step.requires_grad = False


@physicsnemo.sym.main(config_path="conf", config_name="conf_thermal")
def run(cfg: PhysicsNeMoConfig) -> None:
    if DistributedManager().distributed:
        print("Multi-GPU currently not supported for this example. Exiting.")
        return

    # make list of nodes to unroll graph on
    ad = AdvectionDiffusion(
        T="theta_f", rho=nd_fluid_density, D=nd_fluid_diffusivity, dim=3, time=False
    )
    dif = FluxDiffusion(D=nd_copper_diffusivity)
    flow_grad_norm = GradNormal("theta_f", dim=3, time=False)
    solid_grad_norm = FluxGradNormal()
    integrate_flux_dif = FluxIntegrateDiffusion()
    robin_flux = FluxRobin(
        theta_f_conductivity=nd_fluid_conductivity,
        theta_s_conductivity=nd_copper_conductivity,
        h=500.0,
    )
    dirichlet = Dirichlet(lhs="theta_f", rhs="theta_s")
    flow_net = FourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
    )
    f_net = FourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("z")], output_keys=[Key("theta_f")]
    )
    thermal_f_net = hFTBArch(f_net)
    thermal_s_net = FourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("z")], output_keys=[Key("theta_s")]
    )
    flux_s_net = FourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[
            Key("flux_theta_s_x"),
            Key("flux_theta_s_y"),
            Key("flux_theta_s_z"),
        ],
    )
    thermal_nodes = (
        ad.make_nodes(detach_names=["u", "v", "w"])
        + dif.make_nodes()
        + flow_grad_norm.make_nodes()
        + solid_grad_norm.make_nodes()
        + integrate_flux_dif.make_nodes(
            detach_names=["flux_theta_s_x", "flux_theta_s_y", "flux_theta_s_z"]
        )
        + robin_flux.make_nodes(
            detach_names=[
                "theta_f_prev_step",
                "theta_f_prev_step__x",
                "theta_f_prev_step__y",
                "theta_f_prev_step__z",
            ]
        )
        + dirichlet.make_nodes(detach_names=["theta_s"])
        + [flow_net.make_node(name="flow_network", optimize=False, jit=cfg.jit)]
        + [
            thermal_f_net.make_node(
                name="thermal_fluid_network", optimize=True, jit=cfg.jit
            )
        ]
        + [
            thermal_s_net.make_node(
                name="thermal_solid_network", optimize=True, jit=cfg.jit
            )
        ]
        + [flux_s_net.make_node(name="flux_solid_network", optimize=True, jit=cfg.jit)]
    )

    # make domain for first cycle of hFTB
    cycle_1_domain = Domain("cycle_1")

    # add constraints to solver
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    import time as time

    tic = time.time()

    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=limerock.inlet,
        outvar={"theta_f": nd_inlet_temp},
        batch_size=cfg.batch_size.inlet,
        batch_per_epoch=50,
        lambda_weighting={"theta_f": 1000.0},
    )
    cycle_1_domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=limerock.outlet,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.outlet,
        lambda_weighting={"normal_gradient_theta_f": 1.0},
    )
    cycle_1_domain.add_constraint(outlet, "outlet")

    # channel walls insulating
    walls = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=limerock.geo,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.no_slip,
        criteria=Or(
            Or(
                Eq(y, limerock.geo_bounds_lower[1]), Eq(z, limerock.geo_bounds_lower[2])
            ),
            Or(
                Eq(y, limerock.geo_bounds_upper[1]), Eq(z, limerock.geo_bounds_upper[2])
            ),
        ),
        lambda_weighting={"normal_gradient_theta_f": 1.0},
    )
    cycle_1_domain.add_constraint(walls, name="ChannelWalls")

    # flow interior low res away from heat sink
    lr_interior_f = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=limerock.geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.lr_interior_f,
        criteria=Or(
            (x < limerock.heat_sink_bounds[0]), (x > limerock.heat_sink_bounds[1])
        ),
        lambda_weighting={"advection_diffusion_theta_f": 1000.0},
    )
    cycle_1_domain.add_constraint(lr_interior_f, "lr_interior_f")

    # flow interiror high res near heat sink
    hr_interior_f = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=limerock.geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.hr_interior_f,
        lambda_weighting={"advection_diffusion_theta_f": 1000.0},
        criteria=And(
            (x > limerock.heat_sink_bounds[0]), (x < limerock.heat_sink_bounds[1])
        ),
    )
    cycle_1_domain.add_constraint(hr_interior_f, "hr_interior_f")

    # fluid solid interface
    interface = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=limerock.geo_solid,
        outvar={"theta_f": 0.05},
        batch_size=cfg.batch_size.interface,
        criteria=z > limerock.geo_bounds_lower[2],
        lambda_weighting={"theta_f": 100.0},
    )
    cycle_1_domain.add_constraint(interface, "interface")

    # add inferencer data
    vtk_obj = VTKUniformGrid(
        bounds=[limerock.geo_bounds[x], limerock.geo_bounds[y], limerock.geo_bounds[z]],
        npoints=[256, 128, 256],
        export_map={"u": ["u", "v", "w"], "p": ["p"], "theta_f": ["theta_f"]},
    )

    def mask_fn(x, y, z):
        sdf = limerock.geo.sdf({"x": x, "y": y, "z": z}, {})
        return sdf["sdf"] < 0

    grid_inferencer = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=thermal_nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        output_names=["u", "v", "w", "p", "theta_f"],
        mask_fn=mask_fn,
        mask_value=np.nan,
        requires_grad=False,
        batch_size=100000,
    )
    cycle_1_domain.add_inferencer(grid_inferencer, "grid_inferencer")

    # make domain for all other cycles
    cycle_n_domain = Domain("cycle_n")

    # inlet
    cycle_n_domain.add_constraint(inlet, "inlet")

    # outlet
    cycle_n_domain.add_constraint(outlet, "outlet")

    # channel walls insulating
    cycle_n_domain.add_constraint(walls, name="ChannelWalls")

    # flow interior low res away from heat sink
    cycle_n_domain.add_constraint(lr_interior_f, "lr_interior_f")

    # flow interiror high res near heat sink
    cycle_n_domain.add_constraint(hr_interior_f, "hr_interior_f")

    # diffusion dictionaries
    diff_outvar = {
        "diffusion_theta_s": 0,
        "compatibility_theta_s_x_y": 0,
        "compatibility_theta_s_x_z": 0,
        "compatibility_theta_s_y_z": 0,
        "integrate_diffusion_theta_s_x": 0,
        "integrate_diffusion_theta_s_y": 0,
        "integrate_diffusion_theta_s_z": 0,
    }
    diff_lambda = {
        "diffusion_theta_s": 1000000.0,
        "compatibility_theta_s_x_y": 1.0,
        "compatibility_theta_s_x_z": 1.0,
        "compatibility_theta_s_y_z": 1.0,
        "integrate_diffusion_theta_s_x": 1.0,
        "integrate_diffusion_theta_s_y": 1.0,
        "integrate_diffusion_theta_s_z": 1.0,
    }

    # solid interior
    interior_s = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=limerock.geo_solid,
        outvar=diff_outvar,
        batch_size=cfg.batch_size.interior_s,
        lambda_weighting=diff_lambda,
    )
    cycle_n_domain.add_constraint(interior_s, "interior_s")

    # limerock base
    sharpen_tanh = 60.0
    source_func_xl = (tanh(sharpen_tanh * (x - source_origin[0])) + 1.0) / 2.0
    source_func_xh = (
        tanh(sharpen_tanh * ((source_origin[0] + source_dim[0]) - x)) + 1.0
    ) / 2.0
    source_func_yl = (tanh(sharpen_tanh * (y - source_origin[1])) + 1.0) / 2.0
    source_func_yh = (
        tanh(sharpen_tanh * ((source_origin[1] + source_dim[1]) - y)) + 1.0
    ) / 2.0
    gradient_normal = (
        nd_source_term
        * source_func_xl
        * source_func_xh
        * source_func_yl
        * source_func_yh
    )
    base = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=limerock.geo_solid,
        outvar={"normal_gradient_flux_theta_s": gradient_normal},
        batch_size=cfg.batch_size.base,
        criteria=Eq(z, limerock.geo_bounds_lower[2]),
        lambda_weighting={"normal_gradient_flux_theta_s": 10.0},
    )
    cycle_n_domain.add_constraint(base, "base")

    # fluid solid interface
    interface = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=limerock.geo_solid,
        outvar={"dirichlet_theta_s_theta_f": 0, "robin_theta_s": 0},
        batch_size=cfg.batch_size.interface,
        criteria=z > limerock.geo_bounds_lower[2],
        lambda_weighting={"dirichlet_theta_s_theta_f": 100.0, "robin_theta_s": 1.0},
    )
    cycle_n_domain.add_constraint(interface, "interface")

    # add fluid inferencer data
    cycle_n_domain.add_inferencer(grid_inferencer, "grid_inferencer")

    # add solid inferencer data
    vtk_obj = VTKUniformGrid(
        bounds=[
            limerock.geo_hr_bounds[x],
            limerock.geo_hr_bounds[y],
            limerock.geo_hr_bounds[z],
        ],
        npoints=[128, 128, 512],
        export_map={"theta_s": ["theta_s"]},
    )

    def mask_fn(x, y, z):
        sdf = limerock.geo.sdf({"x": x, "y": y, "z": z}, {})
        return sdf["sdf"] > 0

    grid_inferencer = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=thermal_nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        output_names=["theta_s"],
        mask_fn=mask_fn,
        mask_value=np.nan,
        requires_grad=False,
        batch_size=100000,
    )
    cycle_n_domain.add_inferencer(grid_inferencer, "grid_inferencer_solid")

    # peak temperature monitor
    invar_temp = limerock.geo_solid.sample_boundary(
        10000, criteria=Eq(z, limerock.geo_bounds_lower[2])
    )
    peak_temp_monitor = PointwiseMonitor(
        invar_temp,
        output_names=["theta_s"],
        metrics={"peak_temp": lambda var: torch.max(var["theta_s"])},
        nodes=thermal_nodes,
    )
    cycle_n_domain.add_monitor(peak_temp_monitor)

    # make solver
    slv = SequentialSolver(
        cfg,
        [(1, cycle_1_domain), (20, cycle_n_domain)],
        custom_update_operation=thermal_f_net.move_network,
    )

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
