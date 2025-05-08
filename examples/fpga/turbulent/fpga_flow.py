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


from fpga_geometry import *

import os
import warnings

import sys
import torch
import physicsnemo.sym
from sympy import Symbol, Eq, Abs, tanh, And, Or
import numpy as np

from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes, Curl
from physicsnemo.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from physicsnemo.sym.eq.pdes.basic import NormalDotVec, GradNormal
from physicsnemo.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # params for simulation
    #############
    # Real Params
    #############
    # fluid params
    fluid_viscosity = 1.84e-05  # kg/m-s
    fluid_density = 1.1614  # kg/m3

    # boundary params
    length_scale = 0.04  # m
    inlet_velocity = 5.24386  # m/s

    ##############################
    # Nondimensionalization Params
    ##############################
    # fluid params
    nu = fluid_viscosity / (fluid_density * inlet_velocity * length_scale)
    rho = 1
    normalize_inlet_vel = 1.0

    # heat params
    D_solid = 0.1
    D_fluid = 0.02
    inlet_T = 0
    source_grad = 1.5
    source_area = source_dim[0] * source_dim[2]

    u_profile = (
        normalize_inlet_vel
        * tanh((0.5 - Abs(y)) / 0.02)
        * tanh((0.5625 - Abs(z)) / 0.02)
    )
    volumetric_flow = 1.0668  # value via integration of inlet profile

    # make list of nodes to unroll graph on
    ze = ZeroEquation(nu=nu, dim=3, time=False, max_distance=0.5)
    ns = NavierStokes(nu=ze.equations["nu"], rho=rho, dim=3, time=False)
    normal_dot_vel = NormalDotVec()
    equation_nodes = ns.make_nodes() + ze.make_nodes() + normal_dot_vel.make_nodes()

    # determine inputs outputs of the network
    input_keys = [Key("x"), Key("y"), Key("z")]
    if cfg.custom.parameterized:
        input_keys += [Key("HS_height"), Key("HS_length")]
        HS_height_range = (0.40625, 0.8625)
        HS_length_range = (0.35, 0.65)
        param_ranges = {HS_height: HS_height_range, HS_length: HS_length_range}
        validation_param_ranges = {HS_height: 0.8625, HS_length: 0.65}
        fixed_param_ranges = {
            HS_height: lambda batch_size: np.full(
                (batch_size, 1), np.random.uniform(*HS_height_range)
            ),
            HS_length: lambda batch_size: np.full(
                (batch_size, 1), np.random.uniform(*HS_length_range)
            ),
        }
    else:
        param_ranges, validation_param_ranges, fixed_param_ranges = {}, {}, {}

    output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]

    # select the network and the specific configs
    if cfg.custom.arch == "FourierNetArch":
        flow_net = FourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            frequencies=("axis", [i for i in range(35)]),
            frequencies_params=("axis", [i for i in range(35)]),
        )
    else:
        sys.exit(
            "Network not configured for this script. Please include the network in the script"
        )

    flow_nodes = equation_nodes + [flow_net.make_node(name="flow_network")]

    # make flow domain
    flow_domain = Domain()

    # inlet
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=inlet,
        outvar={"u": u_profile, "v": 0, "w": 0},
        batch_size=cfg.batch_size.inlet,
        criteria=Eq(x, channel_origin[0]),
        lambda_weighting={"u": 1.0, "v": 1.0, "w": 1.0},
        batch_per_epoch=5000,
        parameterization=param_ranges,
    )
    flow_domain.add_constraint(constraint_inlet, "inlet")

    # outlet
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_origin[0] + channel_dim[0]),
        batch_per_epoch=5000,
        parameterization=param_ranges,
    )
    flow_domain.add_constraint(constraint_outlet, "outlet")

    # no slip for channel walls
    no_slip = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        batch_per_epoch=5000,
        parameterization=param_ranges,
    )
    flow_domain.add_constraint(no_slip, "no_slip")

    # flow interior low res away from fpga
    lr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.lr_interior,
        criteria=Or(x < flow_box_origin[0], x > (flow_box_origin[0] + flow_box_dim[0])),
        compute_sdf_derivatives=True,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        batch_per_epoch=5000,
        parameterization=param_ranges,
    )
    flow_domain.add_constraint(lr_interior, "lr_interior")

    # flow interiror high res near fpga
    hr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_z": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.hr_interior,
        criteria=And(
            x > flow_box_origin[0], x < (flow_box_origin[0] + flow_box_dim[0])
        ),
        compute_sdf_derivatives=True,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        batch_per_epoch=5000,
        parameterization=param_ranges,
    )
    flow_domain.add_constraint(hr_interior, "hr_interior")

    # integral continuity
    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=integral_plane,
        outvar={"normal_dot_vel": volumetric_flow},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        criteria=integral_criteria,
        lambda_weighting={"normal_dot_vel": 1.0},
        parameterization={**x_pos_range, **param_ranges},
        batch_per_epoch=5000,
    )
    flow_domain.add_constraint(integral_continuity, "integral_continuity")

    # flow data
    # validation data fluid
    file_path = "../openfoam/FPGA_re13239.6_tanh_OF_blockMesh_fullFake.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "Points:0": "x",
            "Points:1": "y",
            "Points:2": "z",
            "U:0": "u",
            "U:1": "v",
            "U:2": "w",
            "p_rgh": "p",
        }
        openfoam_var = csv_to_dict(
            to_absolute_path(file_path),
            mapping,
        )

        # normalize values
        openfoam_var["x"] = openfoam_var["x"] / length_scale + channel_origin[0]
        openfoam_var["y"] = openfoam_var["y"] / length_scale + channel_origin[1]
        openfoam_var["z"] = openfoam_var["z"] / length_scale + channel_origin[2]
        openfoam_var["u"] = openfoam_var["u"] / inlet_velocity
        openfoam_var["v"] = openfoam_var["v"] / inlet_velocity
        openfoam_var["w"] = openfoam_var["w"] / inlet_velocity
        openfoam_var["p"] = (openfoam_var["p"]) / (inlet_velocity**2 * fluid_density)

        if cfg.custom.parameterized:
            openfoam_var["HS_height"] = (
                np.ones_like(openfoam_var["x"])
                * validation_param_ranges[Symbol("HS_height")]
            )
            openfoam_var["HS_length"] = (
                np.ones_like(openfoam_var["x"])
                * validation_param_ranges[Symbol("HS_length")]
            )
            openfoam_invar_numpy = {
                key: value
                for key, value in openfoam_var.items()
                if key in ["x", "y", "z", "HS_height", "HS_length"]
            }
        else:
            openfoam_invar_numpy = {
                key: value
                for key, value in openfoam_var.items()
                if key in ["x", "y", "z"]
            }

        openfoam_outvar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "w", "p"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=flow_nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
        )
        flow_domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/Modulus_sym_examples_supplemental_materials"
        )

    # add pressure monitor
    invar_front_pressure = integral_plane.sample_boundary(
        1024,
        parameterization={
            x_pos: heat_sink_base_origin[0] - 0.65,
            **fixed_param_ranges,
        },
    )
    pressure_monitor = PointwiseMonitor(
        invar_front_pressure,
        output_names=["p"],
        metrics={"front_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(pressure_monitor)
    invar_back_pressure = integral_plane.sample_boundary(
        1024,
        parameterization={
            x_pos: heat_sink_base_origin[0] + 2 * 0.65,
            **fixed_param_ranges,
        },
    )
    pressure_monitor = PointwiseMonitor(
        invar_back_pressure,
        output_names=["p"],
        metrics={"back_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(pressure_monitor)

    # make solver
    flow_slv = Solver(cfg, flow_domain)

    # start flow solver
    flow_slv.solve()


if __name__ == "__main__":
    run()
