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

import os
import warnings

import numpy as np
from sympy import Symbol, Eq, And, Or

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from physicsnemo.sym.utils.sympy.functions import parabola
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.models.fourier_net import FourierNetArch


@physicsnemo.sym.main(config_path="conf_2d_solid_fluid", config_name="config_flow")
def run(cfg: PhysicsNeMoConfig) -> None:
    #############
    # Real Params
    #############
    fluid_kinematic_viscosity = 0.004195088  # m**2/s
    fluid_density = 1.1614  # kg/m**3
    fluid_specific_heat = 1005  # J/(kg K)
    fluid_conductivity = 0.0261  # W/(m K)

    # copper params
    copper_density = 8930  # kg/m3
    copper_specific_heat = 385  # J/(kg K)
    copper_conductivity = 385  # W/(m K)

    # boundary params
    inlet_velocity = 5.24386  # m/s
    inlet_temp = 25.0  # C
    copper_heat_flux = 51.948051948  # W / m2

    ################
    # Non dim params
    ################
    length_scale = 0.04  # m
    time_scale = 0.007627968710072352  # s
    mass_scale = 7.43296e-05  # kg
    temp_scale = 1.0  # K
    velocity_scale = length_scale / time_scale  # m/s
    pressure_scale = mass_scale / (length_scale * time_scale**2)  # kg / (m s**2)
    density_scale = mass_scale / length_scale**3  # kg/m3
    watt_scale = (mass_scale * length_scale**2) / (time_scale**3)  # kg m**2 / s**3
    joule_scale = (mass_scale * length_scale**2) / (
        time_scale**2
    )  # kg * m**2 / s**2

    ##############################
    # Nondimensionalization Params
    ##############################
    # fluid params
    nd_fluid_kinematic_viscosity = fluid_kinematic_viscosity / (
        length_scale**2 / time_scale
    )
    nd_fluid_density = fluid_density / density_scale
    nd_fluid_specific_heat = fluid_specific_heat / (
        joule_scale / (mass_scale * temp_scale)
    )
    nd_fluid_conductivity = fluid_conductivity / (
        watt_scale / (length_scale * temp_scale)
    )
    nd_fluid_diffusivity = nd_fluid_conductivity / (
        nd_fluid_specific_heat * nd_fluid_density
    )

    # copper params
    nd_copper_density = copper_density / (mass_scale / length_scale**3)
    nd_copper_specific_heat = copper_specific_heat / (
        joule_scale / (mass_scale * temp_scale)
    )
    nd_copper_conductivity = copper_conductivity / (
        watt_scale / (length_scale * temp_scale)
    )
    nd_copper_diffusivity = nd_copper_conductivity / (
        nd_copper_specific_heat * nd_copper_density
    )

    # boundary params
    nd_inlet_velocity = inlet_velocity / (length_scale / time_scale)
    nd_inlet_temp = inlet_temp / temp_scale
    nd_copper_source_grad = copper_heat_flux * length_scale / temp_scale

    # make list of nodes to unroll graph on
    ns = NavierStokes(
        nu=nd_fluid_kinematic_viscosity, rho=nd_fluid_density, dim=2, time=False
    )
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = FourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        frequencies=("axis", [i / 5.0 for i in range(25)]),
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # add constraints to solver
    # simulation params
    channel_length = (-2.5, 5.0)
    channel_width = (-0.5, 0.5)
    chip_pos = -1.0
    chip_height = 0.6
    chip_width = 1.0

    # define sympy variables to parametrize domain curves
    x, y = Symbol("x"), Symbol("y")

    # define geometry
    channel = Channel2D(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )
    inlet = Line(
        (channel_length[0], channel_width[0]),
        (channel_length[0], channel_width[1]),
        normal=1,
    )
    outlet = Line(
        (channel_length[1], channel_width[0]),
        (channel_length[1], channel_width[1]),
        normal=1,
    )
    rec = Rectangle(
        (chip_pos, channel_width[0]),
        (chip_pos + chip_width, channel_width[0] + chip_height),
    )
    geo = channel - rec
    x_pos = Symbol("x_pos")
    integral_line = Line((x_pos, channel_width[0]), (x_pos, channel_width[1]), 1)
    x_pos_range = {
        x_pos: lambda batch_size: np.full(
            (batch_size, 1), np.random.uniform(channel_length[0], channel_length[1])
        )
    }

    # make domain
    domain = Domain()

    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": nd_inlet_velocity, "v": 0},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior lr
    interior_lr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_lr,
        criteria=Or(x < (chip_pos - 0.25), x > (chip_pos + chip_width + 0.25)),
        lambda_weighting={
            "continuity": 2 * Symbol("sdf"),
            "momentum_x": 2 * Symbol("sdf"),
            "momentum_y": 2 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior_lr, "interior_lr")

    # interior hr
    interior_hr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_hr,
        criteria=And(x > (chip_pos - 0.25), x < (chip_pos + chip_width + 0.25)),
        lambda_weighting={
            "continuity": 2 * Symbol("sdf"),
            "momentum_x": 2 * Symbol("sdf"),
            "momentum_y": 2 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior_hr, "interior_hr")

    # integral continuity
    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel": 1},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 1},
        criteria=integral_criteria,
        parameterization=x_pos_range,
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    # add validation data
    file_path = "openfoam/2d_real_cht_fluid.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "x-coordinate": "x",
            "y-coordinate": "y",
            "x-velocity": "u",
            "y-velocity": "v",
            "pressure": "p",
            "temperature": "theta_f",
        }
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["x"] = openfoam_var["x"] / length_scale - 2.5  # normalize pos
        openfoam_var["y"] = openfoam_var["y"] / length_scale - 0.5
        openfoam_var["p"] = (openfoam_var["p"] + 400.0) / pressure_scale
        openfoam_var["u"] = openfoam_var["u"] / velocity_scale
        openfoam_var["v"] = openfoam_var["v"] / velocity_scale

        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["u", "v", "p"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
