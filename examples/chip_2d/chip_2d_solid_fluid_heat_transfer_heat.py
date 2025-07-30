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

import torch
import numpy as np
from sympy import Symbol, Eq, Or, And

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from physicsnemo.sym.eq.pdes.navier_stokes import GradNormal
from physicsnemo.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch


@physicsnemo.sym.main(config_path="conf_2d_solid_fluid", config_name="config_heat")
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
    copper_heat_flux = 51.948051948  # K/m (20000 W/m**2)

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
    joule_scale = (mass_scale * length_scale**2) / (time_scale**2)  # kg * m**2 / s**2

    ##############################
    # Nondimensionalization Params
    ##############################
    # fluid params
    fluid_kinematic_viscosity / (length_scale**2 / time_scale)
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
    print("nd_copper_diffusivity", nd_copper_diffusivity)

    # boundary params
    inlet_velocity / (length_scale / time_scale)
    nd_inlet_temp = inlet_temp / temp_scale
    nd_copper_source_grad = copper_heat_flux * length_scale / temp_scale

    # make list of nodes to unroll graph on
    ad = AdvectionDiffusion(
        T="theta_f", rho=nd_fluid_density, D=nd_fluid_diffusivity, dim=2, time=False
    )
    diff = Diffusion(T="theta_s", D=1.0, dim=2, time=False)
    interface = DiffusionInterface(
        "theta_f",
        "theta_s",
        nd_fluid_conductivity,
        nd_copper_conductivity,
        dim=2,
        time=False,
    )
    gn_theta_f = GradNormal("theta_f", dim=2, time=False)
    gn_theta_s = GradNormal("theta_s", dim=2, time=False)

    flow_net = FourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        frequencies=("axis", [i / 5.0 for i in range(25)]),
    )
    solid_heat_net = ModifiedFourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("theta_s_star")],
        layer_size=256,
        frequencies=("gaussian", 2, 128),
    )
    fluid_heat_net = ModifiedFourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("theta_f_star")],
        layer_size=256,
        frequencies=("gaussian", 2, 128),
    )

    nodes = (
        ad.make_nodes(detach_names=["u", "v"])
        + diff.make_nodes()
        + interface.make_nodes()
        + gn_theta_f.make_nodes()
        + gn_theta_s.make_nodes()
        + [
            Node.from_sympy(Symbol("theta_s_star") + 170.0, "theta_s")
        ]  # Normalize the outputs
        + [
            Node.from_sympy(Symbol("theta_f_star") + 70.0, "theta_f")
        ]  # Normalize the outputs
        + [flow_net.make_node(name="flow_network", optimize=False)]
        + [solid_heat_net.make_node(name="solid_heat_network")]
        + [fluid_heat_net.make_node(name="fluid_heat_network")]
    )

    # add constraints to solver
    # simulation params
    channel_length = (-2.5, 5.0)
    channel_width = (-0.5, 0.5)
    chip_pos = -1.0
    chip_height = 0.6
    chip_width = 1.0
    source_origin = (-0.7, -0.5)
    source_dim = (0.4, 0.0)

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
    chip2d = rec
    geo = channel - rec
    x_pos = Symbol("x_pos")
    Line((x_pos, channel_width[0]), (x_pos, channel_width[1]), 1)
    {
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
        outvar={"theta_f": nd_inlet_temp},
        batch_size=cfg.batch_size.inlet,
        lambda_weighting={"theta_f": 100.0},
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet, "outlet")

    # channel walls insulating
    def walls_criteria(invar, params):
        sdf = chip2d.sdf(invar, params)
        return np.less(sdf["sdf"], -1e-5)

    walls = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.walls,
        criteria=walls_criteria,
    )
    domain.add_constraint(walls, "channel_walls")

    # fluid interior lr
    interior_lr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.interior_lr,
        criteria=Or(x < (chip_pos - 0.25), x > (chip_pos + chip_width + 0.25)),
        lambda_weighting={"advection_diffusion_theta_f": 1.0},
    )
    domain.add_constraint(interior_lr, "fluid_interior_lr")

    # fluid interior hr
    interior_hr = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.interior_hr,
        criteria=And(x > (chip_pos - 0.25), x < (chip_pos + chip_width + 0.25)),
        lambda_weighting={"advection_diffusion_theta_f": 1.0},
    )
    domain.add_constraint(interior_hr, "fluid_interior_hr")

    # solid interior
    interiorS = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=chip2d,
        outvar={"diffusion_theta_s": 0},
        batch_size=cfg.batch_size.interiorS,
        lambda_weighting={"diffusion_theta_s": 1.0},
    )
    domain.add_constraint(interiorS, "solid_interior")

    # fluid-solid interface
    def interface_criteria(invar, params):
        sdf = channel.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    interface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chip2d,
        outvar={
            "diffusion_interface_dirichlet_theta_f_theta_s": 0,
            "diffusion_interface_neumann_theta_f_theta_s": 0,
        },
        batch_size=cfg.batch_size.interface,
        lambda_weighting={
            "diffusion_interface_dirichlet_theta_f_theta_s": 1,
            "diffusion_interface_neumann_theta_f_theta_s": 1e-4,
        },
        criteria=interface_criteria,
    )
    domain.add_constraint(interface, name="interface")

    heat_source = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chip2d,
        outvar={"normal_gradient_theta_s": nd_copper_source_grad},
        batch_size=cfg.batch_size.heat_source,
        lambda_weighting={"normal_gradient_theta_s": 100},
        criteria=(
            Eq(y, source_origin[1])
            & (x >= source_origin[0])
            & (x <= (source_origin[0] + source_dim[0]))
        ),
    )
    domain.add_constraint(heat_source, name="heat_source")

    # chip walls
    chip_walls = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chip2d,
        outvar={"normal_gradient_theta_s": 0},
        batch_size=cfg.batch_size.chip_walls,
        criteria=(
            Eq(y, source_origin[1])
            & ((x < source_origin[0]) | (x > (source_origin[0] + source_dim[0])))
        ),
    )
    domain.add_constraint(chip_walls, name="chip_walls")

    # add monitor
    monitor = PointwiseMonitor(
        chip2d.sample_boundary(10000, criteria=Eq(y, source_origin[1])),
        output_names=["theta_s"],
        metrics={
            "peak_temp": lambda var: torch.max(var["theta_s"]),
        },
        nodes=nodes,
    )
    domain.add_monitor(monitor)

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
        openfoam_var["theta_f"] = openfoam_var["theta_f"] - 273.15

        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "p", "theta_f"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            plotter=ValidatorPlotter(),
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # add solid validation data
    file_path = "openfoam/2d_real_cht_solid.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"x-coordinate": "x", "y-coordinate": "y", "temperature": "theta_s"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["x"] = openfoam_var["x"] / length_scale - 2.5  # normalize pos
        openfoam_var["y"] = openfoam_var["y"] / length_scale - 0.5
        openfoam_var["theta_s"] = openfoam_var["theta_s"] - 273.15

        openfoam_solid_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y"]
        }
        openfoam_solid_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["theta_s"]
        }
        openfoam_solid_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_solid_invar_numpy,
            true_outvar=openfoam_solid_outvar_numpy,
            plotter=ValidatorPlotter(),
        )
        domain.add_validator(openfoam_solid_validator)
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
