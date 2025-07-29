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

import csv
import sys
import torch
import physicsnemo.sym
import numpy as np
from sympy import Symbol, Eq, And, Or

from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # params for simulation
    # fluid params
    nu = 0.02
    rho = 1
    inlet_vel = 1.0
    volumetric_flow = 1.125 / 2

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu, rho=rho, dim=3, time=False)
    normal_dot_vel = NormalDotVec()
    equation_nodes = ns.make_nodes() + normal_dot_vel.make_nodes()

    # determine inputs outputs of the network
    input_keys = [Key("x"), Key("y"), Key("z")]
    output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]

    # select the network and the specific configs
    if cfg.custom.arch == "FourierNetArch":
        flow_net = FourierNetArch(input_keys=input_keys, output_keys=output_keys)
    else:
        sys.exit(
            "Network not configured for this script. Please include the network in the script"
        )

    flow_nodes = equation_nodes + [flow_net.make_node(name="flow_network")]

    # make flow domain
    flow_domain = Domain()

    # inlet
    def channel_sdf(x, y, z):
        sdf = channel.sdf({"x": x, "y": y, "z": z}, {})
        return sdf["sdf"]

    constraint_inlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=inlet,
        outvar={"u": inlet_vel, "v": 0, "w": 0},
        batch_size=cfg.batch_size.inlet,
        criteria=Eq(x, channel_origin[0]),
        lambda_weighting={"u": channel_sdf, "v": 1.0, "w": 1.0},  # weight zero on edges
    )
    flow_domain.add_constraint(constraint_inlet, "inlet")

    # outlet
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_origin[0] + channel_dim[0]),
    )
    flow_domain.add_constraint(constraint_outlet, "outlet")

    # no slip for channel walls
    no_slip = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        criteria=z < channel_origin[2] + channel_dim[2] / 2.0,
    )
    flow_domain.add_constraint(no_slip, "no_slip")

    # symmetry channel
    symmetry = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo,
        outvar={"w": 0, "u__z": 0, "v__z": 0, "p__z": 0},
        batch_size=cfg.batch_size.symmetry,
        criteria=Eq(z, channel_origin[2] + channel_dim[2] / 2.0),
    )
    flow_domain.add_constraint(symmetry, "symmetry")

    # flow interior low res away from fpga
    lr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.lr_interior,
        criteria=Or(x < flow_box_origin[0], x > (flow_box_origin[0] + flow_box_dim[0])),
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
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
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
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
    )
    flow_domain.add_constraint(integral_continuity, "integral_continuity")

    # flow data
    file_path = "../openfoam/fpga_heat_fluid0.csv"
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
        filename = to_absolute_path(file_path)
        values = np.loadtxt(filename, skiprows=1, delimiter=",", unpack=False)
        values = values[
            values[:, -1] + channel_origin[2] <= 0.0, :
        ]  # remove redundant data due to symmetry
        # get column keys
        csvfile = open(filename)
        reader = csv.reader(csvfile)
        first_line = next(iter(reader))

        # set dictionary
        csv_dict = {}
        for i, name in enumerate(first_line):
            if mapping is not None:
                if name.strip() in mapping.keys():
                    csv_dict[mapping[name.strip()]] = values[:, i : i + 1]
            else:
                csv_dict[name.strip()] = values[:, i : i + 1]
        openfoam_var = csv_dict

        openfoam_var["x"] = openfoam_var["x"] + channel_origin[0]
        openfoam_var["y"] = openfoam_var["y"] + channel_origin[1]
        openfoam_var["z"] = openfoam_var["z"] + channel_origin[2]
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y", "z"]
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
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # add pressure monitor
    invar_front_pressure = integral_plane.sample_boundary(
        1024,
        parameterization={
            x_pos: heat_sink_base_origin[0] - heat_sink_base_dim[0],
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
            x_pos: heat_sink_base_origin[0] + 2 * heat_sink_base_dim[0],
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
