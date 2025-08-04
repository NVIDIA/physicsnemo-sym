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

from sympy import Symbol, Eq
import numpy as np
import torch

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry import Bounds, Parameterization, Parameter
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Circle
from physicsnemo.sym.utils.sympy.functions import parabola
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("r")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # add constraints to solver
    # specify params
    channel_length = (-6.732, 6.732)
    channel_width = (-1.0, 1.0)
    cylinder_center = (0.0, 0.0)
    outer_cylinder_radius = 2.0
    inner_cylinder_radius = Parameter("r")
    inner_cylinder_radius_ranges = (0.75, 1.0)
    inlet_vel = 1.5
    parameterization = Parameterization(
        {inner_cylinder_radius: inner_cylinder_radius_ranges}
    )

    # make geometry
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )
    outer_circle = Circle(cylinder_center, outer_cylinder_radius)
    inner_circle = Circle(
        (0, 0), inner_cylinder_radius, parameterization=parameterization
    )
    geo = (rec + outer_circle) - inner_circle

    # make annular ring domain
    domain = Domain()

    # inlet
    inlet_sympy = parabola(
        y, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel
    )
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": inlet_sympy, "v": 0},
        batch_size=cfg.batch_size.inlet,
        batch_per_epoch=4000,
        criteria=Eq(x, channel_length[0]),
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        batch_per_epoch=4000,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.no_slip,
        batch_per_epoch=4000,
        criteria=(x > channel_length[0]) & (x < channel_length[1]),
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior,
        batch_per_epoch=4000,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")

    # integral continuity
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"normal_dot_vel": 2},
        batch_size=10,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    # add validation data
    mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    file_path_1 = "../openfoam/bend_finerInternal0.csv"
    file_path_2 = "../openfoam/annularRing_r_0.8750.csv"
    file_path_3 = "../openfoam/annularRing_r_0.750.csv"

    # r1
    if os.path.exists(to_absolute_path(file_path_1)):
        openfoam_var_r1 = csv_to_dict(to_absolute_path(file_path_1), mapping)
        openfoam_var_r1["x"] += channel_length[0]  # center OpenFoam data
        openfoam_var_r1["r"] = np.zeros_like(openfoam_var_r1["x"]) + 1.0
        openfoam_invar_r1_numpy = {
            key: value
            for key, value in openfoam_var_r1.items()
            if key in ["x", "y", "r"]
        }
        openfoam_outvar_r1_numpy = {
            key: value
            for key, value in openfoam_var_r1.items()
            if key in ["u", "v", "p"]
        }

        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_r1_numpy,
            true_outvar=openfoam_outvar_r1_numpy,
            batch_size=1024,
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path_1} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # r875
    if os.path.exists(to_absolute_path(file_path_2)):
        openfoam_var_r875 = csv_to_dict(to_absolute_path(file_path_2), mapping)
        openfoam_var_r875["x"] += channel_length[0]  # center OpenFoam data
        openfoam_var_r875["r"] = np.zeros_like(openfoam_var_r875["x"]) + 0.875
        openfoam_invar_r875_numpy = {
            key: value
            for key, value in openfoam_var_r875.items()
            if key in ["x", "y", "r"]
        }
        openfoam_outvar_r875_numpy = {
            key: value
            for key, value in openfoam_var_r875.items()
            if key in ["u", "v", "p"]
        }

        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_r875_numpy,
            true_outvar=openfoam_outvar_r875_numpy,
            batch_size=1024,
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path_2} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # r75
    if os.path.exists(to_absolute_path(file_path_3)):
        openfoam_var_r75 = csv_to_dict(to_absolute_path(file_path_3), mapping)
        openfoam_var_r75["x"] += channel_length[0]  # center OpenFoam data
        openfoam_var_r75["r"] = np.zeros_like(openfoam_var_r75["x"]) + 0.75
        openfoam_invar_r75_numpy = {
            key: value
            for key, value in openfoam_var_r75.items()
            if key in ["x", "y", "r"]
        }
        openfoam_outvar_r75_numpy = {
            key: value
            for key, value in openfoam_var_r75.items()
            if key in ["u", "v", "p"]
        }

        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_r75_numpy,
            true_outvar=openfoam_outvar_r75_numpy,
            batch_size=1024,
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path_3} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # add inferencer data
    for i, radius in enumerate(
        np.linspace(
            inner_cylinder_radius_ranges[0], inner_cylinder_radius_ranges[1], 10
        )
    ):
        radius = float(radius)
        sampled_interior = geo.sample_interior(
            1024,
            bounds=Bounds(
                {x: channel_length, y: (-outer_cylinder_radius, outer_cylinder_radius)}
            ),
            parameterization={inner_cylinder_radius: radius},
        )
        point_cloud_inference = PointwiseInferencer(
            nodes=nodes,
            invar=sampled_interior,
            output_names=["u", "v", "p"],
            batch_size=1024,
        )
        domain.add_inferencer(point_cloud_inference, "inf_data" + str(i).zfill(5))

    # add monitors
    # metric for mass and momentum imbalance
    global_monitor = PointwiseMonitor(
        geo.sample_interior(
            1024,
            bounds=Bounds(
                {x: channel_length, y: (-outer_cylinder_radius, outer_cylinder_radius)}
            ),
        ),
        output_names=["continuity", "momentum_x", "momentum_y"],
        metrics={
            "mass_imbalance": lambda var: torch.sum(
                var["area"] * torch.abs(var["continuity"])
            ),
            "momentum_imbalance": lambda var: torch.sum(
                var["area"]
                * (torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"]))
            ),
        },
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(global_monitor)

    # metric for force on inner sphere
    for i, radius in enumerate(
        np.linspace(inner_cylinder_radius_ranges[0], inner_cylinder_radius_ranges[1], 3)
    ):
        radius = float(radius)
        force_monitor = PointwiseMonitor(
            inner_circle.sample_boundary(
                1024,
                parameterization={inner_cylinder_radius: radius},
            ),
            output_names=["p"],
            metrics={
                "force_x_r" + str(radius): lambda var: torch.sum(
                    var["normal_x"] * var["area"] * var["p"]
                ),
                "force_y_r" + str(radius): lambda var: torch.sum(
                    var["normal_y"] * var["area"] * var["p"]
                ),
            },
            nodes=nodes,
        )
        domain.add_monitor(force_monitor)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
