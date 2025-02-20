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

from sympy import Symbol, Eq
from typing import Dict

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry import Bounds
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Circle
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
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.geometry.adf import ADF


class HardBC(ADF):
    def __init__(self):
        super().__init__()

        # domain measures
        self.channel_length = (-6.732, 6.732)
        self.channel_width = (-1.0, 1.0)
        self.cylinder_center = (0.0, 0.0)
        self.outer_cylinder_radius = 2.0
        self.inner_cylinder_radius = 1.0
        self.delta = 0.267949
        self.center = (0.0, 0.0)
        self.r = self.outer_cylinder_radius
        self.inlet_vel = 1.5

        # parameters
        self.eps: float = 1e-9
        self.mu: float = 2.0
        self.m: float = 2.0

    def forward(self, invar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forms the solution anstaz for the annular ring example
        """
        outvar = {}
        x, y = invar["x"], invar["y"]

        # ADFs
        # left line
        omega_0 = ADF.line_segment_adf(
            (x, y),
            (self.channel_length[0], self.channel_width[0]),
            (self.channel_length[0], self.channel_width[1]),
        )
        # right line
        omega_1 = ADF.line_segment_adf(
            (x, y),
            (self.channel_length[1], self.channel_width[0]),
            (self.channel_length[1], self.channel_width[1]),
        )
        # top left line
        omega_2 = ADF.line_segment_adf(
            (x, y),
            (self.channel_length[0], self.channel_width[1]),
            (-self.outer_cylinder_radius + self.delta, self.channel_width[1]),
        )
        # top right line
        omega_3 = ADF.line_segment_adf(
            (x, y),
            (self.outer_cylinder_radius - self.delta, self.channel_width[1]),
            (self.channel_length[1], self.channel_width[1]),
        )
        # bottom left line
        omega_4 = ADF.line_segment_adf(
            (x, y),
            (self.channel_length[0], self.channel_width[0]),
            (-self.outer_cylinder_radius + self.delta, self.channel_width[0]),
        )
        # bottom right line
        omega_5 = ADF.line_segment_adf(
            (x, y),
            (self.outer_cylinder_radius - self.delta, self.channel_width[0]),
            (self.channel_length[1], self.channel_width[0]),
        )
        # inner circle
        omega_6 = ADF.circle_adf((x, y), self.inner_cylinder_radius, self.center)
        # top arch
        omega_7 = ADF.trimmed_circle_adf(
            (x, y),
            (-self.outer_cylinder_radius, self.channel_width[1]),
            (self.outer_cylinder_radius, self.channel_width[1]),
            -1,
            self.outer_cylinder_radius,
            self.center,
        )
        # bottom arch
        omega_8 = ADF.trimmed_circle_adf(
            (x, y),
            (-self.outer_cylinder_radius, self.channel_width[0]),
            (self.outer_cylinder_radius, self.channel_width[0]),
            1,
            self.outer_cylinder_radius,
            self.center,
        )

        # r equivalence
        omega_E_u = ADF.r_equivalence(
            [omega_0, omega_2, omega_3, omega_4, omega_5, omega_6, omega_7, omega_8],
            self.m,
        )
        omega_E_v = ADF.r_equivalence(
            [omega_0, omega_2, omega_3, omega_4, omega_5, omega_6, omega_7, omega_8],
            self.m,
        )
        omega_E_p = omega_1

        # u BC
        bases = [
            omega_0**self.mu,
            omega_2**self.mu,
            omega_3**self.mu,
            omega_4**self.mu,
            omega_5**self.mu,
            omega_6**self.mu,
            omega_7**self.mu,
            omega_8**self.mu,
        ]
        w = [
            ADF.transfinite_interpolation(bases, idx, self.eps)
            for idx in range(len(bases))
        ]
        dirichlet_bc = [self.inlet_vel - (3 * (y**2) / 2), 0, 0, 0, 0, 0, 0, 0]
        g = sum([w[i] * dirichlet_bc[i] for i in range(len(w))])
        outvar["u"] = g + omega_E_u * invar["u_star"]

        # v BC
        outvar["v"] = omega_E_v * invar["v_star"]

        # p BC
        outvar["p"] = omega_E_p * invar["p_star"]

        return outvar


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    hard_bc = HardBC()
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False, mixed_form=True)
    normal_dot_vel = NormalDotVec(["u", "v"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[
            Key("u_star"),
            Key("v_star"),
            Key("p_star"),
            Key("u_x"),
            Key("u_y"),
            Key("v_x"),
            Key("v_y"),
        ],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + [
            Node(
                inputs=["x", "y", "u_star", "v_star", "p_star"],
                outputs=["u", "v", "p"],
                evaluate=hard_bc,
            )
        ]
    )

    # add constraints to solver
    # specify params
    channel_length = hard_bc.channel_length
    channel_width = hard_bc.channel_width
    cylinder_center = hard_bc.cylinder_center
    outer_cylinder_radius = hard_bc.outer_cylinder_radius
    inner_cylinder_radius = hard_bc.inner_cylinder_radius
    inlet_vel = hard_bc.inlet_vel

    # make geometry
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )
    outer_circle = Circle(cylinder_center, outer_cylinder_radius)
    inner_circle = Circle((0, 0), inner_cylinder_radius)
    geo = (rec + outer_circle) - inner_circle

    # make annular ring domain
    domain = Domain()

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "continuity": 0,
            "momentum_x": 0,
            "momentum_y": 0,
            "compatibility_u_x": 0,
            "compatibility_u_y": 0,
            "compatibility_v_x": 0,
            "compatibility_v_y": 0,
        },
        batch_size=cfg.batch_size.interior,
        bounds=Bounds(
            {x: channel_length, y: (-outer_cylinder_radius, outer_cylinder_radius)}
        ),
        lambda_weighting={
            "continuity": 5.0 * Symbol("sdf"),
            "momentum_x": 2.0 * Symbol("sdf"),
            "momentum_y": 2.0 * Symbol("sdf"),
            "compatibility_u_x": 0.1 * Symbol("sdf"),
            "compatibility_u_y": 0.1 * Symbol("sdf"),
            "compatibility_v_x": 0.1 * Symbol("sdf"),
            "compatibility_v_y": 0.1 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")

    # integral continuity
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"normal_dot_vel": 2},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(integral_continuity, "integral_continuity")

    # add validation data
    file_path = "../openfoam/bend_finerInternal0.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["x"] += channel_length[0]  # center OpenFoam data
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
            batch_size=1024,
        )
        domain.add_validator(openfoam_validator)

        # add inferencer data
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            output_names=["u", "v", "p"],
            batch_size=1024,
        )
        domain.add_inferencer(grid_inference, "inf_data")
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/Modulus_sym_examples_supplemental_materials"
        )

    # add monitors
    # metric for mass and momentum imbalance
    global_monitor = PointwiseMonitor(
        geo.sample_interior(1024),
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
    force_monitor = PointwiseMonitor(
        inner_circle.sample_boundary(1024),
        output_names=["p"],
        metrics={
            "force_x": lambda var: torch.sum(var["normal_x"] * var["area"] * var["p"]),
            "force_y": lambda var: torch.sum(var["normal_y"] * var["area"] * var["p"]),
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
