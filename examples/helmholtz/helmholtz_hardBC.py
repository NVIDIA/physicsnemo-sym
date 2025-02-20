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
from sympy import Symbol, pi, sin
from typing import List, Tuple, Dict

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.geometry.adf import ADF
from physicsnemo.sym.eq.pdes.wave_equation import HelmholtzEquation


class HardBC(ADF):
    def __init__(self):
        super().__init__()

        # domain measures
        self.domain_height: float = 2.0
        self.domain_width: float = 2.0

        # boundary conditions (bottom, right, top, left)
        self.g: List[float] = [0.0, 0.0, 0.0, 0.0]

        # parameters
        self.eps: float = 1e-9
        self.mu: float = 2.0
        self.m: float = 2.0

    def forward(self, invar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forms the solution anstaz for the Helmholtz example
        """

        outvar = {}
        x, y = invar["x"], invar["y"]
        omega_0 = ADF.line_segment_adf(
            (x, y),
            (-self.domain_width / 2, -self.domain_height / 2),
            (self.domain_width / 2, -self.domain_height / 2),
        )
        omega_1 = ADF.line_segment_adf(
            (x, y),
            (self.domain_width / 2, -self.domain_height / 2),
            (self.domain_width / 2, self.domain_height / 2),
        )
        omega_2 = ADF.line_segment_adf(
            (x, y),
            (self.domain_width / 2, self.domain_height / 2),
            (-self.domain_width / 2, self.domain_height / 2),
        )
        omega_3 = ADF.line_segment_adf(
            (x, y),
            (-self.domain_width / 2, self.domain_height / 2),
            (-self.domain_width / 2, -self.domain_height / 2),
        )
        omega_E_u = ADF.r_equivalence([omega_0, omega_1, omega_2, omega_3], self.m)

        bases = [
            omega_0**self.mu,
            omega_1**self.mu,
            omega_2**self.mu,
            omega_3**self.mu,
        ]
        w = [
            ADF.transfinite_interpolation(bases, idx, self.eps)
            for idx in range(len(self.g))
        ]
        g = w[0] * self.g[0] + w[1] * self.g[1] + w[2] * self.g[2] + w[3] * self.g[3]
        outvar["u"] = g + omega_E_u * invar["u_star"]
        return outvar


@physicsnemo.sym.main(config_path="conf", config_name="config_hardBC")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    wave = HelmholtzEquation(u="u", k=1.0, dim=2, mixed_form=True)
    hard_bc = HardBC()
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u_star"), Key("u_x"), Key("u_y")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        wave.make_nodes()
        + [Node(inputs=["x", "y", "u_star"], outputs=["u"], evaluate=hard_bc)]
        + [wave_net.make_node(name="wave_network")]
    )

    # add constraints to solver
    # make geometry
    x, y = Symbol("x"), Symbol("y")
    height = hard_bc.domain_height
    width = hard_bc.domain_width
    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # make domain
    domain = Domain()

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "helmholtz": -(
                -((pi) ** 2) * sin(pi * x) * sin(4 * pi * y)
                - ((4 * pi) ** 2) * sin(pi * x) * sin(4 * pi * y)
                + 1 * sin(pi * x) * sin(4 * pi * y)
            ),
            "compatibility_u_x": 0,
            "compatibility_u_y": 0,
        },
        batch_size=cfg.batch_size.interior,
        bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
        lambda_weighting={
            "helmholtz": Symbol("sdf"),
            "compatibility_u_x": 0.5,
            "compatibility_u_y": 0.5,
        },
    )
    domain.add_constraint(interior, "interior")

    # validation data
    file_path = "validation/helmholtz.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"x": "x", "y": "y", "z": "u"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["u"]
        }

        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            batch_size=1024,
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/Modulus_sym_examples_supplemental_materials"
        )

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
