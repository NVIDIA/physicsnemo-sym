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

""" This PDE problem was taken from,
"A Physics-Informed Neural Network Framework
for PDEs on 3D Surfaces: Time Independent
Problems" by Zhiwei Fang and Justin Zhan.
"""
from sympy import Symbol, Function

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Point1D
from physicsnemo.sym.geometry.primitives_3d import Sphere
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pde import PDE


# define Poisson equation with sympy
class SurfacePoisson(PDE):
    name = "SurfacePoisson"

    def __init__(self):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x, normal_y, normal_z = (
            Symbol("normal_x"),
            Symbol("normal_y"),
            Symbol("normal_z"),
        )

        # u
        u = Function("u")(x, y, z)

        # set equations
        self.equations = {}
        self.equations["poisson_u"] = u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)
        self.equations["flux_u"] = (
            normal_x * u.diff(x) + normal_y * u.diff(y) + normal_z * u.diff(z)
        )


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    sp = SurfacePoisson()
    poisson_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = sp.make_nodes() + [poisson_net.make_node(name="poisson_network")]

    # add constraints to solver
    # make geometry
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    center = (0, 0, 0)
    radius = 1
    geo = Sphere(center, radius)
    p = Point1D(
        1, parameterization=Parameterization({Parameter("y"): 0, Parameter("z"): 0})
    )

    # make domain
    domain = Domain()

    # sphere surface
    surface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"poisson_u": -18.0 * x * y * z, "flux_u": 0},
        batch_size=cfg.batch_size.surface,
        lambda_weighting={"poisson_u": 1.0, "flux_u": 1.0},
    )
    domain.add_constraint(surface, "surface")

    # single point
    point = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=p,
        outvar={"u": 0.0},
        batch_size=2,
        lambda_weighting={"u": 1.0},
    )
    domain.add_constraint(point, "point")

    # validation data
    surface_points = geo.sample_boundary(10000)
    true_solution = {
        "u": surface_points["x"] * surface_points["y"] * surface_points["z"]
    }
    validator = PointwiseValidator(
        nodes=nodes, invar=surface_points, true_outvar=true_solution, batch_size=128
    )
    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
