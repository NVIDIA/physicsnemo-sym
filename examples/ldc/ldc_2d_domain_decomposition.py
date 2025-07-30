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

from sympy import Symbol, Eq, Abs, Function, Heaviside

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
from physicsnemo.sym.node import Node

import copy


def generate_pde_copies(eq, num_copies=2):
    """
    Generate multiple copies of a equation to use it
    for different domains.
    """

    eq_copies = []
    for i in range(num_copies):
        temp_eq = copy.deepcopy(eq)

        # Find all the functions that need substituion
        eq_sum = 0
        for k, v in temp_eq.equations.items():
            eq_sum += v  # Generate a single equation to find all the terms.

        for func in eq_sum.atoms(Function):
            func_new_str = func.func.__name__ + "_" + str(i + 1)
            args = func.args
            func_new = Function(func_new_str)(*args)
            temp_eq.subs(func, func_new)

        # Generate the functions to be substituted
        equations_new = {}
        for k, v in temp_eq.equations.items():
            equations_new[k + "_" + str(i + 1)] = v

        # Replace the equations dict
        temp_eq.equations = equations_new
        eq_copies.append(temp_eq)

    return eq_copies


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)

    copies = generate_pde_copies(ns, num_copies=2)
    for eq in copies:
        print(eq.equations)

    # two networks for two different domains
    flow_net_1 = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u_1"), Key("v_1"), Key("p_1")],
        cfg=cfg.arch.fully_connected,
    )
    flow_net_2 = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u_2"), Key("v_2"), Key("p_2")],
        cfg=cfg.arch.fully_connected,
    )

    # nodes for interface condition. Currently only dirichlet are used for this
    # demo problem. However, it can also be extended to include higher order
    # derivative continuity conditions too (e.g. neumann).
    # dirichlet BCs
    interface_nodes = [Node.from_sympy(Symbol("u_1") - Symbol("u_2"), "dirichlet_u")]
    interface_nodes += [Node.from_sympy(Symbol("v_1") - Symbol("v_2"), "dirichlet_v")]
    interface_nodes += [Node.from_sympy(Symbol("p_1") - Symbol("p_2"), "dirichlet_p")]

    # nodes for inferencing to merge the results of the two networks
    # For this problem, the division of domain is done at y=0

    custom_nodes = [
        Node.from_sympy(
            Symbol("u_1") * Heaviside(-Symbol("y"))
            + Symbol("u_2") * Heaviside(Symbol("y")),
            "u",
        )
    ]
    custom_nodes += [
        Node.from_sympy(
            Symbol("v_1") * Heaviside(-Symbol("y"))
            + Symbol("v_2") * Heaviside(Symbol("y")),
            "v",
        )
    ]
    custom_nodes += [
        Node.from_sympy(
            Symbol("p_1") * Heaviside(-Symbol("y"))
            + Symbol("p_2") * Heaviside(Symbol("y")),
            "p",
        )
    ]

    nodes = (
        copies[0].make_nodes()
        + copies[1].make_nodes()
        + interface_nodes
        + [flow_net_1.make_node(name="flow_network_1")]
        + [flow_net_2.make_node(name="flow_network_2")]
    )

    nodes_infer = custom_nodes

    # add constraints to solver
    # make geometry
    height = 0.1
    width = 0.1
    x, y = Symbol("x"), Symbol("y")
    rec_1 = Rectangle((-width / 2, -height / 2), (width / 2, 0))
    rec_2 = Rectangle((-width / 2, 0), (width / 2, height / 2))

    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # make ldc domain
    ldc_domain = Domain()

    # top wall
    top_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec_2,
        outvar={"u_2": 1.0, "v_2": 0},
        batch_size=cfg.batch_size.TopWall,
        lambda_weighting={
            "u_2": 1.0 - 20 * Abs(x),
            "v_2": 1.0,
        },  # weight edges to be zero
        criteria=Eq(y, height / 2),
    )
    ldc_domain.add_constraint(top_wall, "top_wall")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u_1": 0, "v_1": 0, "u_2": 0, "v_2": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=y < height / 2,
    )
    ldc_domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity_2": 0, "momentum_x_2": 0, "momentum_y_2": 0},
        batch_size=cfg.batch_size.Interior // 2,
        lambda_weighting={
            "continuity_2": Symbol("sdf"),
            "momentum_x_2": Symbol("sdf"),
            "momentum_y_2": Symbol("sdf"),
        },
        criteria=y > 0,
    )
    ldc_domain.add_constraint(interior, "interior_2")

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity_1": 0, "momentum_x_1": 0, "momentum_y_1": 0},
        batch_size=cfg.batch_size.Interior // 2,
        lambda_weighting={
            "continuity_1": Symbol("sdf"),
            "momentum_x_1": Symbol("sdf"),
            "momentum_y_1": Symbol("sdf"),
        },
        criteria=y < 0,
    )
    ldc_domain.add_constraint(interior, "interior_1")

    # interface
    interface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec_1,
        outvar={"dirichlet_u": 0, "dirichlet_v": 0, "dirichlet_p": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=Eq(y, 0),
    )
    ldc_domain.add_constraint(interface, "interface")

    # add validator
    file_path = "openfoam/cavity_uniformVel0.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_var["x"] += -width / 2  # center OpenFoam data
        openfoam_var["y"] += -height / 2  # center OpenFoam data
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["u", "v"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes + nodes_infer,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            batch_size=1024,
            plotter=ValidatorPlotter(),
        )
        ldc_domain.add_validator(openfoam_validator)

        samples = rec_1.sample_interior(10000)
        # add inferencer data
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={"x": samples["x"], "y": samples["y"]},
            output_names=["u_1", "v_1", "p_1"],
            batch_size=1024,
            plotter=InferencerPlotter(),
        )
        ldc_domain.add_inferencer(grid_inference, "inf_data_1")

        samples = rec_2.sample_interior(10000)
        # add inferencer data
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={"x": samples["x"], "y": samples["y"]},
            output_names=["u_2", "v_2", "p_2"],
            batch_size=1024,
            plotter=InferencerPlotter(),
        )
        ldc_domain.add_inferencer(grid_inference, "inf_data_2")

    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # make solver
    slv = Solver(cfg, ldc_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
