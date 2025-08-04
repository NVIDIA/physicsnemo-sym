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
import matplotlib.pyplot as plt

import numpy as np
from sympy import Symbol, Eq, Abs, tanh, lambdify

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


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=0.01, rho=1.0, dim=2, time=False)

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

    # define the basis functions. we will use one neural network for y<0 and the other
    # y >=0. To achieve the division at y=0, we will use tanh functions as below
    x, y = Symbol("x"), Symbol("y")
    basis_function_1 = 0.5 * (tanh(25 * (0 - y)) + tanh(25 * (y + 1)))
    basis_function_2 = 0.5 * (tanh(25 * (1 - y)) + tanh(25 * (y + 0)))

    # plot the basis functions for visualization
    basis_function_1_lf = lambdify(y, basis_function_1, "numpy")
    basis_function_2_lf = lambdify(y, basis_function_2, "numpy")
    y_vals = np.linspace(-0.5, 0.5, 100)

    out_bf_1 = basis_function_1_lf(y_vals)
    out_bf_2 = basis_function_2_lf(y_vals)

    plt.figure()
    plt.plot(y_vals, out_bf_1, label="basis_function_1", color="blue")
    plt.plot(y_vals, out_bf_2, label="basis_function_2", color="green")

    plt.legend()
    plt.savefig(to_absolute_path("./basis_function_viz.png"))

    # nodes to merge the results of the two networks
    custom_nodes = [
        Node.from_sympy(
            Symbol("u_1") * basis_function_1 + Symbol("u_2") * basis_function_2,
            "u",
        )
    ]
    custom_nodes += [
        Node.from_sympy(
            Symbol("v_1") * basis_function_1 + Symbol("v_2") * basis_function_2,
            "v",
        )
    ]
    custom_nodes += [
        Node.from_sympy(
            Symbol("p_1") * basis_function_1 + Symbol("p_2") * basis_function_2,
            "p",
        )
    ]

    nodes = (
        ns.make_nodes()
        + custom_nodes
        + [flow_net_1.make_node(name="flow_network_1")]
        + [flow_net_2.make_node(name="flow_network_2")]
    )

    # add constraints to solver
    # make geometry
    height = 0.1
    width = 0.1

    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # make ldc domain
    ldc_domain = Domain()

    # top wall
    top_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 1.0, "v": 0},
        batch_size=cfg.batch_size.TopWall,
        lambda_weighting={
            "u": 1.0 - 20 * Abs(x),
            "v": 1.0,
        },  # weight edges to be zero
        criteria=Eq(y, height / 2),
    )
    ldc_domain.add_constraint(top_wall, "top_wall")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.NoSlip,
        criteria=y < height / 2,
    )
    ldc_domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
        },
    )
    ldc_domain.add_constraint(interior, "interior")

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
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            batch_size=1024,
            plotter=ValidatorPlotter(),
        )
        ldc_domain.add_validator(openfoam_validator)

        samples = rec.sample_interior(10000)
        # add inferencer data
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar={"x": samples["x"], "y": samples["y"]},
            output_names=["u", "v", "p"],
            batch_size=1024,
            plotter=InferencerPlotter(),
        )
        ldc_domain.add_inferencer(grid_inference, "inf_data")

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
