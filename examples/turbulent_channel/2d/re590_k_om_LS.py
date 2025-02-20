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

import torch
import numpy as np
from sympy import Symbol, Eq, sin, cos, Min, Max, Abs, log, exp
import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

from custom_k_om_ls import kOmegaInit, kOmega, kOmegaLSWF


@physicsnemo.sym.main(config_path="conf_re590_k_om_LS", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # simulation parameters
    Re = 590
    nu = 1 / Re
    y_plus = 30
    karman_constant = 0.4187
    resolved_y_start = y_plus * nu
    channel_width = (-1, 1)
    channel_length = (-np.pi / 2, np.pi / 2)

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

    geo_sdf = Channel2D(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )

    # geometry where the equations are solved
    geo_resolved = Channel2D(
        (channel_length[0], channel_width[0] + resolved_y_start),
        (channel_length[1], channel_width[1] - resolved_y_start),
    )

    # make list of nodes to unroll graph on
    init = kOmegaInit(nu=nu, rho=1.0)
    eq = kOmega(nu=nu, rho=1.0)
    wf = kOmegaLSWF(nu=nu, rho=1.0)

    flow_net = instantiate_arch(
        input_keys=[Key("x_sin"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    p_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("p")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    k_net = instantiate_arch(
        input_keys=[Key("x_sin"), Key("y")],
        output_keys=[Key("k_star")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    om_net = instantiate_arch(
        input_keys=[Key("x_sin"), Key("y")],
        output_keys=[Key("om_star")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    nodes = (
        init.make_nodes()
        + eq.make_nodes()
        + wf.make_nodes()
        + [
            Node.from_sympy(
                sin(2 * np.pi * Symbol("x") / (channel_length[1] - channel_length[0])),
                "x_sin",
            )
        ]
        + [Node.from_sympy(Min(log(1 + exp(Symbol("k_star"))) + 1e-4, 20), "k")]
        + [Node.from_sympy(Min(log(1 + exp(Symbol("om_star"))) + 1e-4, 20), "om_plus")]
        + [flow_net.make_node(name="flow_network")]
        + [p_net.make_node(name="p_network")]
        + [k_net.make_node(name="k_network")]
        + [om_net.make_node(name="om_network")]
    )

    # add constraints to solver
    p_grad = 1.0

    x, y = Symbol("x"), Symbol("y")

    # make domain
    domain = Domain()

    # Point where wall funciton is applied
    wf_pt = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo_resolved,
        outvar={
            "velocity_wall_normal_wf": 0,
            "velocity_wall_parallel_wf": 0,
            "om_plus_wf": 0,
            "wall_shear_stress_x_wf": 0,
            "wall_shear_stress_y_wf": 0,
        },
        lambda_weighting={
            "velocity_wall_normal_wf": 100,
            "velocity_wall_parallel_wf": 100,
            "om_plus_wf": 10,
            "wall_shear_stress_x_wf": 100,
            "wall_shear_stress_y_wf": 100,
        },
        batch_size=cfg.batch_size.wf_pt,
        parameterization={"normal_distance": resolved_y_start},
    )
    domain.add_constraint(wf_pt, "WF")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_resolved,
        outvar={
            "continuity": 0,
            "momentum_x": 0,
            "momentum_y": 0,
            "k_equation": 0,
            "om_plus_equation": 0,
        },
        lambda_weighting={
            "continuity": 100,
            "momentum_x": 1000,
            "momentum_y": 1000,
            "k_equation": 10,
            "om_plus_equation": 0.1,
        },
        batch_size=cfg.batch_size.interior,
        bounds={x: channel_length, y: channel_width},
    )
    domain.add_constraint(interior, "Interior")

    # pressure pc
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"p": p_grad * (channel_length[1] - channel_length[0])},
        lambda_weighting={"p": 10},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "Inlet")

    # pressure pc
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        lambda_weighting={"p": 10},
        batch_size=cfg.batch_size.outlet,
    )
    domain.add_constraint(outlet, "Outlet")

    # flow initialization
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_resolved,
        outvar={"u_init": 0, "v_init": 0, "k_init": 0, "p_init": 0, "om_plus_init": 0},
        batch_size=cfg.batch_size.interior_init,
        bounds={x: channel_length, y: channel_width},
    )
    domain.add_constraint(interior, "InteriorInit")

    # add inferencing and monitor
    invar_wf_pt = geo_resolved.sample_boundary(
        1024,
    )
    u_tau_monitor = PointwiseMonitor(
        invar_wf_pt,
        output_names=["k"],
        metrics={
            "mean_u_tau": lambda var: torch.mean((0.09**0.25) * torch.sqrt(var["k"]))
        },
        nodes=nodes,
    )
    domain.add_monitor(u_tau_monitor)

    # add inferencer data
    inference = PointwiseInferencer(
        nodes=nodes,
        invar=geo_resolved.sample_interior(
            5000, bounds={x: channel_length, y: channel_width}
        ),
        output_names=["u", "v", "p", "k", "om_plus"],
    )
    domain.add_inferencer(inference, "inf_interior")

    inference = PointwiseInferencer(
        nodes=nodes,
        invar=geo_resolved.sample_boundary(
            10, parameterization={"normal_distance": resolved_y_start}
        ),
        output_names=["u", "v", "p", "k", "om_plus", "normal_distance"],
    )
    domain.add_inferencer(inference, "inf_wf")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
