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

import numpy as np

from sympy import Symbol, sin, cos

import physicsnemo.sym
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.solver import SequentialSolver
from physicsnemo.sym.domain import Domain

from physicsnemo.sym.loss.loss import CausalLossNorm


from physicsnemo.sym.geometry.primitives_3d import Box
from physicsnemo.sym.geometry.parameterization import OrderedParameterization

from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.moving_time_window import MovingTimeWindowArch
from physicsnemo.sym.domain.constraint import (
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.utils.io import (
    VTKUniformGrid,
)
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # time window parameters
    time_window_size = 1.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 10

    # make navier stokes equations
    ns = NavierStokes(nu=0.002, rho=1.0, dim=3, time=True)

    # define sympy variables to parametrize domain curves
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    # make geometry for problem
    channel_length = (0.0, 2 * np.pi)
    channel_width = (0.0, 2 * np.pi)
    channel_height = (0.0, 2 * np.pi)
    box_bounds = {x: channel_length, y: channel_width, z: channel_height}

    # define geometry
    rec = Box(
        (channel_length[0], channel_width[0], channel_height[0]),
        (channel_length[1], channel_width[1], channel_height[1]),
        parameterization=OrderedParameterization(time_range, key=t_symbol),
    )

    # make network for current step and previous step
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        periodicity={"x": channel_length, "y": channel_width, "z": channel_height},
        layer_size=256,
    )
    time_window_net = MovingTimeWindowArch(flow_net, time_window_size)

    # make nodes to unroll graph on
    nodes = ns.make_nodes() + [time_window_net.make_node(name="time_window_network")]

    # make initial condition domain
    ic_domain = Domain("initial_conditions")

    # make moving window domain
    window_domain = Domain("window")

    # make initial condition
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "u": sin(x) * cos(y) * cos(z),
            "v": -cos(x) * sin(y) * cos(z),
            "w": 0,
            "p": 1.0 / 16 * (cos(2 * x) + cos(2 * y)) * (cos(2 * z) + 2),
        },
        batch_size=cfg.batch_size.initial_condition,
        bounds=box_bounds,
        lambda_weighting={"u": 100, "v": 100, "w": 100, "p": 100},
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic, name="ic")

    # make constraint for matching previous windows initial condition
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0, "w_prev_step_diff": 0},
        batch_size=cfg.batch_size.interior,
        bounds=box_bounds,
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
            "w_prev_step_diff": 100,
        },
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic, name="ic")

    # make interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        bounds=box_bounds,
        batch_size=4000,
        loss=CausalLossNorm(eps=1.0),
        fixed_dataset=False,
        shuffle=False,
    )
    ic_domain.add_constraint(interior, name="interior")
    window_domain.add_constraint(interior, name="interior")

    # add inference data for time slices
    for i, specific_time in enumerate(np.linspace(0, time_window_size, 10)):
        vtk_obj = VTKUniformGrid(
            bounds=[(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)],
            npoints=[128, 128, 128],
            export_map={"u": ["u", "v", "w"], "p": ["p"]},
        )
        grid_inference = PointVTKInferencer(
            vtk_obj=vtk_obj,
            nodes=nodes,
            input_vtk_map={"x": "x", "y": "y", "z": "z"},
            output_names=["u", "v", "w", "p"],
            requires_grad=False,
            invar={"t": np.full([128**3, 1], specific_time)},
            batch_size=100000,
        )
        ic_domain.add_inferencer(grid_inference, name="time_slice_" + str(i).zfill(4))
        window_domain.add_inferencer(
            grid_inference, name="time_slice_" + str(i).zfill(4)
        )

    # make solver
    slv = SequentialSolver(
        cfg,
        [(1, ic_domain), (nr_time_windows, window_domain)],
        custom_update_operation=time_window_net.move_window,
    )

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
