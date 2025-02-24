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

import numpy as np
from sympy import Symbol, Eq

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.domain import Domain
from modulus.sym.geometry import Bounds
from modulus.sym.geometry.primitives_2d import Line, Circle, Channel2D
from modulus.sym.eq.pdes.navier_stokes import NavierStokesIncompressible
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)


from modulus.sym.key import Key
from modulus.sym import quantity
from modulus.sym.eq.non_dim import NonDimensionalizer, Scaler
from modulus.sym.models.moving_time_window import MovingTimeWindowArch
from modulus.sym.domain.inferencer import PointVTKInferencer
from modulus.sym.utils.io import (
    VTKUniformGrid,
)
from modulus.sym.solver import SequentialSolver
from sympy import Symbol, Function, Number


from modulus.sym.eq.pde import PDE


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # physical quantities
    nu = quantity(1.48e-5, "kg/(m*s)")
    # nu = quantity(8.9e-3, "m^2/s")
    rho = quantity(1.225, "kg/m^3")
    inlet_u = quantity(1.5, "m/s")
    inlet_v = quantity(0.0, "m/s")
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    outlet_p = quantity(0.0, "pa")
    time_window_size = quantity(2, "s")
    t_symbol = Symbol("t")

    nr_time_windows = 20
    velocity_scale = inlet_u
    density_scale = rho
    length_scale = quantity(20, "m")
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale**3),
    )
    time_range = {t_symbol: (nd.ndim(quantity(0.0, "s")),
                             nd.ndim(time_window_size))}

    # geometry
    channel_length = (quantity(-10, "m"), quantity(30, "m"))
    channel_width = (quantity(-10, "m"), quantity(10, "m"))
    cylinder_center = (quantity(0, "m"), quantity(0, "m"))
    cylinder_radius = quantity(0.5, "m")
    channel_length_nd = tuple(map(lambda x: nd.ndim(x), channel_length))
    channel_width_nd = tuple(map(lambda x: nd.ndim(x), channel_width))
    cylinder_center_nd = tuple(map(lambda x: nd.ndim(x), cylinder_center))
    cylinder_radius_nd = nd.ndim(cylinder_radius)

    channel = Channel2D(
        (channel_length_nd[0], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
    )
    inlet = Line(
        (channel_length_nd[0], channel_width_nd[0]),
        (channel_length_nd[0], channel_width_nd[1]),
        normal=1,
    )
    outlet = Line(
        (channel_length_nd[1], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
        normal=1,
    )
    wall_top = Line(
        (channel_length_nd[1], channel_width_nd[0]),
        (channel_length_nd[1], channel_width_nd[1]),
        normal=1,
    )
    cylinder = Circle(cylinder_center_nd, cylinder_radius_nd)
    volume_geo = channel - cylinder

    volume_geo_small = channel.scale(0.2) - cylinder

    cylinder_interior = cylinder.scale(2) - cylinder

    # make list of nodes to unroll graph on
    ns = NavierStokesIncompressible(nu=nd.ndim(
        nu), rho=nd.ndim(rho), dim=2, time=True)
    normal_dot_vel = NormalDotVec(["u", "v"])

    flow_net = instantiate_arch(
        # Include time as an input key
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    time_window_net = MovingTimeWindowArch(flow_net, nd.ndim(time_window_size))

    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [time_window_net.make_node(name="time_window_network")]
        + Scaler(
            ["u", "v", "p", "t"],
            ["u_scaled", "v_scaled", "p_scaled", "t_scaled"],
            ["m/s", "m/s", "m^2/s^2"],
            nd,
        ).make_node()
    )

    # make domain
    # make initial condition domain
    ic_domain = Domain("initial_conditions")

    # make moving window domain
    window_domain = Domain("window")
    x, y = Symbol("x"), Symbol("y")

    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo,
        outvar={
            "u": nd.ndim(inlet_u),
            "v": nd.ndim(inlet_v),
            "p": np.ndim(outlet_p)
        },
        batch_size=cfg.batch_size.initial_condition,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        parameterization={t_symbol: 0}
    )
    ic_domain.add_constraint(ic, name="ic")

    # make constraint for matching previous windows initial condition
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo,
        outvar={"u_prev_step_diff": 0,
                "v_prev_step_diff": 0},
        batch_size=cfg.batch_size.interior,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic, name="ic")

    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.inlet,
        parameterization=time_range,
    )
    window_domain.add_constraint(inlet, "inlet")
    ic_domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": nd.ndim(outlet_p)},
        batch_size=cfg.batch_size.outlet,
        parameterization=time_range,
    )
    window_domain.add_constraint(outlet, "outlet")
    ic_domain.add_constraint(outlet, "outlet")

    # full slip (channel walls)
    walls = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=channel,
        outvar={"u": nd.ndim(inlet_u), "v": nd.ndim(inlet_v)},
        batch_size=cfg.batch_size.walls,
        parameterization=time_range,
    )
    window_domain.add_constraint(walls, "walls")
    ic_domain.add_constraint(walls, "walls")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=cylinder,
        outvar={"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v)},
        batch_size=cfg.batch_size.no_slip,
        parameterization=time_range,
    )
    window_domain.add_constraint(no_slip, "no_slip")
    ic_domain.add_constraint(no_slip, "no_slip")

    # interior contraints
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        parameterization=time_range,
    )
    window_domain.add_constraint(interior, "interior")
    ic_domain.add_constraint(interior, "interior")

    interior_cylinder = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=cylinder_interior,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_cylinder,
        parameterization=time_range,
    )
    window_domain.add_constraint(interior_cylinder, "interior_cylinder")
    ic_domain.add_constraint(interior_cylinder, "interior_cylinder")

    interior_small = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo_small,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior_small,
        parameterization=time_range,
    )

    window_domain.add_constraint(interior_small, "interior_small")
    ic_domain.add_constraint(interior_small, "interior_small")

    bounds_nd = [
        # Normalized bounds for x-direction
        [channel_length_nd[0], channel_length_nd[1]],
        # Normalized bounds for y-direction
        [channel_width_nd[0], channel_width_nd[1]]
    ]

    for i, specific_time in enumerate(np.linspace(0, nd.ndim(time_window_size), 100)):
        vtk_obj = VTKUniformGrid(
            bounds=bounds_nd,
            npoints=[256, 256],
            export_map={"u": ["u", "v"], "p": ["p"]},
        )
        grid_inference = PointVTKInferencer(
            vtk_obj=vtk_obj,
            nodes=nodes,
            input_vtk_map={"x": "x", "y": "y"},
            output_names=["u", "v", "p"],
            requires_grad=False,
            invar={"t": np.full([256**2, 1], specific_time)},
            batch_size=100000,
        )
        ic_domain.add_inferencer(
            grid_inference, name="time_slice_" + str(i).zfill(4))
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
