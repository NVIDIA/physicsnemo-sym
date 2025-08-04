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
from sympy import Symbol, Or, And

import physicsnemo.sym
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.inferencer import PointVTKInferencer
from physicsnemo.sym.utils.io import (
    VTKUniformGrid,
)
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from physicsnemo.sym.eq.pdes.basic import NormalDotVec

from limerock_properties import *


@physicsnemo.sym.main(config_path="conf", config_name="conf_flow")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    ze = ZeroEquation(
        nu=nd_fluid_viscosity / nd_fluid_density, dim=3, time=False, max_distance=0.479
    )
    ns = NavierStokes(nu=ze.equations["nu"], rho=nd_fluid_density, dim=3, time=False)
    normal_dot_vel = NormalDotVec()
    flow_net = FourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
    )
    flow_nodes = (
        ns.make_nodes()
        + ze.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # make flow domain
    flow_domain = Domain()

    # add constraints to solver
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    # inlet
    def channel_sdf(x, y, z):
        sdf = limerock.channel.sdf({"x": x, "y": y, "z": z}, {})
        return sdf["sdf"]

    inlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=limerock.inlet,
        outvar={"u": nd_inlet_velocity, "v": 0, "w": 0},
        batch_size=cfg.batch_size.inlet,
        lambda_weighting={"u": channel_sdf, "v": 1.0, "w": 1.0},
    )
    flow_domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=limerock.outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
    )
    flow_domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=limerock.geo,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        batch_per_epoch=15000,
    )
    flow_domain.add_constraint(no_slip, "no_slip")

    # flow interior low res away from limerock
    lr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=limerock.geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.lr_interior,
        compute_sdf_derivatives=True,
        lambda_weighting={
            "continuity": 3 * Symbol("sdf"),
            "momentum_x": 3 * Symbol("sdf"),
            "momentum_y": 3 * Symbol("sdf"),
            "momentum_z": 3 * Symbol("sdf"),
        },
        criteria=Or(
            (x < limerock.heat_sink_bounds[0]), (x > limerock.heat_sink_bounds[1])
        ),
        batch_per_epoch=2000,
    )
    flow_domain.add_constraint(lr_interior, "lr_interior")

    # flow interior high res near limerock
    hr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=limerock.geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.hr_interior,
        compute_sdf_derivatives=True,
        lambda_weighting={
            "continuity": 3 * Symbol("sdf"),
            "momentum_x": 3 * Symbol("sdf"),
            "momentum_y": 3 * Symbol("sdf"),
            "momentum_z": 3 * Symbol("sdf"),
        },
        criteria=And(
            (x > limerock.heat_sink_bounds[0]), (x < limerock.heat_sink_bounds[1])
        ),
        batch_per_epoch=2000,
    )
    flow_domain.add_constraint(hr_interior, "hr_interior")

    # integral continuity
    def integral_criteria(invar, params):
        sdf = limerock.geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=limerock.integral_plane,
        outvar={"normal_dot_vel": nd_volumetric_flow},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        criteria=integral_criteria,
    )
    flow_domain.add_constraint(integral_continuity, "integral_continuity")

    # add inferencer data
    vtk_obj = VTKUniformGrid(
        bounds=[limerock.geo_bounds[x], limerock.geo_bounds[y], limerock.geo_bounds[z]],
        npoints=[256, 128, 256],
        export_map={"u": ["u", "v", "w"], "p": ["p"]},
    )

    def mask_fn(x, y, z):
        sdf = limerock.geo.sdf({"x": x, "y": y, "z": z}, {})
        return sdf["sdf"] < 0

    grid_inference = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=flow_nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        output_names=["u", "v", "w", "p"],
        mask_fn=mask_fn,
        mask_value=np.nan,
        requires_grad=False,
        batch_size=100000,
    )
    flow_domain.add_inferencer(grid_inference, "grid_inference")

    # add monitor
    # front pressure
    plane_param_ranges = {Symbol("x_pos"): -0.7}
    invar_pressure = limerock.integral_plane.sample_boundary(
        5000,
        parameterization=plane_param_ranges,
    )
    front_pressure_monitor = PointwiseMonitor(
        invar_pressure,
        output_names=["p"],
        metrics={"front_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(front_pressure_monitor)

    # back pressure
    plane_param_ranges = {Symbol("x_pos"): 0.7}
    invar_pressure = limerock.integral_plane.sample_boundary(
        5000,
        parameterization=plane_param_ranges,
    )
    back_pressure_monitor = PointwiseMonitor(
        invar_pressure,
        output_names=["p"],
        metrics={"back_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(back_pressure_monitor)

    # make solver
    slv = Solver(cfg, flow_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
