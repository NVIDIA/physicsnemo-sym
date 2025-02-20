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
from torch.utils.data import DataLoader, Dataset

import numpy as np
from sympy import Symbol, Eq, tanh, Or, And

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from physicsnemo.sym.eq.pdes.basic import NormalDotVec, GradNormal

from limerock_properties import *


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    ze = ZeroEquation(nu=nu, dim=3, time=False, max_distance=0.5)
    ns = NavierStokes(nu=ze.equations["nu"], rho=rho, dim=3, time=False)
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
        outvar={"u": inlet_velocity_normalized, "v": 0, "w": 0},
        batch_size=cfg.batch_size.inlet,
        batch_per_epoch=5000,
        lambda_weighting={"u": channel_sdf, "v": 1.0, "w": 1.0},
    )
    flow_domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=limerock.outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        batch_per_epoch=5000,
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
        batch_per_epoch=5000,
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
    )
    flow_domain.add_constraint(lr_interior, "lr_interior")

    # flow interior high res near limerock
    hr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=limerock.geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.hr_interior,
        batch_per_epoch=5000,
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
    )
    flow_domain.add_constraint(hr_interior, "hr_interior")

    # integral continuity
    def integral_criteria(invar, params):
        sdf = limerock.geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=limerock.integral_plane,
        outvar={"normal_dot_vel": volumetric_flow},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        criteria=integral_criteria,
    )
    flow_domain.add_constraint(integral_continuity, "integral_continuity")

    print("finished generating points")

    """# add inferencer data
    invar_flow_numpy = limerock.geo.sample_interior(10000, bounds=limerock.geo_bounds)
    point_cloud_inference = PointwiseInferencer(invar_flow_numpy, ["u", "v", "w", "p"], flow_nodes)
    flow_domain.add_inferencer(point_cloud_inference, "inf_data")"""

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
