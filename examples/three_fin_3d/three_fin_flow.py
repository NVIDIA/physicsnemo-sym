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
from sympy import Symbol, Eq, Abs, tanh, Or, And
import numpy as np
import itertools

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.models.fully_connected import FullyConnectedArch

from three_fin_geometry import *


@physicsnemo.sym.main(config_path="conf", config_name="conf_flow")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make navier stokes equations
    if cfg.custom.turbulent:
        ze = ZeroEquation(nu=0.002, dim=3, time=False, max_distance=0.5)
        ns = NavierStokes(nu=ze.equations["nu"], rho=1.0, dim=3, time=False)
        navier_stokes_nodes = ns.make_nodes() + ze.make_nodes()
    else:
        ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
        navier_stokes_nodes = ns.make_nodes()
    normal_dot_vel = NormalDotVec()

    # make network arch
    if cfg.custom.parameterized:
        input_keys = [
            Key("x"),
            Key("y"),
            Key("z"),
            Key("fin_height_m"),
            Key("fin_height_s"),
            Key("fin_length_m"),
            Key("fin_length_s"),
            Key("fin_thickness_m"),
            Key("fin_thickness_s"),
        ]
    else:
        input_keys = [Key("x"), Key("y"), Key("z")]
    flow_net = FullyConnectedArch(
        input_keys=input_keys, output_keys=[Key("u"), Key("v"), Key("w"), Key("p")]
    )

    # make list of nodes to unroll graph on
    flow_nodes = (
        navier_stokes_nodes
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    geo = ThreeFin(parameterized=cfg.custom.parameterized)

    # params for simulation
    # fluid params
    inlet_vel = 1.0
    volumetric_flow = 1.0

    # make flow domain
    flow_domain = Domain()

    # inlet
    u_profile = inlet_vel * tanh((0.5 - Abs(y)) / 0.02) * tanh((0.5 - Abs(z)) / 0.02)
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.inlet,
        outvar={"u": u_profile, "v": 0, "w": 0},
        batch_size=cfg.batch_size.Inlet,
        criteria=Eq(x, channel_origin[0]),
        lambda_weighting={
            "u": 1.0,
            "v": 1.0,
            "w": 1.0,
        },  # weight zero on edges
        parameterization=geo.pr,
        batch_per_epoch=5000,
    )
    flow_domain.add_constraint(constraint_inlet, "inlet")

    # outlet
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.Outlet,
        criteria=Eq(x, channel_origin[0] + channel_dim[0]),
        lambda_weighting={"p": 1.0},
        parameterization=geo.pr,
        batch_per_epoch=5000,
    )
    flow_domain.add_constraint(constraint_outlet, "outlet")

    # no slip for channel walls
    no_slip = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.NoSlip,
        lambda_weighting={
            "u": 1.0,
            "v": 1.0,
            "w": 1.0,
        },  # weight zero on edges
        parameterization=geo.pr,
        batch_per_epoch=5000,
    )
    flow_domain.add_constraint(no_slip, "no_slip")

    # flow interior low res away from three fin
    lr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.InteriorLR,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        compute_sdf_derivatives=True,
        parameterization=geo.pr,
        batch_per_epoch=5000,
        criteria=Or(x < -1.1, x > 0.5),
    )
    flow_domain.add_constraint(lr_interior, "lr_interior")

    # flow interiror high res near three fin
    hr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_z": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.InteriorHR,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        compute_sdf_derivatives=True,
        parameterization=geo.pr,
        batch_per_epoch=5000,
        criteria=And(x > -1.1, x < 0.5),
    )
    flow_domain.add_constraint(hr_interior, "hr_interior")

    # integral continuity
    def integral_criteria(invar, params):
        sdf = geo.geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.integral_plane,
        outvar={"normal_dot_vel": volumetric_flow},
        batch_size=5,
        integral_batch_size=cfg.batch_size.IntegralContinuity,
        criteria=integral_criteria,
        lambda_weighting={"normal_dot_vel": 1.0},
        parameterization={**geo.pr, **{x_pos: (-1.1, 0.1)}},
        fixed_dataset=False,
        num_workers=4,
    )
    flow_domain.add_constraint(integral_continuity, "integral_continuity")

    # flow data
    file_path = "../openfoam/"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "Points:0": "x",
            "Points:1": "y",
            "Points:2": "z",
            "U:0": "u",
            "U:1": "v",
            "U:2": "w",
            "p_rgh": "p",
        }
        if cfg.custom.turbulent:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_zeroEq_re500_fluid.csv"),
                mapping,
            )
        else:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_fluid0.csv"), mapping
            )
        openfoam_var = {key: value[0::4] for key, value in openfoam_var.items()}
        openfoam_var["x"] = openfoam_var["x"] + channel_origin[0]
        openfoam_var["y"] = openfoam_var["y"] + channel_origin[1]
        openfoam_var["z"] = openfoam_var["z"] + channel_origin[2]
        openfoam_var.update({"fin_height_m": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_height_s": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_thickness_m": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_thickness_s": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_length_m": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_var.update({"fin_length_s": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_invar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key
            in [
                "x",
                "y",
                "z",
                "fin_height_m",
                "fin_height_s",
                "fin_thickness_m",
                "fin_thickness_s",
                "fin_length_m",
                "fin_length_s",
            ]
        }
        openfoam_outvar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "w", "p"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=flow_nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
        )
        flow_domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # add pressure monitor
    invar_inlet_pressure = geo.integral_plane.sample_boundary(
        1024, parameterization={**fixed_param_ranges, **{x_pos: -2}}
    )
    pressure_monitor = PointwiseMonitor(
        invar_inlet_pressure,
        output_names=["p"],
        metrics={"inlet_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(pressure_monitor)

    # add pressure drop for design optimization
    # run only for parameterized cases and in eval mode
    if cfg.custom.parameterized and cfg.run_mode == "eval":
        # define candidate designs
        num_samples = cfg.custom.num_samples
        inference_param_tuple = itertools.product(
            np.linspace(*height_m_range, num_samples),
            np.linspace(*height_s_range, num_samples),
            np.linspace(*length_m_range, num_samples),
            np.linspace(*length_s_range, num_samples),
            np.linspace(*thickness_m_range, num_samples),
            np.linspace(*thickness_s_range, num_samples),
        )
        for (
            HS_height_m_,
            HS_height_s_,
            HS_length_m_,
            HS_length_s_,
            HS_thickness_m_,
            HS_thickness_s_,
        ) in inference_param_tuple:
            HS_height_m = float(HS_height_m_)
            HS_height_s = float(HS_height_s_)
            HS_length_m = float(HS_length_m_)
            HS_length_s = float(HS_length_s_)
            HS_thickness_m = float(HS_thickness_m_)
            HS_thickness_s = float(HS_thickness_s_)
            specific_param_ranges = {
                fin_height_m: HS_height_m,
                fin_height_s: HS_height_s,
                fin_length_m: HS_length_m,
                fin_length_s: HS_length_s,
                fin_thickness_m: HS_thickness_m,
                fin_thickness_s: HS_thickness_s,
            }

            # add metrics for front pressure
            plane_param_ranges = {
                **specific_param_ranges,
                **{x_pos: heat_sink_base_origin[0] - heat_sink_base_dim[0]},
            }
            metric = (
                "front_pressure"
                + str(HS_height_m)
                + "_"
                + str(HS_height_s)
                + "_"
                + str(HS_length_m)
                + "_"
                + str(HS_length_s)
                + "_"
                + str(HS_thickness_m)
                + "_"
                + str(HS_thickness_s)
            )
            invar_pressure = geo.integral_plane.sample_boundary(
                1024,
                parameterization=plane_param_ranges,
            )
            front_pressure_monitor = PointwiseMonitor(
                invar_pressure,
                output_names=["p"],
                metrics={metric: lambda var: torch.mean(var["p"])},
                nodes=flow_nodes,
            )
            flow_domain.add_monitor(front_pressure_monitor)

            # add metrics for back pressure
            plane_param_ranges = {
                **specific_param_ranges,
                **{x_pos: heat_sink_base_origin[0] + 2 * heat_sink_base_dim[0]},
            }
            metric = (
                "back_pressure"
                + str(HS_height_m)
                + "_"
                + str(HS_height_s)
                + "_"
                + str(HS_length_m)
                + "_"
                + str(HS_length_s)
                + "_"
                + str(HS_thickness_m)
                + "_"
                + str(HS_thickness_s)
            )
            invar_pressure = geo.integral_plane.sample_boundary(
                1024,
                parameterization=plane_param_ranges,
            )
            back_pressure_monitor = PointwiseMonitor(
                invar_pressure,
                output_names=["p"],
                metrics={metric: lambda var: torch.mean(var["p"])},
                nodes=flow_nodes,
            )
            flow_domain.add_monitor(back_pressure_monitor)

    # make solver
    flow_slv = Solver(cfg, flow_domain)

    # start flow solver
    flow_slv.solve()


if __name__ == "__main__":
    run()
