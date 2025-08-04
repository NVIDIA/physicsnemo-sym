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

from sympy import Symbol, Eq

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Circle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.linear_elasticity import LinearElasticityPlaneStress


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # specify Panel properties
    E = 73.0 * 10**9  # Pa
    nu = 0.33
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))  # Pa
    mu_real = E / (2 * (1 + nu))  # Pa
    lambda_ = lambda_ / mu_real  # Dimensionless
    mu = 1.0  # Dimensionless

    # make list of nodes to unroll graph on
    le = LinearElasticityPlaneStress(lambda_=lambda_, mu=mu)
    elasticity_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("sigma_hoop")],
        output_keys=[
            Key("u"),
            Key("v"),
            Key("sigma_xx"),
            Key("sigma_yy"),
            Key("sigma_xy"),
        ],
        cfg=cfg.arch.fully_connected,
    )
    nodes = le.make_nodes() + [elasticity_net.make_node(name="elasticity_network")]

    # add constraints to solver
    # make geometry
    x, y, sigma_hoop = Symbol("x"), Symbol("y"), Symbol("sigma_hoop")
    panel_origin = (-0.5, -0.9)
    panel_dim = (1, 1.8)  # Panel width is the characteristic length.
    window_origin = (-0.125, -0.2)
    window_dim = (0.25, 0.4)
    panel_aux1_origin = (-0.075, -0.2)
    panel_aux1_dim = (0.15, 0.4)
    panel_aux2_origin = (-0.125, -0.15)
    panel_aux2_dim = (0.25, 0.3)
    hr_zone_origin = (-0.2, -0.4)
    hr_zone_dim = (0.4, 0.8)
    circle_nw_center = (-0.075, 0.15)
    circle_ne_center = (0.075, 0.15)
    circle_se_center = (0.075, -0.15)
    circle_sw_center = (-0.075, -0.15)
    circle_radius = 0.05
    panel = Rectangle(
        panel_origin, (panel_origin[0] + panel_dim[0], panel_origin[1] + panel_dim[1])
    )
    window = Rectangle(
        window_origin,
        (window_origin[0] + window_dim[0], window_origin[1] + window_dim[1]),
    )
    panel_aux1 = Rectangle(
        panel_aux1_origin,
        (
            panel_aux1_origin[0] + panel_aux1_dim[0],
            panel_aux1_origin[1] + panel_aux1_dim[1],
        ),
    )
    panel_aux2 = Rectangle(
        panel_aux2_origin,
        (
            panel_aux2_origin[0] + panel_aux2_dim[0],
            panel_aux2_origin[1] + panel_aux2_dim[1],
        ),
    )
    hr_zone = Rectangle(
        hr_zone_origin,
        (hr_zone_origin[0] + hr_zone_dim[0], hr_zone_origin[1] + hr_zone_dim[1]),
    )
    circle_nw = Circle(circle_nw_center, circle_radius)
    circle_ne = Circle(circle_ne_center, circle_radius)
    circle_se = Circle(circle_se_center, circle_radius)
    circle_sw = Circle(circle_sw_center, circle_radius)
    corners = (
        window - panel_aux1 - panel_aux2 - circle_nw - circle_ne - circle_se - circle_sw
    )
    window = window - corners
    geo = panel - window
    hr_geo = geo & hr_zone

    # Parameterization
    characteristic_length = panel_dim[0]
    characteristic_disp = 0.001 * window_dim[0]
    sigma_normalization = characteristic_length / (mu_real * characteristic_disp)
    sigma_hoop_lower = 46 * 10**6 * sigma_normalization
    sigma_hoop_upper = 56.5 * 10**6 * sigma_normalization
    sigma_hoop_range = (sigma_hoop_lower, sigma_hoop_upper)
    param_ranges = {sigma_hoop: sigma_hoop_range}
    inference_param_ranges = {sigma_hoop: 46 * 10**6 * sigma_normalization}

    # bounds
    bounds_x = (panel_origin[0], panel_origin[0] + panel_dim[0])
    bounds_y = (panel_origin[1], panel_origin[1] + panel_dim[1])
    hr_bounds_x = (hr_zone_origin[0], hr_zone_origin[0] + hr_zone_dim[0])
    hr_bounds_y = (hr_zone_origin[1], hr_zone_origin[1] + hr_zone_dim[1])

    # make domain
    domain = Domain()

    # left wall
    panel_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0.0, "traction_y": 0.0},
        batch_size=cfg.batch_size.panel_left,
        criteria=Eq(x, panel_origin[0]),
        parameterization=param_ranges,
    )
    domain.add_constraint(panel_left, "panel_left")

    # right wall
    panel_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0.0, "traction_y": 0.0},
        batch_size=cfg.batch_size.panel_right,
        criteria=Eq(x, panel_origin[0] + panel_dim[0]),
        parameterization=param_ranges,
    )
    domain.add_constraint(panel_right, "panel_right")

    # bottom wall
    panel_bottom = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"v": 0.0},
        batch_size=cfg.batch_size.panel_bottom,
        criteria=Eq(y, panel_origin[1]),
        parameterization=param_ranges,
    )
    domain.add_constraint(panel_bottom, "panel_bottom")

    # corner point
    panel_corner = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.panel_corner,
        criteria=Eq(x, panel_origin[0])
        & (y > panel_origin[1])
        & (y < panel_origin[1] + 1e-3),
        parameterization=param_ranges,
    )
    domain.add_constraint(panel_corner, "panel_corner")

    # top wall
    panel_top = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0.0, "traction_y": sigma_hoop},
        batch_size=cfg.batch_size.panel_top,
        criteria=Eq(y, panel_origin[1] + panel_dim[1]),
        parameterization=param_ranges,
    )
    domain.add_constraint(panel_top, "panel_top")

    # pannel window
    panel_window = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=window,
        outvar={"traction_x": 0.0, "traction_y": 0.0},
        batch_size=cfg.batch_size.panel_window,
        parameterization=param_ranges,
    )
    domain.add_constraint(panel_window, "panel_window")

    # low-resolution interior
    lr_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "equilibrium_x": 0.0,
            "equilibrium_y": 0.0,
            "stress_disp_xx": 0.0,
            "stress_disp_yy": 0.0,
            "stress_disp_xy": 0.0,
        },
        batch_size=cfg.batch_size.lr_interior,
        bounds={x: bounds_x, y: bounds_y},
        lambda_weighting={
            "equilibrium_x": Symbol("sdf"),
            "equilibrium_y": Symbol("sdf"),
            "stress_disp_xx": Symbol("sdf"),
            "stress_disp_yy": Symbol("sdf"),
            "stress_disp_xy": Symbol("sdf"),
        },
        parameterization=param_ranges,
    )
    domain.add_constraint(lr_interior, "lr_interior")

    # high-resolution interior
    hr_interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=hr_geo,
        outvar={
            "equilibrium_x": 0.0,
            "equilibrium_y": 0.0,
            "stress_disp_xx": 0.0,
            "stress_disp_yy": 0.0,
            "stress_disp_xy": 0.0,
        },
        batch_size=cfg.batch_size.hr_interior,
        bounds={x: hr_bounds_x, y: hr_bounds_y},
        lambda_weighting={
            "equilibrium_x": Symbol("sdf"),
            "equilibrium_y": Symbol("sdf"),
            "stress_disp_xx": Symbol("sdf"),
            "stress_disp_yy": Symbol("sdf"),
            "stress_disp_xy": Symbol("sdf"),
        },
        parameterization=param_ranges,
    )
    domain.add_constraint(hr_interior, "hr_interior")

    # add inferencer data
    invar_numpy = geo.sample_interior(
        100000,
        bounds={x: bounds_x, y: bounds_y},
        parameterization=inference_param_ranges,
    )
    point_cloud_inference = PointwiseInferencer(
        nodes=nodes,
        invar=invar_numpy,
        output_names=["u", "v", "sigma_xx", "sigma_yy", "sigma_xy"],
        batch_size=4096,
    )
    domain.add_inferencer(point_cloud_inference, "inf_data")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
