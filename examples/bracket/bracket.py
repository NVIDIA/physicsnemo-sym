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

"""
Reference: https://www.mathworks.com/help/pde/ug/deflection-analysis-of-a-bracket.html
"""
import os
import warnings

import numpy as np
from sympy import Symbol, Eq, And

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_3d import Box, Cylinder
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.linear_elasticity import LinearElasticity


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Specify parameters
    nu = 0.3
    E = 100e9
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    mu_c = 0.01 * mu
    lambda_ = lambda_ / mu_c
    mu = mu / mu_c
    characteristic_length = 1.0
    characteristic_displacement = 1e-4
    sigma_normalization = characteristic_length / (characteristic_displacement * mu_c)
    T = -4e4 * sigma_normalization

    # make list of nodes to unroll graph on
    le = LinearElasticity(lambda_=lambda_, mu=mu, dim=3)
    disp_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w")],
        cfg=cfg.arch.fully_connected,
    )
    stress_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[
            Key("sigma_xx"),
            Key("sigma_yy"),
            Key("sigma_zz"),
            Key("sigma_xy"),
            Key("sigma_xz"),
            Key("sigma_yz"),
        ],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        le.make_nodes()
        + [disp_net.make_node(name="displacement_network")]
        + [stress_net.make_node(name="stress_network")]
    )

    # add constraints to solver
    # make geometry
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    support_origin = (-1, -1, -1)
    support_dim = (0.25, 2, 2)
    bracket_origin = (-0.75, -1, -0.1)
    bracket_dim = (1.75, 2, 0.2)
    cylinder_radius = 0.1
    cylinder_height = 2.0
    aux_lower_origin = (-0.75, -1, -0.1 - cylinder_radius)
    aux_lower_dim = (cylinder_radius, 2, cylinder_radius)
    aux_upper_origin = (-0.75, -1, 0.1)
    aux_upper_dim = (cylinder_radius, 2, cylinder_radius)
    cylinder_lower_center = (-0.75 + cylinder_radius, 0, 0)
    cylinder_upper_center = (-0.75 + cylinder_radius, 0, 0)
    cylinder_hole_radius = 0.7
    cylinder_hole_height = 0.5
    cylinder_hole_center = (0.125, 0, 0)

    support = Box(
        support_origin,
        (
            support_origin[0] + support_dim[0],
            support_origin[1] + support_dim[1],
            support_origin[2] + support_dim[2],
        ),
    )
    bracket = Box(
        bracket_origin,
        (
            bracket_origin[0] + bracket_dim[0],
            bracket_origin[1] + bracket_dim[1],
            bracket_origin[2] + bracket_dim[2],
        ),
    )
    aux_lower = Box(
        aux_lower_origin,
        (
            aux_lower_origin[0] + aux_lower_dim[0],
            aux_lower_origin[1] + aux_lower_dim[1],
            aux_lower_origin[2] + aux_lower_dim[2],
        ),
    )
    aux_upper = Box(
        aux_upper_origin,
        (
            aux_upper_origin[0] + aux_upper_dim[0],
            aux_upper_origin[1] + aux_upper_dim[1],
            aux_upper_origin[2] + aux_upper_dim[2],
        ),
    )
    cylinder_lower = Cylinder(cylinder_lower_center, cylinder_radius, cylinder_height)
    cylinder_upper = Cylinder(cylinder_upper_center, cylinder_radius, cylinder_height)
    cylinder_hole = Cylinder(
        cylinder_hole_center, cylinder_hole_radius, cylinder_hole_height
    )
    cylinder_lower = cylinder_lower.rotate(np.pi / 2, "x")
    cylinder_upper = cylinder_upper.rotate(np.pi / 2, "x")
    cylinder_lower = cylinder_lower.translate([0, 0, -0.1 - cylinder_radius])
    cylinder_upper = cylinder_upper.translate([0, 0, 0.1 + cylinder_radius])

    curve_lower = aux_lower - cylinder_lower
    curve_upper = aux_upper - cylinder_upper
    geo = support + bracket + curve_lower + curve_upper - cylinder_hole

    # Doamin bounds
    bounds_x = (-1, 1)
    bounds_y = (-1, 1)
    bounds_z = (-1, 1)
    bounds_support_x = (-1, -0.65)
    bounds_support_y = (-1, 1)
    bounds_support_z = (-1, 1)
    bounds_bracket_x = (-0.65, 1)
    bounds_bracket_y = (-1, 1)
    bounds_bracket_z = (-0.1, 0.1)

    # make domain
    domain = Domain()

    # back BC
    backBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.backBC,
        lambda_weighting={"u": 10, "v": 10, "w": 10},
        criteria=Eq(x, support_origin[0]),
    )
    domain.add_constraint(backBC, "backBC")

    # front BC
    frontBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0, "traction_y": 0, "traction_z": T},
        batch_size=cfg.batch_size.frontBC,
        criteria=Eq(x, bracket_origin[0] + bracket_dim[0]),
    )
    domain.add_constraint(frontBC, "frontBC")

    # surface BC
    surfaceBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"traction_x": 0, "traction_y": 0, "traction_z": 0},
        batch_size=cfg.batch_size.surfaceBC,
        criteria=And((x > support_origin[0]), (x < bracket_origin[0] + bracket_dim[0])),
    )
    domain.add_constraint(surfaceBC, "surfaceBC")

    # support interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "equilibrium_x": 0.0,
            "equilibrium_y": 0.0,
            "equilibrium_z": 0.0,
            "stress_disp_xx": 0.0,
            "stress_disp_yy": 0.0,
            "stress_disp_zz": 0.0,
            "stress_disp_xy": 0.0,
            "stress_disp_xz": 0.0,
            "stress_disp_yz": 0.0,
        },
        batch_size=cfg.batch_size.interior_support,
        bounds={x: bounds_support_x, y: bounds_support_y, z: bounds_support_z},
        lambda_weighting={
            "equilibrium_x": Symbol("sdf"),
            "equilibrium_y": Symbol("sdf"),
            "equilibrium_z": Symbol("sdf"),
            "stress_disp_xx": Symbol("sdf"),
            "stress_disp_yy": Symbol("sdf"),
            "stress_disp_zz": Symbol("sdf"),
            "stress_disp_xy": Symbol("sdf"),
            "stress_disp_xz": Symbol("sdf"),
            "stress_disp_yz": Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior_support")

    # bracket interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "equilibrium_x": 0.0,
            "equilibrium_y": 0.0,
            "equilibrium_z": 0.0,
            "stress_disp_xx": 0.0,
            "stress_disp_yy": 0.0,
            "stress_disp_zz": 0.0,
            "stress_disp_xy": 0.0,
            "stress_disp_xz": 0.0,
            "stress_disp_yz": 0.0,
        },
        batch_size=cfg.batch_size.interior_bracket,
        bounds={x: bounds_bracket_x, y: bounds_bracket_y, z: bounds_bracket_z},
        lambda_weighting={
            "equilibrium_x": Symbol("sdf"),
            "equilibrium_y": Symbol("sdf"),
            "equilibrium_z": Symbol("sdf"),
            "stress_disp_xx": Symbol("sdf"),
            "stress_disp_yy": Symbol("sdf"),
            "stress_disp_zz": Symbol("sdf"),
            "stress_disp_xy": Symbol("sdf"),
            "stress_disp_xz": Symbol("sdf"),
            "stress_disp_yz": Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior_bracket")

    # add validation data

    mapping = {
        "X Location (m)": "x",
        "Y Location (m)": "y",
        "Z Location (m)": "z",
        "Directional Deformation (m)": "u",
    }
    mapping_v = {"Directional Deformation (m)": "v"}
    mapping_w = {"Directional Deformation (m)": "w"}
    mapping_sxx = {"Normal Stress (Pa)": "sigma_xx"}
    mapping_syy = {"Normal Stress (Pa)": "sigma_yy"}
    mapping_szz = {"Normal Stress (Pa)": "sigma_zz"}
    mapping_sxy = {"Shear Stress (Pa)": "sigma_xy"}
    mapping_sxz = {"Shear Stress (Pa)": "sigma_xz"}
    mapping_syz = {"Shear Stress (Pa)": "sigma_yz"}

    file_path = "commercial_solver"
    if os.path.exists(to_absolute_path(file_path)):
        commercial_solver_var = csv_to_dict(
            to_absolute_path("commercial_solver/deformation_x.txt"),
            mapping,
            delimiter="\t",
        )
        commercial_solver_var_v = csv_to_dict(
            to_absolute_path("commercial_solver/deformation_y.txt"),
            mapping_v,
            delimiter="\t",
        )
        commercial_solver_var_w = csv_to_dict(
            to_absolute_path("commercial_solver/deformation_z.txt"),
            mapping_w,
            delimiter="\t",
        )
        commercial_solver_var_sxx = csv_to_dict(
            to_absolute_path("commercial_solver/normal_x.txt"),
            mapping_sxx,
            delimiter="\t",
        )
        commercial_solver_var_syy = csv_to_dict(
            to_absolute_path("commercial_solver/normal_y.txt"),
            mapping_syy,
            delimiter="\t",
        )
        commercial_solver_var_szz = csv_to_dict(
            to_absolute_path("commercial_solver/normal_z.txt"),
            mapping_szz,
            delimiter="\t",
        )
        commercial_solver_var_sxy = csv_to_dict(
            to_absolute_path("commercial_solver/shear_xy.txt"),
            mapping_sxy,
            delimiter="\t",
        )
        commercial_solver_var_sxz = csv_to_dict(
            to_absolute_path("commercial_solver/shear_xz.txt"),
            mapping_sxz,
            delimiter="\t",
        )
        commercial_solver_var_syz = csv_to_dict(
            to_absolute_path("commercial_solver/shear_yz.txt"),
            mapping_syz,
            delimiter="\t",
        )
        commercial_solver_var["x"] = commercial_solver_var["x"]
        commercial_solver_var["y"] = commercial_solver_var["y"]
        commercial_solver_var["z"] = commercial_solver_var["z"]
        commercial_solver_var["u"] = (
            commercial_solver_var["u"] / characteristic_displacement
        )
        commercial_solver_var["v"] = (
            commercial_solver_var_v["v"] / characteristic_displacement
        )
        commercial_solver_var["w"] = (
            commercial_solver_var_w["w"] / characteristic_displacement
        )
        commercial_solver_var["sigma_xx"] = (
            commercial_solver_var_sxx["sigma_xx"] * sigma_normalization
        )
        commercial_solver_var["sigma_yy"] = (
            commercial_solver_var_syy["sigma_yy"] * sigma_normalization
        )
        commercial_solver_var["sigma_zz"] = (
            commercial_solver_var_szz["sigma_zz"] * sigma_normalization
        )
        commercial_solver_var["sigma_xy"] = (
            commercial_solver_var_sxy["sigma_xy"] * sigma_normalization
        )
        commercial_solver_var["sigma_xz"] = (
            commercial_solver_var_sxz["sigma_xz"] * sigma_normalization
        )
        commercial_solver_var["sigma_yz"] = (
            commercial_solver_var_syz["sigma_yz"] * sigma_normalization
        )
        commercial_solver_invar = {
            key: value
            for key, value in commercial_solver_var.items()
            if key in ["x", "y", "z"]
        }
        commercial_solver_outvar = {
            key: value
            for key, value in commercial_solver_var.items()
            if key
            in [
                "u",
                "v",
                "w",
                "sigma_xx",
                "sigma_yy",
                "sigma_zz",
                "sigma_xy",
                "sigma_xz",
                "sigma_yz",
            ]
        }
        commercial_solver_validator = PointwiseValidator(
            nodes=nodes,
            invar=commercial_solver_invar,
            true_outvar=commercial_solver_outvar,
            batch_size=128,
        )
        domain.add_validator(commercial_solver_validator)

        # add inferencer data
        grid_inference = PointwiseInferencer(
            nodes=nodes,
            invar=commercial_solver_invar,
            output_names=[
                "u",
                "v",
                "w",
                "sigma_xx",
                "sigma_yy",
                "sigma_zz",
                "sigma_xy",
                "sigma_xz",
                "sigma_yz",
            ],
            batch_size=128,
        )
        domain.add_inferencer(grid_inference, "inf_data")
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/Modulus_sym_examples_supplemental_materials"
        )

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
