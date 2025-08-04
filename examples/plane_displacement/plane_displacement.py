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
from sympy import Symbol, Eq

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.key import Key
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.dataset import DictVariationalDataset
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    VariationalConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.utils.io.plotter import InferencerPlotter
from physicsnemo.sym.loss import Loss

# VPINN imports
from physicsnemo.sym.utils.vpinn.test_functions import (
    Test_Function,
    Legendre_test,
    Trig_test,
    Vector_Test,
)
from physicsnemo.sym.utils.vpinn.integral import tensor_int

x, y = Symbol("x"), Symbol("y")
# parameters
E = 10.0  # MPa
nu = 0.2
lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
domain_origin = (-0.5, -0.5)
domain_dim = (1, 1)
# bounds
bounds_x = (domain_origin[0], domain_origin[0] + domain_dim[0])
bounds_y = (domain_origin[1], domain_origin[1] + domain_dim[1])


class DGLoss(Loss):
    def __init__(self):
        super().__init__()
        test_fn = Test_Function(
            name_ord_dict={
                Legendre_test: [k for k in range(10)],
                Trig_test: [k for k in range(10)],
            },
            box=[
                [domain_origin[0], domain_origin[1]],
                [domain_origin[0] + domain_dim[0], domain_origin[1] + domain_dim[1]],
            ],
            diff_list=["grad"],
        )

        self.v = Vector_Test(test_fn, test_fn, mix=0.02)

    def forward(
        self,
        list_invar,
        list_outvar,
        step: int,
    ):
        torch.cuda.nvtx.range_push("Make_DGLoss")
        torch.cuda.nvtx.range_push("Make_DGLoss_Get_Data")
        # self.v.sample_vector_test()
        # get points on the interior
        x_interior = list_invar[2]["x"]
        y_interior = list_invar[2]["y"]
        area_interior = list_invar[2]["area"]

        # compute solution for the interior
        u_x_interior = list_outvar[2]["u__x"]
        u_y_interior = list_outvar[2]["u__y"]
        v_x_interior = list_outvar[2]["v__x"]
        v_y_interior = list_outvar[2]["v__y"]

        # get points on the boundary
        x_bottom_dir = list_invar[0]["x"]
        y_bottom_dir = list_invar[0]["y"]
        normal_x_bottom_dir = list_invar[0]["normal_x"]
        normal_y_bottom_dir = list_invar[0]["normal_y"]
        area_bottom_dir = list_invar[0]["area"]

        x_top_dir = list_invar[1]["x"]
        y_top_dir = list_invar[1]["y"]
        normal_x_top_dir = list_invar[1]["normal_x"]
        normal_y_top_dir = list_invar[1]["normal_y"]
        area_top_dir = list_invar[1]["area"]

        # compute solution for the boundary
        u_x_bottom_dir = list_outvar[0]["u__x"]
        u_y_bottom_dir = list_outvar[0]["u__y"]
        v_x_bottom_dir = list_outvar[0]["v__x"]
        v_y_bottom_dir = list_outvar[0]["v__y"]

        u_x_top_dir = list_outvar[1]["u__x"]
        u_y_top_dir = list_outvar[1]["u__y"]
        v_x_top_dir = list_outvar[1]["v__x"]
        v_y_top_dir = list_outvar[1]["v__y"]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Make_DGLoss_Test_Function")
        # test functions
        vx_x_interior, vy_x_interior = self.v.eval_test("vx", x_interior, y_interior)
        vx_y_interior, vy_y_interior = self.v.eval_test("vy", x_interior, y_interior)
        vx_bottom_dir, vy_bottom_dir = self.v.eval_test("v", x_bottom_dir, y_bottom_dir)
        vx_top_dir, vy_top_dir = self.v.eval_test("v", x_top_dir, y_top_dir)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Make_DGLoss_Computation")
        w_z_interior = -lambda_ / (lambda_ + 2 * mu) * (u_x_interior + v_y_interior)
        sigma_xx_interior = (
            lambda_ * (u_x_interior + v_y_interior + w_z_interior)
            + 2 * mu * u_x_interior
        )
        sigma_yy_interior = (
            lambda_ * (u_x_interior + v_y_interior + w_z_interior)
            + 2 * mu * v_y_interior
        )
        sigma_xy_interior = mu * (u_y_interior + v_x_interior)

        w_z_bottom_dir = (
            -lambda_ / (lambda_ + 2 * mu) * (u_x_bottom_dir + v_y_bottom_dir)
        )
        sigma_xx_bottom_dir = (
            lambda_ * (u_x_bottom_dir + v_y_bottom_dir + w_z_bottom_dir)
            + 2 * mu * u_x_bottom_dir
        )
        sigma_yy_bottom_dir = (
            lambda_ * (u_x_bottom_dir + v_y_bottom_dir + w_z_bottom_dir)
            + 2 * mu * v_y_bottom_dir
        )
        sigma_xy_bottom_dir = mu * (u_y_bottom_dir + v_x_bottom_dir)

        w_z_top_dir = -lambda_ / (lambda_ + 2 * mu) * (u_x_top_dir + v_y_top_dir)
        sigma_xx_top_dir = (
            lambda_ * (u_x_top_dir + v_y_top_dir + w_z_top_dir) + 2 * mu * u_x_top_dir
        )
        sigma_yy_top_dir = (
            lambda_ * (u_x_top_dir + v_y_top_dir + w_z_top_dir) + 2 * mu * v_y_top_dir
        )
        sigma_xy_top_dir = mu * (u_y_top_dir + v_x_top_dir)

        traction_x_bottom_dir = (
            sigma_xx_bottom_dir * normal_x_bottom_dir
            + sigma_xy_bottom_dir * normal_y_bottom_dir
        )
        traction_y_bottom_dir = (
            sigma_xy_bottom_dir * normal_x_bottom_dir
            + sigma_yy_bottom_dir * normal_y_bottom_dir
        )
        traction_x_top_dir = (
            sigma_xx_top_dir * normal_x_top_dir + sigma_xy_top_dir * normal_y_top_dir
        )
        traction_y_top_dir = (
            sigma_xy_top_dir * normal_x_top_dir + sigma_yy_top_dir * normal_y_top_dir
        )
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Make_DGLoss_Integral")
        interior_loss = tensor_int(
            area_interior,
            sigma_xx_interior * vx_x_interior
            + sigma_yy_interior * vy_y_interior
            + sigma_xy_interior * (vx_y_interior + vy_x_interior),
        )
        boundary_loss1 = tensor_int(
            area_bottom_dir,
            traction_x_bottom_dir * vx_bottom_dir
            + traction_y_bottom_dir * vy_bottom_dir,
        )
        boundary_loss2 = tensor_int(
            area_top_dir,
            traction_x_top_dir * vx_top_dir + traction_y_top_dir * vy_top_dir,
        )
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Make_DGLoss_Register_Loss")
        losses = {
            "variational_plane": torch.abs(
                interior_loss - boundary_loss1 - boundary_loss2
            )
            .pow(2)
            .sum()
        }
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return losses


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    elasticity_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = [elasticity_net.make_node(name="elasticity_net")]
    # domain
    square = Rectangle(
        domain_origin,
        (domain_origin[0] + domain_dim[0], domain_origin[1] + domain_dim[1]),
    )
    geo = square

    # make domain
    domain = Domain()
    bottomBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0.0, "v": 0.0},
        batch_size=cfg.batch_size.bottom,
        batch_per_epoch=5000,
        lambda_weighting={"u": 10.0, "v": 10.0},
        criteria=Eq(y, domain_origin[1]),
    )
    domain.add_constraint(bottomBC, "bottomBC_differential")

    topBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0.0, "v": 0.1},
        batch_size=cfg.batch_size.top,
        batch_per_epoch=5000,
        lambda_weighting={"u": 10.0, "v": 10.0},
        criteria=Eq(y, domain_origin[1] + domain_dim[1])
        & (x <= domain_origin[0] + domain_dim[0] / 2.0),
    )
    domain.add_constraint(topBC, "topBC_differential")

    # register variational data
    batch_per_epoch = 1
    variational_datasets = {}
    batch_sizes = {}
    # bottomBC, index : 0
    invar = geo.sample_boundary(
        batch_per_epoch * cfg.batch_size.bottom,
        criteria=Eq(y, domain_origin[1]),
        quasirandom=True,
    )
    invar["area"] *= batch_per_epoch
    variational_datasets["bottom_bc"] = DictVariationalDataset(
        invar=invar,
        outvar_names=["u__x", "u__y", "v__x", "v__y"],
    )
    batch_sizes["bottom_bc"] = cfg.batch_size.bottom

    # topBC, index : 1
    invar = geo.sample_boundary(
        batch_per_epoch * cfg.batch_size.top,
        criteria=Eq(y, domain_origin[1] + domain_dim[1])
        & (x <= domain_origin[0] + domain_dim[0] / 2.0),
        quasirandom=True,
    )
    invar["area"] *= batch_per_epoch
    variational_datasets["top_bc"] = DictVariationalDataset(
        invar=invar,
        outvar_names=["u__x", "u__y", "v__x", "v__y"],
    )
    batch_sizes["top_bc"] = cfg.batch_size.top

    # Interior, index : 2
    invar = geo.sample_interior(
        batch_per_epoch * cfg.batch_size.interior,
        bounds={x: bounds_x, y: bounds_y},
        quasirandom=True,
    )
    invar["area"] *= batch_per_epoch
    variational_datasets["interior"] = DictVariationalDataset(
        invar=invar,
        outvar_names=["u__x", "u__y", "v__x", "v__y"],
    )
    batch_sizes["interior"] = cfg.batch_size.interior

    # make variational constraints
    variational_constraint = VariationalConstraint(
        datasets=variational_datasets,
        batch_sizes=batch_sizes,
        nodes=nodes,
        num_workers=1,
        loss=DGLoss(),
    )
    domain.add_constraint(variational_constraint, "variational")

    # add inferencer data
    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=geo.sample_interior(
            2 * cfg.batch_size.interior,
            bounds={x: bounds_x, y: bounds_y},
        ),
        output_names=["u", "v"],
        batch_size=2048,
        plotter=InferencerPlotter(),
    )
    domain.add_inferencer(inferencer)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
