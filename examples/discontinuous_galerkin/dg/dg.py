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

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.geometry import Bounds
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.utils.vpinn.test_functions import (
    RBF_Function,
    Test_Function,
    Legendre_test,
    Trig_test,
)
from physicsnemo.sym.utils.vpinn.integral import tensor_int, Quad_Rect, Quad_Collection
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    VariationalConstraint,
)
from physicsnemo.sym.dataset import DictVariationalDataset
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from physicsnemo.sym.loss import Loss
from sympy import Symbol, Heaviside, Eq
import numpy as np
import quadpy


# custom variational loss
class DGLoss(Loss):
    def __init__(self, test_function):
        super().__init__()
        # make test function
        self.test_function = test_function
        if test_function == "rbf":
            self.v = RBF_Function(dim=2, diff_list=["grad"])
            self.eps = 10.0
        elif test_function == "legendre":
            self.v = Test_Function(
                name_ord_dict={
                    Legendre_test: [k for k in range(10)],
                    Trig_test: [k for k in range(5)],
                },
                diff_list=["grad"],
            )

    def forward(
        self,
        list_invar,
        list_outvar,
        step: int,
    ):
        # calculate test function
        if self.test_function == "rbf":
            v_outside = self.v.eval_test(
                "v",
                x=list_invar[0]["x"],
                y=list_invar[0]["y"],
                x_center=list_invar[3]["x"],
                y_center=list_invar[3]["y"],
                eps=self.eps,
            )
            v_center = self.v.eval_test(
                "v",
                x=list_invar[1]["x"],
                y=list_invar[1]["y"],
                x_center=list_invar[3]["x"],
                y_center=list_invar[3]["y"],
                eps=self.eps,
            )
            v_interior = self.v.eval_test(
                "v",
                x=list_invar[2]["x"],
                y=list_invar[2]["y"],
                x_center=list_invar[3]["x"],
                y_center=list_invar[3]["y"],
                eps=self.eps,
            )
            vx_interior = self.v.eval_test(
                "vx",
                x=list_invar[2]["x"],
                y=list_invar[2]["y"],
                x_center=list_invar[3]["x"],
                y_center=list_invar[3]["y"],
                eps=self.eps,
            )
            vy_interior = self.v.eval_test(
                "vy",
                x=list_invar[2]["x"],
                y=list_invar[2]["y"],
                x_center=list_invar[3]["x"],
                y_center=list_invar[3]["y"],
                eps=self.eps,
            )
        elif self.test_function == "legendre":
            v_outside = self.v.eval_test(
                "v", x=list_invar[0]["x"], y=list_invar[0]["y"]
            )
            v_center = self.v.eval_test("v", x=list_invar[1]["x"], y=list_invar[1]["y"])
            v_interior = self.v.eval_test(
                "v", x=list_invar[2]["x"], y=list_invar[2]["y"]
            )
            vx_interior = self.v.eval_test(
                "vx", x=list_invar[2]["x"], y=list_invar[2]["y"]
            )
            vy_interior = self.v.eval_test(
                "vy", x=list_invar[2]["x"], y=list_invar[2]["y"]
            )

        # calculate du/dn on surface
        dudn = (
            list_invar[0]["normal_x"] * list_outvar[0]["u__x"]
            + list_invar[0]["normal_y"] * list_outvar[0]["u__y"]
        )

        # form integrals of interior
        f = -2.0
        uxvx = list_outvar[2]["u__x"] * vx_interior
        uyvy = list_outvar[2]["u__y"] * vy_interior
        fv = f * v_interior

        # calculate integrals
        int_outside = tensor_int(list_invar[0]["area"], v_outside, dudn)
        int_center = tensor_int(list_invar[1]["area"], 2.0 * v_center)
        int_interior = tensor_int(list_invar[2]["area"], uxvx + uyvy - fv)

        losses = {
            "variational_poisson": torch.abs(int_interior - int_center - int_outside)
            .pow(2)
            .sum()
        }
        return losses


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    df = Diffusion(T="u", D=1.0, Q=-2.0, dim=2, time=False)
    dg_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = df.make_nodes() + [dg_net.make_node(name="dg_net")]

    # add constraints to solver
    x, y = Symbol("x"), Symbol("y")

    # make geometry
    rec_1 = Rectangle((0, 0), (0.5, 1))
    rec_2 = Rectangle((0.5, 0), (1, 1))
    rec = rec_1 + rec_2

    # make training domain for traditional PINN
    eps = 0.02
    Rectangle((0 + eps, 0 + eps), (0.5 - eps, 1 - eps)) + Rectangle(
        (0.5 + eps, 0 + eps), (1 - eps, 1 - eps)
    )

    # make domain
    domain = Domain()

    # PINN constraint
    # interior = PointwiseInteriorConstraint(
    #     nodes=nodes,
    #     geometry=rec_pinn,
    #     outvar={"diffusion_u": 0},
    #     batch_size=4000,
    #     bounds={x: (0 + eps, 1 - eps), y: (0 + eps, 1 - eps)},
    #     lambda_weighting={"diffusion_u": 1.},
    # )
    # domain.add_constraint(interior, "interior")

    # exterior boundary
    g = ((x - 1) ** 2 * Heaviside(x - 0.5)) + (x**2 * Heaviside(-x + 0.5))
    boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": g},
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={"u": 10.0},  # weight edges to be zero
        criteria=~Eq(x, 0.5),
    )
    domain.add_constraint(boundary, "boundary")

    batch_per_epoch = 100
    variational_datasets = {}
    batch_sizes = {}
    # Middle line boundary
    invar = rec.sample_boundary(
        batch_per_epoch * cfg.batch_size.boundary, criteria=~Eq(x, 0.5)
    )
    invar["area"] *= batch_per_epoch
    variational_datasets["boundary1"] = DictVariationalDataset(
        invar=invar,
        outvar_names=["u__x", "u__y"],
    )
    batch_sizes["boundary1"] = cfg.batch_size.boundary
    # Middle line boundary
    invar = rec_1.sample_boundary(
        batch_per_epoch * cfg.batch_size.boundary, criteria=Eq(x, 0.5)
    )
    invar["area"] *= batch_per_epoch
    variational_datasets["boundary2"] = DictVariationalDataset(
        invar=invar,
        outvar_names=["u__x"],
    )
    batch_sizes["boundary2"] = cfg.batch_size.boundary

    # Interior points
    if cfg.training.use_quadratures:
        paras = [
            [
                [[0, 0.5], [0, 1]],
                20,
                True,
                lambda n: quadpy.c2.product(quadpy.c1.gauss_legendre(n)),
            ],
            [
                [[0.5, 1], [0, 1]],
                20,
                True,
                lambda n: quadpy.c2.product(quadpy.c1.gauss_legendre(n)),
            ],
        ]
        quad_rec = Quad_Collection(Quad_Rect, paras)
        invar = {
            "x": quad_rec.points_numpy[:, 0:1],
            "y": quad_rec.points_numpy[:, 1:2],
            "area": np.expand_dims(quad_rec.weights_numpy, -1),
        }

        variational_datasets["interior"] = DictVariationalDataset(
            invar=invar,
            outvar_names=["u__x", "u__y"],
        )
        batch_sizes["interior"] = min(
            [quad_rec.points_numpy.shape[0], cfg.batch_size.interior]
        )
    else:
        invar = rec.sample_interior(
            batch_per_epoch * cfg.batch_size.interior,
            bounds=Bounds({x: (0.0, 1.0), y: (0.0, 1.0)}),
        )
        invar["area"] *= batch_per_epoch
        variational_datasets["interior"] = DictVariationalDataset(
            invar=invar,
            outvar_names=["u__x", "u__y"],
        )
        batch_sizes["interior"] = cfg.batch_size.interior

    # Add points for RBF
    if cfg.training.test_function == "rbf":
        invar = rec.sample_interior(
            batch_per_epoch * cfg.batch_size.rbf_functions,
            bounds=Bounds({x: (0.0, 1.0), y: (0.0, 1.0)}),
        )
        invar["area"] *= batch_per_epoch
        variational_datasets["rbf"] = DictVariationalDataset(
            invar=invar,
            outvar_names=[],
        )
        batch_sizes["rbf"] = cfg.batch_size.rbf_functions

    variational_constraint = VariationalConstraint(
        datasets=variational_datasets,
        batch_sizes=batch_sizes,
        nodes=nodes,
        num_workers=1,
        loss=DGLoss(cfg.training.test_function),
    )
    domain.add_constraint(variational_constraint, "variational")

    # add validation data
    delta_x = 0.01
    delta_y = 0.01
    x0 = np.arange(0, 1, delta_x)
    y0 = np.arange(0, 1, delta_y)
    x_grid, y_grid = np.meshgrid(x0, y0)
    x_grid = np.expand_dims(x_grid.flatten(), axis=-1)
    y_grid = np.expand_dims(y_grid.flatten(), axis=-1)
    u = np.where(x_grid <= 0.5, x_grid**2, (x_grid - 1) ** 2)
    invar_numpy = {"x": x_grid, "y": y_grid}
    outvar_numpy = {"u": u}
    openfoam_validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_numpy,
        true_outvar=outvar_numpy,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(openfoam_validator)

    # add inferencer data
    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=invar_numpy,
        output_names=["u"],
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
