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
    Test_Function,
    Legendre_test,
    Trig_test,
)
from physicsnemo.sym.utils.vpinn.integral import tensor_int
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    VariationalDomainConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.utils.io.plotter import InferencerPlotter
from physicsnemo.sym.loss import Loss
from sympy import Symbol
from physicsnemo.sym.constants import tf_dt


# custom variational loss
class DGLoss(Loss):
    def __init__(self):
        super().__init__()
        # make test function
        self.v = Test_Function(
            name_ord_dict={
                Legendre_test: [k for k in range(10)],
                Trig_test: [k for k in range(5)],
            },
            box=[[-0.5, -0.5], [0.5, 0.5]],
            diff_list=["grad"],
        )

    def forward(
        self,
        list_invar,
        list_outvar,
        step: int,
    ):
        # calculate test function
        v_outside = self.v.eval_test("v", x=list_invar[0]["x"], y=list_invar[0]["y"])
        vx_interior = self.v.eval_test("vx", x=list_invar[1]["x"], y=list_invar[1]["y"])
        vy_interior = self.v.eval_test("vy", x=list_invar[1]["x"], y=list_invar[1]["y"])
        v_source = self.v.eval_test(
            "v",
            x=torch.zeros(1, 1, device=list_invar[1]["x"].device, dtype=tf_dt),
            y=torch.zeros(1, 1, device=list_invar[1]["x"].device, dtype=tf_dt),
        )

        # calculate du/dn on surface
        dudn = (
            list_invar[0]["normal_x"] * list_outvar[0]["u__x"]
            + list_invar[0]["normal_y"] * list_outvar[0]["u__y"]
        )

        # form integrals of interior
        uxvx = list_outvar[1]["u__x"] * vx_interior
        uyvy = list_outvar[1]["u__y"] * vy_interior
        fv = v_source

        # calculate integrals
        int_outside = tensor_int(list_invar[0]["area"], v_outside, dudn)
        int_interior = tensor_int(list_invar[1]["area"], uxvx + uyvy) - fv

        losses = {
            "variational_poisson": torch.abs(int_interior - int_outside).pow(2).sum()
        }
        return losses


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # make list of nodes to unroll graph on
    df = Diffusion(T="u", D=1.0, dim=2, time=False)
    dg_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = df.make_nodes() + [dg_net.make_node(name="dg_net")]

    # add constraints to solver
    x, y = Symbol("x"), Symbol("y")

    # make geometry
    rec = Rectangle((-0.5, -0.5), (0.5, 0.5))

    # make domain
    domain = Domain()

    Wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0.0},
        lambda_weighting={"u": 10.0},
        batch_size=cfg.batch_size.boundary,
        fixed_dataset=False,
        batch_per_epoch=1,
        quasirandom=True,
    )
    domain.add_constraint(Wall, name="OutsideWall")

    # PINN constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"diffusion_u": 0.0},
        batch_size=cfg.batch_size.interior,
        bounds=Bounds({x: (-0.5, 0.5), y: (-0.5, 0.5)}),
        lambda_weighting={"diffusion_u": (x**2 + y**2)},
        fixed_dataset=False,
        batch_per_epoch=1,
        quasirandom=True,
    )
    domain.add_constraint(interior, "interior")

    # Variational contraint
    variational = VariationalDomainConstraint(
        nodes=nodes,
        geometry=rec,
        outvar_names=["u__x", "u__y"],
        boundary_batch_size=cfg.batch_size.boundary,
        interior_batch_size=cfg.batch_size.interior,
        interior_bounds=Bounds({x: (-0.5, 0.5), y: (-0.5, 0.5)}),
        loss=DGLoss(),
        batch_per_epoch=1,
        quasirandom=True,
    )
    domain.add_constraint(variational, "variational")

    # add inferencer data
    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=rec.sample_interior(10000),
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
