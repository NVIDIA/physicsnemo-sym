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
from sympy import Symbol, Eq, cos, sin, pi
from physicsnemo.sym.node import Node
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.geometry.primitives_3d import Plane
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    VariationalDomainConstraint,
)
from physicsnemo.sym.loss import Loss
from physicsnemo.sym.geometry.parameterization import Parameterization, Bounds

# TODO: Add some more complex geometery that is the union of multiple shapes to check boundary sampling


def test_PointwiseBoundaryConstraint():
    "define a sinusodial node, create pointwise boundary constraints over it and check their losses are zero"

    ntests = 10
    for fixed_dataset in [True, False]:
        # define sinusodial node
        x, y = Symbol("x"), Symbol("y")
        node = Node.from_sympy(cos(x) + sin(y), "u")

        # make geometry
        height = pi
        width = pi
        rec = Rectangle((0, 0), (width, height))

        # top wall
        top_wall = PointwiseBoundaryConstraint(
            nodes=[node],
            geometry=rec,
            outvar={"u": cos(x) + sin(height)},
            batch_size=1000,
            criteria=Eq(y, height),
            fixed_dataset=fixed_dataset,
            batch_per_epoch=2 * ntests,
        )

        # right wall
        right_wall = PointwiseBoundaryConstraint(
            nodes=[node],
            geometry=rec,
            outvar={"u": cos(width) + sin(y)},
            batch_size=1000,
            criteria=Eq(x, width),
            fixed_dataset=fixed_dataset,
            batch_per_epoch=2 * ntests,
        )

        # bottom wall
        bottom_wall = PointwiseBoundaryConstraint(
            nodes=[node],
            geometry=rec,
            outvar={"u": cos(x) + sin(0)},
            batch_size=1000,
            criteria=Eq(y, 0),
            fixed_dataset=fixed_dataset,
            batch_per_epoch=2 * ntests,
        )

        # left wall
        left_wall = PointwiseBoundaryConstraint(
            nodes=[node],
            geometry=rec,
            outvar={"u": cos(0) + sin(y)},
            batch_size=1000,
            criteria=Eq(x, 0),
            fixed_dataset=fixed_dataset,
            batch_per_epoch=2 * ntests,
        )

        height = float(height)
        width = float(width)
        for _ in range(ntests):
            # check losses are zero
            top_wall.load_data()
            top_wall.forward()
            loss = top_wall.loss(step=0)
            assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-5, atol=1e-5)

            right_wall.load_data()
            right_wall.forward()
            loss = right_wall.loss(step=0)
            assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-5, atol=1e-5)

            bottom_wall.load_data()
            bottom_wall.forward()
            loss = bottom_wall.loss(step=0)
            assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-5, atol=1e-5)

            left_wall.load_data()
            left_wall.forward()
            loss = left_wall.loss(step=0)
            assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-5, atol=1e-5)

            # check invars correct
            invar, _, _ = next(top_wall.dataloader)
            assert torch.allclose(
                invar["y"], height * torch.ones_like(invar["y"]), rtol=1e-5, atol=1e-5
            )
            assert torch.all(torch.logical_and(invar["x"] <= width, invar["x"] >= 0))

            invar, _, _ = next(right_wall.dataloader)
            assert torch.allclose(
                invar["x"], width * torch.ones_like(invar["x"]), rtol=1e-5, atol=1e-5
            )
            assert torch.all(torch.logical_and(invar["y"] <= height, invar["y"] >= 0))

            invar, _, _ = next(bottom_wall.dataloader)
            assert torch.allclose(
                invar["y"], torch.zeros_like(invar["y"]), rtol=1e-5, atol=1e-5
            )
            assert torch.all(torch.logical_and(invar["x"] <= width, invar["x"] >= 0))

            invar, _, _ = next(left_wall.dataloader)
            assert torch.allclose(
                invar["x"], torch.zeros_like(invar["x"]), rtol=1e-5, atol=1e-5
            )
            assert torch.all(torch.logical_and(invar["y"] <= height, invar["y"] >= 0))


def test_PointwiseInteriorConstraint():
    "define a sinusodial node, create pointwise interior constraint over it and check its loss is zero"

    ntests = 10
    for fixed_dataset in [True, False]:
        # define sinusodial node
        x, y = Symbol("x"), Symbol("y")
        node = Node.from_sympy(cos(x) + sin(y), "u")

        # make geometry
        height = 3.14159
        width = 3.14159
        rec = Rectangle((0, 0), (width, height))

        constraint = PointwiseInteriorConstraint(
            nodes=[node],
            geometry=rec,
            outvar={"u": cos(x) + sin(y)},
            bounds=Bounds({x: (0, width), y: (0, height)}),
            batch_size=1000,
            fixed_dataset=fixed_dataset,
            batch_per_epoch=2 * ntests,
        )

        height = float(height)
        width = float(width)
        for _ in range(ntests):
            # check loss is zero
            constraint.load_data()
            constraint.forward()
            loss = constraint.loss(step=0)
            assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-5, atol=1e-5)

            # check invar correct
            invar, _, _ = next(constraint.dataloader)
            assert torch.all(torch.logical_and(invar["x"] <= width, invar["x"] >= 0))
            assert torch.all(torch.logical_and(invar["y"] <= height, invar["y"] >= 0))


def test_IntegralBoundaryConstraint():
    "define a parabola node, create integral boundary constraint over it and check its loss is zero"

    ntests = 10
    for fixed_dataset in [True, False]:
        # define parabola node
        node = Node.from_sympy(Symbol("z") ** 2, "u")

        # make geometry
        plane = Plane((0, 0, 0), (0, 2, 1), 1)

        # make constraint
        constraint = IntegralBoundaryConstraint(
            nodes=[node],
            geometry=plane,
            outvar={"u": 1.0 / 3.0},
            batch_size=1,
            integral_batch_size=100000,
            batch_per_epoch=ntests,
            fixed_dataset=fixed_dataset,
            criteria=Symbol("y") > 1,
        )

        for _ in range(ntests):
            # check loss is zero
            constraint.load_data()
            constraint.forward()
            loss = constraint.loss(step=0)
            assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-3, atol=1e-3)

        # define parabola node
        node = Node.from_sympy(Symbol("z") ** 3 + Symbol("y") ** 3, "u")

        # make geometry
        z_len = Symbol("z_len")
        y_len = Symbol("y_len")
        plane = Plane((0, -y_len, -z_len), (0, y_len, z_len), 1)

        # make constraint
        constraint = IntegralBoundaryConstraint(
            nodes=[node],
            geometry=plane,
            outvar={"u": 0},
            batch_size=1,
            integral_batch_size=100000,
            batch_per_epoch=ntests,
            fixed_dataset=fixed_dataset,
            parameterization=Parameterization({y_len: (0.1, 1.0), z_len: (0.1, 1.0)}),
        )

        for _ in range(ntests):
            # check loss is zero
            constraint.load_data()
            constraint.forward()
            loss = constraint.loss(step=0)
            assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-3, atol=1e-3)


def test_VariationalDomainConstraint():
    "define a parabola node, create variational domain constraint over it and check its loss is zero"

    ntests = 10

    # define parabola node
    x, y = Symbol("x"), Symbol("y")
    node = Node.from_sympy(x**2 + y**2, "u")

    # make geometry
    rec = Rectangle((-0.5, -0.5), (0.5, 0.5))

    # define variational loss
    class VariationalLoss(Loss):
        "fake loss for testing only"

        def forward(self, list_invar, list_outvar, step):
            losses = []
            for invar, outvar in zip(list_invar, list_outvar):
                expected = invar["x"] ** 2 + invar["y"] ** 2
                losses.append(torch.sum(outvar["u"] - expected))
            return {"u": sum(losses)}

    # make constraint
    constraint = VariationalDomainConstraint(
        nodes=[node],
        geometry=rec,
        outvar_names=["u"],
        boundary_batch_size=1000,
        interior_batch_size=2000,
        batch_per_epoch=ntests,
        interior_bounds=Bounds({x: (-0.5, 0.5), y: (-0.5, 0.5)}),
        loss=VariationalLoss(),
    )

    for _ in range(ntests):
        # check loss is zero
        constraint.load_data()
        constraint.forward()
        loss = constraint.loss(step=0)
        assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_PointwiseBoundaryConstraint()

    test_PointwiseInteriorConstraint()

    test_IntegralBoundaryConstraint()

    test_VariationalDomainConstraint()
