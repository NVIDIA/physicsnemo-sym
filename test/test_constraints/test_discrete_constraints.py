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
from sympy import Symbol

from physicsnemo.sym.node import Node
from physicsnemo.sym.domain.constraint.discrete import (
    SupervisedGridConstraint,
    DeepONetConstraint_Data,
    DeepONetConstraint_Physics,
)
from physicsnemo.sym.dataset import DictGridDataset


def test_SupervisedGridConstraint():
    "define a parabola node, create grid constraint over it and check its loss is zero"

    # define parabola node
    node = Node.from_sympy(Symbol("x") ** 2 + Symbol("y") ** 2, "u")

    # define 2D grid inputs
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))

    # define targets
    u = x**2 + y**2

    # make dataset
    dataset = DictGridDataset(
        invar={"x": x[np.newaxis, :], "y": y[np.newaxis, :]},
        outvar={"u": u[np.newaxis, :]},
    )

    # make constraint
    constraint = SupervisedGridConstraint(
        nodes=[node],
        dataset=dataset,
        batch_size=1,
    )

    # check loss is zero
    constraint.load_data()
    constraint.forward()
    loss = constraint.loss(step=0)
    assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-5, atol=1e-5)


def test_DeepONetConstraints():
    "define a parabola node, create deeponet constraints over it and check their losses are zero"

    # define parabola node
    node = Node.from_sympy(Symbol("x") ** 2 + Symbol("y") ** 2, "u")

    # define 2D grid inputs
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))

    # define targets
    u = x**2 + y**2

    # make dataset
    invar_branch = {"x": x[np.newaxis, :]}
    invar_trunk = {"y": y[np.newaxis, :]}
    outvar = {"u": u[np.newaxis, :]}

    # make constraint
    constraint = DeepONetConstraint_Data(
        nodes=[node],
        invar_branch=invar_branch,
        invar_trunk=invar_trunk,
        outvar=outvar,
        batch_size=1,
    )

    # check loss is zero
    constraint.load_data()
    constraint.forward()
    loss = constraint.loss(step=0)
    assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-5, atol=1e-5)

    # define parabola node
    class Parabola(torch.nn.Module):
        def forward(self, invar):
            x, y = invar["x"], invar["y"]
            u = x**2 + y**2
            u = u.reshape((-1, 1))  # reshape output
            return {"u": u}

    node = Node(inputs=["x", "y"], outputs="u", evaluate=Parabola())

    # make constraint
    constraint = DeepONetConstraint_Physics(
        nodes=[node],
        invar_branch=invar_branch,
        invar_trunk=invar_trunk,
        outvar=outvar,
        batch_size=1,
    )

    # check loss is zero
    constraint.load_data()
    constraint.forward()
    loss = constraint.loss(step=0)
    assert torch.isclose(loss["u"], torch.tensor(0.0), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_SupervisedGridConstraint()

    test_DeepONetConstraints()
