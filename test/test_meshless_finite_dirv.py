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

from physicsnemo.sym.eq.derivatives import MeshlessFiniteDerivative
from physicsnemo.sym.node import Node
from physicsnemo.sym.key import Key
from physicsnemo.sym.graph import Graph


class SineNet(torch.nn.Module):
    def forward(self, inputs):
        return {
            "y": (inputs["w"] ** 3) * torch.sin(inputs["x"]),
            "z": inputs["w"] * torch.cos(inputs["x"]),
        }


class ParabolaNet(torch.nn.Module):
    def forward(self, inputs):
        return {
            "p": (inputs["nu"] ** 3) + inputs["x"],
            "q": 2 * inputs["z"],
        }


def test_meshless_finite_deriv():
    # Define sinisoidal function node
    function_node = Node(
        inputs=[Key("w"), Key("x")],
        outputs=[Key("y"), Key("z")],
        evaluate=SineNet(),
        name="Test Node",
    )
    # Define finite derivative node
    deriv = MeshlessFiniteDerivative.make_node(
        node_model=function_node,
        derivatives=[
            Key("y", derivatives=[Key("x"), Key("w")]),
            Key("y", derivatives=[Key("x")]),
            Key("y", derivatives=[Key("w"), Key("w"), Key("w")]),
            Key("z", derivatives=[Key("x"), Key("x")]),
            Key("z", derivatives=[Key("x"), Key("x"), Key("x"), Key("x")]),
        ],
        dx=0.01,
        order=2,
        max_batch_size=15,
    )

    inputs = {"x": torch.randn(5, 1).double(), "w": torch.randn(5, 1).double()}
    inputs.update(function_node.evaluate(inputs))  # Forward to get y
    outputs = deriv.evaluate(inputs)

    assert torch.allclose(
        outputs["y__x"].double(), (inputs["w"] ** 3) * torch.cos(inputs["x"]), atol=1e-3
    ), "First derivative test failed"
    assert torch.allclose(
        outputs["z__x__x"].double(), -inputs["w"] * torch.cos(inputs["x"]), atol=1e-3
    ), "Second derivative test failed"
    assert torch.allclose(
        outputs["y__x__w"].double(),
        3 * inputs["w"] ** 2 * torch.cos(inputs["x"]),
        atol=1e-3,
    ), "Mixed second derivative test failed"
    assert torch.allclose(
        outputs["y__w__w__w"].double(), 6 * torch.sin(inputs["x"]), atol=1e-3
    ), "Third derivative test failed"
    assert torch.allclose(
        outputs["z__x__x__x__x"].double(),
        inputs["w"] * torch.cos(inputs["x"]),
        atol=1e-3,
    ), "Forth derivative test failed"

    # Testing forth order derivs
    deriv = MeshlessFiniteDerivative.make_node(
        node_model=function_node,
        derivatives=[
            Key("y", derivatives=[Key("x")]),
            Key("z", derivatives=[Key("x"), Key("x")]),
        ],
        dx=0.01,
        order=4,
        max_batch_size=20,
    )

    inputs = {"x": torch.randn(5, 1).double(), "w": torch.randn(5, 1).double()}
    inputs.update(function_node.evaluate(inputs))  # Forward to get y
    outputs = deriv.evaluate(inputs)

    assert torch.allclose(
        outputs["y__x"].double(), (inputs["w"] ** 3) * torch.cos(inputs["x"]), atol=1e-2
    ), "Forth order first derivative test failed"
    assert torch.allclose(
        outputs["z__x__x"].double(), -inputs["w"] * torch.cos(inputs["x"]), atol=1e-2
    ), "Forth order second derivative test failed"

    # Multinode checks
    function_node_2 = Node(
        inputs=[Key("nu"), Key("w"), Key("z")],
        outputs=[Key("p"), Key("q")],
        evaluate=ParabolaNet(),
        name="Test Node 2",
    )

    # Define finite derivative node
    deriv = MeshlessFiniteDerivative.make_node(
        node_model=Graph(
            nodes=[function_node, function_node_2],
            invar=[Key("w"), Key("x"), Key("nu")],
            req_names=[Key("p"), Key("q")],
        ),
        derivatives=[
            Key("p", derivatives=[Key("nu")]),
            Key("q", derivatives=[Key("x"), Key("w")]),
        ],
        dx=0.01,
    )

    inputs = {
        "x": torch.randn(5, 1).double(),
        "w": torch.randn(5, 1).double(),
        "nu": torch.randn(5, 1).double(),
    }
    outputs = deriv.evaluate(inputs)

    assert torch.allclose(
        outputs["p__nu"].double(), 3 * (inputs["nu"] ** 2), atol=1e-3
    ), "Multi-node first derivative test failed"
    assert torch.allclose(
        outputs["q__x__w"].double(), 2 * -torch.sin(inputs["x"]), atol=1e-3
    ), "Multi-node second derivative test failed"

    # Testing callable dx
    def dx_func(count: int):
        # First pass should be inaccurate
        if count == 1:
            return 10.0
        else:
            return 0.01

    deriv = MeshlessFiniteDerivative.make_node(
        node_model=function_node,
        derivatives=[
            Key("y", derivatives=[Key("x")]),
        ],
        dx=dx_func,
        order=2,
    )

    inputs = {"x": torch.randn(5, 1).double(), "w": torch.randn(5, 1).double()}
    inputs.update(function_node.evaluate(inputs))  # Forward to get y
    outputs_1 = deriv.evaluate(inputs)  # Inaccruate pass
    outputs_2 = deriv.evaluate(inputs)  # Accruate pass

    assert not torch.allclose(
        outputs_1["y__x"].double(),
        (inputs["w"] ** 3) * torch.cos(inputs["x"]),
        atol=1e-3,
    ), "Callable dx first derivative test failed"
    assert torch.allclose(
        outputs_2["y__x"].double(),
        (inputs["w"] ** 3) * torch.cos(inputs["x"]),
        atol=1e-3,
    ), "Callable dx first derivative test failed"


class GradModel(torch.nn.Module):
    def forward(self, inputs):
        return {"u": torch.cos(inputs["x"]), "v": torch.sin(inputs["y"])}


def test_meshless_finite_deriv_grads():
    # Testing gradient calcs
    # TODO: Grad tests for every grad
    model = GradModel()
    dx = 0.01
    deriv = MeshlessFiniteDerivative.make_node(
        node_model=model,
        derivatives=[
            Key("u", derivatives=[Key("x")]),
            Key("v", derivatives=[Key("y"), Key("y")]),
        ],
        dx=dx,
    )

    # == First derivative test ==
    inputs_mfd = {"x": torch.randn(5, 1).double(), "y": torch.randn(5, 1).double()}
    inputs_mfd["x"].requires_grad = True
    inputs_mfd["y"].requires_grad = True

    inputs_mfd.update(model.forward(inputs_mfd))
    outputs = deriv.evaluate(inputs_mfd)
    loss = outputs["u__x"].sum()
    loss.backward()

    # Auto diff calc
    inputs_auto = inputs_mfd["x"].detach().clone()
    inputs_auto.requires_grad = True
    inputs_up1 = torch.cos(inputs_auto + dx)
    inputs_um1 = torch.cos(inputs_auto - dx)
    grad = (inputs_up1 - inputs_um1) / (2.0 * dx)
    loss = grad.sum()
    loss.backward()

    assert torch.allclose(
        inputs_auto.grad,
        inputs_mfd["x"].grad,
        atol=1e-3,
    ), "First derivative gradient test failed"

    # == Second derivative test ==
    loss = outputs["v__y__y"].sum()
    loss.backward()

    # Auto diff calc
    inputs_auto = inputs_mfd["y"].detach().clone()
    inputs_auto.requires_grad = True
    inputs = torch.sin(inputs_auto)
    inputs_up1 = torch.sin(inputs_auto + dx)
    inputs_um1 = torch.sin(inputs_auto - dx)
    grad = (inputs_up1 - 2 * inputs + inputs_um1) / (dx * dx)
    loss = grad.sum()
    loss.backward()

    assert torch.allclose(
        inputs_auto.grad,
        inputs_mfd["y"].grad,
        atol=1e-3,
    ), "Second derivative gradient test failed"


if __name__ == "__main__":
    test_meshless_finite_deriv()
    test_meshless_finite_deriv_grads()
