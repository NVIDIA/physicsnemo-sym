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

import time
import torch
from typing import Dict, List, Optional
from physicsnemo.sym.key import Key
from physicsnemo.sym.constants import diff
from physicsnemo.sym.node import Node
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.eq.derivatives import MeshlessFiniteDerivative


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, y, z = inputs["x"], inputs["y"], inputs["z"]
        return {
            "u": 1.5 * x * x + torch.sin(y) + torch.exp(z),
            "v": 2 * x * x + torch.cos(y) + torch.exp(-z),
            "w": 1.5 * x * x + torch.sin(y) + torch.exp(z),
            "p": 2 * x * x + torch.cos(y) + torch.exp(-z),
        }


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_keys: List[str] = [diff("u", "x"), diff("v", "y"), diff("w", "z")]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        divergence = (
            inputs[self.input_keys[0]]
            + inputs[self.input_keys[1]]
            + inputs[self.input_keys[2]]
        )
        return {"divergence_loss": torch.square(divergence).mean()}


def validate_divergence_loss(x, y, z, divergence_loss, rtol=1e-5, atol=1e-8):
    dudx = 3 * x
    dvdy = -torch.sin(y)
    dwdz = torch.exp(z)
    divergence_loss_exact = torch.square(dudx + dvdy + dwdz).mean()
    assert torch.allclose(divergence_loss, divergence_loss_exact, rtol, atol)


def test_graph():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up input coordinates
    batch_size = 128
    x = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)
    z = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)

    # Instantiate the model and compute outputs
    model = torch.jit.script(Model()).to(device)
    model_node = Node(["x", "y", "z"], ["u", "v", "w", "p"], model, name="Model")

    loss = torch.jit.script(Loss()).to(device)
    loss_node = Node(
        [diff("u", "x"), diff("v", "y"), diff("w", "z")],
        ["divergence_loss"],
        loss,
        name="Loss",
    )

    nodes = [model_node, loss_node]

    input_vars = [Key.from_str("x"), Key.from_str("y"), Key.from_str("z")]
    output_vars = [
        Key.from_str("u"),
        Key.from_str("v"),
        Key.from_str("w"),
        Key.from_str("p"),
        Key.from_str("divergence_loss"),
    ]

    graph = Graph(nodes, input_vars, output_vars)

    input_dict = dict(zip((str(v) for v in input_vars), [x, y, z]))
    output_dict = graph(input_dict)

    validate_divergence_loss(x, y, z, output_dict["divergence_loss"])


def test_graph_no_loss_node():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up input coordinates
    batch_size = 128
    x = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)
    z = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)

    # Instantiate the model and compute outputs
    model = torch.jit.script(Model()).to(device)
    model_node = Node(["x", "y", "z"], ["u", "v", "w", "p"], model, name="Model")

    loss = torch.jit.script(Loss()).to(device)
    loss_node = Node(
        [diff("u", "x"), diff("v", "y"), diff("w", "z")],
        ["divergence_loss"],
        loss,
        name="Loss",
    )

    nodes = [model_node]

    input_vars = [Key.from_str("x"), Key.from_str("y"), Key.from_str("z")]
    output_vars = [
        Key.from_str("u__x"),
        Key.from_str("v__y"),
        Key.from_str("w__z"),
    ]

    graph = Graph(nodes, input_vars, output_vars)

    input_dict = dict(zip((str(v) for v in input_vars), [x, y, z]))
    output_dict = graph(input_dict)

    # Calc loss manually
    loss = Loss()
    output_dict.update(loss(output_dict))

    validate_divergence_loss(x, y, z, output_dict["divergence_loss"])


def test_mfd_graph():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up input coordinates
    batch_size = 32
    x = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)
    z = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)

    # Instantiate the model and compute outputs
    model = torch.jit.script(Model()).to(device)
    model_node = Node(["x", "y", "z"], ["u", "v", "w", "p"], model, name="Model")

    loss = torch.jit.script(Loss()).to(device)
    loss_node = Node(
        [diff("u", "x"), diff("v", "y"), diff("w", "z")],
        ["divergence_loss"],
        loss,
        name="Loss",
    )

    nodes = [model_node, loss_node]

    input_vars = [Key.from_str("x"), Key.from_str("y"), Key.from_str("z")]
    output_vars = [
        Key.from_str("u"),
        Key.from_str("v"),
        Key.from_str("w"),
        Key.from_str("p"),
        Key.from_str("divergence_loss"),
    ]

    # Test meshless finite derivative node in graph
    mfd_node = MeshlessFiniteDerivative.make_node(
        node_model=model,
        derivatives=[
            Key("u", derivatives=[Key("x")]),
            Key("v", derivatives=[Key("y")]),
            Key("w", derivatives=[Key("z")]),
        ],
        dx=0.001,
    )

    graph = Graph(nodes + [mfd_node], input_vars, output_vars)

    input_dict = dict(zip((str(v) for v in input_vars), [x, y, z]))
    output_dict = graph(input_dict)
    # Need to raise allclose atol here because finite diff is approximate
    validate_divergence_loss(x, y, z, output_dict["divergence_loss"], atol=1e-3)


if __name__ == "__main__":
    test_graph()
    test_graph_no_loss_node()
    test_mfd_graph()
