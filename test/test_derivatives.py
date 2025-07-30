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
from physicsnemo.sym.key import Key
from physicsnemo.sym.constants import diff
from physicsnemo.sym.eq.derivatives import Derivative


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return (
            1.5 * x * x + torch.sin(y) + torch.exp(z),
            2 * x * x + torch.cos(y) + torch.exp(-z),
            1.5 * x * x + torch.sin(y) + torch.exp(z),
            2 * x * x + torch.cos(y) + torch.exp(-z),
        )


def validate_gradients(
    x, y, z, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, dpdx, dpdy, dpdz
):
    # Check against exact solution
    assert torch.allclose(dudx, 3 * x), "x derivative of u failed"
    assert torch.allclose(dudy, torch.cos(y)), "y derivative of u  failed"
    assert torch.allclose(dudz, torch.exp(z)), "z derivative of u  failed"

    assert torch.allclose(dvdx, 4 * x), "x derivative of v failed"
    assert torch.allclose(dvdy, -torch.sin(y)), "y derivative of v failed"
    assert torch.allclose(dvdz, -torch.exp(-z)), "z derivative of v failed"

    assert torch.allclose(dwdx, 3 * x), "x derivative of w failed"
    assert torch.allclose(dwdy, torch.cos(y)), "y derivative of w failed"
    assert torch.allclose(dwdz, torch.exp(z)), "z derivative of w failed"

    assert torch.allclose(dpdx, 4 * x), "x derivative of p failed"
    assert torch.allclose(dpdy, -torch.sin(y)), "y derivative of p failed"
    assert torch.allclose(dpdz, -torch.exp(-z)), "z derivative of p failed"


def test_derivative_node():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up input coordinates
    batch_size = 128
    x = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)
    z = torch.rand(batch_size, 1, dtype=torch.float32, requires_grad=True).to(device)

    # Instantiate the model and compute outputs
    model = torch.jit.script(Model()).to(device)
    u, v, w, p = model(x, y, z)

    input_vars = [
        Key.from_str("x"),
        Key.from_str("y"),
        Key.from_str("z"),
        Key.from_str("u"),
        Key.from_str("v"),
        Key.from_str("w"),
        Key.from_str("p"),
    ]
    derivs = [
        Key.from_str(diff("u", "x")),
        Key.from_str(diff("u", "y")),
        Key.from_str(diff("u", "z")),
        Key.from_str(diff("v", "x")),
        Key.from_str(diff("v", "y")),
        Key.from_str(diff("v", "z")),
        Key.from_str(diff("w", "x")),
        Key.from_str(diff("w", "y")),
        Key.from_str(diff("w", "z")),
        Key.from_str(diff("p", "x")),
        Key.from_str(diff("p", "y")),
        Key.from_str(diff("p", "z")),
    ]
    dnode = Derivative.make_node(input_vars, derivs, jit=False)

    input_dict = dict(zip((str(v) for v in input_vars), [x, y, z, u, v, w, p]))
    derivs_dict = dnode.evaluate(input_dict)
    validate_gradients(x, y, z, *(derivs_dict[str(d)] for d in derivs))


if __name__ == "__main__":
    test_derivative_node()
