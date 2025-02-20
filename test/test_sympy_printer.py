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

import numpy as np
import torch
from physicsnemo.sym.utils.sympy import torch_lambdify
import sympy


def test_lambdify():
    # Define SymPy symbol and expression
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    expr = sympy.Max(sympy.sin(x), sympy.cos(y))

    # Get numpy reference
    x_np = np.random.random(10)
    y_np = np.random.random(10)
    expr_np = np.maximum(np.sin(x_np), np.cos(y_np))

    # Compile SymPy expression to the framework
    lam_tf = torch_lambdify(expr, ["x", "y"])

    # Choose device to run on and copy data from numpy
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_th = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_th = torch.tensor(y_np, dtype=torch.float32, device=device)
    assert np.allclose(x_th.cpu().detach().numpy(), x_np)
    assert np.allclose(y_th.cpu().detach().numpy(), y_np)

    # Run the compiled function on input tensors
    expr_th = lam_tf([x_th, y_th])
    expr_th_out = expr_th.cpu().detach().numpy()

    assert np.allclose(expr_th_out, expr_np, rtol=1.0e-3), "SymPy printer test failed!"


if __name__ == "__main__":
    test_lambdify()
