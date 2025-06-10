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

from physicsnemo.sym.eq.pdes.basic import GradNormal, Curl
import torch
import numpy as np
import os


def test_normal_gradient_equation():
    # test data for normal gradient
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    normal_x = np.random.rand(1024, 1)
    normal_y = np.random.rand(1024, 1)
    normal_z = np.random.rand(1024, 1)

    u = np.exp(2 * x + y + z + t)
    u__x = 2 * np.exp(2 * x + y + z + t)
    u__y = 1 * np.exp(2 * x + y + z + t)
    u__z = 1 * np.exp(2 * x + y + z + t)

    normal_gradient_u_true = normal_x * u__x + normal_y * u__y + normal_z * u__z

    normal_gradient_eq = GradNormal(T="u", dim=3, time=True)
    evaluations = normal_gradient_eq.make_nodes()[0].evaluate(
        {
            "u__x": torch.tensor(u__x, dtype=torch.float32),
            "u__y": torch.tensor(u__y, dtype=torch.float32),
            "u__z": torch.tensor(u__z, dtype=torch.float32),
            "normal_x": torch.tensor(normal_x, dtype=torch.float32),
            "normal_y": torch.tensor(normal_y, dtype=torch.float32),
            "normal_z": torch.tensor(normal_z, dtype=torch.float32),
        }
    )

    normal_gradient_u_eval_pred = evaluations["normal_gradient_u"].numpy()

    # verify PDE computation
    assert np.allclose(
        normal_gradient_u_eval_pred, normal_gradient_u_true
    ), "Test Failed!"


def test_curl():
    # test data for curl equation
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)

    a = np.exp(2 * x + y + z)
    b = np.exp(x + 2 * y + z)
    c = np.exp(x + y + 2 * z)
    a__x = 2 * np.exp(2 * x + y + z)
    a__y = 1 * np.exp(2 * x + y + z)
    a__z = 1 * np.exp(2 * x + y + z)
    b__x = 1 * np.exp(x + 2 * y + z)
    b__y = 2 * np.exp(x + 2 * y + z)
    b__z = 1 * np.exp(x + 2 * y + z)
    c__x = 1 * np.exp(x + y + 2 * z)
    c__y = 1 * np.exp(x + y + 2 * z)
    c__z = 2 * np.exp(x + y + 2 * z)

    u_true = c__y - b__z
    v_true = a__z - c__x
    w_true = b__x - a__y

    curl_eq = Curl(("a", "b", "c"), ("u", "v", "w"))
    evaluations_u = curl_eq.make_nodes()[0].evaluate(
        {
            "c__y": torch.tensor(c__y, dtype=torch.float32),
            "b__z": torch.tensor(b__z, dtype=torch.float32),
        }
    )
    evaluations_v = curl_eq.make_nodes()[1].evaluate(
        {
            "a__z": torch.tensor(a__z, dtype=torch.float32),
            "c__x": torch.tensor(c__x, dtype=torch.float32),
        }
    )
    evaluations_w = curl_eq.make_nodes()[2].evaluate(
        {
            "b__x": torch.tensor(b__x, dtype=torch.float32),
            "a__y": torch.tensor(a__y, dtype=torch.float32),
        }
    )

    u_eval_pred = evaluations_u["u"].numpy()
    v_eval_pred = evaluations_v["v"].numpy()
    w_eval_pred = evaluations_w["w"].numpy()

    # verify PDE computation
    assert np.allclose(u_eval_pred, u_true, atol=1e-4), "Test Failed!"
    assert np.allclose(v_eval_pred, v_true, atol=1e-4), "Test Failed!"
    assert np.allclose(w_eval_pred, w_true, atol=1e-4), "Test Failed!"


if __name__ == "__main__":
    test_normal_gradient_equation()
    test_curl()
