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
from physicsnemo.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface


def test_diffusion_equation():
    # test data for diffusion equation
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)

    np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)

    D = 0.1
    Q = 0.1

    u__t = -np.sin(x) * np.sin(y) * np.sin(z) * np.sin(t)
    np.cos(x) * np.sin(y) * np.sin(z) * np.cos(t)
    np.sin(x) * np.cos(y) * np.sin(z) * np.cos(t)
    np.sin(x) * np.sin(y) * np.cos(z) * np.cos(t)

    u__x__x = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u__y__y = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u__z__z = -np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)

    diffusion_equation_true = u__t - D * u__x__x - D * u__y__y - D * u__z__z - Q

    # evaluate the equation
    eq = Diffusion(T="u", D=D, Q=Q, dim=3, time=True)
    evaluations = eq.make_nodes()[0].evaluate(
        {
            "u__x__x": torch.tensor(u__x__x, dtype=torch.float32),
            "u__y__y": torch.tensor(u__y__y, dtype=torch.float32),
            "u__z__z": torch.tensor(u__z__z, dtype=torch.float32),
            "u__t": torch.tensor(u__t, dtype=torch.float32),
        }
    )
    eq_eval = evaluations["diffusion_u"].numpy()

    # verify PDE computation
    assert np.allclose(eq_eval, diffusion_equation_true, rtol=1e-3), "Test Failed!"


def test_diffusion_interface():
    # test data for diffusion interface
    x = np.random.rand(1024, 1)
    y = np.random.rand(1024, 1)
    z = np.random.rand(1024, 1)
    t = np.random.rand(1024, 1)
    normal_x = np.random.rand(1024, 1)
    normal_y = np.random.rand(1024, 1)
    normal_z = np.random.rand(1024, 1)

    u_1 = np.sin(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u_2 = np.cos(x) * np.cos(y) * np.cos(z) * np.sin(t)

    D_1 = 0.1
    D_2 = 100

    u_1__x = np.cos(x) * np.sin(y) * np.sin(z) * np.cos(t)
    u_1__y = np.sin(x) * np.cos(y) * np.sin(z) * np.cos(t)
    u_1__z = np.sin(x) * np.sin(y) * np.cos(z) * np.cos(t)

    u_2__x = -np.sin(x) * np.cos(y) * np.cos(z) * np.sin(t)
    u_2__y = -np.cos(x) * np.sin(y) * np.cos(z) * np.sin(t)
    u_2__z = -np.cos(x) * np.cos(y) * np.sin(z) * np.sin(t)

    diffusion_interface_dirichlet_u_1_u_2_true = u_1 - u_2
    diffusion_interface_neumann_u_1_u_2_true = D_1 * (
        normal_x * u_1__x + normal_y * u_1__y + normal_z * u_1__z
    ) - D_2 * (normal_x * u_2__x + normal_y * u_2__y + normal_z * u_2__z)

    # evaluate the equation
    eq = DiffusionInterface(T_1="u_1", T_2="u_2", D_1=D_1, D_2=D_2, dim=3, time=True)
    evaluations = eq.make_nodes()[0].evaluate(
        {
            "u_1": torch.tensor(u_1, dtype=torch.float32),
            "u_2": torch.tensor(u_2, dtype=torch.float32),
        }
    )
    eq_1_eval = evaluations["diffusion_interface_dirichlet_u_1_u_2"].numpy()
    evaluations = eq.make_nodes()[1].evaluate(
        {
            "u_1": torch.tensor(u_1, dtype=torch.float32),
            "u_2": torch.tensor(u_2, dtype=torch.float32),
            "u_1__x": torch.tensor(u_1__x, dtype=torch.float32),
            "u_1__y": torch.tensor(u_1__y, dtype=torch.float32),
            "u_1__z": torch.tensor(u_1__z, dtype=torch.float32),
            "u_2__x": torch.tensor(u_2__x, dtype=torch.float32),
            "u_2__y": torch.tensor(u_2__y, dtype=torch.float32),
            "u_2__z": torch.tensor(u_2__z, dtype=torch.float32),
            "normal_x": torch.tensor(normal_x, dtype=torch.float32),
            "normal_y": torch.tensor(normal_y, dtype=torch.float32),
            "normal_z": torch.tensor(normal_z, dtype=torch.float32),
        }
    )
    eq_2_eval = evaluations["diffusion_interface_neumann_u_1_u_2"].numpy()

    # verify PDE computation
    assert np.allclose(eq_1_eval, diffusion_interface_dirichlet_u_1_u_2_true), (
        "Test Failed!"
    )
    assert np.allclose(eq_2_eval, diffusion_interface_neumann_u_1_u_2_true), (
        "Test Failed!"
    )


if __name__ == "__main__":
    test_diffusion_equation()
    test_diffusion_interface()
