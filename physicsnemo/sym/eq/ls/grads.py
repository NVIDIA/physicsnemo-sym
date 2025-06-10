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
from typing import List

Tensor = torch.Tensor


class FirstDeriv(torch.nn.Module):
    """Module to compute first derivative with 2nd order accuracy using least squares method"""

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        assert (
            self.dim > 1
        ), "First Derivative through least squares method only supported for 2D and 3D inputs"

    def forward(self, coords, connectivity_tensor, y) -> List[Tensor]:
        p1 = coords[connectivity_tensor[:, :, 0]]
        p2 = coords[connectivity_tensor[:, :, 1]]
        dx = p1[:, :, 0] - p2[:, :, 0]
        dy = p1[:, :, 1] - p2[:, :, 1]

        f1 = y[connectivity_tensor[:, :, 0]]
        f2 = y[connectivity_tensor[:, :, 1]]

        du = (f1 - f2).squeeze(-1)

        result = []
        if self.dim == 2:
            w = 1 / torch.sqrt(dx**2 + dy**2)
            w = torch.where(torch.isinf(w), torch.tensor(1.0).to(w.device), w)
            mask = torch.ones_like(dx)

            a1 = torch.sum((w**2 * dx * dx) * mask, dim=1)
            b1 = torch.sum((w**2 * dx * dy) * mask, dim=1)
            d1 = torch.sum((w**2 * du * dx) * mask, dim=1)

            a2 = torch.sum((w**2 * dx * dy) * mask, dim=1)
            b2 = torch.sum((w**2 * dy * dy) * mask, dim=1)
            d2 = torch.sum((w**2 * du * dy) * mask, dim=1)

            detA = torch.linalg.det(
                torch.stack(
                    [
                        torch.stack([a1, a2], dim=1),
                        torch.stack([b1, b2], dim=1),
                    ],
                    dim=2,
                )
            )
            dudx = (
                torch.linalg.det(
                    torch.stack(
                        [
                            torch.stack([d1, d2], dim=1),
                            torch.stack([b1, b2], dim=1),
                        ],
                        dim=2,
                    )
                )
                / detA
            )
            dudy = (
                torch.linalg.det(
                    torch.stack(
                        [
                            torch.stack([a1, a2], dim=1),
                            torch.stack([d1, d2], dim=1),
                        ],
                        dim=2,
                    )
                )
                / detA
            )
            result.append(dudx.unsqueeze(dim=1))
            result.append(dudy.unsqueeze(dim=1))
            return result
        elif self.dim == 3:
            dz = p1[:, :, 2] - p2[:, :, 2]

            w = 1 / torch.sqrt(dx**2 + dy**2 + dz**2)
            w = torch.where(torch.isinf(w), torch.tensor(1.0).to(w.device), w)
            mask = torch.ones_like(dx)

            a1 = torch.sum((w**2 * dx * dx) * mask, dim=1)
            b1 = torch.sum((w**2 * dx * dy) * mask, dim=1)
            c1 = torch.sum((w**2 * dx * dz) * mask, dim=1)
            d1 = torch.sum((w**2 * du * dx) * mask, dim=1)

            a2 = torch.sum((w**2 * dx * dy) * mask, dim=1)
            b2 = torch.sum((w**2 * dy * dy) * mask, dim=1)
            c2 = torch.sum((w**2 * dy * dz) * mask, dim=1)
            d2 = torch.sum((w**2 * du * dy) * mask, dim=1)

            a3 = torch.sum((w**2 * dx * dz) * mask, dim=1)
            b3 = torch.sum((w**2 * dy * dz) * mask, dim=1)
            c3 = torch.sum((w**2 * dz * dz) * mask, dim=1)
            d3 = torch.sum((w**2 * du * dz) * mask, dim=1)

            detA = torch.linalg.det(
                torch.stack(
                    [
                        torch.stack([a1, a2, a3], dim=1),
                        torch.stack([b1, b2, b3], dim=1),
                        torch.stack([c1, c2, c3], dim=1),
                    ],
                    dim=2,
                )
            )
            dudx = (
                torch.linalg.det(
                    torch.stack(
                        [
                            torch.stack([d1, d2, d3], dim=1),
                            torch.stack([b1, b2, b3], dim=1),
                            torch.stack([c1, c2, c3], dim=1),
                        ],
                        dim=2,
                    )
                )
                / detA
            )
            dudy = (
                torch.linalg.det(
                    torch.stack(
                        [
                            torch.stack([a1, a2, a3], dim=1),
                            torch.stack([d1, d2, d3], dim=1),
                            torch.stack([c1, c2, c3], dim=1),
                        ],
                        dim=2,
                    )
                )
                / detA
            )
            dudz = (
                torch.linalg.det(
                    torch.stack(
                        [
                            torch.stack([a1, a2, a3], dim=1),
                            torch.stack([b1, b2, b3], dim=1),
                            torch.stack([d1, d2, d3], dim=1),
                        ],
                        dim=2,
                    )
                )
                / detA
            )

            result.append(dudx.unsqueeze(dim=1))
            result.append(dudy.unsqueeze(dim=1))
            result.append(dudz.unsqueeze(dim=1))
            return result
