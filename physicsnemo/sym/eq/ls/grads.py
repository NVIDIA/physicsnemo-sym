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
        """
        Compute first derivatives using least squares method with fully vectorized computation.

        Parameters
        ----------
        coords : torch.Tensor
            Node coordinates of shape [N, dim]
        connectivity_tensor : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of (offsets, indices, neighbor_matrix) representing connectivity
        y : torch.Tensor
            Function values at nodes of shape [N, 1]

        Returns
        -------
        List[torch.Tensor]
            List of gradient components [dudx, dudy, dudz] for each node
        """
        _, _, neighbor_matrix = connectivity_tensor

        num_nodes = coords.shape[0]
        max_neighbors = neighbor_matrix.shape[1]

        # Create mask for valid neighbors
        valid_mask = (neighbor_matrix != -1)  # [N, max_neighbors]

        # neighbor_matrix: [N, max_neighbors] -> neighbor_coords: [N, max_neighbors, dim]
        neighbor_coords = coords[neighbor_matrix]  # [N, max_neighbors, dim]
        neighbor_values = y[neighbor_matrix]  # [N, max_neighbors, 1]

        center_coords = coords.unsqueeze(1)  # [N, 1, dim]
        center_values = y.unsqueeze(1)  # [N, 1, 1]

        dv = neighbor_coords - center_coords  # [N, max_neighbors, dim]
        du = neighbor_values - center_values  # [N, max_neighbors, 1]

        mask_expanded = valid_mask.unsqueeze(-1)  # [N, max_neighbors, 1]
        dv = dv * mask_expanded
        du = du * mask_expanded

        dv_batched = dv.unsqueeze(0)  # [1, N, max_neighbors, dim]
        du_batched = du.unsqueeze(0)  # [1, N, max_neighbors, 1]

        grad_u = self.compute_ls_grads(dv_batched, du_batched)  # [1, N, dim, 1]
        grad_u = grad_u.squeeze(0).squeeze(-1)  # [N, dim]

        # Split into individual components
        result = []
        for i in range(self.dim):
            result.append(grad_u[:, i:i+1])  # [N, 1]

        return result

    def compute_ls_grads(self, dv, du):
        """Given du and dv, compute the grads (batched)"""

        w_squared = 1 / ((dv**2).sum(dim=3) + 1e-8) # Sum along the coordinate dim
        W = torch.diag_embed(w_squared)
        A = torch.matmul(torch.matmul(dv.transpose(-2, -1), W), dv)   # Should be [1, batch_size, 3, 3]
        B = torch.matmul(torch.matmul(dv.transpose(-2, -1), W), du)   # Should be [1, batch_size, 3, 5]
        grad_u, _, _, _ = torch.linalg.lstsq(A, B) # Should be [1, batchsize, 3, 5]

        return grad_u