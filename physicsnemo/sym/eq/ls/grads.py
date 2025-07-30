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

    def __init__(self, dim: int, direct_input: bool = False):
        super().__init__()

        self.dim = dim
        self.direct_input = direct_input
        assert self.dim > 1, (
            "First Derivative through least squares method only supported for 2D and 3D inputs"
        )

    def forward(self, coords, connectivity_tensor, y, du=None, dv=None) -> list[Tensor]:
        """
        Compute first derivatives using least squares method with fully vectorized computation.

        Parameters
        ----------
        coords : torch.Tensor
            Node coordinates of shape [N, dim] (ignored if direct_input=True)
        connectivity_tensor : tuple[torch.Tensor, torch.Tensor] or tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Either (offsets, indices) for sparse format or (offsets, indices, neighbor_matrix) for batched format (ignored if direct_input=True)
        y : torch.Tensor
            Function values at nodes of shape [N, 1] (ignored if direct_input=True)
        du : torch.Tensor, optional
            Direct input for function differences, shape [N, max_neighbors, 1] or [batch_size, N, max_neighbors, 1] (required if direct_input=True)
        dv : torch.Tensor, optional
            Direct input for coordinate differences, shape [N, max_neighbors, dim] or [batch_size, N, max_neighbors, dim] (required if direct_input=True)

        Returns
        -------
        List[torch.Tensor]
            List of gradient components [dudx, dudy, dudz] for each node
        """
        if self.direct_input:
            if du is None or dv is None:
                raise ValueError("du and dv must be provided when direct_input=True")
            return self._forward_direct(du, dv)

        # Handle different connectivity formats
        if len(connectivity_tensor) == 2:
            offsets, indices = connectivity_tensor
            return self._forward_sparse(coords, offsets, indices, y)
        elif len(connectivity_tensor) == 3:
            _, _, neighbor_matrix = connectivity_tensor
            return self._forward_batched(coords, neighbor_matrix, y)
        else:
            raise ValueError(
                f"connectivity_tensor must be tuple of length 2 or 3; got {len(connectivity_tensor)=}"
            )

    def _forward_direct(self, du, dv) -> list[Tensor]:
        """
        Compute derivatives directly from provided du and dv tensors.

        Parameters
        ----------
        du : torch.Tensor
            Function differences, shape [N, max_neighbors, 1] or [batch_size, N, max_neighbors, 1]
        dv : torch.Tensor
            Coordinate differences, shape [N, max_neighbors, dim] or [batch_size, N, max_neighbors, dim]

        Returns
        -------
        List[torch.Tensor]
            List of gradient components [dudx, dudy, dudz] for each node
        """

        grad_u = self.compute_ls_grads(dv, du)  # [batch_size, N, dim, 1]
        grad_u = grad_u.squeeze(-1)  # [batch_size, N, dim]

        # Split into individual components
        result = [grad_u[:, [i]] for i in range(self.dim)]

        return result

    def _forward_sparse(self, coords, offsets, indices, y) -> list[Tensor]:
        """
        Compute derivatives using sparse connectivity format with parallel processing.
        Optimized for cases where all nodes have the same number of neighbors.
        """
        num_nodes = coords.shape[0]

        # Check if all nodes have the same number of neighbors
        neighbor_counts = offsets[1:] - offsets[:-1]  # [N]
        unique_counts = torch.unique(neighbor_counts)

        if len(unique_counts) == 1:
            num_neighbors = int(unique_counts[0].item())

            neighbor_matrix = torch.zeros(
                num_nodes, num_neighbors, dtype=torch.long, device=coords.device
            )

            for i in range(num_nodes):
                start_idx = offsets[i]
                end_idx = offsets[i + 1]
                neighbor_matrix[i] = indices[start_idx:end_idx]

            return self._forward_batched(coords, neighbor_matrix, y)

        else:
            max_neighbors = int(torch.max(neighbor_counts).item())

            all_dv = torch.zeros(
                num_nodes, max_neighbors, self.dim, device=coords.device
            )
            all_du = torch.zeros(num_nodes, max_neighbors, 1, device=coords.device)
            all_weights = torch.zeros(num_nodes, max_neighbors, device=coords.device)

            for i in range(num_nodes):
                start_idx = offsets[i]
                end_idx = offsets[i + 1]
                neighbor_indices = indices[start_idx:end_idx]

                if len(neighbor_indices) == 0:
                    continue

                p_center = coords[i : i + 1]  # [1, dim]
                p_neighbors = coords[neighbor_indices]  # [num_neighbors, dim]

                f_center = y[i : i + 1]  # [1, 1]
                f_neighbors = y[neighbor_indices]  # [num_neighbors, 1]

                dv = p_neighbors - p_center  # [num_neighbors, dim]
                du = f_neighbors - f_center  # [num_neighbors, 1]

                weights = 1 / (torch.sum(dv**2, dim=1) + 1e-8)  # [num_neighbors]

                num_neighbors = len(neighbor_indices)
                all_dv[i, :num_neighbors] = dv
                all_du[i, :num_neighbors] = du
                all_weights[i, :num_neighbors] = weights

            grad_u = self.compute_ls_grads(all_dv, all_du)  # [N, dim, 1]
            grad_u = grad_u.squeeze(-1)  # [N, dim]

            # Split into individual components
            result = [grad_u[:, [i]] for i in range(self.dim)]

            return result

    def _forward_batched(self, coords, neighbor_matrix, y) -> list[Tensor]:
        """
        Compute derivatives using batched connectivity format.
        """
        num_nodes = coords.shape[0]
        max_neighbors = neighbor_matrix.shape[1]

        # Create mask for valid neighbors
        valid_mask = neighbor_matrix != -1  # [N, max_neighbors]

        neighbor_coords = coords[neighbor_matrix]  # [N, max_neighbors, dim]
        neighbor_values = y[neighbor_matrix]  # [N, max_neighbors, 1]

        center_coords = coords.unsqueeze(1)  # [N, 1, dim]
        center_values = y.unsqueeze(1)  # [N, 1, 1]

        dv = neighbor_coords - center_coords  # [N, max_neighbors, dim]
        du = neighbor_values - center_values  # [N, max_neighbors, 1]

        # Apply mask to zero out invalid neighbors
        mask_expanded = valid_mask.unsqueeze(-1)  # [N, max_neighbors, 1]
        dv = dv * mask_expanded
        du = du * mask_expanded

        grad_u = self.compute_ls_grads(dv, du)  # [N, dim, 1]
        grad_u = grad_u.squeeze(-1)  # [N, dim]

        # Split into individual components
        result = [grad_u[:, [i]] for i in range(self.dim)]

        return result

    def compute_ls_grads(self, dv: torch.Tensor, du: torch.Tensor) -> torch.Tensor:
        """Given du and dv, compute the grads (batched)"""

        w_squared = 1 / (torch.einsum("bni,bni->bn", dv, dv) + 1e-8)
        A = torch.einsum("bni,bn,bnj->bij", dv, w_squared, dv)
        B = torch.einsum("bni,bn,bnk->bik", dv, w_squared, du)
        grad_u, _, _, _ = torch.linalg.lstsq(A, B)

        return grad_u
