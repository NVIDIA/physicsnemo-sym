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

import logging
from typing import List, Optional, Union

import numpy as np
from numba import njit
from typing import List, Optional, Union
import torch
from physicsnemo.sym.eq.derivatives import gradient_autodiff
from physicsnemo.sym.eq.fd import grads as fd_grads
from physicsnemo.sym.eq.ls import grads as ls_grads
from physicsnemo.sym.eq.mfd import grads as mfd_grads


@njit
def edges_to_adjacency(
    sorted_bidirectional_edges: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a sorted bidirectional edge list to an adjacency list using numba.

    Parameters
    ----------
    sorted_bidirectional_edges : np.ndarray
        A 2D array of shape (n_edges, 2) where each row contains the start
        and end indices of an edge. Edges are sorted by increasing start index,
        then increasing end index.
    n_points : int
        The number of points in the mesh.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (offsets, indices) arrays.
    """
    n_edges = len(sorted_bidirectional_edges)
    offsets = np.zeros(n_points + 1, dtype=sorted_bidirectional_edges.dtype)
    indices = np.zeros(n_edges, dtype=sorted_bidirectional_edges.dtype)

    edge_idx = 0
    for adj_index in range(n_points):
        start_offset = offsets[adj_index]
        while edge_idx < n_edges:
            start_idx = sorted_bidirectional_edges[edge_idx, 0]
            if start_idx == adj_index:
                indices[start_offset] = sorted_bidirectional_edges[edge_idx, 1]
                start_offset += 1
            elif start_idx > adj_index:
                break
            edge_idx += 1
        offsets[adj_index + 1] = start_offset

    return offsets, indices


edges_to_adjacency(np.zeros((0, 2), dtype=np.int64), 0)  # Does the precompilation


def unique_axis0_fast(array):
    """
    A faster version of np.unique(array, axis=0) for 2D arrays using numba.
    """
    if len(array) == 0:
        return array

    idxs = np.lexsort(array.T[::-1])
    array = array[idxs]
    unique_idxs = np.empty(len(array), dtype=np.bool_)
    unique_idxs[0] = True
    unique_idxs[1:] = np.any(array[:-1, :] != array[1:, :], axis=-1)
    return array[unique_idxs]


def compute_stencil2d(coords, model, dx, return_mixed_derivs=False):
    """Compute 2D stencil required for MFD"""
    # compute stencil points
    posx = coords[:, 0:1] + torch.ones_like(coords[:, 0:1]) * dx
    negx = coords[:, 0:1] - torch.ones_like(coords[:, 0:1]) * dx
    posy = coords[:, 1:2] + torch.ones_like(coords[:, 0:1]) * dx
    negy = coords[:, 1:2] - torch.ones_like(coords[:, 0:1]) * dx

    uposx = model(torch.cat([posx, coords[:, 1:2], coords[:, 2:3]], dim=1))
    unegx = model(torch.cat([negx, coords[:, 1:2], coords[:, 2:3]], dim=1))
    uposy = model(torch.cat([coords[:, 0:1], posy, coords[:, 2:3]], dim=1))
    unegy = model(torch.cat([coords[:, 0:1], negy, coords[:, 2:3]], dim=1))

    if return_mixed_derivs:
        uposxposy = model(torch.cat([posx, posy, coords[:, 2:3]], dim=1))
        uposxnegy = model(torch.cat([posx, negy, coords[:, 2:3]], dim=1))
        unegxposy = model(torch.cat([negx, posy, coords[:, 2:3]], dim=1))
        unegxnegy = model(torch.cat([negx, negy, coords[:, 2:3]], dim=1))

    if return_mixed_derivs:
        return (
            uposx,
            unegx,
            uposy,
            unegy,
            uposxposy,
            uposxnegy,
            unegxposy,
            unegxnegy,
        )
    else:
        return uposx, unegx, uposy, unegy


def compute_stencil3d(coords, model, dx, return_mixed_derivs=False):
    """Compute 3D stencil required for MFD"""
    # compute stencil points
    posx = coords[:, 0:1] + torch.ones_like(coords[:, 0:1]) * dx
    negx = coords[:, 0:1] - torch.ones_like(coords[:, 0:1]) * dx
    posy = coords[:, 1:2] + torch.ones_like(coords[:, 0:1]) * dx
    negy = coords[:, 1:2] - torch.ones_like(coords[:, 0:1]) * dx
    posz = coords[:, 2:3] + torch.ones_like(coords[:, 0:1]) * dx
    negz = coords[:, 2:3] - torch.ones_like(coords[:, 0:1]) * dx

    uposx = model(torch.cat([posx, coords[:, 1:2], coords[:, 2:3]], dim=1))
    unegx = model(torch.cat([negx, coords[:, 1:2], coords[:, 2:3]], dim=1))
    uposy = model(torch.cat([coords[:, 0:1], posy, coords[:, 2:3]], dim=1))
    unegy = model(torch.cat([coords[:, 0:1], negy, coords[:, 2:3]], dim=1))
    uposz = model(torch.cat([coords[:, 0:1], coords[:, 1:2], posz], dim=1))
    unegz = model(torch.cat([coords[:, 0:1], coords[:, 1:2], negz], dim=1))

    if return_mixed_derivs:
        uposxposy = model(torch.cat([posx, posy, coords[:, 2:3]], dim=1))
        uposxnegy = model(torch.cat([posx, negy, coords[:, 2:3]], dim=1))
        unegxposy = model(torch.cat([negx, posy, coords[:, 2:3]], dim=1))
        unegxnegy = model(torch.cat([negx, negy, coords[:, 2:3]], dim=1))

        uposxposz = model(torch.cat([posx, coords[:, 1:2], posz], dim=1))
        uposxnegz = model(torch.cat([posx, coords[:, 1:2], negz], dim=1))
        unegxposz = model(torch.cat([negx, coords[:, 1:2], posz], dim=1))
        unegxnegz = model(torch.cat([negx, coords[:, 1:2], negz], dim=1))

        uposyposz = model(torch.cat([coords[:, 0:1], posy, posz], dim=1))
        uposynegz = model(torch.cat([coords[:, 0:1], posy, negz], dim=1))
        unegyposz = model(torch.cat([coords[:, 0:1], negy, posz], dim=1))
        unegynegz = model(torch.cat([coords[:, 0:1], negy, negz], dim=1))

    if return_mixed_derivs:
        return (
            uposx,
            unegx,
            uposy,
            unegy,
            uposz,
            unegz,
            uposxposy,
            uposxnegy,
            unegxposy,
            unegxnegy,
            uposxposz,
            uposxnegz,
            unegxposz,
            unegxnegz,
            uposyposz,
            uposynegz,
            unegyposz,
            unegynegz,
        )
    else:
        return uposx, unegx, uposy, unegy, uposz, unegz


def compute_connectivity_tensor(nodes, edges, max_neighbors=None):
    """
    Compute connectivity tensor for given nodes and edges using optimized numba functions.

    Parameters
    ----------
    nodes :
        Node ids of the nodes in the mesh in [N, 1] format.
        Where N is the number of nodes.
    edges :
        Edges of the mesh in [M, 2] format.
        Where M is the number of edges.
    max_neighbors : int, optional
        Maximum number of neighbors to pad to. If None, uses the maximum found in the mesh.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - offsets: Tensor of shape [N+1] containing the start index for each node's neighbors
        - indices: Tensor containing the neighbor indices for all nodes concatenated
        - neighbor_matrix: Tensor of shape [N, max_neighbors] for batched computation
    """
    edges_np = edges.cpu().numpy()
    nodes_np = nodes.cpu().numpy().flatten()

    bidirectional_edges = np.concatenate((edges_np, edges_np[:, ::-1]), axis=0)

    order = np.lexsort((bidirectional_edges[:, 1], bidirectional_edges[:, 0]))
    sorted_bidirectional_edges = bidirectional_edges[order]

    unique_edges = unique_axis0_fast(sorted_bidirectional_edges)

    num_nodes = len(nodes_np)
    offsets, indices = edges_to_adjacency(unique_edges, num_nodes)

    offsets_tensor = torch.tensor(offsets, dtype=torch.long, device=nodes.device)
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=nodes.device)

    if max_neighbors is None:
        neighbor_counts = offsets[1:] - offsets[:-1]
        max_neighbors = int(np.max(neighbor_counts))

    neighbor_matrix = torch.full(
        (num_nodes, max_neighbors), -1, dtype=torch.long, device=nodes.device
    )

    for i in range(num_nodes):
        start_idx = offsets[i]
        end_idx = offsets[i + 1]
        num_neighbors = end_idx - start_idx
        if num_neighbors > 0:
            neighbor_matrix[i, :num_neighbors] = torch.tensor(
                indices[start_idx:end_idx], dtype=torch.long, device=nodes.device
            )

    return offsets_tensor, indices_tensor, neighbor_matrix


class GradientsAutoDiff(torch.nn.Module):
    """
    Compute spatial derivatives using Automatic differentiation.

    Parameters
    ----------
    invar : str
        Variable whose gradients are computed.
    dim : int, optional
        Dimensionality of the input (1D, 2D, or 3D), by default 3
    order : int, optional
        Order of the derivatives, by default 1 which returns the first order
        derivatives (e.g. `u__x`, `u__y`, `u__z`). Max order 2 is supported.
    return_mixed_derivs : bool, optional
        Whether to include mixed derivatives such as `u__x__y`, by default False
    """

    def __init__(
        self,
        invar: str,
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()

        self.invar = invar
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs

        assert self.order < 3, "Derivatives only upto 2nd order are supported"

        if self.return_mixed_derivs:
            assert self.dim > 1, "Mixed Derivatives only supported for 2D and 3D inputs"
            assert self.order == 2, (
                "Mixed Derivatives not possible for first order derivatives"
            )

    def forward(self, input_dict):
        y = input_dict[self.invar]
        x = input_dict["coordinates"]

        if not x.requires_grad:
            logging.warning(
                "The input tensor does not have requries_grad set to true, "
                + "the computations might be incorrect!"
            )

        assert x.shape[1] == self.dim, (
            f"Expected shape (N, {self.dim}), but got {x.shape}"
        )

        grad = gradient_autodiff(y, [x])

        result = {}
        axis_list = ["x", "y", "z"]
        if self.order == 1:
            for axis in range(self.dim):
                result[f"{self.invar}__{axis_list[axis]}"] = grad[0][:, axis : axis + 1]
        elif self.order == 2:
            for axis in range(self.dim):
                result[f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"] = (
                    gradient_autodiff(grad[0][:, axis : axis + 1], [x])[0][
                        :, axis : axis + 1
                    ]
                )
            if self.return_mixed_derivs:
                # Need to compute them manually due to how pytorch builds graph
                if self.dim == 2:
                    grad_x = grad[0][:, 0:1]
                    ggrad_mixed_xy = gradient_autodiff(grad_x, [x])[0][:, 1:2]
                    result[f"{self.invar}__x__y"] = ggrad_mixed_xy
                    result[f"{self.invar}__y__x"] = ggrad_mixed_xy
                elif self.dim == 3:
                    grad_x = grad[0][:, 0:1]
                    grad_y = grad[0][:, 1:2]
                    ggrad_mixed_xy = gradient_autodiff(grad_x, [x])[0][:, 1:2]
                    ggrad_mixed_xz = gradient_autodiff(grad_x, [x])[0][:, 2:3]
                    ggrad_mixed_yz = gradient_autodiff(grad_y, [x])[0][:, 2:3]
                    result[f"{self.invar}__x__y"] = ggrad_mixed_xy
                    result[f"{self.invar}__y__x"] = ggrad_mixed_xy
                    result[f"{self.invar}__x__z"] = ggrad_mixed_xz
                    result[f"{self.invar}__z__x"] = ggrad_mixed_xz
                    result[f"{self.invar}__y__z"] = ggrad_mixed_yz
                    result[f"{self.invar}__z__y"] = ggrad_mixed_yz

        return result


class GradientsMeshlessFiniteDifference(torch.nn.Module):
    """
    Compute spatial derivatives using Meshless Finite Differentiation. The gradients are
    computed using 2nd order finite difference stencils.
    For more details, refer: https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/features/performance.html#meshless-finite-derivatives

    Parameters
    ----------
    invar : str
        Variable whose gradients are computed.
    dx : Union[Union[float, int]]
        dx for the finite difference calculation.
    dim : int, optional
        Dimensionality of the input (1D, 2D, or 3D), by default 3
    order : int, optional
        Order of the derivatives, by default 1 which returns the first order
        derivatives (e.g. `u__x`, `u__y`, `u__z`). Max order 2 is supported.
    return_mixed_derivs : bool, optional
        Whether to include mixed derivatives such as `u__x__y`, by default False
    """

    def __init__(
        self,
        invar: str,
        dx: Union[Union[float, int]],
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()

        self.invar = invar
        self.dx = dx
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs

        if isinstance(self.dx, (float, int)):
            self.dx = [self.dx for _ in range(self.dim)]

        assert self.order < 3, "Derivatives only upto 2nd order are supported"
        assert len(self.dx) == self.dim, f"Mismatch in {self.dim} and {self.dx}"

        if self.return_mixed_derivs:
            assert self.dim > 1, "Mixed Derivatives only supported for 2D and 3D inputs"
            assert self.order == 2, (
                "Mixed Derivatives not possible for first order derivatives"
            )

        self.init_derivative_operators()

    def init_derivative_operators(self):
        self.first_deriv_ops = {}
        self.second_deriv_ops = {}

        if self.order == 1:
            for axis in range(self.dim):
                axis_name = ["x", "y", "z"][axis]
                self.first_deriv_ops[axis] = mfd_grads.FirstDerivSecondOrder(
                    var=self.invar,
                    indep_var=axis_name,
                    out_name=f"{self.invar}__{axis_name}",
                )
        elif self.order == 2:
            for axis in range(self.dim):
                axis_name = ["x", "y", "z"][axis]
                self.second_deriv_ops[axis] = mfd_grads.SecondDerivSecondOrder(
                    var=self.invar,
                    indep_var=axis_name,
                    out_name=f"{self.invar}__{axis_name}__{axis_name}",
                )
            if self.return_mixed_derivs:
                self.mixed_deriv_ops = {}
                self.mixed_deriv_ops["dxdy"] = mfd_grads.MixedSecondDerivSecondOrder(
                    var=self.invar,
                    indep_vars=["x", "y"],
                    out_name=f"{self.invar}__x__y",
                )
                if self.dim == 3:
                    self.mixed_deriv_ops["dxdz"] = (
                        mfd_grads.MixedSecondDerivSecondOrder(
                            var=self.invar,
                            indep_vars=["x", "z"],
                            out_name=f"{self.invar}__x__z",
                        )
                    )
                    self.mixed_deriv_ops["dydz"] = (
                        mfd_grads.MixedSecondDerivSecondOrder(
                            var=self.invar,
                            indep_vars=["y", "z"],
                            out_name=f"{self.invar}__y__z",
                        )
                    )

    def forward(self, input_dict):
        result = {}
        axis_list = ["x", "y", "z"]
        if self.order == 1:
            for axis, op in self.first_deriv_ops.items():
                result[f"{self.invar}__{axis_list[axis]}"] = op.forward(
                    input_dict, self.dx[axis]
                )[f"{self.invar}__{axis_list[axis]}"]
        elif self.order == 2:
            for axis, op in self.second_deriv_ops.items():
                result[f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"] = (
                    op.forward(input_dict, self.dx[axis])[
                        f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"
                    ]
                )
            if self.return_mixed_derivs:
                result[f"{self.invar}__x__y"] = self.mixed_deriv_ops["dxdy"].forward(
                    input_dict, self.dx[0]
                )[f"{self.invar}__x__y"]  # TODO: enable different dx and dy?
                result[f"{self.invar}__y__x"] = self.mixed_deriv_ops["dxdy"].forward(
                    input_dict, self.dx[0]
                )[f"{self.invar}__x__y"]
                if self.dim == 3:
                    result[f"{self.invar}__x__z"] = self.mixed_deriv_ops[
                        "dxdz"
                    ].forward(input_dict, self.dx[0])[f"{self.invar}__x__z"]
                    result[f"{self.invar}__z__x"] = self.mixed_deriv_ops[
                        "dxdz"
                    ].forward(input_dict, self.dx[0])[f"{self.invar}__x__z"]
                    result[f"{self.invar}__y__z"] = self.mixed_deriv_ops[
                        "dydz"
                    ].forward(input_dict, self.dx[0])[f"{self.invar}__y__z"]
                    result[f"{self.invar}__z__y"] = self.mixed_deriv_ops[
                        "dydz"
                    ].forward(input_dict, self.dx[0])[f"{self.invar}__y__z"]

        return result


class GradientsFiniteDifference(torch.nn.Module):
    """
    Compute spatial derivatives using Finite Differentiation. The gradients are
    computed using 2nd order finite difference stencils using convolution operation.

    Parameters
    ----------
    invar : str
        Variable whose gradients are computed.
    dx : Union[Union[float, int]]
        dx for the finite difference calculation.
    dim : int, optional
        Dimensionality of the input (1D, 2D, or 3D), by default 3
    order : int, optional
        Order of the derivatives, by default 1 which returns the first order
        derivatives (e.g. `u__x`, `u__y`, `u__z`). Max order 2 is supported.
    return_mixed_derivs : bool, optional
        Whether to include mixed derivatives such as `u__x__y`, by default False
    """

    def __init__(
        self,
        invar: str,
        dx: Union[Union[float, int], List[float]],
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()

        self.invar = invar
        self.dx = dx
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs

        if isinstance(self.dx, (float, int)):
            self.dx = [self.dx for _ in range(self.dim)]

        assert self.order < 3, "Derivatives only upto 2nd order are supported"
        assert len(self.dx) == self.dim, f"Mismatch in {self.dim} and {self.dx}"

        if self.return_mixed_derivs:
            assert self.dim > 1, "Mixed Derivatives only supported for 2D and 3D inputs"
            assert self.order == 2, (
                "Mixed Derivatives not possible for first order derivatives"
            )

        if self.order == 1:
            self.deriv_modulue = fd_grads.FirstDerivSecondOrder(self.dim, self.dx)
        elif self.order == 2:
            self.deriv_modulue = fd_grads.SecondDerivSecondOrder(self.dim, self.dx)
            if self.return_mixed_derivs:
                self.mixed_deriv_module = fd_grads.MixedSecondDerivSecondOrder(
                    self.dim, self.dx
                )

    def forward(self, input_dict):
        u = input_dict[self.invar]

        assert (u.dim() - 2) == self.dim, (
            f"Expected a {self.dim + 2} dimensional tensor, but got {u.dim()} dimensional tensor"
        )

        # compute finite difference based on convolutional operation
        result = {}
        derivatives = self.deriv_modulue(u)
        axis_list = ["x", "y", "z"]
        if self.order == 1:
            for axis, derivative in enumerate(derivatives):
                result[f"{self.invar}__{axis_list[axis]}"] = derivative
        elif self.order == 2:
            for axis, derivative in enumerate(derivatives):
                result[f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"] = (
                    derivative
                )
            if self.return_mixed_derivs:
                result[f"{self.invar}__x__y"] = self.mixed_deriv_module.forward(u)[0]
                result[f"{self.invar}__y__x"] = self.mixed_deriv_module.forward(u)[0]
                if self.dim == 3:
                    result[f"{self.invar}__x__z"] = self.mixed_deriv_module.forward(u)[
                        1
                    ]
                    result[f"{self.invar}__z__x"] = self.mixed_deriv_module.forward(u)[
                        1
                    ]
                    result[f"{self.invar}__y__z"] = self.mixed_deriv_module.forward(u)[
                        2
                    ]
                    result[f"{self.invar}__z__y"] = self.mixed_deriv_module.forward(u)[
                        2
                    ]

        return result


class GradientsSpectral(torch.nn.Module):
    """
    Compute spatial derivatives using Spectral Differentiation using FFTs.

    Parameters
    ----------
    invar : str
        Variable whose gradients are computed.
    ell : Union[Union[float, int]]
        bounds for the domain.
    dim : int, optional
        Dimensionality of the input (1D, 2D, or 3D), by default 3
    order : int, optional
        Order of the derivatives, by default 1 which returns the first order
        derivatives (e.g. `u__x`, `u__y`, `u__z`). Max order 2 is supported.
    return_mixed_derivs : bool, optional
        Whether to include mixed derivatives such as `u__x__y`, by default False
    """

    def __init__(
        self,
        invar: str,
        ell: Union[Union[float, int]],
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()

        self.invar = invar
        self.ell = ell
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs

        if isinstance(self.ell, (float, int)):
            self.ell = [self.ell for _ in range(self.dim)]

        assert self.order < 3, "Derivatives only upto 2nd order are supported"
        assert len(self.ell) == self.dim, f"Mismatch in {self.dim} and {self.ell}"

        if self.return_mixed_derivs:
            assert self.dim > 1, "Mixed Derivatives only supported for 2D and 3D inputs"
            assert self.order == 2, (
                "Mixed Derivatives not possible for first order derivatives"
            )

    def forward(self, input_dict):
        u = input_dict[self.invar]

        pi = float(np.pi)

        n = tuple(u.shape[2:])
        assert len(n) == self.dim, (
            f"Expected a {self.dim + 2} dimensional tensor, but got {u.dim()} dimensional tensor"
        )

        # compute the fourier transform
        u_h = torch.fft.fftn(u, dim=list(range(2, self.dim + 2)))

        # make wavenumbers
        kx = []
        for i, nx in enumerate(n):
            kx.append(
                torch.cat(
                    (
                        torch.arange(start=0, end=nx // 2, step=1, device=u.device),
                        torch.arange(start=-nx // 2, end=0, step=1, device=u.device),
                    ),
                    0,
                ).reshape((i + 2) * [1] + [nx] + (self.dim - i - 1) * [1])
            )

        # compute laplacian in the fourier space
        j = torch.complex(
            torch.tensor([0.0], device=u.device), torch.tensor([1.0], device=u.device)
        )

        wx_h = [j * k_x_i * u_h * (2 * pi / self.ell[i]) for i, k_x_i in enumerate(kx)]

        result = {}
        axis_list = ["x", "y", "z"]
        if self.order == 1:
            # inverse fourier transform out
            wx = torch.cat(
                [
                    torch.fft.ifftn(wx_h_i, dim=list(range(2, self.dim + 2))).real
                    for wx_h_i in wx_h
                ],
                dim=1,
            )
            for axis in range(self.dim):
                result[f"{self.invar}__{axis_list[axis]}"] = wx[:, axis : axis + 1]

        elif self.order == 2:
            wxx_h = [
                j * k_x_i * wx_h_i * (2 * pi / self.ell[i])
                for i, (wx_h_i, k_x_i) in enumerate(zip(wx_h, kx))
            ]

            # inverse fourier transform out
            wxx = torch.cat(
                [
                    torch.fft.ifftn(wxx_h_i, dim=list(range(2, self.dim + 2))).real
                    for wxx_h_i in wxx_h
                ],
                dim=1,
            )

            for axis in range(self.dim):
                result[f"{self.invar}__{axis_list[axis]}__{axis_list[axis]}"] = wxx[
                    :, axis : axis + 1
                ]

            if self.return_mixed_derivs:
                w_xy_h = (
                    -(2 * pi / self.ell[0])
                    * (2 * pi / self.ell[1])
                    * kx[0]
                    * kx[1]
                    * u_h
                )
                w_xy = torch.fft.ifftn(w_xy_h, dim=list(range(2, self.dim + 2))).real
                result[f"{self.invar}__x__y"] = w_xy
                result[f"{self.invar}__y__x"] = w_xy
                if self.dim == 3:
                    w_xz_h = (
                        -(2 * pi / self.ell[0])
                        * (2 * pi / self.ell[2])
                        * kx[0]
                        * kx[2]
                        * u_h
                    )
                    w_xz = torch.fft.ifftn(
                        w_xz_h, dim=list(range(2, self.dim + 2))
                    ).real
                    result[f"{self.invar}__x__z"] = w_xz
                    result[f"{self.invar}__z__x"] = w_xz

                    w_yz_h = (
                        -(2 * pi / self.ell[1])
                        * (2 * pi / self.ell[2])
                        * kx[1]
                        * kx[2]
                        * u_h
                    )
                    w_yz = torch.fft.ifftn(
                        w_yz_h, dim=list(range(2, self.dim + 2))
                    ).real
                    result[f"{self.invar}__y__z"] = w_yz
                    result[f"{self.invar}__z__y"] = w_yz

        return result


class GradientsLeastSquares(torch.nn.Module):
    """
    Compute spatial derivatives using Least Squares technique modified to compute
    gradients on nodes. Useful for mesh based representations (i.e. unstructured grids)

    Reference: https://scientific-sims.com/cfdlab/Dimitri_Mavriplis/HOME/assets/papers/aiaa20033986.pdf

    Parameters
    ----------
    invar : str
        Variable whose gradients are computed.
    dim : int, optional
        Dimensionality of the input (2D, or 3D), by default 3
    order : int, optional
        Order of the derivatives, by default 1 which returns the first order
        derivatives (e.g. `u__x`, `u__y`, `u__z`). Max order 2 is supported.
    return_mixed_derivs : bool, optional
        Whether to include mixed derivatives such as `u__x__y`, by default False
    """

    def __init__(
        self,
        invar: str,
        dim: int = 3,
        order: int = 1,
        return_mixed_derivs: bool = False,
    ):
        super().__init__()
        self.invar = invar
        self.cache = {}
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs

        assert self.dim > 1, (
            "1D gradients using Least squares is not supported. Please try other methods."
        )
        assert self.order < 3, "Derivatives only upto 2nd order are supported"

        if self.return_mixed_derivs:
            assert self.order == 2, (
                "Mixed Derivatives not possible for first order derivatives"
            )

        # TODO add a seperate SecondDeriv module
        self.deriv_module = ls_grads.FirstDeriv(self.dim)

    def forward(self, input_dict):
        coords = input_dict["coordinates"]

        assert coords.shape[1] == self.dim, (
            f"Expected shape (N, {self.dim}), but got {coords.shape}"
        )

        connectivity_tensor = input_dict["connectivity_tensor"]

        result = {}
        if self.dim == 2:
            derivs = self.deriv_module.forward(
                coords, connectivity_tensor, input_dict[self.invar]
            )

            if self.order == 1:
                result[f"{self.invar}__x"] = derivs[0]
                result[f"{self.invar}__y"] = derivs[1]
                return result
            elif self.order == 2:
                dderivs_x = self.deriv_module.forward(
                    coords, connectivity_tensor, derivs[0]
                )
                dderivs_y = self.deriv_module.forward(
                    coords, connectivity_tensor, derivs[1]
                )
                result[f"{self.invar}__x__x"] = dderivs_x[0]
                result[f"{self.invar}__y__y"] = dderivs_y[1]

                if self.return_mixed_derivs:
                    result[f"{self.invar}__x__y"] = dderivs_x[1]  # same as dderivs_y[0]
                    result[f"{self.invar}__y__x"] = dderivs_x[1]
                return result

        elif self.dim == 3:
            derivs = self.deriv_module.forward(
                coords, connectivity_tensor, input_dict[self.invar]
            )

            if self.order == 1:
                result[f"{self.invar}__x"] = derivs[0]
                result[f"{self.invar}__y"] = derivs[1]
                result[f"{self.invar}__z"] = derivs[2]
                return result
            elif self.order == 2:
                dderivs_x = self.deriv_module.forward(
                    coords, connectivity_tensor, derivs[0]
                )
                dderivs_y = self.deriv_module.forward(
                    coords, connectivity_tensor, derivs[1]
                )
                dderivs_z = self.deriv_module.forward(
                    coords, connectivity_tensor, derivs[2]
                )
                result[f"{self.invar}__x__x"] = dderivs_x[0]
                result[f"{self.invar}__y__y"] = dderivs_y[1]
                result[f"{self.invar}__z__z"] = dderivs_z[2]

                if self.return_mixed_derivs:
                    result[f"{self.invar}__x__y"] = dderivs_x[1]
                    result[f"{self.invar}__y__x"] = dderivs_x[1]
                    result[f"{self.invar}__x__z"] = dderivs_x[2]
                    result[f"{self.invar}__z__x"] = dderivs_x[2]
                    result[f"{self.invar}__y__z"] = dderivs_y[2]
                    result[f"{self.invar}__z__y"] = dderivs_y[2]

                return result


class GradientCalculator:
    """
    Unified Gradient calculator class.

    Parameters
    ----------
    device : Optional[str], optional
        The device to use for computation. Options are "cuda" or "cpu". If not
        specified, the computation defaults to "cpu".

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.sym.eq.spatial_grads.spatial_grads import GradientCalculator
    >>> coords = torch.rand(10, 3).requires_grad_(True)
    >>> u = coords[:, 0:1] ** 2 * coords[:, 1:2] ** 3 * coords[:, 2:3] ** 4
    >>> grad_calculator = GradientCalculator()
    >>> input_dict = {"coordinates": coords, "u": u}
    >>> grad_u_autodiff = grad_calculator.compute_gradients(
    ... input_dict,
    ... method_name="autodiff",
    ... invar="u"
    ... )
    >>> sorted(grad_u_autodiff.keys())
    ['u__x', 'u__y', 'u__z']
    >>> grad_u_autodiff = grad_calculator.compute_gradients(
    ... input_dict,
    ... method_name="autodiff",
    ... order=2,
    ... return_mixed_derivs=True,
    ... invar="u"
    ... )
    >>> sorted(grad_u_autodiff.keys())
    ['u__x__x', 'u__x__y', 'u__x__z', 'u__y__x', 'u__y__y', 'u__y__z', 'u__z__x', 'u__z__y', 'u__z__z']
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device if device is not None else torch.device("cpu")
        self.methods = {}
        self._register_methods()

    def _register_methods(self):
        self.methods["autodiff"] = GradientsAutoDiff
        self.methods["meshless_finite_difference"] = GradientsMeshlessFiniteDifference
        self.methods["finite_difference"] = GradientsFiniteDifference
        self.methods["spectral"] = GradientsSpectral
        self.methods["least_squares"] = GradientsLeastSquares

    def get_gradient_module(self, method_name, invar, **kwargs):
        """Return the gradient module"""
        module = self.methods[method_name](invar, **kwargs)
        module.to(self.device)
        return module

    def compute_gradients(self, input_dict, method_name=None, invar=None, **kwargs):
        """Compute the gradients"""
        module = self.get_gradient_module(method_name, invar, **kwargs)
        return module.forward(input_dict)

    @staticmethod
    def clip_gradients():
        pass

    @staticmethod
    def visualize():
        pass

    @staticmethod
    def scale_grads():
        pass

    @staticmethod
    def unscale_grads():
        pass
