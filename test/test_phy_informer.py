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
import numpy as np
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.eq.spatial_grads.spatial_grads import (
    compute_stencil3d,
)
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
import pytest
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_fields(field, name):
    fig, axs = plt.subplots(1, 3)
    for i, ax in enumerate(axs):
        if i < 2:
            im = ax.imshow(field.detach().cpu().numpy()[:, :, field.shape[2] // 2])
        else:
            im = ax.imshow(field.detach().cpu().numpy()[field.shape[2] // 2, :, :])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # compute u, v, w, p using trignometric functions
        u = (
            torch.sin(1 * x[:, 0:1])
            + torch.sin(8 * x[:, 1:2])
            + torch.sin(4 * x[:, 2:3])
        )
        v = (
            torch.sin(8 * x[:, 0:1])
            + torch.sin(2 * x[:, 1:2])
            + torch.sin(1 * x[:, 2:3])
        )
        w = (
            torch.sin(2 * x[:, 0:1])
            + torch.sin(2 * x[:, 1:2])
            + torch.sin(9 * x[:, 2:3])
        )
        p = (
            torch.sin(1 * x[:, 0:1])
            + torch.sin(1 * x[:, 1:2])
            + torch.sin(1 * x[:, 2:3])
        )

        return torch.cat([u, v, w, p], dim=1)


@pytest.fixture
def general_setup(request):
    device = request.param
    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)
    y = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)
    z = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)
    coords_unstructured = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    model = Model().to(device)

    # Analytical residuals
    u = (
        torch.sin(1 * coords[:, 0:1])
        + torch.sin(8 * coords[:, 1:2])
        + torch.sin(4 * coords[:, 2:3])
    )
    v = (
        torch.sin(8 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(1 * coords[:, 2:3])
    )
    w = (
        torch.sin(2 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(9 * coords[:, 2:3])
    )
    (
        torch.sin(1 * coords[:, 0:1])
        + torch.sin(1 * coords[:, 1:2])
        + torch.sin(1 * coords[:, 2:3])
    )

    p__x = torch.cos(1 * coords[:, 0:1])
    u__x = torch.cos(1 * coords[:, 0:1])
    u__y = 8 * torch.cos(8 * coords[:, 1:2])
    u__z = 4 * torch.cos(4 * coords[:, 2:3])
    v__y = 2 * torch.cos(2 * coords[:, 1:2])
    w__z = 9 * torch.cos(9 * coords[:, 2:3])
    u__x__x = -1 * torch.sin(1 * coords[:, 0:1])
    u__y__y = -64 * torch.sin(8 * coords[:, 1:2])
    u__z__z = -16 * torch.sin(4 * coords[:, 2:3])

    true_cont = u__x + v__y + w__z
    true_mom_x = (
        u * u__x
        + v * u__y
        + w * u__z
        + p__x
        - 0.01 * u__x__x
        - 0.01 * u__y__y
        - 0.01 * u__z__z
    )

    residuals_analytical = {"continuity": true_cont, "momentum_x": true_mom_x}

    return coords, coords_unstructured, residuals_analytical, model


@pytest.fixture
def least_squares_setup(request):
    device = request.param
    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)
    y = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)
    z = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True).to(device)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)
    coords_unstructured = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    # Connectivity information
    indices = torch.arange(steps, device=device)
    i, j, k = torch.meshgrid(indices, indices, indices, indexing="ij")

    i = i.flatten()
    j = j.flatten()
    k = k.flatten()

    index = i * steps * steps + j * steps + k

    edges = []

    if steps > 1:
        # Edges in the i-direction
        edges_i = torch.stack([index[: -steps * steps], index[steps * steps :]], dim=1)
        edges.append(edges_i)

        # Edges in the j-direction
        edges_j = torch.stack([index[:-steps], index[steps:]], dim=1)
        edges.append(edges_j)

        # Edges in the k-direction
        edges_k = torch.stack([index[:-1], index[1:]], dim=1)
        edges.append(edges_k)

    # Concatenate all edges and move to device
    edges = torch.cat(edges).to(device)

    node_ids = torch.arange(coords_unstructured.size(0)).reshape(-1, 1).to(device)

    model = Model().to(device)

    # Analytical residuals
    u = (
        torch.sin(1 * coords[:, 0:1])
        + torch.sin(8 * coords[:, 1:2])
        + torch.sin(4 * coords[:, 2:3])
    )
    v = (
        torch.sin(8 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(1 * coords[:, 2:3])
    )
    w = (
        torch.sin(2 * coords[:, 0:1])
        + torch.sin(2 * coords[:, 1:2])
        + torch.sin(9 * coords[:, 2:3])
    )
    (
        torch.sin(1 * coords[:, 0:1])
        + torch.sin(1 * coords[:, 1:2])
        + torch.sin(1 * coords[:, 2:3])
    )

    p__x = torch.cos(1 * coords[:, 0:1])
    u__x = torch.cos(1 * coords[:, 0:1])
    u__y = 8 * torch.cos(8 * coords[:, 1:2])
    u__z = 4 * torch.cos(4 * coords[:, 2:3])
    v__y = 2 * torch.cos(2 * coords[:, 1:2])
    w__z = 9 * torch.cos(9 * coords[:, 2:3])
    u__x__x = -1 * torch.sin(1 * coords[:, 0:1])
    u__y__y = -64 * torch.sin(8 * coords[:, 1:2])
    u__z__z = -16 * torch.sin(4 * coords[:, 2:3])

    true_cont = u__x + v__y + w__z
    true_mom_x = (
        u * u__x
        + v * u__y
        + w * u__z
        + p__x
        - 0.01 * u__x__x
        - 0.01 * u__y__y
        - 0.01 * u__z__z
    )

    residuals_analytical = {"continuity": true_cont, "momentum_x": true_mom_x}

    return coords, coords_unstructured, residuals_analytical, model, node_ids, edges


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_residuals_autodiff(general_setup):
    coords, coords_unstructured, residuals_analytical, model = general_setup
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="autodiff",
        device=coords.device,
    )
    pred_outvar = model(coords_unstructured)
    residuals_autodiff = phy_informer.forward(
        {
            "coordinates": coords_unstructured,
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )

    # Validate and assert error
    pad = 2
    for key in residuals_analytical.keys():
        # plot_fields(residuals_autodiff["momentum_x"].reshape(100, 100, 100), "autodiff_momentum_x")
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals_autodiff[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
            )
        )
        assert error < 0.5, f"Autodiff gradient error too high for {key}: {error}"


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_residuals_meshless_fd(general_setup):
    coords, coords_unstructured, residuals_analytical, model = general_setup
    # Compute stencil
    po_posx, po_negx, po_posy, po_negy, po_posz, po_negz = compute_stencil3d(
        coords_unstructured, model, dx=0.001
    )

    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="meshless_finite_difference",
        fd_dx=0.001,
        device=coords.device,
    )
    pred_outvar = model(coords_unstructured)
    residuals_meshless_fd = phy_informer.forward(
        {
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
            "u>>x::1": po_posx[:, 0:1],
            "v>>x::1": po_posx[:, 1:2],
            "w>>x::1": po_posx[:, 2:3],
            "p>>x::1": po_posx[:, 3:4],
            "u>>x::-1": po_negx[:, 0:1],
            "v>>x::-1": po_negx[:, 1:2],
            "w>>x::-1": po_negx[:, 2:3],
            "p>>x::-1": po_negx[:, 3:4],
            "u>>y::1": po_posy[:, 0:1],
            "v>>y::1": po_posy[:, 1:2],
            "w>>y::1": po_posy[:, 2:3],
            "p>>y::1": po_posy[:, 3:4],
            "u>>y::-1": po_negy[:, 0:1],
            "v>>y::-1": po_negy[:, 1:2],
            "w>>y::-1": po_negy[:, 2:3],
            "p>>y::-1": po_negy[:, 3:4],
            "u>>z::1": po_posz[:, 0:1],
            "v>>z::1": po_posz[:, 1:2],
            "w>>z::1": po_posz[:, 2:3],
            "p>>z::1": po_posz[:, 3:4],
            "u>>z::-1": po_negz[:, 0:1],
            "v>>z::-1": po_negz[:, 1:2],
            "w>>z::-1": po_negz[:, 2:3],
            "p>>z::-1": po_negz[:, 3:4],
        },
    )

    # Validate and assert error
    pad = 2
    for key in residuals_analytical.keys():
        # plot_fields(residuals_meshless_fd["momentum_x"].reshape(100, 100, 100), "meshless_fd_momentum_x")
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals_meshless_fd[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
            )
        )
        assert error < 0.5, f"Meshless FD gradient error too high for {key}: {error}"


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_residuals_finite_difference(general_setup):
    coords, coords_unstructured, residuals_analytical, model = general_setup
    steps = 100
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="finite_difference",
        fd_dx=(2 * np.pi / steps),  # computed based on the grid spacing
        device=coords.device,
    )
    pred_outvar = model(coords)
    residuals_fd = phy_informer.forward(
        {
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )

    # Validate and assert error
    pad = 2
    for key in residuals_analytical.keys():
        # plot_fields(residuals_fd["momentum_x"].reshape(100, 100, 100), "finite_difference_momentum_x")
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals_fd[key].reshape(100, 100, 100)[pad:-pad, pad:-pad, pad:-pad]
            )
        )
        assert error < 0.5, (
            f"Finite Difference gradient error too high for {key}: {error}"
        )


@pytest.mark.parametrize("general_setup", ["cuda"], indirect=True)
def test_residuals_spectral(general_setup):
    coords, coords_unstructured, residuals_analytical, model = general_setup
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="spectral",
        bounds=[2 * np.pi, 2 * np.pi, 2 * np.pi],
        device=coords.device,
    )
    pred_outvar = model(coords)
    residuals_spectral = phy_informer.forward(
        {
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )

    # Validate and assert error
    pad = 2
    for key in residuals_analytical.keys():
        # plot_fields(residuals_spectral["momentum_x"].reshape(100, 100, 100), "spectral_momentum_x")
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals_spectral[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
            )
        )
        assert error < 0.5, f"Spectral gradient error too high for {key}: {error}"


@pytest.mark.parametrize("least_squares_setup", ["cuda"], indirect=True)
def test_residuals_least_squares(least_squares_setup):
    (
        coords,
        coords_unstructured,
        residuals_analytical,
        model,
        node_ids,
        edges,
    ) = least_squares_setup
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="least_squares",
        device=coords.device,
    )
    pred_outvar = model(coords_unstructured)
    residuals_ls = phy_informer.forward(
        {
            "coordinates": coords_unstructured,
            "nodes": node_ids,
            "edges": edges,
            "u": pred_outvar[:, 0:1],
            "v": pred_outvar[:, 1:2],
            "w": pred_outvar[:, 2:3],
            "p": pred_outvar[:, 3:4],
        },
    )

    # Validate and assert error
    pad = 2
    for key in residuals_analytical.keys():
        # plot_fields(residuals_ls["momentum_x"].reshape(100, 100, 100), "least_squares_momentum_x")
        error = torch.mean(
            torch.abs(
                residuals_analytical[key].reshape(100, 100, 100)[
                    pad:-pad, pad:-pad, pad:-pad
                ]
                - residuals_ls[key].reshape(100, 100, 100)[pad:-pad, pad:-pad, pad:-pad]
            )
        )
        assert error < 0.5, f"Least Squares gradient error too high for {key}: {error}"
