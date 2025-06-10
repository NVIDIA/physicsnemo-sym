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
from ops import FlowOps
import torch
from typing import Tuple
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes


def compute_velocity_grad(velocity: torch.Tensor, dx: float = None) -> torch.Tensor:
    """
    Compute the Velocity gradient using finite difference
    Args:
        velocity (torch.Tensor): 3D Velocity tensor (N, 3, nx, ny, nz)
        dx (float, optional): The spacing of the grid. Defaults to None, in which case, the
        spacing is computed asuming bounds of 2*pi.

    Returns:
        torch.Tensor: Computed velocity gradient tensor (N, 9, nx, ny, nz)
    """
    if dx == None:
        dx = 2 * np.pi / velocity.shape[-1]

    ops = FlowOps().to(velocity.device)
    vel_grad_dict = ops.get_velocity_grad(velocity, dx, dx, dx)
    u_vel_grad = torch.cat(
        (
            vel_grad_dict["u__x"],
            vel_grad_dict["u__y"],
            vel_grad_dict["u__z"],
        ),
        axis=1,
    )
    v_vel_grad = torch.cat(
        (
            vel_grad_dict["v__x"],
            vel_grad_dict["v__y"],
            vel_grad_dict["v__z"],
        ),
        axis=1,
    )
    w_vel_grad = torch.cat(
        (
            vel_grad_dict["w__x"],
            vel_grad_dict["w__y"],
            vel_grad_dict["w__z"],
        ),
        axis=1,
    )

    vel_grad_tensor = torch.stack((u_vel_grad, v_vel_grad, w_vel_grad), axis=2)

    return vel_grad_tensor


def compute_p_q_r(
    velocity_grad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the P, Q and R invariants of the velocity gradient tensor.
    The Q and R are normalized. Uses Finite Difference to compute the gradients.

    Args:
        velocity_grad (torch.Tensor): 3D Velocity gradient tensor (N, 9, nx, ny, nz)
        spacing is computed asuming bounds of 2*pi.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Computed P, Q and R.
    """

    bs = velocity_grad.shape[0]
    J = velocity_grad.permute(0, 3, 4, 5, 1, 2).reshape(bs, -1, 3, 3)
    strain = 0.5 * (J + torch.permute(J, (0, 1, 3, 2)))

    # Combine the points across all batches
    J = J.reshape(-1, 3, 3)
    strain = strain.reshape(-1, 3, 3)

    # Compute J^2 and J^3 for each 3x3 matrix
    J2 = torch.bmm(J, J)
    J3 = torch.bmm(J, J2)
    strain2 = torch.bmm(strain, strain)

    # Reshape back to have batch dimension
    J = J.reshape(bs, -1, 3, 3)
    J2 = J2.reshape(bs, -1, 3, 3)
    J3 = J3.reshape(bs, -1, 3, 3)
    strain2 = strain2.reshape(bs, -1, 3, 3)

    # Compute traces
    trace_J1 = J.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_J2 = J2.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_J3 = J3.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_strain2 = strain2.diagonal(dim1=-2, dim2=-1).sum(-1)

    # Compute P, Q and R invariants
    P = -1 * trace_J1
    Q = -0.5 * trace_J2
    R = -1 / 3 * trace_J3

    # Normalize P, Q and R
    P = P / torch.mean(trace_strain2**0.5, dim=-1, keepdim=True)
    Q = Q / torch.mean(trace_strain2, dim=-1, keepdim=True)
    R = R / torch.mean(trace_strain2**1.5, dim=-1, keepdim=True)

    return P, Q, R


def compute_continuity(field: torch.Tensor, dx: float = None) -> torch.Tensor:
    """
    Compute Continuity residual. Uses Finite Difference to compute the gradients.

    Args:
        field (torch.Tensor): 3D Velocity tensor (N, 3, nx, ny, nz)
        dx (float, optional): The spacing of the grid. Defaults to None, in which case, the
        spacing is computed asuming bounds of 2*pi.

    Returns:
        torch.Tensor: _description_
    """
    # setup Navier stokes node
    ns = NavierStokes(nu=0.1, rho=1, dim=3, time=False)
    ns_node = ns.make_nodes()

    if dx == None:
        dx = 2 * np.pi / field.shape[-1]

    ops = FlowOps().to(field.device)
    vel_grad_dict = ops.get_velocity_grad(field, dx, dx, dx)

    continuity = ns_node[0].evaluate(
        {
            "u__x": vel_grad_dict["u__x"],
            "v__y": vel_grad_dict["v__y"],
            "w__z": vel_grad_dict["w__z"],
        }
    )["continuity"]

    return continuity


def compute_tke_spectrum(
    field: np.ndarray, l: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the turbulent kinetic energy spectrum

    Args:
        field (np.ndarray): Velocity tensor (3, nx, ny, nz)
        l (float, optional): Length of the domain. Defaults to None, in which case, the
        spacing is computed asuming bounds of 2*pi.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: wave numbers and TKE raw
        and binned.
    """

    if l == None:
        lx, ly, lz = 2 * np.pi, 2 * np.pi, 2 * np.pi

    u, v, w = field[0, :, :, :], field[1, :, :, :], field[2, :, :, :]

    nx = len(u[:, 0, 0])
    ny = len(v[0, :, 0])
    nz = len(w[0, 0, :])

    nt = nx * ny * nz

    # Compute FFT of the velocity components
    uhat = np.fft.fftn(u) / nt
    vhat = np.fft.fftn(v) / nt
    what = np.fft.fftn(w) / nt
    uhat_conj = np.conjugate(uhat)
    vhat_conj = np.conjugate(vhat)
    what_conj = np.conjugate(what)

    kx = np.fft.fftfreq(nx, lx / nx)
    ky = np.fft.fftfreq(ny, ly / ny)
    kz = np.fft.fftfreq(nz, lz / nz)

    kx_g, ky_g, kz_g = np.meshgrid(kx, ky, kz, indexing="ij")
    mk = np.sqrt(kx_g**2 + ky_g**2 + kz_g**2)
    E = 0.5 * (uhat * uhat_conj + vhat * vhat_conj + what * what_conj).real

    # Perform binning
    wave_numbers = np.arange(0, nx + 1) * 2 * np.pi
    tke_spectrum = np.zeros(wave_numbers.shape)

    # Filter out under-sampled regions
    # https://scicomp.stackexchange.com/questions/21360/computing-turbulent-energy-spectrum-from-isotropic-turbulence-flow-field-in-a-bo
    for rkx in range(nx):
        for rky in range(ny):
            for rkz in range(nz):
                rk = int(np.round(np.sqrt(rkx * rkx + rky * rky + rkz * rkz)))
                if rk < len(tke_spectrum):
                    tke_spectrum[rk] += E[rkx, rky, rkz]

    E = E.flatten()
    mk = mk.flatten()
    idx = mk.argsort()
    mk = mk[idx]
    E = E[idx]

    return mk, E, wave_numbers, tke_spectrum
