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
from typing import Union, List

Tensor = torch.Tensor


class FirstDerivSecondOrder(torch.nn.Module):
    """Module to compute first derivative with 2nd order accuracy"""

    def __init__(self, dim: int, dx: Union[float, List[float]]):
        super().__init__()
        self.dim = dim
        if isinstance(dx, float):
            dx = [dx for _ in range(dim)]
        self.dx = dx

        self.register_buffer("ddx1D", torch.Tensor([-1.0 / 2.0, 0.0, 1.0 / 2.0]))

    def get_convolution_kernel(self, axis):
        shape = [1, 1] + axis * [1] + [-1] + (self.dim - axis - 1) * [1]
        # multiply with 1/dx
        return (1 / self.dx[axis]) * torch.reshape(self.ddx1D, shape)

    def apply_convolution(self, u, kernel, axis):
        if self.dim == 1:
            conv_result = torch.nn.functional.conv1d(
                u, kernel, stride=1, padding=0, bias=None
            )
        elif self.dim == 2:
            conv_result = torch.nn.functional.conv2d(
                u, kernel, stride=1, padding=0, bias=None
            )
        elif self.dim == 3:
            conv_result = torch.nn.functional.conv3d(
                u, kernel, stride=1, padding=0, bias=None
            )
        slice_index = (kernel.flatten().shape[-1] - 1) // 2
        if self.dim == 1:
            return conv_result
        elif self.dim == 2:
            if axis == 0:
                return conv_result[:, :, :, slice_index:-slice_index]
            elif axis == 1:
                return conv_result[:, :, slice_index:-slice_index, :]
        elif self.dim == 3:
            if axis == 0:
                return conv_result[
                    :, :, :, slice_index:-slice_index, slice_index:-slice_index
                ]
            elif axis == 1:
                return conv_result[
                    :, :, slice_index:-slice_index, :, slice_index:-slice_index
                ]
            elif axis == 2:
                return conv_result[
                    :, :, slice_index:-slice_index, slice_index:-slice_index, :
                ]

    def forward(self, u) -> List[Tensor]:
        u = torch.nn.functional.pad(u, self.dim * (1, 1), "replicate")
        result = []
        for axis in range(self.dim):
            kernel = self.get_convolution_kernel(axis)
            conv_result = self.apply_convolution(u, kernel, axis)
            result.append(conv_result)
        return result


class SecondDerivSecondOrder(torch.nn.Module):
    """Module to compute second derivative with 2nd order accuracy"""

    def __init__(self, dim: int, dx: Union[float, List[float]]):
        super().__init__()
        self.dim = dim
        if isinstance(dx, float):
            dx = [dx for _ in range(dim)]
        self.dx = dx

        self.register_buffer("d2dx21D", torch.Tensor([1.0, -2.0, 1.0]))

    def get_convolution_kernel(self, axis):
        shape = [1, 1] + axis * [1] + [-1] + (self.dim - axis - 1) * [1]
        # multiply with 1/dx**2
        return (1 / (self.dx[axis] ** 2)) * torch.reshape(self.d2dx21D, shape)

    def apply_convolution(self, u, kernel, axis):
        if self.dim == 1:
            conv_result = torch.nn.functional.conv1d(
                u, kernel, stride=1, padding=0, bias=None
            )
        elif self.dim == 2:
            conv_result = torch.nn.functional.conv2d(
                u, kernel, stride=1, padding=0, bias=None
            )
        elif self.dim == 3:
            conv_result = torch.nn.functional.conv3d(
                u, kernel, stride=1, padding=0, bias=None
            )
        slice_index = (kernel.flatten().shape[-1] - 1) // 2
        if self.dim == 1:
            return conv_result
        elif self.dim == 2:
            if axis == 0:
                return conv_result[:, :, :, slice_index:-slice_index]
            elif axis == 1:
                return conv_result[:, :, slice_index:-slice_index, :]
        elif self.dim == 3:
            if axis == 0:
                return conv_result[
                    :, :, :, slice_index:-slice_index, slice_index:-slice_index
                ]
            elif axis == 1:
                return conv_result[
                    :, :, slice_index:-slice_index, :, slice_index:-slice_index
                ]
            elif axis == 2:
                return conv_result[
                    :, :, slice_index:-slice_index, slice_index:-slice_index, :
                ]

    def forward(self, u) -> List[Tensor]:
        u = torch.nn.functional.pad(u, self.dim * (1, 1), "replicate")
        result = []
        for axis in range(self.dim):
            kernel = self.get_convolution_kernel(axis)
            conv_result = self.apply_convolution(u, kernel, axis)
            result.append(conv_result)
        return result


class MixedSecondDerivSecondOrder(torch.nn.Module):
    """Module to compute second mixed derivative with 2nd order accuracy

    For 2d, this returns [d2f/dxdy]
    For 3d, this returns [d2f/dxdy, d2f/dxdz, d2f/dydz]

    Ref: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119083405.app1
    """

    def __init__(self, dim: int, dx: Union[float, List[float]]):
        super().__init__()
        self.dim = dim
        assert self.dim > 1, "Mixed Derivatives only supported for 2D and 3D inputs"
        if isinstance(dx, float):
            dx = [dx for _ in range(dim)]
        self.dx = dx
        assert len(self.dx) == self.dim, "Mismatch between dx and dim"

        self.register_buffer("ddx1D", torch.Tensor([-1.0, 0.0, 1.0]))

    def get_convolution_kernel(self, axis):
        shape = [1, 1] + axis * [1] + [-1] + (self.dim - axis - 1) * [1]
        return torch.reshape(self.ddx1D, shape)

    def apply_convolution(self, u, kernel, axis):
        if self.dim == 2:
            conv_result = torch.nn.functional.conv2d(
                u, kernel, stride=1, padding=0, bias=None
            )
        elif self.dim == 3:
            conv_result = torch.nn.functional.conv3d(
                u, kernel, stride=1, padding=0, bias=None
            )
        slice_index = (kernel.flatten().shape[-1] - 1) // 2
        if self.dim == 2:
            if axis == 0:
                return conv_result[:, :, :, slice_index:-slice_index]
            elif axis == 1:
                return conv_result[:, :, slice_index:-slice_index, :]
        elif self.dim == 3:
            if axis == 0:
                return conv_result[
                    :, :, :, slice_index:-slice_index, slice_index:-slice_index
                ]
            elif axis == 1:
                return conv_result[
                    :, :, slice_index:-slice_index, :, slice_index:-slice_index
                ]
            elif axis == 2:
                return conv_result[
                    :, :, slice_index:-slice_index, slice_index:-slice_index, :
                ]

    def forward(self, u) -> List[Tensor]:
        # get u_i+1 and u_i-1
        pad = (self.dim - 1) * (0, 0) + (1, 1)
        u_pad = torch.nn.functional.pad(u, pad, "constant")

        u_xplus = u_pad[:, :, :-2]
        u_xminus = u_pad[:, :, 2:]
        u_xplus = torch.nn.functional.pad(u_xplus, self.dim * (1, 1), "replicate")
        u_xminus = torch.nn.functional.pad(u_xminus, self.dim * (1, 1), "replicate")

        result = []
        for axis in range(self.dim - 1):
            kernel = self.get_convolution_kernel(
                axis + 1
            )  # Apply convolution except x dim
            conv_result_1 = self.apply_convolution(u_xplus, kernel, axis + 1)
            conv_result_2 = self.apply_convolution(u_xminus, kernel, axis + 1)
            conv_result = (1 / (4 * self.dx[0] * self.dx[axis])) * (
                conv_result_2 - conv_result_1
            )
            result.append(conv_result)  # for 2d, gives dxdy; for 3d gives dxdy, dxdz

        if self.dim == 3:  # compute dydz
            pad = (0, 0, 1, 1, 0, 0)
            u_pad = torch.nn.functional.pad(u, pad, "constant")

            u_yplus = u_pad[:, :, :, :-2]
            u_yminus = u_pad[:, :, :, 2:]
            u_yplus = torch.nn.functional.pad(u_yplus, self.dim * (1, 1), "replicate")
            u_yminus = torch.nn.functional.pad(u_yminus, self.dim * (1, 1), "replicate")

            kernel = self.get_convolution_kernel(2)  # Apply convolution on z dim
            conv_result_1 = self.apply_convolution(u_yplus, kernel, 2)
            conv_result_2 = self.apply_convolution(u_yminus, kernel, 2)
            conv_result = (1 / (4 * self.dx[1] * self.dx[2])) * (
                conv_result_2 - conv_result_1
            )
            result.append(conv_result)

        return result
