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
"""
@Author : Clement Etienam
"""

import enum
import math
from typing import Callable
from typing import Optional
from typing import Union
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Activation(enum.Enum):
    ELU = enum.auto()
    LEAKY_RELU = enum.auto()
    MISH = enum.auto()
    POLY = enum.auto()
    RELU = enum.auto()
    GELU = enum.auto()
    SELU = enum.auto()
    PRELU = enum.auto()
    SIGMOID = enum.auto()
    SILU = enum.auto()
    SIN = enum.auto()
    SQUAREPLUS = enum.auto()
    SOFTPLUS = enum.auto()
    TANH = enum.auto()
    IDENTITY = enum.auto()


def identity(x: Tensor) -> Tensor:
    return x


def squareplus(x: Tensor) -> Tensor:
    b = 4
    return 0.5 * (x + torch.sqrt(x * x + b))


def gelu(x: Tensor) -> Tensor:
    # Applies GELU approximation, slower than sigmoid but more accurate. See: https://github.com/hendrycks/GELUs
    # Standard GELU that is present in PyTorch does not JIT compile!
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    # return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class WeightNormLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_g = nn.Parameter(torch.empty((out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.weight_g, 1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input: Tensor) -> Tensor:
        norm = self.weight.norm(dim=1, p=2, keepdim=True)
        weight = self.weight_g * self.weight / norm
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def get_activation_fn(
    activation: Union[Activation, Callable[[Tensor], Tensor]],
    module: bool = False,
    **kwargs,  # Optional parameters
) -> Callable[[Tensor], Tensor]:
    activation_mapping = {
        Activation.ELU: F.elu,
        Activation.LEAKY_RELU: F.leaky_relu,
        Activation.MISH: F.mish,
        Activation.RELU: F.relu,
        Activation.GELU: F.gelu,
        Activation.SELU: F.selu,
        Activation.SIGMOID: torch.sigmoid,
        Activation.SILU: F.silu,
        Activation.SIN: torch.sin,
        Activation.SQUAREPLUS: squareplus,
        Activation.SOFTPLUS: F.softplus,
        Activation.TANH: torch.tanh,
        Activation.IDENTITY: identity,
    }
    # Some activations have parameters in them thus must
    # be in a Module before forward call
    module_activation_mapping = {
        Activation.ELU: nn.ELU,
        Activation.LEAKY_RELU: nn.LeakyReLU,
        Activation.MISH: nn.Mish,
        Activation.RELU: nn.ReLU,
        Activation.GELU: nn.GLU,
        Activation.SELU: nn.SELU,
        Activation.PRELU: nn.PReLU,
        Activation.SIGMOID: nn.Sigmoid,
        Activation.SILU: nn.SiLU,
        Activation.TANH: nn.Tanh,
    }

    if activation in activation_mapping and not module:
        activation_fn = activation_mapping[activation]
    elif activation in module_activation_mapping:
        activation_fn = module_activation_mapping[activation](**kwargs)
    else:
        activation_fn = activation

    return activation_fn


class FCLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        weight_norm: bool = False,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()

        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(activation_fn)
        self.weight_norm = weight_norm
        self.activation_par = activation_par

        if weight_norm:
            self.linear = WeightNormLinear(in_features, out_features, bias=True)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=True)
        self.reset_parameters()

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def reset_parameters(self) -> None:
        nn.init.constant_(self.linear.bias, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.weight_norm:
            nn.init.constant_(self.linear.weight_g, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x


# FC like layer for image channels
class ConvFCLayer(nn.Module):
    def __init__(
        self,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(activation_fn)
        self.activation_par = activation_par

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def apply_activation(self, x: Tensor) -> Tensor:
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x


class Conv1dFCLayer(ConvFCLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv2dFCLayer(ConvFCLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.conv.bias, 0)
        self.conv.bias.requires_grad = False
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv3dFCLayer(ConvFCLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class SirenLayerType(enum.Enum):
    FIRST = enum.auto()
    HIDDEN = enum.auto()
    LAST = enum.auto()


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SirenLayerType = SirenLayerType.HIDDEN,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.layer_type = layer_type
        self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.apply_activation = layer_type in {
            SirenLayerType.FIRST,
            SirenLayerType.HIDDEN,
        }

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight_ranges = {
            SirenLayerType.FIRST: 1.0 / self.in_features,
            SirenLayerType.HIDDEN: math.sqrt(6.0 / self.in_features) / self.omega_0,
            SirenLayerType.LAST: math.sqrt(6.0 / self.in_features),
        }
        weight_range = weight_ranges[self.layer_type]
        nn.init.uniform_(self.linear.weight, -weight_range, weight_range)

        k_sqrt = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.linear.bias, -k_sqrt, k_sqrt)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.apply_activation:
            x = torch.sin(self.omega_0 * x)
        return x


class FourierLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        frequencies,
    ) -> None:
        super().__init__()

        # To do: Need more robust way for these params
        if isinstance(frequencies[0], str):
            if "gaussian" in frequencies[0]:
                nr_freq = frequencies[2]
                np_f = (
                    np.random.normal(0, 1, size=(nr_freq, in_features)) * frequencies[1]
                )
            else:
                nr_freq = len(frequencies[1])
                np_f = []
                if "full" in frequencies[0]:
                    np_f_i = np.meshgrid(
                        *[np.array(frequencies[1]) for _ in range(in_features)],
                        indexing="ij",
                    )
                    np_f.append(
                        np.reshape(
                            np.stack(np_f_i, axis=-1),
                            (nr_freq**in_features, in_features),
                        )
                    )
                if "axis" in frequencies[0]:
                    np_f_i = np.zeros((nr_freq, in_features, in_features))
                    for i in range(in_features):
                        np_f_i[:, i, i] = np.reshape(
                            np.array(frequencies[1]), (nr_freq)
                        )
                    np_f.append(
                        np.reshape(np_f_i, (nr_freq * in_features, in_features))
                    )
                if "diagonal" in frequencies[0]:
                    np_f_i = np.reshape(np.array(frequencies[1]), (nr_freq, 1, 1))
                    np_f_i = np.tile(np_f_i, (1, in_features, in_features))
                    np_f_i = np.reshape(np_f_i, (nr_freq * in_features, in_features))
                    np_f.append(np_f_i)
                np_f = np.concatenate(np_f, axis=-2)

        else:
            np_f = frequencies  # [nr_freq, in_features]

        frequencies = torch.tensor(np_f, dtype=torch.get_default_dtype())
        frequencies = frequencies.t().contiguous()
        self.register_buffer("frequencies", frequencies)

    def out_features(self) -> int:
        return int(self.frequencies.size(1) * 2)

    def forward(self, x: Tensor) -> Tensor:
        x_hat = torch.matmul(x, self.frequencies)
        x_sin = torch.sin(2.0 * math.pi * x_hat)
        x_cos = torch.cos(2.0 * math.pi * x_hat)
        x_i = torch.cat([x_sin, x_cos], dim=-1)
        return x_i


class FourierFilter(nn.Module):
    def __init__(
        self,
        in_features: int,
        layer_size: int,
        nr_layers: int,
        input_scale: float,
    ) -> None:
        super().__init__()

        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)

        self.frequency = nn.Parameter(torch.empty(in_features, layer_size))
        self.phase = nn.Parameter(torch.empty(1, layer_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.frequency)
        nn.init.uniform_(self.phase, -math.pi, math.pi)

    def forward(self, x: Tensor) -> Tensor:
        frequency = self.weight_scale * self.frequency

        x_i = torch.sin(torch.matmul(x, 2.0 * math.pi * frequency) + self.phase)
        return x_i


class GaborFilter(nn.Module):
    def __init__(
        self,
        in_features: int,
        layer_size: int,
        nr_layers: int,
        input_scale: float,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__()

        self.layer_size = layer_size
        self.alpha = alpha
        self.beta = beta

        self.weight_scale = input_scale / math.sqrt(nr_layers + 1)

        self.frequency = nn.Parameter(torch.empty(in_features, layer_size))
        self.phase = nn.Parameter(torch.empty(1, layer_size))
        self.mu = nn.Parameter(torch.empty(in_features, layer_size))
        self.gamma = nn.Parameter(torch.empty(1, layer_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.frequency)
        nn.init.uniform_(self.phase, -math.pi, math.pi)
        nn.init.uniform_(self.mu, -1.0, 1.0)
        with torch.no_grad():
            self.gamma.copy_(
                torch.from_numpy(
                    np.random.gamma(self.alpha, 1.0 / self.beta, (1, self.layer_size)),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        frequency = self.weight_scale * (self.frequency * self.gamma.sqrt())

        x_c = x.unsqueeze(-1)
        x_c = x_c - self.mu
        x_c = torch.square(x_c.norm(p=2, dim=1))
        x_c = torch.exp(-0.5 * x_c * self.gamma)
        x_i = x_c * torch.sin(torch.matmul(x, 2.0 * math.pi * frequency) + self.phase)
        return x_i


class DGMLayer(nn.Module):
    def __init__(
        self,
        in_features_1: int,
        in_features_2: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        weight_norm: bool = False,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()

        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(activation_fn)
        self.weight_norm = weight_norm
        self.activation_par = activation_par

        if weight_norm:
            self.linear_1 = WeightNormLinear(in_features_1, out_features, bias=False)
            self.linear_2 = WeightNormLinear(in_features_2, out_features, bias=False)
        else:
            self.linear_1 = nn.Linear(in_features_1, out_features, bias=False)
            self.linear_2 = nn.Linear(in_features_2, out_features, bias=False)
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.bias, 0)
        if self.weight_norm:
            nn.init.constant_(self.linear_1.weight_g, 1.0)
            nn.init.constant_(self.linear_2.weight_g, 1.0)

    def forward(self, input_1: Tensor, input_2: Tensor) -> Tensor:
        x = self.linear_1(input_1) + self.linear_2(input_2) + self.bias
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, 2)
        )
        self.reset_parameters()

    # Complex multiplication
    def compl_mul1d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bix,iox->box", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            bsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1],
            self.weights1,
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    def reset_parameters(self):
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.reset_parameters()

    # Complex multiplication
    def compl_mul2d(self, input: Tensor, weights: Tensor) -> Tensor:
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights2 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights3 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights4 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.reset_parameters()

    # Complex multiplication
    def compl_mul3d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * torch.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * torch.rand(self.weights4.data.shape)


def fourier_derivatives(x: Tensor, l: List[float]) -> Tuple[Tensor, Tensor]:
    # check that input shape maches domain length
    assert len(x.shape) - 2 == len(l), "input shape doesn't match domain dims"

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    x.size(0)
    n = x.shape[2:]
    dim = len(l)

    # get device
    device = x.device

    # compute fourier transform
    x_h = torch.fft.fftn(x, dim=list(range(2, dim + 2)))

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1, device=device),
                    torch.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )

    # compute laplacian in fourier space
    j = torch.complex(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )  # Cuda graphs does not work here
    wx_h = [j * k_x_i * x_h * (2 * pi / l[i]) for i, k_x_i in enumerate(k_x)]
    wxx_h = [
        j * k_x_i * wx_h_i * (2 * pi / l[i])
        for i, (wx_h_i, k_x_i) in enumerate(zip(wx_h, k_x))
    ]

    # inverse fourier transform out
    wx = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in wx_h],
        dim=1,
    )
    wxx = torch.cat(
        [
            torch.fft.ifftn(wxx_h_i, dim=list(range(2, dim + 2))).real
            for wxx_h_i in wxx_h
        ],
        dim=1,
    )
    return (wx, wxx)


def hessian_tanh_fc_layer(
    u: Tensor,
    ux: Tensor,
    uxx: Tensor,
    weights_1: Tensor,
    weights_2: Tensor,
    bias_1: Tensor,
) -> Tuple[Tensor]:
    # dim for einsum
    dim = len(u.shape) - 2
    dim_str = "xyz"[:dim]

    # compute first order derivatives of input
    # compute first layer
    if dim == 1:
        u_hidden = F.conv1d(u, weights_1, bias_1)
    elif dim == 2:
        u_hidden = F.conv2d(u, weights_1, bias_1)
    elif dim == 3:
        u_hidden = F.conv3d(u, weights_1, bias_1)

    # compute derivative hidden layer
    diff_tanh = 1 / torch.cosh(u_hidden) ** 2

    # compute diff(f(g))
    diff_fg = torch.einsum(
        "mi" + dim_str + ",bm" + dim_str + ",km" + dim_str + "->bi" + dim_str,
        weights_1,
        diff_tanh,
        weights_2,
    )

    # compute diff(f(g)) * diff(g)
    vx = [
        torch.einsum("bi" + dim_str + ",bi" + dim_str + "->b" + dim_str, diff_fg, w)
        for w in ux
    ]
    vx = [torch.unsqueeze(w, dim=1) for w in vx]

    # compute diagonal of hessian
    # double derivative of hidden layer
    diff_diff_tanh = -2 * diff_tanh * torch.tanh(u_hidden)

    # compute diff(g) * hessian(f) * diff(g)
    vxx1 = [
        torch.einsum(
            "bi"
            + dim_str
            + ",mi"
            + dim_str
            + ",bm"
            + dim_str
            + ",mj"
            + dim_str
            + ",bj"
            + dim_str
            + "->b"
            + dim_str,
            w,
            weights_1,
            weights_2 * diff_diff_tanh,
            weights_1,
            w,
        )
        for w in ux
    ]  # (b,x,y,t)

    # compute diff(f) * hessian(g)
    vxx2 = [
        torch.einsum("bi" + dim_str + ",bi" + dim_str + "->b" + dim_str, diff_fg, w)
        for w in uxx
    ]
    vxx = [torch.unsqueeze(a + b, dim=1) for a, b in zip(vxx1, vxx2)]

    return (vx, vxx)
