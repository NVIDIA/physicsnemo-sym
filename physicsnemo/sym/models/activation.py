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

import enum
from typing import Callable
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from physicsnemo.sym.manager import JitManager, JitArchMode


class ActivationMeta(enum.EnumMeta):
    def __getitem__(self, name):
        try:
            return super().__getitem__(name.upper())
        except (KeyError) as error:
            raise KeyError(f"Invalid activation function {name}")


class Activation(enum.Enum, metaclass=ActivationMeta):
    ELU = enum.auto()
    LEAKY_RELU = enum.auto()
    MISH = enum.auto()
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
    STAN = enum.auto()
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


class Stan(nn.Module):
    """
    Self-scalable Tanh (Stan)
    References: Gnanasambandam, Raghav and Shen, Bo and Chung, Jihoon and Yue, Xubo and others.
    Self-scalable Tanh (Stan): Faster Convergence and Better Generalization
    in Physics-informed Neural Networks. arXiv preprint arXiv:2204.12589, 2022.
    """

    def __init__(self, out_features=1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        if x.shape[-1] != self.beta.shape[-1]:
            raise ValueError(
                f"The last dimension of the input must be equal to the dimension of Stan parameters. Got inputs: {x.shape}, params: {self.beta.shape}"
            )
        return torch.tanh(x) * (1.0 + self.beta * x)


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
        Activation.STAN: Stan,
    }

    if activation in activation_mapping and not module:
        activation_fn_ = activation_mapping[activation]
        # wraps the function because torch.sin and F.gelu could not be scripted directly
        def activation_fn(x: Tensor) -> Tensor:
            return activation_fn_(x)

    elif activation in module_activation_mapping:
        activation_fn = module_activation_mapping[activation](**kwargs)
    else:
        activation_fn = activation

    if JitManager().enabled and JitManager().arch_mode == JitArchMode.ONLY_ACTIVATION:
        activation_fn = torch.jit.script(activation_fn)

    return activation_fn
