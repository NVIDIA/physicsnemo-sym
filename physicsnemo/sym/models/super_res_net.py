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
from typing import List, Dict

from physicsnemo.models.srrn import SRResNet
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.models.activation import Activation, get_activation_fn


Tensor = torch.Tensor


class SRResNetArch(Arch):
    """3D super resolution network

    Based on the implementation:
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    large_kernel_size : int, optional
        convolutional kernel size for first and last convolution, by default 7
    small_kernel_size : int, optional
        convolutional kernel size for internal convolutions, by default 3
    conv_layer_size : int, optional
        Latent channel size, by default 32
    n_resid_blocks : int, optional
        Number of residual blocks before , by default 8
    scaling_factor : int, optional
        Scaling factor to increase the output feature size compared to the input (2, 4, or 8), by default 8
    activation_fn : Activation, optional
        Activation function, by default Activation.PRELU
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        large_kernel_size: int = 7,
        small_kernel_size: int = 3,
        conv_layer_size: int = 32,
        n_resid_blocks: int = 8,
        scaling_factor: int = 8,
        activation_fn: Activation = Activation.PRELU,
    ):
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )
        in_channels = sum(self.input_key_dict.values())
        out_channels = sum(self.output_key_dict.values())
        activation_fn = get_activation_fn(activation_fn)
        self.srrn = SRResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            conv_layer_size=conv_layer_size,
            n_resid_blocks=n_resid_blocks,
            scaling_factor=scaling_factor,
            activation_fn=activation_fn,
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )

        output = self.srrn(input)

        return self.prepare_output(
            output, self.output_key_dict, dim=1, output_scales=self.output_scales
        )
