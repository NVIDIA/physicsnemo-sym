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
import numpy as np

from physicsnemo.sym.key import Key
from physicsnemo.sym.models.activation import Activation, get_activation_fn
from physicsnemo.sym.models.arch import Arch

from physicsnemo.models.pix2pix import Pix2Pix

Tensor = torch.Tensor


class Pix2PixArch(Arch):
    """Convolutional encoder-decoder based on pix2pix generator models.

    Note
    ----
    The pix2pix architecture supports options for 1D, 2D and 3D fields which can
    be constroled using the `dimension` parameter.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list. The key dimension size should equal the variables channel dim.
    output_keys : List[Key]
        Output key list. The key dimension size should equal the variables channel dim.
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    conv_layer_size : int, optional
        Latent channel size after first convolution, by default 64
    n_downsampling : int, optional
        Number of downsampling/upsampling blocks, by default 3
    n_blocks : int, optional
        Number of residual blocks in middle of model, by default 3
    scaling_factor : int, optional
        Scaling factor to increase the output feature size compared to the input
        (1, 2, 4, or 8), by default 1
    activation_fn : Activation, optional
        Activation function, by default :obj:`Activation.RELU`
    batch_norm : bool, optional
        Batch normalization, by default False
    padding_type : str, optional
        Padding type ('constant', 'reflect', 'replicate' or 'circular'),
        by default "reflect"

    Variable Shape
    --------------
    Input variable tensor shape:

    - 1D: :math:`[N, size, W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Output variable tensor shape:

    - 1D: :math:`[N, size,  W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Note
    ----
    Reference:  Isola, Phillip, et al. “Image-To-Image translation with conditional
    adversarial networks” Conference on Computer Vision and Pattern Recognition, 2017.
    https://arxiv.org/abs/1611.07004

    Reference: Wang, Ting-Chun, et al. “High-Resolution image synthesis and semantic
    manipulation with conditional GANs” Conference on Computer Vision and Pattern
    Recognition, 2018. https://arxiv.org/abs/1711.11585

    Note
    ----
    Based on the implementation: https://github.com/NVIDIA/pix2pixHD
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        dimension: int,
        detach_keys: List[Key] = [],
        conv_layer_size: int = 64,
        n_downsampling: int = 3,
        n_blocks: int = 3,
        scaling_factor: int = 1,
        activation_fn: Activation = Activation.RELU,
        batch_norm: bool = False,
        padding_type="reflect",
    ):
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )
        in_channels = sum(self.input_key_dict.values())
        out_channels = sum(self.output_key_dict.values())
        self.var_dim = 1
        activation_fn = get_activation_fn(activation_fn, module=True, inplace=True)

        # Scaling factor must be 1, 2, 4, or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {
            1,
            2,
            4,
            8,
        }, "The scaling factor must be 1, 2, 4, or 8!"
        n_upsampling = n_downsampling + int(np.log2(scaling_factor))

        self._impl = Pix2Pix(
            in_channels,
            out_channels,
            dimension,
            conv_layer_size,
            n_downsampling,
            n_upsampling,
            n_blocks,
            activation_fn,
            batch_norm,
            padding_type,
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=1,
            input_scales=self.input_scales,
        )
        output = self._impl(input)
        return self.prepare_output(
            output, self.output_key_dict, dim=1, output_scales=self.output_scales
        )
