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

from typing import List, Dict

import torch
import torch.nn as nn
from torch import Tensor


from physicsnemo.models.layers import FCLayer, DGMLayer
from physicsnemo.sym.models.activation import Activation, get_activation_fn
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.key import Key


class DGMArch(Arch):
    """
    A variation of the fully connected network.
    Reference: Sirignano, J. and Spiliopoulos, K., 2018.
    DGM: A deep learning algorithm for solving partial differential equations.
    Journal of computational physics, 375, pp.1339-1364.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int = 512
        Layer size for every hidden layer of the model.
    nr_layers : int = 6
        Number of hidden layers of the model.
    skip_connections : bool = False
        If true then apply skip connections every 2 hidden layers.
    activation_fn : Activation = Activation.SILU
        Activation function used by network.
    adaptive_activations : bool = False
        If True then use an adaptive activation function as described here
        https://arxiv.org/abs/1906.01170.
    weight_norm : bool = True
        Use weight norm on fully connected layers.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        activation_fn=Activation.SIN,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        self.fc_start = FCLayer(
            in_features=in_features,
            out_features=layer_size,
            activation_fn=get_activation_fn(activation_fn, out_features=out_features),
            weight_norm=weight_norm,
        )

        self.dgm_layers = nn.ModuleList()

        for _ in range(nr_layers - 1):
            single_layer = {}
            for key in ["z", "g", "r", "h"]:
                single_layer[key] = DGMLayer(
                    in_features_1=in_features,
                    in_features_2=layer_size,
                    out_features=layer_size,
                    activation_fn=get_activation_fn(
                        activation_fn, out_features=out_features
                    ),
                    weight_norm=weight_norm,
                    activation_par=activation_par,
                )
            self.dgm_layers.append(nn.ModuleDict(single_layer))

        self.fc_end = FCLayer(
            in_features=layer_size,
            out_features=out_features,
            activation_fn=None,
            weight_norm=False,
            activation_par=None,
        )

    def _tensor_forward(self, x: Tensor) -> Tensor:
        x = self.process_input(
            x,
            self.input_scales_tensor,
            periodicity=self.periodicity,
            input_dict=self.input_key_dict,
            dim=-1,
        )
        s = self.fc_start(x)
        for layer in self.dgm_layers:
            # TODO: this can be optimized, 'z', 'g', 'r' can be merged into a
            # single layer with 3x output size
            z = layer["z"](x, s)
            g = layer["g"](x, s)
            r = layer["r"](x, s)
            h = layer["h"](x, s * r)

            s = h - g * h + z * s

        x = self.fc_end(s)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
        )
        s = self.fc_start(x)
        for layer in self.dgm_layers:
            # TODO: this can be optimized, 'z', 'g', 'r' can be merged into a
            # single layer with 3x output size
            z = layer["z"](x, s)
            g = layer["g"](x, s)
            r = layer["r"](x, s)
            h = layer["h"](x, s * r)

            s = h - g * h + z * s

        x = self.fc_end(s)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
