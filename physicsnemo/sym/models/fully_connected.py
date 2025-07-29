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

from typing import Optional, Dict, Tuple, Union, List
from physicsnemo.sym.key import Key

import torch
import torch.nn as nn
from torch import Tensor

from physicsnemo.sym.amp import (
    amp_manager_scaler_enabled_and_disable_autocast_firstlayer,
)

from physicsnemo.models.layers import FCLayer, Conv1dFCLayer
from physicsnemo.sym.models.activation import Activation, get_activation_fn
from physicsnemo.sym.models.arch import Arch


class FullyConnectedArchCore(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        layer_size: int = 512,
        out_features: int = 512,
        nr_layers: int = 6,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        conv_layers: bool = False,
    ) -> None:
        super().__init__()

        self.skip_connections = skip_connections

        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers
        if conv_layers:
            fc_layer = Conv1dFCLayer
        else:
            fc_layer = FCLayer

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                nr_layers - len(activation_fn)
            )

        self.layers = nn.ModuleList()

        layer_in_features = in_features
        for i in range(nr_layers):
            self.layers.append(
                fc_layer(
                    in_features=layer_in_features,
                    out_features=layer_size,
                    activation_fn=get_activation_fn(
                        activation_fn[i], out_features=out_features
                    ),
                    weight_norm=weight_norm,
                    activation_par=activation_par,
                )
            )
            layer_in_features = layer_size

        self.final_layer = fc_layer(
            in_features=layer_size,
            out_features=out_features,
            activation_fn=None,
            weight_norm=False,
            activation_par=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.layers):
            if i == 0 and amp_manager_scaler_enabled_and_disable_autocast_firstlayer():
                # disable autocast for the first layer
                with torch.cuda.amp.autocast(enabled=False):
                    x = layer(x.float())
            else:
                x = layer(x)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x)
        return x

    def get_weight_list(self):
        weights = [layer.conv.weight for layer in self.layers] + [
            self.final_layer.conv.weight
        ]
        biases = [layer.conv.bias for layer in self.layers] + [
            self.final_layer.conv.bias
        ]
        return weights, biases


class FullyConnectedArch(Arch):
    """Fully Connected Neural Network.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list.
    output_keys : List[Key]
        Output key list.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int, optional
        Layer size for every hidden layer of the model, by default 512
    nr_layers : int, optional
        Number of hidden layers of the model, by default 6
    activation_fn : Activation, optional
        Activation function used by network, by default :obj:`Activation.SILU`
    periodicity : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary of tuples that allows making model give periodic predictions on
        the given bounds in tuple.
    skip_connections : bool, optional
        Apply skip connections every 2 hidden layers, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default True
    adaptive_activations : bool, optional
        Use an adaptive activation functions, by default False

    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size]`
    - Output variable tensor shape: :math:`[N, size]`

    Example
    -------
    Fully-connected model (2 -> 64 -> 64 -> 2)

    >>> arch = .fully_connected.FullyConnectedArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    layer_size = 64,
    >>>    nr_layers = 2)
    >>> model = arch.make_node()
    >>> input = {"x": torch.randn(64, 2)}
    >>> output = model.evaluate(input)

    Fully-connected model with periodic outputs between (0,1)

    >>> arch = .fully_connected.FullyConnectedArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    periodicity={'x': (0, 1)})

    Note
    ----
    For information regarding adaptive activations please refer to
    https://arxiv.org/abs/1906.01170.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        activation_fn=Activation.SILU,
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
    ) -> None:
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
            periodicity=periodicity,
        )

        if self.periodicity is not None:
            in_features = sum(
                [
                    x.size
                    for x in self.input_keys
                    if x.name not in list(periodicity.keys())
                ]
            ) + +sum(
                [
                    2 * x.size
                    for x in self.input_keys
                    if x.name in list(periodicity.keys())
                ]
            )
        else:
            in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        self._impl = FullyConnectedArchCore(
            in_features,
            layer_size,
            out_features,
            nr_layers,
            skip_connections,
            activation_fn,
            adaptive_activations,
            weight_norm,
        )

    def _tensor_forward(self, x: Tensor) -> Tensor:
        x = self.process_input(
            x,
            self.input_scales_tensor,
            periodicity=self.periodicity,
            input_dict=self.input_key_dict,
            dim=-1,
        )
        x = self._impl(x)
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
            periodicity=self.periodicity,
        )
        y = self._impl(x)
        return self.prepare_output(
            y, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )


class ConvFullyConnectedArch(Arch):
    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        activation_fn=Activation.SILU,
        skip_connections: bool = False,
        adaptive_activations: bool = False,
    ) -> None:
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
        )
        self.var_dim = 1
        in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        self._impl = FullyConnectedArchCore(
            in_features,
            layer_size,
            out_features,
            nr_layers,
            skip_connections,
            activation_fn,
            adaptive_activations,
            weight_norm=False,
            conv_layers=True,
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        x_shape = list(x.size())
        x = x.view(x.shape[0], x.shape[1], -1)
        y = self._impl(x)

        x_shape[1] = y.shape[1]
        y = y.view(x_shape)

        return self.prepare_output(
            y, self.output_key_dict, dim=1, output_scales=self.output_scales
        )
