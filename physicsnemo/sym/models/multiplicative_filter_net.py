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
from typing import Optional, List, Dict, Tuple, Union

import torch.nn as nn
from torch import Tensor

from physicsnemo.models.layers import FCLayer, FourierFilter, GaborFilter
from physicsnemo.sym.models.activation import Activation, get_activation_fn
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.constants import NO_OP_NORM


class FilterTypeMeta(enum.EnumMeta):
    def __getitem__(self, name):
        try:
            return super().__getitem__(name.upper())
        except (KeyError) as error:
            raise KeyError(f"Invalid activation function {name}")


class FilterType(enum.Enum, metaclass=FilterTypeMeta):
    FOURIER = enum.auto()
    GABOR = enum.auto()


class MultiplicativeFilterNetArch(Arch):
    """
    Multiplicative Filter Net with Activations
    Reference: Fathony, R., Sahu, A.K., AI, A.A., Willmott, D. and Kolter, J.Z., MULTIPLICATIVE FILTER NETWORKS.

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
    filter_type : FilterType = FilterType.FOURIER
        Filter type for multiplicative filter network, (Fourier or Gabor).
    weight_norm : bool = True
        Use weight norm on fully connected layers.
    input_scale : float = 10.0
        Scale inputs for multiplicative filters.
    gabor_alpha : float = 6.0
        Alpha value for Gabor filter.
    gabor_beta : float = 1.0
        Beta value for Gabor filter.
    normalization : Optional[Dict[str, Tuple[float, float]]] = None
        Normalization of input to network.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        skip_connections: bool = False,
        activation_fn=Activation.IDENTITY,
        filter_type: Union[FilterType, str] = FilterType.FOURIER,
        weight_norm: bool = True,
        input_scale: float = 10.0,
        gabor_alpha: float = 6.0,
        gabor_beta: float = 1.0,
        normalization: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        self.nr_layers = nr_layers
        self.skip_connections = skip_connections
        activation_fn = get_activation_fn(activation_fn)

        if isinstance(filter_type, str):
            filter_type = FilterType[filter_type]

        if filter_type == FilterType.FOURIER:
            self.first_filter = FourierFilter(
                in_features=in_features,
                layer_size=layer_size,
                nr_layers=nr_layers,
                input_scale=input_scale,
            )
        elif filter_type == FilterType.GABOR:
            self.first_filter = GaborFilter(
                in_features=in_features,
                layer_size=layer_size,
                nr_layers=nr_layers,
                input_scale=input_scale,
                alpha=gabor_alpha,
                beta=gabor_beta,
            )
        else:
            raise ValueError

        self.filters = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        for i in range(nr_layers):
            self.fc_layers.append(
                FCLayer(
                    in_features=layer_size,
                    out_features=layer_size,
                    activation_fn=activation_fn,
                    weight_norm=weight_norm,
                )
            )
            if filter_type == FilterType.FOURIER:
                self.filters.append(
                    FourierFilter(
                        in_features=in_features,
                        layer_size=layer_size,
                        nr_layers=nr_layers,
                        input_scale=input_scale,
                    )
                )
            elif filter_type == FilterType.GABOR:
                self.filters.append(
                    GaborFilter(
                        in_features=in_features,
                        layer_size=layer_size,
                        nr_layers=nr_layers,
                        input_scale=input_scale,
                        alpha=gabor_alpha,
                        beta=gabor_beta,
                    )
                )
            else:
                raise ValueError

        self.final_layer = FCLayer(
            in_features=layer_size,
            out_features=out_features,
            activation_fn=None,
            weight_norm=False,
            activation_par=None,
        )

        self.normalization: Optional[Dict[str, Tuple[float, float]]] = normalization
        # iterate input keys and add NO_OP_NORM if it is not specified
        if self.normalization is not None:
            for key in self.input_key_dict:
                if key not in self.normalization:
                    self.normalization[key] = NO_OP_NORM
        self.register_buffer(
            "normalization_tensor",
            self._get_normalization_tensor(self.input_key_dict, self.normalization),
            persistent=False,
        )

    def _tensor_forward(self, x: Tensor) -> Tensor:
        x = self._tensor_normalize(x, self.normalization_tensor)
        x = self.process_input(
            x, self.input_scales_tensor, input_dict=self.input_key_dict, dim=-1
        )

        res = self.first_filter(x)
        res_skip: Optional[Tensor] = None
        for i, (fc_layer, filter) in enumerate(zip(self.fc_layers, self.filters)):
            res_fc = fc_layer(res)
            res_filter = filter(x)
            res = res_fc * res_filter
            if self.skip_connections and i % 2 == 0:
                if res_skip is not None:
                    res, res_skip = res + res_skip, res
                else:
                    res_skip = res

        x = self.final_layer(res)
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
            self._normalize(in_vars, self.normalization),
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
        )

        res = self.first_filter(x)
        res_skip: Optional[Tensor] = None
        for i, (fc_layer, filter) in enumerate(zip(self.fc_layers, self.filters)):
            res_fc = fc_layer(res)
            res_filter = filter(x)
            res = res_fc * res_filter
            if self.skip_connections and i % 2 == 0:
                if res_skip is not None:
                    res, res_skip = res + res_skip, res
                else:
                    res_skip = res

        res = self.final_layer(res)
        return self.prepare_output(
            res, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )

    def _normalize(
        self,
        in_vars: Dict[str, Tensor],
        norms: Optional[Dict[str, Tuple[float, float]]],
    ) -> Dict[str, Tensor]:
        if norms is None:
            return in_vars

        normalized_in_vars = {}
        for k, v in in_vars.items():
            if k in norms:
                v = (v - norms[k][0]) / (norms[k][1] - norms[k][0])
                v = 2 * v - 1
            normalized_in_vars[k] = v
        return normalized_in_vars
