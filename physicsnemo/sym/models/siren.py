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

from typing import List, Dict, Tuple, Optional

import torch.nn as nn
from torch import Tensor

from physicsnemo.models.layers import SirenLayer, SirenLayerType
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.constants import NO_OP_NORM


class SirenArch(Arch):
    """Sinusoidal Representation Network (SIREN).

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
    first_omega : float, optional
        Scales first weight matrix by this factor, by default 30
    omega : float, optional
        Scales the weight matrix of all hidden layers by this factor, by default 30
    normalization : Dict[str, Tuple[float, float]], optional
        Normalization of input to network, by default None


    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size]`
    - Output variable tensor shape: :math:`[N, size]`

    Example
    -------
    Siren model (2 -> 64 -> 64 -> 2)

    >>> arch = .siren.SirenArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    layer_size = 64,
    >>>    nr_layers = 2)
    >>> model = arch.make_node()
    >>> input = {"x": torch.randn(64, 2)}
    >>> output = model.evaluate(input)

    Note
    ----
    Reference: Sitzmann, Vincent, et al.
    Implicit Neural Representations with Periodic Activation Functions.
    https://arxiv.org/abs/2006.09661.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        first_omega: float = 30.0,
        omega: float = 30.0,
        normalization: Dict[str, Tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        layers_list = []

        layers_list.append(
            SirenLayer(
                in_features,
                layer_size,
                SirenLayerType.FIRST,
                first_omega,
            )
        )

        for _ in range(nr_layers - 1):
            layers_list.append(
                SirenLayer(layer_size, layer_size, SirenLayerType.HIDDEN, omega)
            )

        layers_list.append(
            SirenLayer(layer_size, out_features, SirenLayerType.LAST, omega)
        )

        self.layers = nn.Sequential(*layers_list)
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
        x = self.layers(x)
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
        x = self.layers(x)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
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
