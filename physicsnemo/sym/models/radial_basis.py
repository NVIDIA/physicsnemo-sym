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

from typing import Dict
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


from physicsnemo.models.layers import FCLayer
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.key import Key


class RadialBasisArch(Arch):
    """
    Radial Basis Neural Network.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    bounds : Dict[str, Tuple[float, float]]
        Bounds to to randomly generate radial basis functions in.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    nr_centers : int = 128
        number of radial basis functions to use.
    sigma : float = 0.1
        Sigma in radial basis kernel.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        bounds: Dict[str, List[float]],
        detach_keys: List[Key] = [],
        nr_centers: int = 128,
        sigma: float = 0.1,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        out_features = sum(self.output_key_dict.values())

        self.nr_centers = nr_centers
        self.sigma = sigma

        self.centers = nn.Parameter(
            torch.empty(nr_centers, len(bounds)), requires_grad=False
        )
        with torch.no_grad():
            for idx, bound in enumerate(bounds.values()):
                self.centers[:, idx].uniform_(bound[0], bound[1])

        self.fc_layer = FCLayer(
            nr_centers,
            out_features,
            activation_fn=None,
        )

    def _tensor_forward(self, x: Tensor) -> Tensor:
        # no op since no scales
        x = self.process_input(x, input_dict=self.input_key_dict, dim=-1)
        x = x.unsqueeze(-2)
        # no need to unsqueeze(0), we could and we have to rely on broadcast to
        # make BatchedTensor work
        centers = self.centers

        radial_activation = torch.exp(
            -0.5 * torch.square(torch.norm(centers - x, p=2, dim=-1) / self.sigma)
        )
        x = self.fc_layer(radial_activation)
        x = self.process_output(x)  # no op
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
            in_vars, self.input_key_dict.keys(), self.detach_key_dict, -1
        )
        shape = (x.size(0), self.nr_centers, x.size(1))
        x = x.unsqueeze(1).expand(shape)
        centers = self.centers.expand(shape)

        radial_activation = torch.exp(
            -0.5 * torch.square(torch.norm(centers - x, p=2, dim=-1) / self.sigma)
        )

        x = self.fc_layer(radial_activation)
        return self.prepare_output(x, self.output_key_dict, -1)
