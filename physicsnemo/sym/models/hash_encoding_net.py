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
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Dict, List, Tuple
import itertools

import physicsnemo.sym.models.fully_connected as fully_connected
from physicsnemo.sym.models.activation import Activation
from physicsnemo.models.layers.interpolation import (
    _grid_knn_idx,
    _hyper_cube_weighting,
    smooth_step_2,
    linear_step,
)
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.distributed import DistributedManager


class MultiresolutionHashNetArch(Arch):
    """Hash encoding network as seen in,

    Müller, Thomas, et al. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." arXiv preprint arXiv:2201.05989 (2022).
    A reference pytorch implementation can be found, https://github.com/yashbhalgat/HashNeRF-pytorch

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    activation_fn : Activation = Activation.SILU
        Activation function used by network.
    layer_size : int = 64
        Layer size for every hidden layer of the model.
    nr_layers : int = 3
        Number of hidden layers of the model.
    skip_connections : bool = False
        If true then apply skip connections every 2 hidden layers.
    weight_norm : bool = False
        Use weight norm on fully connected layers.
    adaptive_activations : bool = False
        If True then use an adaptive activation function as described here
        https://arxiv.org/abs/1906.01170.
    bounds : List[Tuple[float, float]] = [(-1.0, 1.0), (-1.0, 1.0)]
        List of bounds for hash grid. Each element is a tuple
        of the upper and lower bounds.
    nr_levels :  int = 5
        Number of levels in the hash grid.
    nr_features_per_level : int = 2
        Number of features from each hash grid.
    log2_hashmap_size : int = 19
        Hash map size will be `2**log2_hashmap_size`.
    base_resolution : int = 2
        base resolution of hash grids.
    finest_resolution : int = 32
        Highest resolution of hash grids.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        activation_fn=Activation.SILU,
        layer_size: int = 64,
        nr_layers: int = 3,
        skip_connections: bool = False,
        weight_norm: bool = True,
        adaptive_activations: bool = False,
        bounds: List[Tuple[float, float]] = [(-1.0, 1.0), (-1.0, 1.0)],
        nr_levels: int = 16,
        nr_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 2,
        finest_resolution: int = 32,
    ) -> None:
        super().__init__(
            input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys
        )

        # get needed input output information
        self.xyzt_var = [x for x in self.input_key_dict if x in ["x", "y", "z", "t"]]
        self.params_var = [
            x for x in self.input_key_dict if x not in ["x", "y", "z", "t"]
        ]
        in_features_xyzt = sum(
            (v for k, v in self.input_key_dict.items() if k in self.xyzt_var)
        )
        in_features_params = sum(
            (v for k, v in self.input_key_dict.items() if k in self.params_var)
        )
        in_features = in_features_xyzt + in_features_params
        out_features = sum(self.output_key_dict.values())
        if len(self.params_var) == 0:
            self.params_var = None

        # get device for torch constants used in inference
        self.device = DistributedManager().device

        # store hash grid parameters
        self.bounds = bounds
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = Tensor([base_resolution])
        self.finest_resolution = Tensor([finest_resolution])
        self.nr_levels = nr_levels
        self.nr_features_per_level = nr_features_per_level

        # make embeddings
        self.embedding = nn.Embedding(
            self.nr_levels * 2**self.log2_hashmap_size, self.nr_features_per_level
        )
        nn.init.uniform_(self.embedding.weight, a=-0.001, b=0.001)
        self.b = np.exp(
            (np.log(self.finest_resolution) - np.log(self.base_resolution))
            / (nr_levels - 1)
        )

        # make grid dx and start tensors
        list_dx = []
        list_start = []
        list_resolution = []
        for i in range(self.nr_levels):
            # calculate resolution
            resolution = int(np.floor(self.base_resolution * self.b**i))
            list_resolution.append(
                torch.tensor([resolution]).to(self.device).view(1, 1)
            )

            # make adjust factor
            adjust_factor = ((8253729**i + 2396403) % 32767) / 32767.0

            # compute grid and adjust it
            not_adjusted_dx = [(x[1] - x[0]) / (resolution - 1) for x in self.bounds]
            grid = [
                (
                    b[0] + (-2.0 + adjust_factor) * x,
                    b[1] + (2.0 + adjust_factor) * x,
                    resolution,
                )
                for b, x in zip(self.bounds, not_adjusted_dx)
            ]

            # make grid spacing size tensor
            dx = torch.tensor([(x[1] - x[0]) / (x[2] - 1) for x in grid]).to(
                self.device
            )
            dx = dx.view(1, len(grid))
            list_dx.append(dx)

            # make start tensor of grid
            start = torch.tensor([val[0] for val in grid]).to(self.device)
            start = start.view(1, len(grid))
            list_start.append(start)

        # stack values
        self.resolutions = torch.stack(list_resolution, dim=1)
        self.dx = torch.stack(list_dx, dim=1)
        self.start = torch.stack(list_start, dim=1)

        # hyper cube for adding to lower point index
        self.hyper_cube = (
            torch.tensor(list(itertools.product(*(len(self.bounds) * [[0, 1]]))))
            .to(self.device)
            .view(1, 1, -1, len(bounds))
        )

        # multiply factor for hash encoding to order layers
        list_mul_factor = []
        mul_factor = torch.tensor([1], dtype=torch.int).to(self.device)
        for r in range(self.nr_levels):
            for d in range(len(self.bounds)):
                list_mul_factor.append(mul_factor.clone())
                mul_factor *= self.resolutions[0, r, 0]
                mul_factor %= 20731370  # prevent overflow
        self.mul_factor = torch.stack(list_mul_factor).view(
            1, self.nr_levels, 1, len(self.bounds)
        )

        # make fully connected decoding network
        self.fc = fully_connected.FullyConnectedArchCore(
            in_features=(self.nr_features_per_level * nr_levels) + in_features_params,
            layer_size=layer_size,
            out_features=out_features,
            nr_layers=nr_layers,
            skip_connections=skip_connections,
            activation_fn=activation_fn,
            adaptive_activations=adaptive_activations,
            weight_norm=weight_norm,
        )

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # get spacial inputs and hash encode
        in_xyzt_var = self.prepare_input(
            in_vars,
            self.xyzt_var,
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
        )

        # unsqueeze input to operate on all grids at once
        unsqueezed_xyzt = torch.unsqueeze(in_xyzt_var, 1)

        # get lower and upper bounds cells
        lower_indice = torch.floor((unsqueezed_xyzt - self.start) / self.dx).int()
        all_indice = torch.unsqueeze(lower_indice, -2) + self.hyper_cube
        lower_point = lower_indice * self.dx + self.start
        upper_point = lower_point + self.dx

        # get hash from indices and resolutions
        key = torch.sum(all_indice * self.mul_factor, dim=-1)
        key = 10000003 * key + 124777 * torch.bitwise_xor(
            key, torch.tensor(3563504501)
        )  # shuffle it
        key = (
            torch.tensor(self.nr_levels * (1 << self.log2_hashmap_size) - 1).to(
                key.device
            )
            & key
        )

        # compute embedding
        embed = self.embedding(key)

        # compute smooth step interpolation of embeddings
        smoothed_lower_point = smooth_step_2((unsqueezed_xyzt - lower_point) / self.dx)
        smoother_upper_point = smooth_step_2(-(unsqueezed_xyzt - upper_point) / self.dx)
        weights = _hyper_cube_weighting(smoothed_lower_point, smoother_upper_point)

        # add embedding to list
        hash_xyzt = torch.sum(embed * weights, dim=-2)
        x = torch.reshape(hash_xyzt, [hash_xyzt.shape[0], -1])

        # add other features
        if self.params_var is not None:
            in_params_var = self.prepare_input(
                in_vars,
                self.params_var,
                detach_dict=self.detach_key_dict,
                dim=-1,
                input_scales=self.input_scales,
            )
            x = torch.cat((x, in_params_var), dim=-1)

        x = self.fc(x)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
