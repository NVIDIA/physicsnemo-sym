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

import logging
from typing import List, Dict, Union

import torch

from torch import Tensor
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.manager import GraphManager

logger = logging.getLogger(__name__)


class DeepONetArch(Arch):
    """DeepONet

    Parameters
    ----------
    branch_net : Arch
        Branch net model. Output key should be variable "branch"
    trunk_net : Arch
        Trunk net model. Output key should be variable "trunk"
    output_keys : List[Key], optional
        Output variable keys, by default None
    detach_keys : List[Key], optional
    List of keys to detach gradients, by default []
    branch_dim : Union[None, int], optional
        Dimension of the branch encoding vector. If none, the model will use the
        variable trunk dimension. Should be set for 2D/3D models. By default None
    trunk_dim : Union[None, int], optional
        Dimension of the trunk encoding vector. If none, the model will use the
        variable trunk dimension. Should be set for 2D/3D models. By default None

    Note
    ----
    The branch and trunk net should ideally output to the same dimensionality, but if
    this is not the case the DeepO model will use a linear layer to match both branch/trunk
    dimensionality to (branch_dim + trunk_dim)/2. This vector will then be
    used for the final output multiplication.

    Note
    ----
    Higher dimension branch networks are supported. If the output is not a 1D vector the
    DeepO model will reshape for the final output multiplication.

    Note
    ----
    For more info on DeepONet refer to: https://arxiv.org/abs/1910.03193
    """

    def __init__(
        self,
        branch_net: Arch,
        trunk_net: Arch,
        output_keys: List[Key] = None,
        detach_keys: List[Key] = [],
        branch_dim: Union[None, int] = None,
        trunk_dim: Union[None, int] = None,
    ) -> None:
        super().__init__(
            input_keys=[],
            output_keys=output_keys,
            detach_keys=detach_keys,
        )
        # branch net
        self.branch_net = branch_net
        self.branch_dim = branch_dim
        # trunk net
        self.trunk_net = trunk_net
        self.trunk_dim = trunk_dim

        # Set up input keys not trunk and branch should be initialized
        self.input_keys = self.branch_net.input_keys + self.trunk_net.input_keys
        self.input_key_dict = {str(var): var.size for var in self.input_keys}
        self.input_scales = {str(k): k.scale for k in self.input_keys}

        # Set up output linear layer for multiple variables
        # If output dims have not been defined, attempt to set then through the variables
        if self.trunk_dim is None:
            self.trunk_dim = sum(self.trunk_net.output_key_dict.values())
        if self.branch_dim is None:
            self.branch_dim = sum(self.branch_net.output_key_dict.values())

        self.deepo_dim = (self.trunk_dim + self.branch_dim) // 2
        out_features = sum(self.output_key_dict.values())

        if not self.trunk_dim == self.branch_dim:
            self.branch_linear = torch.nn.Linear(
                self.branch_dim, self.deepo_dim, bias=False
            )
            self.trunk_linear = torch.nn.Linear(
                self.trunk_dim, self.deepo_dim, bias=False
            )
        else:
            self.branch_linear = torch.nn.Identity()
            self.trunk_linear = torch.nn.Identity()

        self.output_linear = torch.nn.Linear(self.deepo_dim, out_features, bias=False)

        # prepare slice indices
        branch_slice_index = self.prepare_slice_index(
            self.input_key_dict, self.branch_net.input_key_dict.keys()
        )
        self.register_buffer("branch_slice_index", branch_slice_index, persistent=False)
        trunk_slice_index = self.prepare_slice_index(
            self.input_key_dict, self.trunk_net.input_key_dict.keys()
        )
        self.register_buffer("trunk_slice_index", trunk_slice_index, persistent=False)

        # Because we directly call `branch_net._tensor_forward` and `trunk_net._tensor_forward`
        # method in `self._tensor_forward`, we have to redirect `self.forward` to
        # `self._dict_forward` if one of them does not support func_arch.
        if not self.supports_func_arch:
            self.forward = self._dict_forward
            if GraphManager().func_arch:
                logger.warning(
                    f"The combination of branch_net ({type(self.branch_net)}) and trunk_net"
                    + f"({type(self.trunk_net)}) does not support FuncArch."
                )

    @property
    def supports_func_arch(self) -> bool:
        return self.branch_net.supports_func_arch and self.trunk_net.supports_func_arch

    def _tensor_forward(self, x: Tensor) -> Tensor:
        assert self.supports_func_arch, (
            f"The combination of branch_net {type(self.branch_net)} and trunk_net "
            + f"{type(self.trunk_net)} does not support FuncArch."
        )
        branch_x = self.slice_input(x, self.branch_slice_index, dim=-1)
        trunk_x = self.slice_input(x, self.trunk_slice_index, dim=-1)

        branch_output = self.branch_net._tensor_forward(branch_x)
        trunk_output = self.trunk_net._tensor_forward(trunk_x)

        # Convert ouputs into 1D feature vectors
        if torch._C._functorch.is_gradtrackingtensor(
            trunk_output
        ) or torch._C._functorch.is_batchedtensor(trunk_output):
            # batched tensor does not have the original shape
            branch_output = branch_output.view(-1)
            trunk_output = trunk_output.view(-1)
        else:
            branch_output = branch_output.view(branch_output.shape[0], -1)
            trunk_output = trunk_output.view(trunk_output.shape[0], -1)

        assert branch_output.size(-1) == self.branch_dim, (
            f"Invalid feature dimension from branch net, expected {self.branch_dim} but found {branch_output.size(-1)}"
        )
        assert trunk_output.size(-1) == self.trunk_dim, (
            f"Invalid feature dimension from trunk net, expected {self.trunk_dim} but found {trunk_output.size(-1)}"
        )

        # Send through final linear layers
        branch_output = self.branch_linear(branch_output)
        trunk_output = self.trunk_linear(trunk_output)
        y = self.output_linear(branch_output * trunk_output)

        y = self.process_output(y, self.output_scales_tensor)
        return y

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
        # Forward pass of branch and trunk net
        branch_output = self.branch_net(in_vars)
        trunk_output = self.trunk_net(in_vars)

        branch_output = branch_output["branch"]
        trunk_output = trunk_output["trunk"]
        # Convert ouputs into 1D feature vectors
        branch_output = branch_output.view(branch_output.shape[0], -1)
        trunk_output = trunk_output.view(trunk_output.shape[0], -1)

        assert branch_output.size(-1) == self.branch_dim, (
            f"Invalid feature dimension from branch net, expected {self.branch_dim} but found {branch_output.size(-1)}"
        )
        assert trunk_output.size(-1) == self.trunk_dim, (
            f"Invalid feature dimension from trunk net, expected {self.trunk_dim} but found {trunk_output.size(-1)}"
        )

        # Send through final linear layers
        branch_output = self.branch_linear(branch_output)
        trunk_output = self.trunk_linear(trunk_output)

        out = self.output_linear(branch_output * trunk_output)

        return self.prepare_output(
            out, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
