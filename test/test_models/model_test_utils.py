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
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import FuncArch, Arch
from typing import List

# ensure torch.rand() is deterministic
_ = torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# disable tf32 for accuracy
torch.backends.cuda.matmul.allow_tf32 = False


def validate_func_arch_net(
    ref_net: Arch,
    deriv_keys: List[Key],
    validate_with_dict_forward: bool,
):
    """
    Using double precision for testing.
    """
    if validate_with_dict_forward:
        ref_net.forward = ref_net._dict_forward
    ref_graph = (
        Graph(
            [
                ref_net.make_node("ref_net", jit=False),
            ],
            ref_net.input_keys,
            deriv_keys + ref_net.output_keys,
            func_arch=False,
        )
        .double()
        .to(device)
    )

    ft_net = FuncArch(arch=ref_net, deriv_keys=deriv_keys).double().to(device)

    # check result
    batch_size = 20
    in_vars = {
        v.name: torch.rand(
            [batch_size, v.size], device=device, dtype=torch.double
        ).requires_grad_()
        for v in ref_net.input_keys
    }
    ft_out = ft_net(in_vars)
    ref_out = ref_graph(in_vars)
    for k in ref_out.keys():
        assert torch.allclose(ref_out[k], ft_out[k])

    return ft_net
