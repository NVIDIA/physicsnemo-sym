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

import warnings
import torch
import pytest
from packaging import version
from physicsnemo.sym.manager import JitManager
from physicsnemo.sym.utils.benchmark import profile, timeit
from physicsnemo.sym.models.activation import Activation, get_activation_fn

# Allow fusing single node, and prevent tiny autodiff graph are inlined/reverted.
# These flags are automatically set when specifying jit_manager.enabled is True.
# User needs to set these flags manually if they would like to fuse activation
# function for standalone code.
#
# torch._C._jit_set_nvfuser_single_node_mode(True)
# torch._C._debug_set_autodiff_subgraph_inlining(False)

skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="There is no GPU to run this test"
)


def test_activation_jit():
    jit_manager = JitManager()
    jit_manager.enabled = True
    jit_manager.arch_mode = "only_activation"

    for act in Activation:
        act_scripted = get_activation_fn(act)
        assert isinstance(
            act_scripted, (torch.jit.ScriptFunction, torch.jit.ScriptModule)
        )

    def sin(x):
        return torch.sin(x)

    sin_scripted = get_activation_fn(sin)
    assert isinstance(sin_scripted, torch.jit.ScriptFunction)


@skip_if_no_gpu
def test_activation_fused_silu():
    """
    Make sure SiLU derivative kernels are fused when jit_manager.arch_mode == "only_activation".
    We need to rely on the fused SiLU derivative kernels for AMP, because the unfused path
    may have intermediate results that overflow the FP16 dynamic range.
    """
    jit_manager = JitManager()
    jit_manager.enabled = True
    jit_manager.arch_mode = "only_activation"
    jit_manager.use_nvfuser = True

    silu_scripted = get_activation_fn(Activation.SILU)
    assert isinstance(silu_scripted, torch.jit.ScriptFunction)

    device = "cuda"
    batch_size = 10000
    x = torch.rand([batch_size, 512], device=device, requires_grad=True)
    I_N = torch.ones_like(x)

    def run(func, order, *args):
        torch.cuda.nvtx.range_push("forward")
        y = func(*args)
        torch.cuda.nvtx.range_pop()

        if order >= 1:
            torch.cuda.nvtx.range_push("1st order")
            (y__x,) = torch.autograd.grad(y, [x], I_N, create_graph=True)
            torch.cuda.nvtx.range_pop()

        if order >= 2:
            torch.cuda.nvtx.range_push("2nd order")
            (y__x__x,) = torch.autograd.grad(y__x, [x], I_N, create_graph=True)
            torch.cuda.nvtx.range_pop()

        if order >= 3:
            torch.cuda.nvtx.range_push("3rd order")
            (y__x__x__x,) = torch.autograd.grad(y__x__x, [x], I_N, create_graph=True)
            torch.cuda.nvtx.range_pop()

    def cleanup_events(event_keys):
        keys = ["cuLaunchKernel", "cudaLaunchKernel", "cudaDeviceSynchronize"]
        for evt in keys:
            if evt in event_keys:
                event_keys.remove(evt)
        return event_keys

    # benchmark
    silu = torch.nn.functional.silu
    silu_1st = timeit(run, silu, 1, x, label="silu_1st", verbose=True)
    silu_scripted_1st = timeit(
        run, silu_scripted, 1, x, label="silu_scripted_1st", verbose=True
    )
    silu_2nd = timeit(run, silu, 2, x, label="silu_2nd", verbose=True)
    silu_scripted_2nd = timeit(
        run, silu_scripted, 2, x, label="silu_scripted_2nd", verbose=True
    )
    silu_3rd = timeit(run, silu, 3, x, label="silu_3rd", verbose=True)
    silu_scripted_3rd = timeit(
        run, silu_scripted, 3, x, label="silu_scripted_3rd", verbose=True
    )

    assert silu_scripted_1st < silu_1st, "silu_scripted_1st is slower than silu_1st"
    assert silu_scripted_2nd < silu_2nd, "silu_scripted_2nd is slower than silu_2nd"
    assert silu_scripted_3rd < silu_3rd, "silu_scripted_3rd is slower than silu_3rd"


if __name__ == "__main__":
    test_activation_jit()
    test_activation_fused_silu()
