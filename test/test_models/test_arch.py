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
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch

# ensure torch.rand() is deterministic
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_slice_input():
    # prepare inputs
    x = torch.rand([100, 1])
    y = torch.rand([100, 2])
    z = torch.rand([100, 1])
    input_variables = {"x": x, "y": y, "z": z}
    input_keys = [Key("x", 1), Key("y", 2), Key("z", 1)]
    input_key_dict = {str(var): var.size for var in input_keys}
    ipt = Arch.prepare_input(input_variables, input_key_dict.keys(), {}, dim=-1)

    slice_keys = ["x", "z"]
    # expected result
    expected = Arch.prepare_input(input_variables, slice_keys, {}, dim=-1)
    # sliced result
    slice_index = Arch.prepare_slice_index(input_key_dict, slice_keys)
    result = Arch.slice_input(ipt, slice_index, dim=-1)
    assert torch.allclose(result, expected)

    slice_keys = ["y", "z"]
    # expected result
    expected = Arch.prepare_input(input_variables, slice_keys, {}, dim=-1)
    # sliced result
    slice_index = Arch.prepare_slice_index(input_key_dict, slice_keys)
    result = Arch.slice_input(ipt, slice_index, dim=-1)

    assert torch.allclose(result, expected)


def validate_process_input_output(input_variables, arch):
    # -------------------------- input --------------------------
    # expected
    expected = Arch.prepare_input(
        input_variables,
        arch.input_key_dict.keys(),
        {},
        dim=-1,
        input_scales=arch.input_scales,
        periodicity=arch.periodicity,
    )
    # result
    result = Arch.concat_input(input_variables, arch.input_key_dict.keys(), {}, dim=-1)
    result = Arch.process_input(
        result, arch.input_scales_tensor, arch.periodicity, arch.input_key_dict, dim=-1
    )
    # check result
    assert torch.allclose(expected, result)

    # -------------------------- output --------------------------
    batch_size, output_size = expected.shape[0], sum(arch.output_key_dict.values())
    y = torch.rand([batch_size, output_size])

    # expected
    expected = Arch.prepare_output(
        y,
        arch.output_key_dict,
        dim=-1,
        output_scales=arch.output_scales,
    )
    # result
    result = Arch.process_output(y, output_scales_tensor=arch.output_scales_tensor)
    result = Arch.split_output(result, output_dict=arch.output_key_dict, dim=-1)
    # check result
    assert expected.keys() == result.keys()
    for key in expected:
        assert torch.allclose(expected[key], result[key])


def test_process_input_output():
    # prepare inputs
    x = torch.ones([100, 1])
    y = torch.ones([100, 2])
    z = torch.ones([100, 1])
    input_variables = {"x": x, "y": y, "z": z}

    # no input scales
    input_keys = [Key("x", 1), Key("y", 2), Key("z", 1)]
    output_keys = [Key("u", 1), Key("v", 1)]

    arch = Arch(input_keys, output_keys)
    validate_process_input_output(input_variables, arch)
    assert arch.input_scales_tensor is None
    assert arch.output_scales_tensor is None

    # input scales
    input_keys = [
        Key("x", 1, scale=(0.0, 1.0)),
        Key("y", 2, scale=(0.0, 2.0)),
        Key("z", 1, scale=(0.0, 3.0)),
    ]
    output_keys = [Key("u", 1, scale=(1.0, 2.0)), Key("v", 1)]

    arch = Arch(input_keys, output_keys)
    validate_process_input_output(input_variables, arch)
    assert torch.allclose(
        arch.input_scales_tensor,
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 2.0, 3.0]]),
    )
    assert torch.allclose(
        arch.output_scales_tensor, torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    )

    # input scales and also periodicity
    arch = Arch(
        input_keys,
        output_keys,
        periodicity={"x": (0.0, 2 * torch.pi), "y": (torch.pi, 4 * torch.pi)},
    )
    validate_process_input_output(input_variables, arch)


if __name__ == "__main__":
    test_slice_input()
    test_process_input_output()
