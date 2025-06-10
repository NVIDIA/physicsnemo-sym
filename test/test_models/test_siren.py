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

from physicsnemo.sym.models.siren import SirenArch
import torch
import numpy as np
from pathlib import Path
from physicsnemo.sym.key import Key
import pytest
from .model_test_utils import validate_func_arch_net

dir_path = Path(__file__).parent


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases")]
    for i in range(nr_layers + 1):
        for pt_name, tf_name in names:
            _dict["layers." + str(i) + ".linear." + pt_name] = (
                "fc" + str(i) + "/" + tf_name + ":0"
            )
    return _dict


def test_siren():
    filename = dir_path / "data/test_siren.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    # create graph
    arch = SirenArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        layer_size=params["layer_size"],
        nr_layers=params["nr_layers"],
        first_omega=params["first_omega"],
        omega=params["omega"],
    )
    name_dict = make_dict(params["nr_layers"])
    for _name, _tensor in arch.named_parameters():
        if _tensor.requires_grad:
            _tensor.data = torch.from_numpy(Wbs[name_dict[_name]].T)

    data_out2 = arch(
        {"x": torch.from_numpy(data_in[:, 0:1]), "y": torch.from_numpy(data_in[:, 1:2])}
    )
    data_out2 = data_out2["u"].detach().numpy()
    # load outputs
    data_out1 = test_data["data_out"]
    # verify
    assert np.allclose(data_out1, data_out2, rtol=1e-3), "Test failed!"
    print("Success!")


def validate_tensor_normalize(input_variables, arch):
    # expected
    expected = arch._normalize(input_variables, arch.normalization)
    expected = SirenArch.concat_input(expected, arch.input_key_dict.keys(), dim=-1)
    # result
    result = SirenArch.concat_input(input_variables, arch.input_key_dict.keys(), dim=-1)
    result = SirenArch._tensor_normalize(result, arch.normalization_tensor)
    # check result
    assert torch.allclose(expected, result)


def test_tensor_normalize():
    # prepare inputs
    x = torch.ones([100, 1])
    y = torch.ones([100, 2])
    z = torch.ones([100, 1])
    input_variables = {"x": x, "y": y, "z": z}
    input_keys = [Key("x", 1), Key("y", 2), Key("z", 1)]
    output_keys = [Key("u", 1), Key("v", 1)]

    # normalization is None
    normalization = None
    arch = SirenArch(input_keys, output_keys, normalization=normalization)
    validate_tensor_normalize(input_variables, arch)
    assert arch.normalization_tensor is None

    # normalization for part of the inputs, z will use no_op_norm
    normalization = {"x": (-2.5, 2.5), "y": (-2.5, 2.5)}
    arch = SirenArch(input_keys, output_keys, normalization=normalization)
    validate_tensor_normalize(input_variables, arch)
    assert torch.allclose(
        arch.normalization_tensor,
        torch.tensor([[-2.5, -2.5, -2.5, -1.0], [2.5, 2.5, 2.5, 1.0]]),
    )

    # normalization for all inputs
    normalization = {"x": (-2.5, 2.5), "y": (-2.5, 2.5), "z": (-3.5, 3.5)}
    arch = SirenArch(input_keys, output_keys, normalization=normalization)
    validate_tensor_normalize(input_variables, arch)
    assert torch.allclose(
        arch.normalization_tensor,
        torch.tensor([[-2.5, -2.5, -2.5, -3.5], [2.5, 2.5, 2.5, 3.5]]),
    )


@pytest.mark.parametrize(
    "input_keys", [[Key("x"), Key("y")], [Key("x"), Key("y", scale=(1.0, 2.0))]]
)
@pytest.mark.parametrize("validate_with_dict_forward", [True, False])
@pytest.mark.parametrize("normalization", [None, {"x": (-2.5, 2.5), "y": (-2.5, 2.5)}])
def test_func_arch_siren(input_keys, validate_with_dict_forward, normalization):
    deriv_keys = [
        Key.from_str("u__x"),
        Key.from_str("u__x__x"),
        Key.from_str("v__y"),
        Key.from_str("v__y__y"),
    ]
    ref_net = SirenArch(
        input_keys=input_keys,
        output_keys=[Key("u"), Key("v")],
        normalization=normalization,
    )
    validate_func_arch_net(ref_net, deriv_keys, validate_with_dict_forward)


if __name__ == "__main__":
    test_siren()
    test_tensor_normalize()
