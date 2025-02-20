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

from physicsnemo.sym.models.radial_basis import RadialBasisArch
import torch
import numpy as np
from pathlib import Path
from physicsnemo.sym.key import Key
import pytest
from .model_test_utils import validate_func_arch_net

dir_path = Path(__file__).parent


def make_dict():
    _dict = dict()
    _dict["fc_layer.linear.weight"] = "fc_final/weights:0"
    _dict["fc_layer.linear.bias"] = "fc_final/biases:0"
    return _dict


def test_radial_basis():
    filename = dir_path / "data/test_radial_basis.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    # create graph
    arch = RadialBasisArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        nr_centers=params["nr_centers"],
        sigma=params["sigma"],
    )
    name_dict = make_dict()
    center_data = np.hstack(
        (Wbs["c_x:0"].reshape((-1, 1)), Wbs["c_y:0"].reshape((-1, 1)))
    )
    for _name, _tensor in arch.named_parameters():
        if _name == "centers":
            _tensor.data = torch.from_numpy(center_data)
        else:
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


@pytest.mark.parametrize(
    "input_keys", [[Key("x"), Key("y")], [Key("x"), Key("y", scale=(1.0, 2.0))]]
)
@pytest.mark.parametrize("validate_with_dict_forward", [True, False])
def test_func_arch_radial_basis(input_keys, validate_with_dict_forward):
    deriv_keys = [
        Key.from_str("u__x"),
        Key.from_str("u__x__x"),
        Key.from_str("v__y"),
        Key.from_str("v__y__y"),
    ]
    ref_net = RadialBasisArch(
        input_keys=input_keys,
        output_keys=[Key("u"), Key("v")],
        bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    validate_func_arch_net(ref_net, deriv_keys, validate_with_dict_forward)


if __name__ == "__main__":
    test_radial_basis()
