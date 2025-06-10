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

from physicsnemo.sym.models.highway_fourier_net import HighwayFourierNetArch
from pathlib import Path
import torch
import numpy as np
from physicsnemo.sym.key import Key
import pytest
from .model_test_utils import validate_func_arch_net

dir_path = Path(__file__).parent


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases"), ("weight_g", "alphas")]
    for layer_name in ("fc_t", "fc_v"):
        for pt_name, tf_name in names:
            _dict[layer_name + ".linear." + pt_name] = layer_name + "/" + tf_name + ":0"
    for i in range(nr_layers):
        for pt_name, tf_name in names:
            _dict["fc_layers." + str(i) + ".linear." + pt_name] = (
                "fc" + str(i) + "/" + tf_name + ":0"
            )
    for pt_name, tf_name in names[:2]:
        _dict["final_layer.linear." + pt_name] = "fc_final/" + tf_name + ":0"
    return _dict


def test_highway_fourier_net():
    filename = dir_path / "data/test_highway_fourier.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    frequencies = test_data["frequencies"]
    frequencies_params = test_data["frequencies_params"]
    # create graph
    arch = HighwayFourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        frequencies=("axis,diagonal", frequencies),
        frequencies_params=("axis,diagonal", frequencies_params),
        layer_size=params["layer_size"],
        nr_layers=params["nr_layers"],
    )
    name_dict = make_dict(params["nr_layers"])
    for _name, _tensor in arch.named_parameters():
        if _tensor.requires_grad:
            _tensor.data = torch.from_numpy(Wbs[name_dict[_name]].T)

    arch.fourier_layer_xyzt.frequencies = torch.from_numpy(
        Wbs["fourier_layer_xyzt:0"].T
    )
    data_out2 = arch(
        {"x": torch.from_numpy(data_in[:, 0:1]), "y": torch.from_numpy(data_in[:, 1:2])}
    )
    data_out2 = data_out2["u"].detach().numpy()
    # load outputs
    data_out1 = test_data["data_out"]
    # verify
    assert np.allclose(data_out1, data_out2, atol=1e-3, rtol=1e-3), "Test failed!"
    print("Success!")


@pytest.mark.parametrize(
    "input_keys", [[Key("x"), Key("y")], [Key("x"), Key("y", scale=(1.0, 2.0))]]
)
@pytest.mark.parametrize("validate_with_dict_forward", [True, False])
def test_func_arch_highway_fourier(input_keys, validate_with_dict_forward):
    deriv_keys = [
        Key.from_str("u__x"),
        Key.from_str("u__x__x"),
        Key.from_str("v__y"),
        Key.from_str("v__y__y"),
    ]
    ref_net = HighwayFourierNetArch(
        input_keys=input_keys,
        output_keys=[Key("u"), Key("v")],
    )
    validate_func_arch_net(ref_net, deriv_keys, validate_with_dict_forward)


test_highway_fourier_net()
