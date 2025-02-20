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

from physicsnemo.sym.models.multiplicative_filter_net import (
    MultiplicativeFilterNetArch,
    FilterType,
)
import torch
import numpy as np
from pathlib import Path
from physicsnemo.sym.key import Key
import pytest
from .model_test_utils import validate_func_arch_net

dir_path = Path(__file__).parent


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases"), ("weight_g", "alphas")]
    tri_names = ("frequency", "phase")
    for tri_name in tri_names:
        _dict["first_filter." + tri_name] = "fourier_filter_first_" + tri_name + ":0"
    for i in range(nr_layers):
        for pt_name, tf_name in names:
            _dict["fc_layers." + str(i) + ".linear." + pt_name] = (
                "fc_" + str(i) + "/" + tf_name + ":0"
            )
        for tri_name in tri_names:
            _dict["filters." + str(i) + "." + tri_name] = (
                "fourier_filter_layer" + str(i) + "_" + tri_name + ":0"
            )
    for pt_name, tf_name in names[:2]:
        _dict["final_layer.linear." + pt_name] = "fc_final/" + tf_name + ":0"
    return _dict


def test_multiplicative_filter():
    filename = dir_path / "data/test_multiplicative_filter.npz"
    test_data = np.load(filename, allow_pickle=True)
    data_in = test_data["data_in"]
    Wbs = test_data["Wbs"][()]
    params = test_data["params"][()]
    # create graph
    arch = MultiplicativeFilterNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        layer_size=params["layer_size"],
        nr_layers=params["nr_layers"],
    )
    name_dict = make_dict(params["nr_layers"])
    for _name, _tensor in arch.named_parameters():
        if _tensor.requires_grad:
            if "filter" in _name:
                _tensor.data = torch.from_numpy(Wbs[name_dict[_name]])
            else:
                _tensor.data = torch.from_numpy(Wbs[name_dict[_name]].T)

    data_out2 = arch(
        {"x": torch.from_numpy(data_in[:, 0:1]), "y": torch.from_numpy(data_in[:, 1:2])}
    )
    data_out2 = data_out2["u"].detach().numpy()
    # load outputs
    data_out1 = test_data["data_out"]
    # verify
    assert np.allclose(data_out1, data_out2, atol=1e-4), "Test failed!"
    print("Success!")


@pytest.mark.parametrize(
    "input_keys", [[Key("x"), Key("y")], [Key("x"), Key("y", scale=(1.0, 2.0))]]
)
@pytest.mark.parametrize("validate_with_dict_forward", [True, False])
@pytest.mark.parametrize("normalization", [None, {"x": (-2.5, 2.5), "y": (-2.5, 2.5)}])
@pytest.mark.parametrize("filter_type", [FilterType.FOURIER, FilterType.GABOR])
def test_func_arch_multiplicative_filter(
    input_keys, validate_with_dict_forward, normalization, filter_type
):
    deriv_keys = [
        Key.from_str("u__x"),
        Key.from_str("u__x__x"),
        Key.from_str("v__y"),
        Key.from_str("v__y__y"),
    ]
    ref_net = MultiplicativeFilterNetArch(
        input_keys=input_keys,
        output_keys=[Key("u"), Key("v")],
        normalization=normalization,
        filter_type=filter_type,
    )
    validate_func_arch_net(ref_net, deriv_keys, validate_with_dict_forward)


if __name__ == "__main__":
    test_multiplicative_filter()
