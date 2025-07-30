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
import numpy as np
from physicsnemo.sym.key import Key

import pytest
from test.pytest_utils import import_or_fail


layer_size_params = [
    pytest.param(128, id="fused_128"),
    pytest.param(256, id="cutlass_256"),
]


def make_dict(nr_layers):
    _dict = dict()
    names = [("weight", "weights"), ("bias", "biases"), ("weight_g", "alphas")]
    for i in range(nr_layers):
        for pt_name, tf_name in names:
            _dict["_impl.layers." + str(i) + ".linear." + pt_name] = (
                "fc" + str(i) + "/" + tf_name + ":0"
            )
    for pt_name, tf_name in names[:2]:
        _dict["_impl.final_layer.linear." + pt_name] = "fc_final/" + tf_name + ":0"
    return _dict


@import_or_fail("tinycudann")
@pytest.mark.parametrize("layer_size", layer_size_params)
def test_fully_fused_mlp(layer_size, pytestconfig):
    from physicsnemo.sym.models.fused_mlp import (
        FusedMLPArch,
    )

    batch_size = 1024

    data_in = np.random.random((batch_size, 2))

    fully_fused = False
    if layer_size in set([16, 32, 64, 128]):
        fully_fused = True

    # create graph
    arch = FusedMLPArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        layer_size=layer_size,
        nr_layers=6,
        fully_fused=fully_fused,
    )

    data_out2 = arch(
        {
            "x": torch.from_numpy(data_in[:, 0:1]).cuda(),
            "y": torch.from_numpy(data_in[:, 1:2]).cuda(),
        }
    )
    data_out2 = data_out2["u"].cpu().detach().numpy()

    # TODO: Figure out arch.params slicing to initialize pytorch model
    #       and compare TCNN output to that
    # assert np.allclose(data_out1, data_out2, rtol=1e-3), "Test failed!"


@import_or_fail("tinycudann")
@pytest.mark.parametrize("layer_size", layer_size_params)
def test_fused_fourier_net(layer_size, pytestconfig):
    from physicsnemo.sym.models.fused_mlp import (
        FusedFourierNetArch,
    )

    batch_size = 1024

    data_in = np.random.random((batch_size, 2))

    fully_fused = False
    if layer_size in set([16, 32, 64, 128]):
        fully_fused = True

    # create graph
    arch = FusedFourierNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        layer_size=layer_size,
        nr_layers=6,
        fully_fused=fully_fused,
        n_frequencies=12,
    )

    data_out2 = arch(
        {
            "x": torch.from_numpy(data_in[:, 0:1]).cuda(),
            "y": torch.from_numpy(data_in[:, 1:2]).cuda(),
        }
    )
    data_out2 = data_out2["u"].cpu().detach().numpy()

    # TODO: Figure out arch.params slicing to initialize pytorch model
    #       and compare TCNN output to that
    # assert np.allclose(data_out1, data_out2, rtol=1e-3), "Test failed!"


@import_or_fail("tinycudann")
@pytest.mark.parametrize("layer_size", layer_size_params)
def test_fused_grid_encoding_net(layer_size, pytestconfig):
    from physicsnemo.sym.models.fused_mlp import (
        FusedGridEncodingNetArch,
    )

    batch_size = 1024

    data_in = np.random.random((batch_size, 2))

    fully_fused = False
    if layer_size in set([16, 32, 64, 128]):
        fully_fused = True

    # create graph
    arch = FusedGridEncodingNetArch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        layer_size=layer_size,
        nr_layers=6,
        fully_fused=fully_fused,
        indexing="Hash",
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=2.0,
        interpolation="Smoothstep",
    )

    data_out2 = arch(
        {
            "x": torch.from_numpy(data_in[:, 0:1]).cuda(),
            "y": torch.from_numpy(data_in[:, 1:2]).cuda(),
        }
    )
    data_out2 = data_out2["u"].cpu().detach().numpy()

    # TODO: Figure out arch.params slicing to initialize pytorch model
    #       and compare TCNN output to that
    # assert np.allclose(data_out1, data_out2, rtol=1e-3), "Test failed!"


if __name__ == "__main__":
    # Fused MLP tests
    test_fully_fused_mlp(128)  # Fully Fused MLP
    test_fully_fused_mlp(256)  # Cutlass MLP

    # Fused Fourier Net tests
    test_fused_fourier_net(128)  # Fully Fused MLP
    test_fused_fourier_net(256)  # Cutlass MLP

    # Fused Grid encoding tests
    test_fused_grid_encoding_net(128)  # Fully Fused MLP
    test_fused_grid_encoding_net(256)  # Cutlass MLP
