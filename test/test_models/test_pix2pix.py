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

import itertools
import torch

from physicsnemo.sym.key import Key
from physicsnemo.sym.models.pix2pix import Pix2PixArch


def test_pix2pix():
    # check 1D
    model = Pix2PixArch(
        input_keys=[Key("x", size=4)],
        output_keys=[Key("y", size=4), Key("z", size=2)],
        dimension=1,
        scaling_factor=2,
    )
    bsize = 4
    x = {"x": torch.randn((bsize, 4, 32))}
    outvar = model.forward(x)
    # Check output size
    assert outvar["y"].shape == (bsize, 4, 64)
    assert outvar["z"].shape == (bsize, 2, 64)

    # check 2D
    model = Pix2PixArch(
        input_keys=[Key("x", size=2)],
        output_keys=[Key("y", size=2), Key("z", size=1)],
        dimension=2,
        n_downsampling=1,
        scaling_factor=4,
    )
    bsize = 4
    x = {"x": torch.randn((bsize, 2, 28, 28))}
    outvar = model.forward(x)
    # Check output size
    assert outvar["y"].shape == (bsize, 2, 112, 112)
    assert outvar["z"].shape == (bsize, 1, 112, 112)

    # check 3D
    model = Pix2PixArch(
        input_keys=[Key("x", size=1)],
        output_keys=[Key("y", size=2), Key("z", size=2)],
        dimension=3,
    )
    bsize = 4
    x = {"x": torch.randn((bsize, 1, 64, 64, 64))}
    outvar = model.forward(x)
    # Check output size
    assert outvar["y"].shape == (bsize, 2, 64, 64, 64)
    assert outvar["z"].shape == (bsize, 2, 64, 64, 64)
