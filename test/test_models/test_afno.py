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
from physicsnemo.sym.models.afno import AFNOArch

########################
# load & verify
########################
def test_afno():
    # Construct FNO model
    model = AFNOArch(
        input_keys=[Key("x", size=2)],
        output_keys=[Key("u", size=2), Key("p")],
        img_shape=(240, 240),
        patch_size=16,
        embed_dim=256,
        depth=4,
        num_blocks=8,
    )
    # Testing JIT
    node = model.make_node(name="AFNO", jit=True)

    bsize = 5
    invar = {
        "x": torch.randn(bsize, 2, 240, 240),
    }
    # Model forward
    outvar = node.evaluate(invar)
    # Check output size
    assert outvar["u"].shape == (bsize, 2, 240, 240)
    assert outvar["p"].shape == (bsize, 1, 240, 240)


test_afno()
