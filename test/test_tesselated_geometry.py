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

from sympy import Symbol
import numpy as np
from pathlib import Path

from physicsnemo.sym.geometry.tessellation import Tessellation
from physicsnemo.sym.geometry import Parameterization

dir_path = Path(__file__).parent


def test_tesselated_geometry():
    # read in cube file
    cube = Tessellation.from_stl(dir_path / "stls/cube.stl")

    # sample boundary
    boundary = cube.sample_boundary(
        1000, parameterization=Parameterization({Symbol("fake_param"): 1})
    )

    # sample interior
    interior = cube.sample_interior(
        1000, parameterization=Parameterization({Symbol("fake_param"): 1})
    )

    # check if surface area is right for boundary
    assert np.isclose(np.sum(boundary["area"]), 6.0)

    # check if volume is right for interior
    assert np.isclose(np.sum(interior["area"]), 1.0)
