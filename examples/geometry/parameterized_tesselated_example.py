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

import glob
import numpy as np

from physicsnemo.sym.geometry.tessellation import Tessellation
from physicsnemo.sym.geometry.discrete_geometry import DiscreteGeometry
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter

if __name__ == "__main__":
    # make geometry for each bracket
    bracket_files = glob.glob("./bracket_stl/*.stl")
    bracket_files.sort()
    brackets = []
    radius = []
    width = []
    for f in bracket_files:
        # get param values
        radius.append(float(f.split("_")[3]))
        width.append(float(f.split("_")[5][:-4]))

        # make geometry
        brackets.append(Tessellation.from_stl(f))

    # make discretely parameterized geometry
    parameterization = Parameterization(
        {
            Parameter("radius"): np.array(radius)[:, None],
            Parameter("width"): np.array(width)[:, None],
        }
    )
    geo = DiscreteGeometry(brackets, parameterization)

    # sample geometry over entire parameter range
    s = geo.sample_boundary(nr_points=1000000)
    var_to_polyvtk(s, "parameterized_bracket_boundary")
    s = geo.sample_interior(nr_points=1000000)
    var_to_polyvtk(s, "parameterized_bracket_interior")
