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

from physicsnemo.sym.geometry.primitives_2d import Rectangle, Circle
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter

if __name__ == "__main__":
    # make plate with parameterized hole
    # make parameterized primitives
    plate = Rectangle(point_1=(-1, -1), point_2=(1, 1))
    y_pos = Parameter("y_pos")
    parameterization = Parameterization({y_pos: (-1, 1)})
    circle = Circle(center=(0, y_pos), radius=0.3, parameterization=parameterization)
    geo = plate - circle

    # sample geometry over entire parameter range
    s = geo.sample_boundary(nr_points=100000)
    var_to_polyvtk(s, "parameterized_boundary")
    s = geo.sample_interior(nr_points=100000)
    var_to_polyvtk(s, "parameterized_interior")

    # sample specific parameter
    s = geo.sample_boundary(
        nr_points=100000, parameterization=Parameterization({y_pos: 0})
    )
    var_to_polyvtk(s, "y_pos_zero_boundary")
    s = geo.sample_interior(
        nr_points=100000, parameterization=Parameterization({y_pos: 0})
    )
    var_to_polyvtk(s, "y_pos_zero_interior")
