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

import numpy as np
from physicsnemo.sym.geometry.tessellation import Tessellation
from physicsnemo.sym.geometry.primitives_3d import Plane
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk

if __name__ == "__main__":
    # number of points to sample
    nr_points = 100000

    # make tesselated geometry from stl file
    geo = Tessellation.from_stl("./stl_files/tessellated_example.stl")

    # tesselated geometries can be combined with primitives
    cut_plane = Plane((0, -1, -1), (0, 1, 1))
    geo = geo & cut_plane

    # sample geometry for plotting in Paraview
    s = geo.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "tessellated_boundary")
    print("Repeated Surface Area: {:.3f}".format(np.sum(s["area"])))
    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "tessellated_interior")
    print("Repeated Volume: {:.3f}".format(np.sum(s["area"])))
