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
from physicsnemo.sym.geometry.primitives_3d import Box, Sphere, Cylinder
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk

if __name__ == "__main__":
    # number of points to sample
    nr_points = 100000

    # make standard constructive solid geometry example
    # make primitives
    box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
    sphere = Sphere(center=(0, 0, 0), radius=1.2)
    cylinder_1 = Cylinder(center=(0, 0, 0), radius=0.5, height=2)
    cylinder_2 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="x")
    cylinder_3 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="y")

    # combine with boolean operations
    all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
    box_minus_sphere = box & sphere
    geo = box_minus_sphere - all_cylinders

    # sample geometry for plotting in Paraview
    s = geo.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "boundary")
    print("Surface Area: {:.3f}".format(np.sum(s["area"])))
    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "interior")
    print("Volume: {:.3f}".format(np.sum(s["area"])))

    # apply transformations
    geo = geo.scale(0.5)
    geo = geo.rotate(angle=np.pi / 4, axis="z")
    geo = geo.rotate(angle=np.pi / 4, axis="y")
    geo = geo.repeat(spacing=4.0, repeat_lower=(-1, -1, -1), repeat_higher=(1, 1, 1))

    # sample geometry for plotting in Paraview
    s = geo.sample_boundary(nr_points=nr_points)
    var_to_polyvtk(s, "repeated_boundary")
    print("Repeated Surface Area: {:.3f}".format(np.sum(s["area"])))
    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=True)
    var_to_polyvtk(s, "repeated_interior")
    print("Repeated Volume: {:.3f}".format(np.sum(s["area"])))
