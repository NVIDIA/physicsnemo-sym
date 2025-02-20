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
from physicsnemo.sym.geometry.primitives_3d import (
    Box,
    Sphere,
    Cylinder,
    VectorizedBoxes,
)
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from stl import mesh as np_mesh
import time


def speed_check(geo, nr_points):
    tic = time.time()
    s = geo.sample_boundary(nr_points=nr_points)
    surface_sample_time = time.time() - tic
    var_to_polyvtk(s, "boundary")
    tic = time.time()
    s = geo.sample_interior(nr_points=nr_points, compute_sdf_derivatives=False)
    volume_sample_time = time.time() - tic
    var_to_polyvtk(s, "interior")
    print(
        "Surface sample (seconds per million point): {:.3e}".format(
            1000000 * surface_sample_time / nr_points
        )
    )
    print(
        "Volume sample (seconds per million point): {:.3e}".format(
            1000000 * volume_sample_time / nr_points
        )
    )


if __name__ == "__main__":
    # number of points to sample for speed test
    nr_points = 1000000

    # tesselated geometry speed test
    mesh = np_mesh.Mesh.from_file("./stl_files/tessellated_example.stl")
    geo = Tessellation(mesh)
    print("Tesselated Speed Test")
    print("Number of triangles: {:d}".format(mesh.vectors.shape[0]))
    speed_check(geo, nr_points)

    # primitives speed test
    box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
    sphere = Sphere(center=(0, 0, 0), radius=1.2)
    cylinder_1 = Cylinder(center=(0, 0, 0), radius=0.5, height=2)
    cylinder_2 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="x")
    cylinder_3 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="y")
    all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
    box_minus_sphere = box & sphere
    geo = box_minus_sphere - all_cylinders
    print("CSG Speed Test")
    speed_check(geo, nr_points)

    # make boxes for many body check
    nr_boxes = [10, 100, 500]
    boxes = []
    for i in range(max(nr_boxes)):
        x_pos = (np.sqrt(5.0) * i % 0.8) + 0.1
        y_pos = (np.sqrt(3.0) * i % 0.8) + 0.1
        z_pos = (np.sqrt(7.0) * i % 0.8) + 0.1
        boxes.append(
            np.array(
                [[x_pos, x_pos + 0.05], [y_pos, y_pos + 0.05], [z_pos, z_pos + 0.05]]
            )
        )
    boxes = np.array(boxes)

    for nr_b in nr_boxes:
        # csg many object speed test
        geo = Box((0, 0, 0), (1, 1, 1))
        for i in range(nr_b):
            geo = geo - Box(tuple(boxes[i, :, 0]), tuple(boxes[i, :, 1]))

        print("CSG Many Box Speed Test, Number of Boxes " + str(nr_b))
        speed_check(geo, nr_points)
