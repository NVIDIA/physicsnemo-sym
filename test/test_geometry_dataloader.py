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
import copy
import pytest
from pathlib import Path
from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe
from physicsnemo.sym.geometry.primitives_3d import Box, Sphere, Cylinder
from physicsnemo.sym.geometry.tessellation import Tessellation
import torch

dir_path = Path(__file__).parent


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_geom_datapipe_csg(device):

    geoms = []
    for i in range(10):
        box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
        sphere = Sphere(center=(0, 0, 0), radius=1 + i / 10)
        cylinder_1 = Cylinder(center=(0, 0, 0), radius=0.5, height=2)
        cylinder_2 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="x")
        cylinder_3 = cylinder_1.rotate(angle=float(np.pi / 2.0), axis="y")

        # combine with boolean operations
        all_cylinders = cylinder_1 + cylinder_2 + cylinder_3
        box_minus_sphere = box & sphere
        geo = box_minus_sphere - all_cylinders
        geoms.append(geo)

    datapipe = GeometryDatapipe(
        geom_objects=geoms,
        sample_type="surface",
        num_points=100,
        batch_size=2,
        num_workers=1,
        device=device,
    )

    for data in datapipe:
        assert data[0]["normal_x"].shape == (2, 100, 1)
        assert all(
            elem in list(data[0].keys())
            for elem in ["x", "y", "z", "area", "normal_x", "normal_y", "normal_z"]
        )

    datapipe = GeometryDatapipe(
        geom_objects=geoms,
        sample_type="volume",
        num_points=100,
        batch_size=2,
        num_workers=1,
        device=device,
    )

    for data in datapipe:
        assert data[0]["sdf"].shape == (2, 100, 1)
        assert all(
            elem in list(data[0].keys())
            for elem in ["x", "y", "z", "area", "sdf", "sdf__x", "sdf__y", "sdf__z"]
        )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_geom_datapipe_stl(device):

    geoms = []
    for i in range(10):
        geo = Tessellation.from_stl(
            dir_path / "stls/cube.stl"
        )  # use the same stl again and again
        geoms.append(geo)

    datapipe = GeometryDatapipe(
        geom_objects=geoms,
        sample_type="surface",
        num_points=100,
        batch_size=2,
        num_workers=1,
        device=device,
    )

    for data in datapipe:
        assert data[0]["normal_x"].shape == (2, 100, 1)
        assert all(
            elem in list(data[0].keys())
            for elem in ["x", "y", "z", "area", "normal_x", "normal_y", "normal_z"]
        )

    datapipe = GeometryDatapipe(
        geom_objects=geoms,
        sample_type="volume",
        num_points=100,
        batch_size=2,
        num_workers=1,
        device=device,
    )

    for data in datapipe:
        assert data[0]["sdf"].shape == (2, 100, 1)
        assert all(
            elem in list(data[0].keys())
            for elem in ["x", "y", "z", "area", "sdf", "sdf__x", "sdf__y", "sdf__z"]
        )


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_geom_datapipe_stl_quasirandom(device):

    geoms = []
    for i in range(4):
        geo = Tessellation.from_stl(
            dir_path / "stls/cube.stl"
        )  # use the same stl again and again
        geoms.append(geo)

    datapipe = GeometryDatapipe(
        geom_objects=geoms,
        sample_type="volume",
        quasirandom=True,
        num_points=100,
        batch_size=2,
        num_workers=1,
        device=device,
    )

    for epoch in range(2):
        for i, data in enumerate(datapipe):
            if (epoch == 0) and (i == 0):
                s1 = copy.deepcopy(data[0])
            if (epoch == 1) and (i == 0):
                s2 = copy.deepcopy(data[0])

    for k in s1.keys():
        assert torch.allclose(s1[k], s2[k])
