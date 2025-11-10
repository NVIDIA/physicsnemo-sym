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

from functools import partial

import numpy as np
import warp as wp
from stl import mesh as np_mesh

from .geometry import Geometry
from .parameterization import Parameterization, Bounds, Parameter
from .curve import Curve
from physicsnemo.sym.constants import diff_str


@wp.kernel
def compute_triangle_areas_warp(
    v0: wp.array(dtype=wp.vec3),
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    areas: wp.array(dtype=float),
):
    """Warp kernel to compute triangle areas."""
    # This is a warp implementation of the _area_of_triangles function in tessellation.py.
    tid = wp.tid()

    # Get triangle vertices.
    vertex0 = v0[tid]
    vertex1 = v1[tid]
    vertex2 = v2[tid]

    # Compute edges.
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0

    # Cross product gives area vector.
    cross = wp.cross(edge1, edge2)
    areas[tid] = wp.length(cross) * 0.5


@wp.kernel
def sample_triangles_warp(
    v0: wp.array(dtype=wp.vec3),
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=int),
    random_r1: wp.array(dtype=float),
    random_r2: wp.array(dtype=float),
    points_x: wp.array(dtype=float),
    points_y: wp.array(dtype=float),
    points_z: wp.array(dtype=float),
    normal_x: wp.array(dtype=float),
    normal_y: wp.array(dtype=float),
    normal_z: wp.array(dtype=float),
):
    """Warp kernel to sample points on triangles using barycentric coordinates."""
    # This is a warp implementation of the _sample_triangle function in tessellation.py.
    tid = wp.tid()

    # Get triangle index for this point.
    tri_idx = triangle_indices[tid]

    # Get random values.
    r1 = random_r1[tid]
    r2 = random_r2[tid]

    # Barycentric sampling (uniform distribution in triangle).
    s1 = wp.sqrt(r1)
    u = 1.0 - s1
    v = (1.0 - r2) * s1
    w = r2 * s1

    # Get triangle vertices.
    vertex0 = v0[tri_idx]
    vertex1 = v1[tri_idx]
    vertex2 = v2[tri_idx]

    # Sample point using barycentric coordinates.
    sampled_point = u * vertex0 + v * vertex1 + w * vertex2

    # Store coordinates.
    points_x[tid] = sampled_point[0]
    points_y[tid] = sampled_point[1]
    points_z[tid] = sampled_point[2]

    # Get and normalize normal.
    normal = normals[tri_idx]
    normal_length = wp.length(normal)

    if normal_length > 0.0:
        normalized_normal = normal / normal_length
        normal_x[tid] = normalized_normal[0]
        normal_y[tid] = normalized_normal[1]
        normal_z[tid] = normalized_normal[2]
    else:
        normal_x[tid] = 0.0
        normal_y[tid] = 0.0
        normal_z[tid] = 1.0


class WarpTessellationSampler:
    """Warp-accelerated tessellation sampling.

    Parameters
    ----------
    mesh : Mesh (numpy-stl)
        A mesh that defines the surface of the geometry.
    device : str, optional
        Device to use for Warp operations ('cuda' or 'cpu').
        If None, automatically sets the device to the current device.
    seed : int, optional
        Seed for the random number generator. If None, uses default seeding.
    """

    def __init__(self, mesh, device=None, seed=None):
        self.num_triangles = len(mesh.v0)
        self.rng = np.random.default_rng(seed)

        self.device = device if device is not None else wp.get_device()

        # Convert mesh data to Warp format.
        v0_data = [(float(v[0]), float(v[1]), float(v[2])) for v in mesh.v0]
        v1_data = [(float(v[0]), float(v[1]), float(v[2])) for v in mesh.v1]
        v2_data = [(float(v[0]), float(v[1]), float(v[2])) for v in mesh.v2]
        normals_data = [(float(n[0]), float(n[1]), float(n[2])) for n in mesh.normals]

        self.v0_wp = wp.array(v0_data, dtype=wp.vec3, device=self.device)
        self.v1_wp = wp.array(v1_data, dtype=wp.vec3, device=self.device)
        self.v2_wp = wp.array(v2_data, dtype=wp.vec3, device=self.device)
        self.normals_wp = wp.array(normals_data, dtype=wp.vec3, device=self.device)

        # Pre-compute triangle areas.
        self.areas_wp = wp.zeros(self.num_triangles, device=self.device)
        wp.launch(
            compute_triangle_areas_warp,
            dim=self.num_triangles,
            inputs=[self.v0_wp, self.v1_wp, self.v2_wp, self.areas_wp],
            device=self.device,
        )

        # Copy areas to CPU for probability computation.
        areas_host = self.areas_wp.numpy()
        self.triangle_areas = areas_host
        self.triangle_probabilities = areas_host / np.sum(areas_host)
        self.total_area = float(np.sum(areas_host))

    def sample_warp(
        self,
        nr_points: int,
        parameterization: Parameterization = None,
        quasirandom: bool = False,
    ) -> tuple[dict, dict]:
        """
        Warp-accelerated sampling function.

        Parameters
        ----------
        nr_points : int
            Number of points to sample.
        parameterization : Parameterization, optional
            Parameterization of the geometry. Default is an empty Parameterization.
        quasirandom : bool, optional
            If True, use quasirandom sampling. Default is False.

        Returns
        -------
        invar : dict
            Dictionary containing the sampled points and normals.
        params : dict
            Dictionary containing the parameters of the sampled points.
        """

        nr_points = int(nr_points)

        # Step 1: Distribute points across triangles based on area.
        triangle_indices = self.rng.choice(
            self.num_triangles, size=nr_points, p=self.triangle_probabilities
        )

        # Step 2: Generate random numbers for barycentric sampling.
        r1 = self.rng.uniform(0, 1, size=nr_points).astype(np.float32)
        r2 = self.rng.uniform(0, 1, size=nr_points).astype(np.float32)

        # Step 3: Create Warp arrays.
        triangle_indices_wp = wp.array(triangle_indices, dtype=int, device=self.device)
        r1_wp = wp.array(r1, dtype=float, device=self.device)
        r2_wp = wp.array(r2, dtype=float, device=self.device)

        # Step 4: Create output arrays.
        points_x_wp = wp.zeros(nr_points, device=self.device)
        points_y_wp = wp.zeros(nr_points, device=self.device)
        points_z_wp = wp.zeros(nr_points, device=self.device)
        normal_x_wp = wp.zeros(nr_points, device=self.device)
        normal_y_wp = wp.zeros(nr_points, device=self.device)
        normal_z_wp = wp.zeros(nr_points, device=self.device)

        # Step 5: Launch sampling kernel.
        wp.launch(
            sample_triangles_warp,
            dim=nr_points,
            inputs=[
                self.v0_wp,
                self.v1_wp,
                self.v2_wp,
                self.normals_wp,
                triangle_indices_wp,
                r1_wp,
                r2_wp,
                points_x_wp,
                points_y_wp,
                points_z_wp,
                normal_x_wp,
                normal_y_wp,
                normal_z_wp,
            ],
            device=self.device,
        )

        # Step 6: Copy results back to CPU.
        x = points_x_wp.numpy().reshape(-1, 1)
        y = points_y_wp.numpy().reshape(-1, 1)
        z = points_z_wp.numpy().reshape(-1, 1)
        nx = normal_x_wp.numpy().reshape(-1, 1)
        ny = normal_y_wp.numpy().reshape(-1, 1)
        nz = normal_z_wp.numpy().reshape(-1, 1)

        # Step 7: Compute area weights (matching original implementation).
        area_weight = self.total_area / nr_points
        area = np.full((nr_points, 1), area_weight, dtype=np.float32)

        # Step 8: Create invar dictionary matching original format.
        invar = {
            "x": x,
            "y": y,
            "z": z,
            "normal_x": nx,
            "normal_y": ny,
            "normal_z": nz,
            "area": area,
        }

        # Step 9: Handle parameterization.
        if parameterization is None:
            parameterization = Parameterization()

        params = parameterization.sample(nr_points, quasirandom=quasirandom)

        return invar, params


class Tessellation(Geometry):
    """
    Tessellation is a geometry that uses Warp to accelerate the sampling of a tessellated geometry.

    Parameters
    ----------
    mesh : Mesh (numpy-stl)
        A mesh that defines the surface of the geometry.
    airtight : bool
        If the geometry is airtight or not. If false sample everywhere for interior.
    parameterization : Parameterization, optional
        Parameterization of the geometry. Default is an empty Parameterization.
    device : str, optional
        Device to use for Warp operations ('cuda' or 'cpu'). If None, automatically
        detects the best available device.
    seed : int, optional
        Seed for the random number generator. If None, uses default seeding.
    """

    def __init__(
        self,
        mesh,
        airtight: bool = True,
        parameterization: Parameterization = Parameterization(),
        device: str = None,
        seed: int = None,
    ):
        # Create curve with Warp-accelerated sampling.
        curves = [
            Curve(
                WarpTessellationSampler(mesh, device=device, seed=seed).sample_warp,
                dims=3,
                parameterization=parameterization,
            )
        ]

        bounds = Bounds(
            {
                Parameter("x"): (
                    float(np.min(mesh.vectors[:, :, 0])),
                    float(np.max(mesh.vectors[:, :, 0])),
                ),
                Parameter("y"): (
                    float(np.min(mesh.vectors[:, :, 1])),
                    float(np.max(mesh.vectors[:, :, 1])),
                ),
                Parameter("z"): (
                    float(np.min(mesh.vectors[:, :, 2])),
                    float(np.max(mesh.vectors[:, :, 2])),
                ),
            },
            parameterization=parameterization,
        )

        super().__init__(
            curves,
            partial(Tessellation.sdf, mesh.vectors, airtight),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )

    @classmethod
    def from_stl(
        cls,
        filename,
        airtight=True,
        parameterization=Parameterization(),
        device=None,
        seed=None,
    ):
        """
        Create a Tessellation geometry from an STL file.

        Parameters
        ----------
        filename : str
            Path to the STL mesh file.
        airtight : bool, optional
            If True, assumes the geometry is airtight. Default is True.
        parameterization : Parameterization, optional
            Parameterization of the geometry. Default is an empty Parameterization.
        device : str, optional
            Device to use for Warp operations ('cuda' or 'cpu'). If None, automatically
            detects the best available device.
        seed : int, optional
            Seed for the random number generator. If None, uses default seeding.

        Returns
        -------
        Tessellation
            An instance of Tessellation initialized with the mesh from the STL file.
        """
        # Read in mesh.
        mesh = np_mesh.Mesh.from_file(filename)
        return cls(mesh, airtight, parameterization, device, seed)

    @staticmethod
    def sdf(triangles, airtight, invar, params, compute_sdf_derivatives=False):
        """Simple copy of the sdf function in tessellation.py."""
        points = np.stack([invar["x"], invar["y"], invar["z"]], axis=1)
        points = np.squeeze(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        minx, maxx, miny, maxy, minz, maxz = Tessellation.find_mins_maxs(points)
        max_dis = max(max((maxx - minx), (maxy - miny)), (maxz - minz))
        store_triangles = np.array(triangles, dtype=np.float32)
        store_triangles[:, :, 0] -= minx
        store_triangles[:, :, 1] -= miny
        store_triangles[:, :, 2] -= minz
        store_triangles *= 1 / max_dis
        store_triangles = store_triangles.reshape(-1, 3)  # (n_vertices, 3)
        points[:, 0] -= minx
        points[:, 1] -= miny
        points[:, 2] -= minz
        points *= 1 / max_dis
        points = points.astype(np.float32)  # Keep as (n_points, 3)

        outputs = {}
        if airtight:
            # Import locally to avoid pickling issues with multiprocessing
            import torch
            from physicsnemo.utils.sdf import signed_distance_field

            torch_vertices = torch.from_numpy(store_triangles)
            torch_indices = torch.arange(store_triangles.shape[0], dtype=torch.int32)
            torch_points = torch.from_numpy(points)

            sdf_field, sdf_derivative = signed_distance_field(
                torch_vertices,
                torch_indices,
                torch_points,
                max_dist=1e8,
                use_sign_winding_number=False,
            )

            sdf_field = sdf_field.numpy()
            sdf_derivative = sdf_derivative.numpy()  # Shape: (n_points, 3)

            sdf_field = -np.expand_dims(max_dis * sdf_field, axis=1)
        else:
            sdf_field = np.zeros_like(invar["x"])
        outputs["sdf"] = sdf_field

        if compute_sdf_derivatives:
            sdf_derivative = -(sdf_derivative - points)
            sdf_derivative = sdf_derivative / np.linalg.norm(
                sdf_derivative, axis=1, keepdims=True
            )
            outputs["sdf" + diff_str + "x"] = sdf_derivative[:, 0:1]
            outputs["sdf" + diff_str + "y"] = sdf_derivative[:, 1:2]
            outputs["sdf" + diff_str + "z"] = sdf_derivative[:, 2:3]

        return outputs

    @staticmethod
    def find_mins_maxs(points):
        minx = float(np.min(points[:, 0]))
        miny = float(np.min(points[:, 1]))
        minz = float(np.min(points[:, 2]))
        maxx = float(np.max(points[:, 0]))
        maxy = float(np.max(points[:, 1]))
        maxz = float(np.max(points[:, 2]))
        return minx, maxx, miny, maxy, minz, maxz
