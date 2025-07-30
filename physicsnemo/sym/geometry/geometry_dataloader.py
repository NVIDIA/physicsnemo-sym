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
import torch

import nvidia.dali as dali
import nvidia.dali.plugin.pytorch as dali_pth

from typing import Iterable, Union, Tuple, Dict, List

from physicsnemo.sym.geometry.geometry import Geometry

Tensor = torch.Tensor


class GeometryDatapipe:
    """
    DALI Datapipe to sample PhysicsNeMo geometry objects.
    Can be used to sample points on surface or inside interior (and exterior) of
    geometry objects generated from PhysicsNeMo' constructive geometry module or the
    tessellation module (stls).

    Parameters
    ----------
    geom_objects : List[Geometry]
        List of PhysicsNeMo Geometry objects. Can be CSG or Tessallation geometries.
    batch_size : int, optional
        Batch Size, by default 1
    num_points : int, optional
        Number of points to sample either on surface or interior, by default 1000
    requested_vars : Union[List[str], None], optional
        List of output variables to output. If None, all variables are outputed which
        include the coordinates (`x`, `y`, `z`), `area` and normals (`normal_x`,
        `normal_y`, `normal_z`) for surface sampling and coordinates (`x`, `y`, `z`),
        `area`, `sdf` and it's derivatives (`sdf__x`, `sdf__y`, `sdf__z`). Default None
    sample_type : str, optional
        Whether to sample surface or volume. Options are "surface" and "volume", by
        default "surface"
    flip_interior : bool, optional
        Whether to sample inside the geometry or outside. by default False which samples
        inside the geometry. Only used when `sample_type` is "volume".
    bounds : Union[Dict[str, float], None]
        Bounds for sampling the geometry during "volume" type sampling, by default None
        where the internal bounds are used (bounding box).
    quasirandom : bool, optional
        If true, points are sampled using Halton Sequences, by default False
    dtype : str, optional
        Typecode to which the output data is cast, by default float32.
    shuffle : bool, optional
        Shuffle dataset, by default True
    num_workers : int, optional
        Number of parallel workers, by default 1
    device : Union[str, torch.device], optional
        Device for DALI pipeline. Options are "cuda" and "cpu", by default "cuda"
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1
    """

    def __init__(
        self,
        geom_objects: List[Geometry],
        batch_size: int = 1,
        num_points: int = 1000,
        requested_vars: Union[List[str], None] = None,
        sample_type: str = "surface",  # options are "volume" and "surface"
        flip_interior: bool = False,  # Whether to sample inside of the geometry or outside
        bounds: Union[Dict[str, float], None] = None,
        quasirandom: bool = False,
        dtype: str = "float32",
        shuffle: bool = True,
        num_workers: int = 1,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
    ):
        self.geom_objects = geom_objects
        self.batch_size = batch_size
        self.num_points = num_points
        self.requested_vars = requested_vars
        self.sample_type = sample_type
        self.flip_interior = flip_interior
        self.bounds = bounds
        self.quasirandom = quasirandom
        self.dtype = dtype
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.process_rank = process_rank
        self.world_size = world_size

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device

        self.parse_dataset_files()

        self.pipe = self._create_pipeline()

        self.output_keys = self.requested_vars

    def parse_dataset_files(self) -> None:
        """
        Parse the geometries.
        """
        # get the var names by sampling one of the geom
        geo = self.geom_objects[0]
        if self.sample_type == "surface":
            samples = geo.sample_boundary(nr_points=self.num_points)
        elif self.sample_type == "volume":
            samples = geo.sample_interior(
                nr_points=self.num_points, compute_sdf_derivatives=True
            )

        available_vars = list(samples.keys())
        if self.requested_vars is None:
            self.requested_vars = available_vars

        if not set(self.requested_vars) <= set(available_vars):
            raise ValueError(
                f"Requested variables not available. Please choose from {available_vars}"
            )

    def _create_pipeline(self) -> dali.Pipeline:
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            source = GeometrySource(
                geom_objects=self.geom_objects,
                batch_size=self.batch_size,
                num_points=self.num_points,
                requested_vars=self.requested_vars,
                sample_type=self.sample_type,
                bounds=self.bounds,
                quasirandom=self.quasirandom,
                dtype=self.dtype,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )

            self.length = len(source) // self.batch_size
            vars_tuple = dali.fn.external_source(
                source,
                num_outputs=len(self.requested_vars),
                parallel=True,
                batch=False,
                device="cpu",
            )

            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that.
                vars_out = [var.gpu() for var in vars_tuple]
            else:
                vars_out = [var for var in vars_tuple]

            pipe.set_outputs(*vars_out)

        return pipe

    def __iter__(self):
        self.pipe.reset()
        return dali_pth.DALIGenericIterator(
            [self.pipe],
            self.output_keys,
            auto_reset=True,
            size=self.length * self.batch_size,
        )

    def __len__(self):
        return self.length


class GeometrySource:
    """
    DALI Source for lazy sampling of geometries.

    Parameters
    ----------
    geom_objects : Iterable[str]
        Geometry objects
    batch_size : int, optional
        Batch size, by default 1
    num_points : int, optional
        Number of points to sample either on surface or interior, by default 1000
    requested_vars : Union[List[str], None], optional
        Number of points to sample either on surface or interior, by default None which
        selects all variables available.
    sample_type : str, optional
        Whether to sample surface or volume. , by default "surface"
    flip_interior : bool, optional
        Whether to sample inside the geometry or outside. by default False which samples
        inside the geometry. Only used when `sample_type` is "volume".
    bounds : Union[Dict[str, float], None]
        Bounds for sampling the geometry during "volume" type sampling, by default None
        where the internal bounds are used (bounding box).
    quasirandom : bool, optional
        If true, points are sampled using Halton Sequences, by default False
    dtype : str, optional
        Typecode to which the output data is cast, by default float32.
    shuffle : bool, optional
        Shuffle dataset, by default True
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1
    """

    def __init__(
        self,
        geom_objects: Iterable[str],
        batch_size: int = 1,
        num_points: int = 1000,
        requested_vars: Union[List[str], None] = None,
        sample_type: str = "surface",
        flip_interior: bool = False,
        bounds: Union[Dict[str, float], None] = None,
        quasirandom: bool = False,
        dtype: str = "float32",
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
    ):
        self.geom_objects = list(geom_objects)
        self.batch_size = batch_size
        self.num_points = num_points
        self.requested_vars = requested_vars
        self.sample_type = sample_type
        self.flip_interior = flip_interior
        self.bounds = bounds
        self.quasirandom = quasirandom
        self.dtype = dtype
        self.shuffle = shuffle

        self.last_epoch = None

        self.indices = np.arange(len(self.geom_objects))
        # Shard from indices if running in parallel
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        self.num_batches = len(self.indices) // self.batch_size

    def __call__(self, sample_info: dali.types.SampleInfo) -> Tuple[np.ndarray]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        # Shuffle before the next epoch starts
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        idx = self.indices[sample_info.idx_in_epoch]
        data = self.geom_objects[idx]

        if self.sample_type == "surface":
            samples = data.sample_boundary(
                nr_points=self.num_points, quasirandom=self.quasirandom
            )  # Note quasirandom for boundary sampling is not yet supported
            for k, v in samples.items():
                samples[k] = v.astype(self.dtype)
        elif self.sample_type == "volume":
            samples = data.sample_interior(
                nr_points=self.num_points,
                compute_sdf_derivatives=True,
                flip_interior=self.flip_interior,
                bounds=self.bounds,
                quasirandom=self.quasirandom,
            )
            for k, v in samples.items():
                samples[k] = v.astype(self.dtype)

        # Add batch dimension
        var = tuple([samples[k] for k in self.requested_vars])
        return var

    def __len__(self):
        return len(self.indices)
