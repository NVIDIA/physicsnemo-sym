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
import matplotlib.pyplot as plt
import torch

try:
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getData
except:
    raise ModuleNotFoundError(
        "This example requires the giverny python package for access to the JHT database.\n"
        + "Find out information here: https://github.com/sciserver/giverny"
    )
from tqdm import *
from typing import List
from pathlib import Path
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.distributed.manager import DistributedManager


def _pos_to_name(dataset, field, time_step, start, end, step, filter_width):
    return (
        "jhtdb_field_"
        + str(field)
        + "_time_step_"
        + str(time_step)
        + "_start_"
        + str(start[0])
        + "_"
        + str(start[1])
        + "_"
        + str(start[2])
        + "_end_"
        + str(end[0])
        + "_"
        + str(end[1])
        + "_"
        + str(end[2])
        + "_step_"
        + str(step[0])
        + "_"
        + str(step[1])
        + "_"
        + str(step[2])
        + "_filter_width_"
        + str(filter_width)
    )


def _name_to_pos(name):
    scrapted_name = name[:4].split("_")
    field = str(scrapted_name[3])
    time_step = int(scrapted_name[6])
    start = [int(x) for x in scrapted_name[7:10]]
    end = [int(x) for x in scrapted_name[11:14]]
    step = [int(x) for x in scrapted_name[15:18]]
    filter_width = int(scrapted_name[-1])
    return field, time_step, start, end, step, filter_width


def _download_sharded_volume(
    loader,
    file_dir: Path,
    field: str,
    physical_time_step: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    subfield_size,
    subfield_idx: int = 0,
    temporal_method: str = "none",
    spatial_method: str  = "none",
    spatial_operator: str = "field"
):
    """
    Break down big query into smaller getData calls to stay withing 2m point limit.
    Only rank 0 executes the download; afterwards the array is saved on
    disk so the other ranks can reload it.
    """
    MAX_POINTS = 2_000_000
    # --- decide sub-cube size -------------------------------------------------
    # maximum linear size that stays within the point limit
    sub_n = int(MAX_POINTS ** (1 / 3))          # ~126 for 2e6
    sub_n = max(1, sub_n)                       # safety
    sub_n = min(sub_n, nx, ny, nz)
    # round sub_n to power of two so that 128 -> 64
    while sub_n & (sub_n - 1):
        sub_n -= 1                              # 125 → 124 → … → 64

    results = np.empty((nx, ny, nz), dtype=np.float32)

    # loop over blocks----------------------------------------------------------
    for i0 in range(0, nx, sub_n):
        i1 = min(i0 + sub_n, nx)
        x_sub = np.linspace(i0 * dx, (i1 - 1) * dx, i1 - i0, dtype=np.float64)

        for j0 in range(0, ny, sub_n):
            j1 = min(j0 + sub_n, ny)
            y_sub = np.linspace(j0 * dy, (j1 - 1) * dy, j1 - j0, dtype=np.float64)

            for k0 in range(0, nz, sub_n):
                k1 = min(k0 + sub_n, nz)
                z_sub = np.linspace(k0 * dz, (k1 - 1) * dz, k1 - k0, dtype=np.float64)

                # build point list for this block
                pts = np.array(
                    [axis.ravel() for axis in np.meshgrid(x_sub, y_sub, z_sub, indexing='ij')],
                    dtype=np.float64
                ).T

                block = getData(
                    loader,
                    field,
                    physical_time_step,
                    temporal_method,
                    spatial_method,
                    spatial_operator,
                    pts
                )
                block = np.array(block[0]).reshape(len(x_sub), len(y_sub), len(z_sub), subfield_size)
                # Swap z-y-x axis order from getData into x-y-z from original getCutout method
                block = block.transpose(2,1,0,3) # 4th dimension - the variable "depth" should stay where it is
                desired_var = block[:,:,:,subfield_idx]
                results[i0:i1, j0:j1, k0:k1] = desired_var

    return results


def get_jhtdb(
    loader, data_dir: Path, dataset, field, time_step, start, end, step, filter_width, domain_size
):
    # Set Dataset params. See https://turbulence.idies.jhu.edu/docs/isotropic/README-isotropic.pdf
    subfield = 0 # u-component of u-v-w of Velocity
    temporal_method = 'none'
    spatial_method = 'none'
    spatial_operator = 'field'
    physical_domain_size = 2 * np.pi
    # Scalar fields have depth 3. Velocity is a vector field of depth 3
    subfield_size = 3 if field == "velocity" else 1
    # Physical Domain steps between measured points
    dx = dy = dz = physical_domain_size / 1024
    # Physical Time between each snapshot. Can be found in the README file of each dataset
    #    e.g. https://turbulence.idies.jhu.edu/docs/isotropic/README-isotropic.pdf
    dt = .002

    subfield = "u"  # 0-th component of velocity, which is a vector of [u,v,w]
    subfield_idx = 0

    file_name = (
        _pos_to_name(dataset, subfield, time_step, start, end, step, filter_width) + ".npy"
    )
    file_dir = data_dir / Path(file_name)

    nx = ny = nz = domain_size

    # getData is now 0-indexed, unlike previous getCutout. Keep this after file_name
    x_points = np.linspace((start[0]-1) * dx, (end[0]-1-64) * dx * step[0], nx, dtype=np.float64)
    y_points = np.linspace((start[1]-1) * dy, (end[1]-1-64) * dy * step[1], ny, dtype=np.float64)
    z_points = np.linspace((start[2]-1) * dz, (end[2]-1-64) * dz * step[2], nz, dtype=np.float64)

    points = np.array(
        [axis.ravel() for axis in np.meshgrid(x_points, y_points, z_points, indexing = 'ij')],
        dtype = np.float64).T
    
    physical_time_step = (time_step-1) * dt

    # check if file exists and if not download it
    try:
        results = np.load(file_dir)

    except FileNotFoundError:
        # Only MPI process 0 can download data
        if DistributedManager().rank == 0:
            # getData is limited to 2,000,000 points. Bigger queries need to be broken down
            results = _download_sharded_volume(
                loader,
                file_dir,
                field,
                physical_time_step,
                nx, ny, nz,
                dx, dy, dz,
                subfield_size,
                subfield_idx = subfield_idx,
            )
            # Optional: Test against predownloaded 
            # res2 = np.array(r2[0]).reshape(nx, ny, nz, subfield_size)
            # # Swap z-y-x axis order from getData into x-y-z from original getCutout method
            # results2 = res2.transpose(2,1,0,3) # 4th dimension - the variable "depth" should stay where it is
            # # results2 = res2[:,:,:,subfield_idx] # nvidia seems to have saved all 3 u-v-w
    
            # assert np.array_equal(results[:64,:64,:64,:], results2)
            np.save(file_dir, results)
        # Wait for all processes to get here
        if DistributedManager().distributed:
            torch.distributed.barrier()
        results = np.load(file_dir)

    return results


def make_jhtdb_dataset(
    nr_samples: int = 128,
    domain_size: int = 64,
    lr_factor: int = 4,
    token: str = "edu.jhu.pha.turbulence.testing-201311",
    data_dir: str = to_absolute_path("datasets/jhtdb_training"),
    time_range: List[int] = [1, 1024],
    dataset_seed: int = 123,
    debug: bool = False,
):
    # make data dir
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = "isotropic1024coarse"

    # initialize JHTDB dataset
    turbulence_dataset = turb_dataset(
        dataset_title=dataset,
        output_path = str(data_dir),
        auth_token = token
    )

    # loop to get dataset
    np.random.seed(dataset_seed)
    list_low_res_u = []
    list_high_res_u = []
    for i in tqdm(range(nr_samples)):
        field = "velocity"
        time_step = int(np.random.randint(time_range[0], time_range[1]))

        start = np.array(
            [np.random.randint(1, 1024 - domain_size) for _ in range(3)], dtype=int
        )
        end = np.array([x + domain_size - 1 for x in start], dtype=int)

        # get high res data
        high_res_u = get_jhtdb(
            turbulence_dataset,
            data_dir,
            dataset,
            field,
            time_step,
            start,
            end,
            np.array(3 * [1], dtype=int),
            1,
            domain_size
        )

        # get low res data
        low_res_u = get_jhtdb(
            turbulence_dataset,
            data_dir,
            dataset,
            field,
            time_step,
            start,
            end,
            np.array(3 * [lr_factor], dtype=int),
            lr_factor,
            domain_size
        )

        # plot
        if debug:
            fig = plt.figure(figsize=(10, 5))
            a = fig.add_subplot(121)
            a.set_axis_off()
            a.imshow(low_res_u[:, :, 0, 0], interpolation="none")
            a = fig.add_subplot(122)
            a.imshow(high_res_u[:, :, 0, 0], interpolation="none")
            plt.savefig("debug_plot_" + str(i))
            plt.close()

        # append to list
        list_low_res_u.append(np.rollaxis(low_res_u, -1, 0))
        list_high_res_u.append(np.rollaxis(high_res_u, -1, 0))

    # concatenate to tensor
    dataset_low_res_u = np.stack(list_low_res_u, axis=0)
    dataset_high_res_u = np.stack(list_high_res_u, axis=0)

    return {"U_lr": dataset_low_res_u}, {"U": dataset_high_res_u}
