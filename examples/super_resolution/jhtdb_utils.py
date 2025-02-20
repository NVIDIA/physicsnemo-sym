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
    import pyJHTDB
    import pyJHTDB.dbinfo
except:
    raise ModuleNotFoundError(
        "This example requires the pyJHTDB python package for access to the JHT database.\n"
        + "Find out information here: https://github.com/idies/pyJHTDB"
    )
from tqdm import *
from typing import List, Dict, Union
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


def get_jhtdb(
    loader, data_dir: Path, dataset, field, time_step, start, end, step, filter_width
):
    # get filename
    file_name = (
        _pos_to_name(dataset, field, time_step, start, end, step, filter_width) + ".npy"
    )
    file_dir = data_dir / Path(file_name)

    # check if file exists and if not download it
    try:
        results = np.load(file_dir)
    except:
        # Only MPI process 0 can download data
        if DistributedManager().rank == 0:
            results = loader.getCutout(
                data_set=dataset,
                field=field,
                time_step=time_step,
                start=start,
                end=end,
                step=step,
                filter_width=filter_width,
            )
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

    # initialize runner
    lJHTDB = pyJHTDB.libJHTDB()
    lJHTDB.initialize()
    lJHTDB.add_token(token)

    # loop to get dataset
    np.random.seed(dataset_seed)
    list_low_res_u = []
    list_high_res_u = []
    for i in tqdm(range(nr_samples)):
        # set download params
        dataset = "isotropic1024coarse"
        field = "u"
        time_step = int(np.random.randint(time_range[0], time_range[1]))
        start = np.array(
            [np.random.randint(1, 1024 - domain_size) for _ in range(3)], dtype=int
        )
        end = np.array([x + domain_size - 1 for x in start], dtype=int)
        step = np.array(3 * [1], dtype=int)

        # get high res data
        high_res_u = get_jhtdb(
            lJHTDB,
            data_dir,
            dataset,
            field,
            time_step,
            start,
            end,
            np.array(3 * [1], dtype=int),
            1,
        )

        # get low res data
        low_res_u = get_jhtdb(
            lJHTDB,
            data_dir,
            dataset,
            field,
            time_step,
            start,
            end,
            np.array(3 * [lr_factor], dtype=int),
            lr_factor,
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
