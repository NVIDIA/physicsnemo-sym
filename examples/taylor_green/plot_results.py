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
import matplotlib.pyplot as plt
from physicsnemo.sym.utils.io import csv_to_dict
import os
import warnings

# get list of steps
window_dirs = glob.glob("./outputs/taylor_green/network_checkpoint/*")
window_dirs.sort()
window_dirs = [x for x in window_dirs if os.path.isdir(x)]

# read each file in each dir and store tke
index = 0
time_points = []
tke_points = []
for i, d in enumerate(window_dirs):
    # get list of slices
    slice_files = glob.glob(d + "/inferencers/time_slice_*.npz")
    slice_files.sort()

    for f in slice_files:
        predicted_data = np.load(f, allow_pickle=True)["arr_0"].item()

        # shift t
        predicted_data["t"] += i
        if float(predicted_data["t"][0, 0, 0]) < 10.0:
            # store time
            time_points.append(float(predicted_data["t"][0, 0, 0]))

            # compute tke and store
            tke = np.mean(
                predicted_data["u"] ** 2 / 2
                + predicted_data["v"] ** 2 / 2
                + predicted_data["w"] ** 2 / 2
            )
            tke_points.append(tke)
            index += 1
tke_points = tke_points / np.max(tke_points)

# load validation tke data
file_path = "validation_tke"
if os.path.exists(to_absolute_path(file_path)):
    validation_tke_128 = csv_to_dict("validation_tke/tke_mean_Re500_N128.csv")
    validation_tke_256 = csv_to_dict("validation_tke/tke_mean_Re500_N256.csv")

    plt.plot(
        validation_tke_128["Time"][:, 0],
        validation_tke_128["TKE_mean"][:, 0],
        label="Spectral Solver (grid res: 128)",
    )
    plt.plot(
        validation_tke_256["Time"][:, 0],
        validation_tke_256["TKE_mean"][:, 0],
        label="Spectral Solver (grid res: 256)",
    )
else:
    warnings.warn(
        f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/Modulus_sym_examples_supplemental_materials"
    )

# plot turbulent kinetic energy decay
plt.plot(time_points, tke_points, label="PhysicsNeMo")

plt.legend()
plt.title("TKE")
plt.ylabel("TKE")
plt.xlabel("time")
plt.savefig("tke_plot.png")
