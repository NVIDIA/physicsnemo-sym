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

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from physicsnemo.sym.utils.io import csv_to_dict

# path for checkpoint
checkpoint = "./outputs/re590_k_ep_LS/network_checkpoint/"

# read data to compute u_tau
data = np.load(checkpoint + "inferencers/inf_wf.npz", allow_pickle=True)
data = np.atleast_1d(data.f.arr_0)[0]
k_wf = data["k"]
u_tau = np.mean((0.09**0.25) * (k_wf**0.5))

# read data to plot profiles
interior_data = np.load(checkpoint + "inferencers/inf_interior.npz", allow_pickle=True)
interior_data = np.atleast_1d(interior_data.f.arr_0)[0]
y = interior_data["y"]
u = interior_data["u"]
k = interior_data["k"]

nu = 1 / 590
u_plus = u / u_tau
y_plus = (1 - np.abs(y)) * u_tau / nu
k_plus = k / u_tau / u_tau
y = 1 - np.abs(y)

fig, ax = plt.subplots(2, figsize=(4.5, 9))

# read validation data
# Fluent data from Turbulence lecture notes: Gianluca Iaccarino: https://web.stanford.edu/class/me469b/handouts/turbulence.pdf
# DNS data from Moser et al.: https://aip.scitation.org/doi/10.1063/1.869966
file_path = "../validation_data"
if os.path.exists(to_absolute_path(file_path)):
    mapping = {"u+": "u_plus", "y+": "y_plus"}
    u_dns_data = csv_to_dict("../validation_data/re590-moser-dns-u_plus.csv", mapping)
    u_fluent_gi_data = csv_to_dict(
        "../validation_data/re590-gi-fluent-u_plus.csv", mapping
    )

    mapping = {"k+": "k_plus", "y/2H": "y"}
    k_dns_data = csv_to_dict("../validation_data/re590-moser-dns-k_plus.csv", mapping)
    k_fluent_gi_data = csv_to_dict(
        "../validation_data/re590-gi-fluent-k_plus.csv", mapping
    )

    ax[0].scatter(k_dns_data["y"], k_dns_data["k_plus"], label="DNS: Moser")
    ax[0].scatter(k_fluent_gi_data["y"], k_fluent_gi_data["k_plus"], label="Fluent: GI")
    ax[1].scatter(u_dns_data["y_plus"], u_dns_data["u_plus"], label="DNS: Moser")
    ax[1].scatter(
        u_fluent_gi_data["y_plus"], u_fluent_gi_data["u_plus"], label="Fluent: GI"
    )
else:
    warnings.warn(
        f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/Modulus_sym_examples_supplemental_materials"
    )

ax[0].scatter(y, k_plus, label="PhysicsNeMo")
ax[0].set(title="TKE: u_tau=" + str(round(u_tau, 3)))
ax[0].set(xlabel="y", ylabel="k+")
ax[0].legend()

ax[1].scatter(y_plus, u_plus, label="PhysicsNeMo")
ax[1].set_xscale("log")
ax[1].set(title="U+: u_tau=" + str(round(u_tau, 3)))
ax[1].set(xlabel="y+", ylabel="u+")
ax[1].legend()

plt.tight_layout()
plt.savefig("results_k_ep_LS.png")
