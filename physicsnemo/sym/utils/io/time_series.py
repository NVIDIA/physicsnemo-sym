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

import matplotlib.pyplot as plt


def plot_time_series(var, base_name, time_axis="step"):
    for plot_var in var.keys():
        if plot_var != time_axis:
            plt.plot(var[time_axis][:, 0], var[plot_var][:, 0], label=plot_var)
    plt.legend()
    plt.xlabel(time_axis)
    plt.savefig(base_name + ".png")
    plt.close()
