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

"""
NOTE: run three_fin_flow and Three_fin_thermal in "eval" mode
after training to get the monitor values for different designs.
"""

# import PhysicsNeMo library
from physicsnemo.sym.utils.io.csv_rw import dict_to_csv
from physicsnemo.sym.hydra import to_absolute_path

# import other libraries
import numpy as np
import os
import sys
import csv

# specify the design optimization requirements
max_pressure_drop = 2.5
num_design = 10
path_flow = to_absolute_path("outputs/run_mode=eval/three_fin_flow")
path_thermal = to_absolute_path("outputs/run_mode=eval/three_fin_thermal")
invar_mapping = [
    "fin_height_middle",
    "fin_height_sides",
    "fin_length_middle",
    "fin_length_sides",
    "fin_thickness_middle",
    "fin_thickness_sides",
]
outvar_mapping = ["pressure_drop", "peak_temp"]


# read the monitor files, and perform a design space search
def DesignOpt(
    path_flow,
    path_thermal,
    num_design,
    max_pressure_drop,
    invar_mapping,
    outvar_mapping,
):
    path_flow += "/monitors"
    path_thermal += "/monitors"
    directory = os.path.join(os.getcwd(), path_flow)
    sys.path.append(path_flow)
    values, configs = [], []

    for _, _, files in os.walk(directory):
        for file in files:
            if file.startswith("back_pressure") & file.endswith(".csv"):
                value = []
                configs.append(file[13:-4])

                # read back pressure
                with open(os.path.join(path_flow, file), "r") as datafile:
                    data = []
                    reader = csv.reader(datafile, delimiter=",")
                    for row in reader:
                        columns = [row[1]]
                        data.append(columns)
                    last_row = float(data[-1][0])
                    value.append(last_row)

                # read front pressure
                with open(
                    os.path.join(path_flow, "front_pressure" + file[13:]), "r"
                ) as datafile:
                    reader = csv.reader(datafile, delimiter=",")
                    data = []
                    for row in reader:
                        columns = [row[1]]
                        data.append(columns)
                    last_row = float(data[-1][0])
                    value.append(last_row)

                # read temperature
                with open(
                    os.path.join(path_thermal, "peak_temp" + file[13:]), "r"
                ) as datafile:
                    data = []
                    reader = csv.reader(datafile, delimiter=",")
                    for row in reader:
                        columns = [row[1]]
                        data.append(columns)
                    last_row = float(data[-1][0])
                    value.append(last_row)
                values.append(value)

    # perform the design optimization
    values = np.array(
        [
            [values[i][1] - values[i][0], values[i][2] * 273.15]
            for i in range(len(values))
        ]
    )
    indices = np.where(values[:, 0] < max_pressure_drop)[0]
    values = values[indices]
    configs = [configs[i] for i in indices]
    opt_design_index = values[:, 1].argsort()[0:num_design]
    opt_design_values = values[opt_design_index]
    opt_design_configs = [configs[i] for i in opt_design_index]

    # Save to a csv file
    opt_design_configs = np.array(
        [
            np.array(opt_design_configs[i][1:].split("_")).astype(float)
            for i in range(num_design)
        ]
    )
    opt_design_configs_dict = {
        key: value.reshape(-1, 1)
        for (key, value) in zip(invar_mapping, opt_design_configs.T)
    }
    opt_design_values_dict = {
        key: value.reshape(-1, 1)
        for (key, value) in zip(outvar_mapping, opt_design_values.T)
    }
    opt_design = {**opt_design_configs_dict, **opt_design_values_dict}
    dict_to_csv(opt_design, "optimal_design")
    print("Finished design optimization!")


if __name__ == "__main__":
    DesignOpt(
        path_flow,
        path_thermal,
        num_design,
        max_pressure_drop,
        invar_mapping,
        outvar_mapping,
    )
