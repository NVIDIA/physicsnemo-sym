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

# import libraries
import numpy as np
import chaospy

# load data
samples = np.loadtxt("samples.txt")
num_samples = len(samples)

# read monitored values
y_vec = []
for i in range(num_samples):
    front_pressure_dir = (
        "./outputs/limerock_flow/tl_" + str(i) + "/monitors/front_pressure.csv"
    )
    back_pressure_dir = (
        "./outputs/limerock_flow/tl_" + str(i) + "/monitors/back_pressure.csv"
    )
    with open(front_pressure_dir, "r", encoding="utf-8", errors="ignore") as scraped:
        front_pressure = float(scraped.readlines()[-1].split(",")[1])
    with open(back_pressure_dir, "r", encoding="utf-8", errors="ignore") as scraped:
        back_pressure = float(scraped.readlines()[-1].split(",")[1])
    pressure_drop = front_pressure - back_pressure
    y_vec.append(pressure_drop)
y_vec = np.array(y_vec)

# Split data into training and validation
val_portion = 0.15
val_idx = np.random.choice(
    np.arange(num_samples, dtype=int), int(val_portion * num_samples), replace=False
)
val_x, val_y = samples[val_idx], y_vec[val_idx]
train_x, train_y = (
    np.delete(samples, val_idx, axis=0).T,
    np.delete(y_vec, val_idx).reshape(-1, 1),
)

# Construct the PCE
distribution = chaospy.J(
    chaospy.Uniform(0.0, np.pi / 6),
    chaospy.Uniform(0.0, np.pi / 6),
    chaospy.Uniform(0.0, np.pi / 6),
    chaospy.Uniform(0.0, np.pi / 6),
)
expansion = chaospy.generate_expansion(2, distribution)
poly = chaospy.fit_regression(expansion, train_x, train_y)

# PCE closed form
print("__________")
print("PCE closd form:")
print(poly)
print("__________")

# Validation
print("PCE evaluatins:")
for i in range(len(val_x)):
    pred = poly(val_x[i, 0], val_x[i, 1], val_x[i, 2], val_x[i, 3])[0]
    print("Sample:", val_x[i])
    print("True val:", val_y[i])
    print("Predicted val:", pred)
    print("Relative error (%):", abs(pred - val_y[i]) / val_y[i] * 100)
print("__________")
