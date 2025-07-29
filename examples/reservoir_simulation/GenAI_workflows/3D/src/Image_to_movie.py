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
@Author : Clement Etienam
"""

import os
from PIL import Image

print("")
print("Now - Creating GIF")
import glob

oldfolder = os.getcwd()
os.chdir(oldfolder)

choice = int(
    input(
        "What Surrogate model did you train for?: \n\
1 = FNO\n\
2 = PINO\n\
"
    )
)


if choice == 1:
    os.chdir("outputs/Forward_problem_FNO/ResSim/validators")
else:
    os.chdir("outputs/Forward_problem_PINO/ResSim/validators")

frames = []
imgs = sorted(glob.glob("*test_pressure_pressure_simulations*"), key=os.path.getmtime)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "pressure_test.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0,
)


frames1 = []
imgs = sorted(
    glob.glob("*test_saturation_saturation_simulations*"), key=os.path.getmtime
)
for i in imgs:
    new_frame = Image.open(i)
    frames1.append(new_frame)

frames1[0].save(
    "saturation_test.gif",
    format="GIF",
    append_images=frames1[1:],
    save_all=True,
    duration=500,
    loop=0,
)


frames = []
imgs = sorted(glob.glob("*val_pressure_pressure_simulations*"), key=os.path.getmtime)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "pressure_val.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0,
)


frames1 = []
imgs = sorted(
    glob.glob("*val_saturation_saturation_simulations*"), key=os.path.getmtime
)
for i in imgs:
    new_frame = Image.open(i)
    frames1.append(new_frame)

frames1[0].save(
    "saturation_val.gif",
    format="GIF",
    append_images=frames1[1:],
    save_all=True,
    duration=500,
    loop=0,
)


from glob import glob

for f3 in glob("*test_saturation_saturation_simulations*"):
    os.remove(f3)

for f4 in glob("*test_pressure_pressure_simulations*"):
    os.remove(f4)

for f5 in glob("*val_pressure_pressure_simulations*"):
    os.remove(f5)

for f6 in glob("*val_saturation_saturation_simulations*"):
    os.remove(f6)

os.chdir(oldfolder)
