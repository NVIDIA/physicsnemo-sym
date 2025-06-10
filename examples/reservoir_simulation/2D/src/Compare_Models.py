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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 21:33:56 2023

@author: clementetienam
"""
import os
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from NVRS import *
from physicsnemo.sym.models.fno import *
from physicsnemo.sym.models.afno.afno import *
import shutil
import pandas as pd
import scipy.io as sio
import torch
import yaml
from PIL import Image

oldfolder = os.getcwd()
os.chdir(oldfolder)

data = []
os.chdir("../COMPARE_RESULTS/FNO")
True_measurement = pd.read_csv("RSM_NUMERICAL.csv")
True_measurement = True_measurement.values.astype(np.float32)[:, 1:]

data.append(True_measurement)


FNO = pd.read_csv("RSM_MODULUS.csv")
FNO = FNO.values.astype(np.float32)[:, 1:]
data.append(FNO)

os.chdir(oldfolder)

os.chdir("../COMPARE_RESULTS/PINO")
PINO = pd.read_csv("RSM_MODULUS.csv")
PINO = PINO.values.astype(np.float32)[:, 1:]
data.append(PINO)
os.chdir(oldfolder)

os.chdir("../COMPARE_RESULTS/AFNOP")
AFNOP = pd.read_csv("RSM_MODULUS.csv")
AFNOP = AFNOP.values.astype(np.float32)[:, 1:]
data.append(AFNOP)
os.chdir(oldfolder)

os.chdir("../COMPARE_RESULTS/AFNOD")
AFNOD = pd.read_csv("RSM_MODULUS.csv")
AFNOD = AFNOD.values.astype(np.float32)[:, 1:]
data.append(AFNOD)
os.chdir(oldfolder)


os.chdir("../COMPARE_RESULTS")
Plot_Models(data)
Plot_bar(data)
os.chdir(oldfolder)
