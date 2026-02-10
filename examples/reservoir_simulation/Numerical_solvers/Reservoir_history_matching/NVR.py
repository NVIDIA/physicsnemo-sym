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

NVIDIA Finite volume reservoir simulator with flexible solver

AMG to solve the pressure and saturation well possed inverse problem

Geostatistics packages are also provided

@Author: Clement Etienam

"""

import os
import sys
import numpy as np


Yet = 0
import cupy as cp
from cupyx.scipy.sparse import spdiags
from numba import cuda

# print(cuda.detect())#Print the GPU information
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)
# import pyamgx
from cupyx.scipy.sparse.linalg import gmres, cg, spsolve, lsqr, LinearOperator, spilu
from cupyx.scipy.sparse import csr_matrix, spmatrix
from cupy import sparse

clementtt = 0


import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os.path

# import torch
from collections import OrderedDict
from scipy import interpolate
import multiprocessing
import mpslib as mps
import numpy.matlib
from pyDOE import lhs
import matplotlib.colors
from matplotlib import cm
from shutil import rmtree
import numpy

# from PIL import Image
from scipy.fftpack import dct
import numpy.matlib


# os.environ['KERAS_BACKEND'] = 'tensorflow'
import os.path
import random

# import dolfin as df
from numpy import *
import scipy.optimize.lbfgsb as lbfgsb
import numpy.linalg
from numpy.linalg import norm
from scipy.fftpack.realtransforms import idct
import numpy.ma as ma
import logging
import os
from FyeldGenerator import generate_field
import warnings
import yaml

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # I have just 1 GPU
from cpuinfo import get_cpu_info

# Prints a json string describing the cpu
s = get_cpu_info()
# print("Cpu info")
# for k,v in s.items():
#     print(f"\t{k}: {v}")
cores = multiprocessing.cpu_count()
import math

logger = logging.getLogger(__name__)
# numpy.random.seed(99)
# print(' ')
# print(' This computer has %d cores, which will all be utilised in parallel '%cores)
# print(' ')
# print('......................DEFINE SOME FUNCTIONS.....................')


def ShowBar(Bar):
    sys.stdout.write(Bar)
    sys.stdout.flush()


def peaks(n):
    """
    Mimic basic of matlab peaks fn
    """
    xp = np.arange(n)
    [x, y] = np.meshgrid(xp, xp)
    z = np.zeros_like(x).astype(float)
    for i in np.xrange(n / 5):
        x0 = random() * n
        y0 = random() * n
        sdx = random() * n / 4.0
        sdy = sdx
        c = random() * 2 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - ((x - x0) / sdx) * ((y - y0) / sdy) * c
        )
        # f /= f.sum()
        f *= random()
        z += f
    return z


def Plot_petrophysical(
    permmean, poroo, nx, ny, nz, Low_K, High_K, Low_P, High_P, injectors, producers
):
    Low_Ka = Low_K
    High_Ka = High_K

    permmean = np.reshape(permmean, (nx, ny, nz), "F")
    poroo = np.reshape(poroo, (nx, ny, nz), "F")

    from skimage.restoration import denoise_nl_means, estimate_sigma

    temp_K = permmean
    temp_phi = poroo
    # fast algorithm
    timk = 5
    for n in range(timk):
        sigma_est1 = np.mean(estimate_sigma(temp_K))
        sigma_est2 = np.mean(estimate_sigma(temp_phi))
        # print(f'estimated noise standard deviation for permeability = {sigma_est1}')
        # print('')
        # print(f'estimated noise standard deviation for porosity = {sigma_est2}')

        patch_kw = dict(patch_size=5, patch_distance=6)  # 5x5 patches
        temp_K = denoise_nl_means(
            temp_K, h=0.8 * sigma_est1, fast_mode=True, **patch_kw
        )
        temp_phi = denoise_nl_means(
            temp_phi, h=0.8 * sigma_est2, fast_mode=True, **patch_kw
        )

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(30, 30))

    plt.subplot(4, 3, 1)
    plt.pcolormesh(XX.T, YY.T, permmean[:, :, 0], cmap="jet")
    plt.title("Layer 1- Permeability", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 2)
    plt.pcolormesh(XX.T, YY.T, permmean[:, :, 1], cmap="jet")
    plt.title("Layer 2- Permeability", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 3)
    plt.pcolormesh(XX.T, YY.T, permmean[:, :, 2], cmap="jet")
    plt.title("Layer 3- Permeability", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 4)
    plt.pcolormesh(XX.T, YY.T, temp_K[:, :, 0], cmap="jet")
    plt.title("Smoothed - Permeability - Layer 1", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 5)
    plt.pcolormesh(XX.T, YY.T, temp_K[:, :, 1], cmap="jet")
    plt.title("Smoothed - Permeability - Layer 2", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 6)
    plt.pcolormesh(XX.T, YY.T, temp_K[:, :, 2], cmap="jet")
    plt.title("Smoothed - Permeability - Layer 2", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 7)
    plt.pcolormesh(XX.T, YY.T, poroo[:, :, 0], cmap="jet")
    plt.title("Porosity Layer 1", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Units", fontsize=13)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 8)
    plt.pcolormesh(XX.T, YY.T, poroo[:, :, 1], cmap="jet")
    plt.title("Porosity Layer 2", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Units", fontsize=13)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 9)
    plt.pcolormesh(XX.T, YY.T, poroo[:, :, 2], cmap="jet")
    plt.title("Porosity Layer 3", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Units", fontsize=13)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 10)
    plt.pcolormesh(XX.T, YY.T, temp_phi[:, :, 0], cmap="jet")
    plt.title("Smoothed Porosity -Layer 1", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Units", fontsize=13)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 11)
    plt.pcolormesh(XX.T, YY.T, temp_phi[:, :, 1], cmap="jet")
    plt.title("Smoothed Porosity -Layer 2", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Units", fontsize=13)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(4, 3, 12)
    plt.pcolormesh(XX.T, YY.T, temp_phi[:, :, 2], cmap="jet")
    plt.title("Smoothed Porosity -Layer 3", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Units", fontsize=13)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Petrophysical Reconstruction", fontsize=25)
    plt.savefig("Petro_Recon.png")
    plt.close()
    plt.clf()


def Plot_mean(
    permbest,
    permmean,
    iniperm,
    nx,
    ny,
    nz,
    Low_K,
    High_K,
    True_perm,
    injectors,
    producers,
):
    Low_Ka = Low_K
    High_Ka = High_K

    permmean = np.reshape(permmean, (nx, ny, nz), "F")
    permbest = np.reshape(permbest, (nx, ny, nz), "F")
    iniperm = np.reshape(iniperm, (nx, ny, nz), "F")
    True_perm = np.reshape(True_perm, (nx, ny, nz), "F")
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(30, 30))

    plt.subplot(3, 4, 3)
    # plt.pcolormesh(XX.T,YY.T,permmean[:,:,0],cmap = 'jet')
    plt.pcolormesh(XX.T, YY.T, permmean[:, :, 0], cmap="jet")
    plt.title(" Layer 1 - mean", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 7)
    # plt.pcolormesh(XX.T,YY.T,permmean[:,:,0],cmap = 'jet')
    plt.pcolormesh(XX.T, YY.T, permmean[:, :, 1], cmap="jet")
    plt.title("Layer 2 - mean", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 11)
    # plt.pcolormesh(XX.T,YY.T,permmean[:,:,0],cmap = 'jet')
    plt.pcolormesh(XX.T, YY.T, permmean[:, :, 2], cmap="jet")
    plt.title("Layer 3 - mean", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 2)
    plt.pcolormesh(XX.T, YY.T, permbest[:, :, 0], cmap="jet")
    plt.title(" Layer 1 - Best", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 6)
    plt.pcolormesh(XX.T, YY.T, permbest[:, :, 1], cmap="jet")
    plt.title(" Layer 2 - Best", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 10)
    plt.pcolormesh(XX.T, YY.T, permbest[:, :, 2], cmap="jet")
    plt.title(" Layer 3 - Best", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 1)
    plt.pcolormesh(XX.T, YY.T, iniperm[:, :, 0], cmap="jet")
    plt.title(" Layer 1 - initial", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 5)
    plt.pcolormesh(XX.T, YY.T, iniperm[:, :, 1], cmap="jet")
    plt.title(" Layer 2 - initial", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 9)
    plt.pcolormesh(XX.T, YY.T, iniperm[:, :, 2], cmap="jet")
    plt.title(" Layer 3 - initial", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 4)
    plt.pcolormesh(XX.T, YY.T, True_perm[:, :, 0], cmap="jet")
    plt.title(" Layer 1 - TRue", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 8)
    plt.pcolormesh(XX.T, YY.T, True_perm[:, :, 1], cmap="jet")
    plt.title(" Layer 2 - TRue", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 4, 12)
    plt.pcolormesh(XX.T, YY.T, True_perm[:, :, 2], cmap="jet")
    plt.title(" Layer 3 - TRue", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Permeability comparison", fontsize=25)
    plt.savefig("Comparison.png")
    plt.close()
    plt.clf()


def read_yaml(fname):
    """Read Yaml file into a dict of parameters"""
    print(f"Read simulation plan from {fname}...")
    with open(fname, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            # print(data)
        except yaml.YAMLError as exc:
            print(exc)
        return data


def field2Metric(input, key):
    """
    Converts field (imperial) units to metric units
    """
    switcher = {
        "psi": input * 6894.757293168,  # psi to Pa
        "lbft3": input * 16.0184634,  # lb/ft3 to kg/m3
        "ms": input * 3.2808399,  # m/s to ft/s
        "ft": input / 3.2808399,  # ft/s to m/s
    }
    return switcher.get(key, "Please choose a correct unit!")


def Gassmann(PORO, Pr, SO, nx, ny, nz):
    """
    Calculates the P- and S-wave velocities, and their corresponding acoustic impedances, using Gassmann's equations.

    Args:
    - PORO (numpy array): array of shape (nx, ny, nz) containing the porosity values of the rock.
    - Pr (numpy array): array of shape (nx, ny, nz) containing the pressure values in psi.
    - SO (numpy array): array of shape (nx, ny, nz) containing the oil saturation values in eclipse format.
    - nx (int): number of grid blocks in the x-direction.
    - ny (int): number of grid blocks in the y-direction.
    - nz (int): number of grid blocks in the z-direction.

    Returns:
    - ImpP (numpy array): array of shape (nx, ny, nz) containing the P-wave acoustic impedance values in g/cm^2/s.
    - ImpS (numpy array): array of shape (nx, ny, nz) containing the S-wave acoustic impedance values in g/cm^2/s.
    - VP (numpy array): array of shape (nx, ny, nz) containing the P-wave velocity values in ft/s.
    - VS (numpy array): array of shape (nx, ny, nz) containing the S-wave velocity values in ft/s.
    """

    # Poro - porosity (array shape)
    # P - pressure (psi, eclipse formatted)
    # SO - oil saturation (eclispe formatted)

    T = 103  # C
    POROVANCOUVER = PORO

    # sorting out indices
    porosity = [cp.empty((nx, ny), dtype=cp.float64) for _ in range(nz)]

    for i in range(nz):
        porosity[i] = POROVANCOUVER[:, :, i]

    # Read in pressure and oil saturation files

    PressureAfter1Year = Pr
    OilSatAfter1Year = SO

    # labeling the imported data
    pressure = [cp.empty((nx, ny), dtype=cp.float64) for _ in range(nz)]
    saturation = [cp.empty((nx, ny), dtype=cp.float64) for _ in range(nz)]

    for i in range(nz):
        # psia to MPa
        pressure[i] = field2Metric(PressureAfter1Year[:, :, i], "psi") * 1e-6

        saturation[i] = OilSatAfter1Year[:, :, i]

    # Water data

    # Compressibility
    CWater = 3.13e-6  # 1/psi

    # Density
    rhoWater = field2Metric(64.00, "lbft3")  # kg/m3

    # Oil data
    # Gas Specific gravity
    G = 0.8515

    # 600 SCF/BBL ->  m3/m3
    RG = 0  # 600*0.0283168466/0.158987295;

    # Oil API
    API = 141.5 / G - 131.5  # need rhoOil in g/cm^3

    # Matrix data
    # From permeability - porosity graphs, assumed made of quartz and feldspar
    # quartz dominant

    # Density
    # From Carmichael (1986)
    # Try different percentage of quartz
    SQuartz = 0.6
    SFeldspar = 1 - SQuartz
    KQuartz = 37e9  # Pa
    KFeldspar = 37.5e9  # Pa
    KMatrix = (
        cp.ones((nx, ny)) * (SQuartz * KQuartz + SFeldspar * KFeldspar) / 2
        + (SQuartz / KQuartz + SFeldspar / KFeldspar) ** (-1) / 2
    )

    # Frame data
    # Lab data
    rhoDry = cp.ones((nx, ny)) * 2169
    KDry = cp.ones((nx, ny)) * field2Metric(2e6, "psi")
    GDry = cp.ones((nx, ny)) * field2Metric(1.368e6, "psi")

    # Calculate parameters from input data for each z value
    KSat = [cp.empty((nx, ny), dtype=cp.float64) for _ in range(nz)]
    rhoSat = [cp.empty((nx, ny), dtype=cp.float64) for _ in range(nz)]

    for i in range(nz):
        P = pressure[i]
        SOil = saturation[i]
        phi = porosity[i]

        # Oil data (Batzle and Wang)
        rho0 = 141.5 / (API + 131.5)  # g/cm3

        # volume formation factor
        B0 = 0.972 + 0.00038 * (2.4 * RG * cp.sqrt(G / rho0) + T + 17.8) ** (1.175)
        # pseudo density
        rhopseudo = rho0 / B0 * (1 + 0.001 * RG) ** -1  # g/cm3

        # density of oil with gas
        rhoG = (rho0 + 0.0012 * G * RG) / B0  # g/cm3

        # density corrected for pressure and temperature
        rhoOil = (
            1000
            * (rho0 + (0.00277 * P - 1.71e-7 * P**3) * (rhoG - 1.15) ** 2 + P * 3.49e-4)
            / (0.972 + 3.81e-4 * (T + 17.78) ** 1.175)
        )  # kg/m3

        # Oil velocity from API (Batzle and Wang)
        # Simplified version of the equation in the report
        VOil = (
            2096 * (rhopseudo / (2.6 - rhopseudo)) ** 0.5
            - 3.7 * T
            + 4.64 * P
            + 0.0115 * T * P * (4.12 * (1.08 / rhopseudo - 1) ** 0.5 - 1)
        )  # m/s

        # Bulk physicsnemo of oil
        KOil = rhoOil * VOil**2  # Pa

        # Bulk physicsnemo of water
        KWater = field2Metric(1.0 / CWater, "psi")  # Pa

        # Woodcock's equation
        KFluid = ((1 - SOil) / KWater + SOil / KOil) ** (-1)  # Pa
        rhoFluid = (1 - SOil) * rhoWater + SOil * rhoOil  # kg/m3

        # Gassmann equations (in SI units)Dry,shape
        KSat[i] = KDry + (1 - KDry / KMatrix) ** 2 / (
            phi / KFluid + (1 - phi) / KMatrix - KDry / KMatrix**2
        )
        GSat = GDry

        # Density equation
        rhoSat[i] = rhoDry + phi * rhoFluid

    # Backus Average
    D = cp.empty((nx, ny), dtype=cp.float64)
    C = cp.empty((nx, ny), dtype=cp.float64)
    rho = cp.empty((nx, ny), dtype=cp.float64)
    for i in range(nx):
        for j in range(ny):
            element = []
            density = []
            for k in range(nz):
                element.append(1 / (KSat[k][i, j] + 4 / 3 * GSat[i, j]))
                density.append(rhoSat[k][i, j])
            D[i, j] = GSat[i, j]
            C[i, j] = cp.mean(cp.asarray(element)) ** (-1)
            rho[i, j] = cp.mean(cp.asarray(density))

    # P and S wave velocities
    VP = (C / rho) ** 0.5  # m/s
    VS = (D / rho) ** 0.5

    # Seismic data (output)
    VP = field2Metric(VP, "ms")  # ft/s
    VS = field2Metric(VS, "ms")

    rho = rho / 1000  # g/cm3

    # Impedance
    ImpP = rho * VP
    ImpS = rho * VS
    return ImpP, ImpS, VP, VS


def simulator_to_python(a):
    kk = a.shape[2]
    anew = []
    for i in range(kk):
        afirst = a[:, :, i]
        afirst = afirst.T
        afirst = cp.reshape(afirst, (-1, 1), "F")
        anew.append(afirst)
    return cp.vstack(anew)


def python_to_simulator(a, ny, nx, nz):
    a = cp.reshape(a, (-1, 1), "F")
    a = cp.reshape(a, (ny, nx, nz), "F")
    anew = []
    for i in range(nz):
        afirst = a[:, :, i]
        afirst = afirst.T
        anew.append(afirst)
    return cp.vstack(anew)


def Add_marker(plt, XX, YY, locc):
    """
    Function to add marker to given coordinates on a matplotlib plot

    less
    Copy code
    Parameters:
        plt: a matplotlib.pyplot object to add the markers to
        XX: a numpy array of X coordinates
        YY: a numpy array of Y coordinates
        locc: a numpy array of locations where markers need to be added

    Return:
        None
    """
    # iterate through each location
    for i in range(locc.shape[0]):
        a = locc[i, :]
        xloc = int(a[0])
        yloc = int(a[1])

        # if the location type is 2, add an upward pointing marker
        if a[2] == 2:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="^",
                color="white",
            )
        # otherwise, add a downward pointing marker
        else:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="v",
                color="white",
            )


def Add_marker2(plt, XX, YY, injectors, producers):
    """
    Function to add marker to given coordinates on a matplotlib plot

    less
    Copy code
    Parameters:
        plt: a matplotlib.pyplot object to add the markers to
        XX: a numpy array of X coordinates
        YY: a numpy array of Y coordinates
        locc: a numpy array of locations where markers need to be added

    Return:
        None
    """

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])

        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="^",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )


def residual(A, b, x):
    return b - A @ x


@cuda.jit
def prolongation_kernel(x, fine_x, fine_grid_size):
    i = cuda.grid(1)
    if i < fine_grid_size:
        fine_x[i] = x[i // 2]


def prolongation(x, fine_grid_size):
    fine_x = cp.zeros((fine_grid_size,), dtype=x.dtype)
    blocks = (
        fine_grid_size + cuda.get_current_device().WARP_SIZE - 1
    ) // cuda.get_current_device().WARP_SIZE
    threads = cuda.get_current_device().WARP_SIZE
    prolongation_kernel[blocks, threads](x, fine_x, fine_grid_size)
    return fine_x


def v_cycle(A, b, x, smoother, levels, tol, smoothing_steps):
    """
    Function to perform V-cycle multigrid method for solving a linear system of equations Ax=b
    Parameters:
        A: a cupyx.scipy.sparse CSR matrix representing the system matrix A
        b: a cupy ndarray representing the right-hand side vector b
        x: a cupy ndarray representing the initial guess for the solution vector x
        smoother: a string representing the type of smoothing method to use
                  (possible values: 'jacobi', 'gauss-seidel', 'SOR')
        levels: an integer representing the number of levels in the multigrid method
        tol: a float representing the tolerance for the residual norm
        smoothing_steps: an integer representing the number of smoothing steps to perform at each level
        color: a boolean representing whether to use coloring for Gauss-Seidel smoother

    Return:
        x: a cupy ndarray representing the solution vector x
    """

    # iterate through each level
    for level in range(levels):
        # if on finest level, solve exactly
        if level == levels - 1:
            # solve exactly on finest grid
            A = check_cupy_sparse_matrix(A)
            x = spsolve(A, b)

            # check if tolerance is met
            if ((cp.sum(b - A @ x)) / b.shape[0]) < tol:
                break
            return x
        else:
            # perform smoothing
            for i in range(smoothing_steps):
                if smoother == "jacobi":
                    x, _ = jacobi(A, b, x, omega=1.0, tol=1e-6, max_iters=100)
                    r = residual(A, b, x)
                elif smoother == "gauss-seidel":
                    x = gauss_seidel(A, b, x, omega=1.0, tol=1e-6, max_iters=100)
                    r = residual(A, b, x)
                elif smoother == "SOR":
                    x = SOR(A, b, omega=1.5, tol=1e-6, max_iter=100)
                    r = residual(A, b, x)
                else:
                    r = residual(A, b, x)
                    x += spsolve(A, r)

            # restrict residual to coarser grid
            coarse_A, coarse_r = restriction(A, r)
            coarse_A = check_cupy_sparse_matrix(coarse_A)

            # solve exactly on coarser grid
            coarse_x = v_cycle(
                coarse_A,
                coarse_r,
                cp.zeros_like(coarse_r),
                smoother,
                levels - 1,
                tol,
                smoothing_steps,
            )

            # interpolate solution back to fine grid
            x += prolongation(coarse_x, fine_grid_size=A.shape[0])

    return x


def gauss_seidel(A, b, x0, omega=1.0, tol=1e-6, max_iters=100):
    """
    Gauss-Seidel method with overrelaxation for solving linear system Ax=b.
    Parameters:
        A: a cupy.sparse matrix representing the system matrix A
        b: a cupy.ndarray representing the right-hand side vector b
        x0: a cupy.ndarray representing the initial guess for the solution vector x
        omega: a float representing the relaxation factor (default=1.0)
        tol: a float representing the tolerance for the residual norm (default=1e-8)
        max_iters: an integer representing the maximum number of iterations (default=1000)
    Returns:
        x: a cupy.ndarray representing the solution vector x
    """
    x = cp.copy(x0)
    residual_norm = tol + 1.0  # initialize with a value larger than the tolerance
    iters = 0
    while residual_norm > tol and iters < max_iters:
        dot_products = A @ x
        x_new = (1 - omega) * x + omega * (
            b - dot_products + A.diagonal() * x
        ) / A.diagonal()
        residual_norm = cp.linalg.norm(A @ x_new - b)
        x = x_new
        iters += 1
    return x


def jacobi(A, b, x, omega=1.0, tol=1e-6, max_iters=100):
    """
    Jacobi iteration for solving linear systems.

    Parameters
    ----------
    A : cupy.sparse.csr_matrix, shape (N, N)
        The coefficient matrix.
    b : cupy.ndarray, shape (N,)
        The right-hand side vector.
    x : cupy.ndarray, shape (N,)
        The initial guess for the solution.
    omega : float, optional
        The relaxation parameter. Default is 1.0 (no relaxation).
    tol : float, optional
        The tolerance for convergence. Default is 1e-6.
    max_iters : int, optional
        The maximum number of iterations. Default is 1000.

    Returns
    -------
    x : cupy.ndarray, shape (N,)
        The solution.
    iters : int
        The number of iterations performed.
    """
    D = A.diagonal()
    R = A - cp.diagflat(D)
    iters = 0
    res = 1.0
    while res > tol and iters < max_iters:
        x_new = (b - R @ x) / D
        x = x + omega * (x_new - x)
        res = cp.linalg.norm(b - A @ x)
        iters += 1
    return x, iters


def SOR(A, b, omega=1.5, tol=1e-6, max_iter=100):
    """
    Use successive over-relaxation to solve Ax=b.

    Parameters
    ----------
    A : cupy.sparse.csr_matrix, shape (N, N)
        The coefficient matrix.
    b : cupy.ndarray, shape (N,)
        The right-hand side vector.
    omega : float, optional
        The relaxation parameter. Default is 1.5.
    tol : float, optional
        The convergence tolerance. Default is 1e-4.
    max_iter : int, optional
        The maximum number of iterations. Default is 1000.

    Returns
    -------
    x : cupy.ndarray, shape (N,)
        The solution vector.
    """

    N = A.shape[0]
    x = cp.zeros(N, dtype=cp.float32)
    omega_a = omega - 1

    # Compute the inverse of the diagonal
    D_inv = cp.reciprocal(A.diagonal())

    for k in range(max_iter):
        # Perform one SOR iteration
        x_new = x + omega_a * x
        x_new += omega * D_inv * (b - A @ x_new)

        # Compute the residual
        res = cp.linalg.norm(b - A @ x_new, ord=2)

        # Check for convergence
        if res < tol:
            break

        # Update x for the next iteration
        x = x_new

    return x_new


def check_cupy_sparse_matrix(A):
    """
    Function to check if a matrix is a Cupy sparse matrix and convert it to a CSR matrix if necessary

    Parameters:
        A: a sparse matrix

    Return:
        A: a CSR matrix
    """

    if not isinstance(A, spmatrix):
        # Convert the matrix to a csr matrix if it is not already a cupy sparse matrix
        A = csr_matrix(A)
    return A


def restriction(A, f):
    """
    Coarsen the matrix A and vector f with aggregation.

    Parameters
    ----------
    A : cupy.sparse.csr_matrix, shape (N, N)
        The coefficient matrix.
    f : cupy.ndarray, shape (N,)
        The right-hand side vector.

    Returns
    -------
    A_coarse : cupy.sparse.csr_matrix, shape (N_coarse, N_coarse)
        The coarsened coefficient matrix.
    f_coarse : cupy.ndarray, shape (N_coarse,)
        The coarsened right-hand side vector.
    """

    N = A.shape[0]

    # Divide the nodes into aggregates
    aggregate_size = 2
    num_aggregates = N // aggregate_size
    aggregates = cp.arange(N) // aggregate_size

    # Compute the indices of the first node in each aggregate
    unique_aggregates, first_node_indices = cp.unique(aggregates, return_index=True)
    reps = cp.zeros((num_aggregates,), dtype=cp.int32)
    reps[unique_aggregates] = first_node_indices

    # Create the aggregation matrix P
    node_counts = cp.diff(A.indptr)  # get number of nonzero entries in each row
    node_counts = cp.clip(node_counts, a_min=1, a_max=None)  # avoid divide-by-zero
    P = cp.zeros((N, num_aggregates), dtype=cp.float32)
    P[cp.arange(N), aggregates] = 1.0 / node_counts[cp.arange(N)]
    P[reps, cp.arange(num_aggregates)] = 1.0

    # Compute the level schedule
    level_schedule = []
    current_level = cp.where(node_counts == 1)[0]  # nodes with degree 1
    while len(current_level) > 0:
        level_schedule.append(current_level)
        neighbors = cp.unique(A[current_level, :].indices)
        next_level = cp.setdiff1d(neighbors, current_level)
        current_level = next_level
    level_schedule.append(cp.arange(N)[node_counts > 1])  # nodes with degree > 1

    # Coarsen the matrix and vector
    A_coarse = A
    f_coarse = f
    for level in level_schedule:
        A_coarse = P[level, :].T @ A_coarse @ P[level, :]
        f_coarse = P[level, :].T @ f_coarse

    return A_coarse, f_coarse


def Plot_RSM_percentilee(pertoutt, True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    P10 = pertoutt[0]
    P50 = pertoutt[1]
    P90 = pertoutt[2]
    arekibest = pertoutt[3]
    # arekimean =  pertoutt[4]

    plt.figure(figsize=(40, 40))

    plt.subplot(4, 4, 1)
    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 1], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 1], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 1], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 1], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 2)
    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 2], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 2], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 2], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 2], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 3)
    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 3], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 3], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 3], color="green", lw="2", label="P90 Model")
    plt.plot(timezz, arekibest[:, 3], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 4)
    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 4], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 4], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 4], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 4], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 5)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 5], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 5], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 5], color="green", lw="2", label="P90 Model")
    plt.plot(timezz, arekibest[:, 5], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 6)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 6], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 6], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 6], color="green", lw="2", label="P90 Model")
    plt.plot(timezz, arekibest[:, 6], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 7)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 7], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 7], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 7], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 7], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 8)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 8], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 8], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 8], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 8], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 9)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 9], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 9], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 9], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 9], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 10)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 10], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 10], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 10], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 10], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 11)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 11], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 11], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 11], color="green", lw="2", label="P90 Model")
    plt.plot(timezz, arekibest[:, 11], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 12)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 12], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 12], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 12], color="green", lw="2", label="P90 Model")
    plt.plot(timezz, arekibest[:, 12], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 13)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 13], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 13], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 13], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 13], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 14)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 14], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 14], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 14], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 14], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 15)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 15], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 15], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 15], color="green", lw="2", label="P90 Model")

    plt.plot(timezz, arekibest[:, 15], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 16)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 16], color="blue", lw="2", label="P10 Model")
    plt.plot(timezz, P50[:, 16], color="c", lw="2", label="P50 Model")
    plt.plot(timezz, P90[:, 16], color="green", lw="2", label="P90 Model")
    plt.plot(timezz, arekibest[:, 16], color="k", lw="2", label="aREKI cum best Model")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        Namesz
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM_percentile(True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    plt.figure(figsize=(40, 40))

    plt.subplot(4, 4, 1)
    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 2)
    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 3)
    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 4)
    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 5)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 6)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 7)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 8)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 9)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 10)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 11)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 12)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 13)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 14)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 15)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 16)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()
    plt.suptitle("FIELD PRODUCTION PROFILE", fontsize=25)

    # os.chdir('RESULTS')
    plt.savefig(
        Namesz
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM_percentile2(True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    plt.figure(figsize=(40, 40))

    plt.subplot(5, 4, 1)
    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 2)
    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 3)
    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 4)
    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 5)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 6)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 7)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 8)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 9)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 10)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 11)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 12)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 13)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 14)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 15)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 16)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 17)
    plt.plot(timezz, True_mat[:, 17], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 18)
    plt.plot(timezz, True_mat[:, 18], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 19)
    plt.plot(timezz, True_mat[:, 19], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(5, 4, 20)
    plt.plot(timezz, True_mat[:, 20], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.suptitle("FIELD PRODUCTION PROFILE", fontsize=25)
    # os.chdir('RESULTS')
    plt.savefig(
        Namesz
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM_single(True_mat, Namesz):
    True_mat = True_mat[0]
    timezz = True_mat[:, 0].reshape(-1, 1)

    plt.figure(figsize=(40, 40))

    plt.subplot(4, 4, 1)
    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 2)
    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 3)
    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 4)
    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 5)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 6)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 7)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 8)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 9)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 10)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 11)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 12)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 13)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 14)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 15)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 16)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        Namesz
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM_singleT(True_mat, Namesz):
    # True_mat = True_mat[0]
    timezz = True_mat[:, 0].reshape(-1, 1)

    plt.figure(figsize=(40, 40))

    plt.subplot(4, 4, 1)
    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 2)
    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 3)
    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 4)
    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 5)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 6)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 7)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 8)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 9)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 10)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 11)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 12)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 13)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 14)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 15)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 16)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        Namesz
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM(predMatrix, True_mat, Namesz, Ne):
    timezz = True_mat[:, 0].reshape(-1, 1)

    Nt = predMatrix[0].shape[0]
    timezz = predMatrix[0][:, 0].reshape(-1, 1)

    BHPA = np.zeros((Nt, Ne))
    BHPB = np.zeros((Nt, Ne))
    BHPC = np.zeros((Nt, Ne))
    BHPD = np.zeros((Nt, Ne))

    WOPRA = np.zeros((Nt, Ne))
    WOPRB = np.zeros((Nt, Ne))
    WOPRC = np.zeros((Nt, Ne))
    WOPRD = np.zeros((Nt, Ne))

    WWPRA = np.zeros((Nt, Ne))
    WWPRB = np.zeros((Nt, Ne))
    WWPRC = np.zeros((Nt, Ne))
    WWPRD = np.zeros((Nt, Ne))

    WWCTA = np.zeros((Nt, Ne))
    WWCTB = np.zeros((Nt, Ne))
    WWCTC = np.zeros((Nt, Ne))
    WWCTD = np.zeros((Nt, Ne))

    for i in range(Ne):
        usef = predMatrix[i]

        BHPA[:, i] = usef[:, 1]
        BHPB[:, i] = usef[:, 2]
        BHPC[:, i] = usef[:, 3]
        BHPD[:, i] = usef[:, 4]

        WOPRA[:, i] = usef[:, 5]
        WOPRB[:, i] = usef[:, 6]
        WOPRC[:, i] = usef[:, 7]
        WOPRD[:, i] = usef[:, 8]

        WWPRA[:, i] = usef[:, 9]
        WWPRB[:, i] = usef[:, 10]
        WWPRC[:, i] = usef[:, 11]
        WWPRD[:, i] = usef[:, 12]

        WWCTA[:, i] = usef[:, 13]
        WWCTB[:, i] = usef[:, 14]
        WWCTC[:, i] = usef[:, 15]
        WWCTD[:, i] = usef[:, 16]

    plt.figure(figsize=(40, 40))

    plt.subplot(4, 4, 1)
    plt.plot(timezz, BHPA[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    plt.title("I1", fontsize=13)

    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 2)
    plt.plot(timezz, BHPB[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    plt.title("I2", fontsize=13)

    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 3)
    plt.plot(timezz, BHPC[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    plt.title("I3", fontsize=13)

    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 4)
    plt.plot(timezz, BHPD[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    plt.title("I4", fontsize=13)

    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 5)
    plt.plot(timezz, WOPRA[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title("P1", fontsize=13)

    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 6)
    plt.plot(timezz, WOPRB[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title("P2", fontsize=13)

    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 7)
    plt.plot(timezz, WOPRC[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title("P3", fontsize=13)

    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 8)
    plt.plot(timezz, WOPRD[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title("P4", fontsize=13)

    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 9)
    plt.plot(timezz, WWPRA[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title("P1", fontsize=13)

    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 10)
    plt.plot(timezz, WWPRB[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title("P2", fontsize=13)

    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 11)
    plt.plot(timezz, WWPRC[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title("P3", fontsize=13)

    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 12)
    plt.plot(timezz, WWPRD[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title("P4", fontsize=13)

    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 13)
    plt.plot(timezz, WWCTA[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    plt.title("P1", fontsize=13)

    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 14)
    plt.plot(timezz, WWCTB[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    plt.title("P1", fontsize=13)

    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 15)
    plt.plot(timezz, WWCTC[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    plt.title("P3", fontsize=13)

    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(4, 4, 16)
    plt.plot(timezz, WWCTD[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    plt.title("P4", fontsize=13)

    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    # os.chdir('RESULTS')
    plt.savefig(
        Namesz
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM_percentile_model(pertoutt, True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    P10 = pertoutt

    plt.figure(figsize=(40, 40))

    plt.subplot(4, 4, 1)
    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 1], color="blue", lw="2", label="aREKI")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 2)
    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 2], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 3)
    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 3], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 4)
    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 4], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 5)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 5], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 6)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 6], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 7)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 7], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 8)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 8], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 9)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 9], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 10)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 10], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 11)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 11], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 12)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 12], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 13)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 13], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 14)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 14], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 15)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 15], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 16)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 16], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        Namesz
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def plot_properties(perm, poro, nx, ny, nz, injectors, producers, path_save):
    if nz == 1:
        permeability = np.reshape(perm, (nx, ny), "F")
        porosity = np.reshape(poro, (nx, ny), "F")

        permeability = cp.asnumpy(permeability)
        porosity = cp.asnumpy(porosity)

        XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.pcolormesh(XX.T, YY.T, permeability, cmap="jet")

        plt.title("permeability ", fontsize=15)
        plt.ylabel("Y", fontsize=13)
        plt.xlabel("X", fontsize=13)
        plt.axis([0, (nx - 1), 0, (ny - 1)])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        cbar1 = plt.colorbar()
        cbar1.ax.set_ylabel(" K (mD)", fontsize=13)
        plt.clim(min(cp.ravel(permeability)), max(cp.ravel(permeability)))
        Add_marker2(plt, XX, YY, injectors, producers)

        plt.subplot(2, 2, 2)
        plt.pcolormesh(XX.T, YY.T, porosity, cmap="jet")
        # Add_marker2(plt,XX,YY,injectors,producers)
        plt.title("porosity ", fontsize=15)
        plt.ylabel("Y", fontsize=13)
        plt.xlabel("X", fontsize=13)
        plt.axis([0, (nx - 1), 0, (ny - 1)])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        cbar1 = plt.colorbar()
        cbar1.ax.set_ylabel(" K (mD)", fontsize=13)
        plt.clim(min(cp.ravel(porosity)), max(cp.ravel(porosity)))
        Add_marker2(plt, XX, YY, injectors, producers)
        plt.savefig(os.path.join(path_save, "properties.png"))
        plt.clf()
        plt.close()
    else:
        permeability = np.reshape(perm, (nx, ny, nz), "F")

        porosity = np.reshape(poro, (nx, ny, nz), "F")

        permeability = cp.asnumpy(permeability)
        porosity = cp.asnumpy(porosity)

        XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

        plt.figure(figsize=(12, 12))

        for i in range(nz):
            plt.subplot(2, 3, i + 1)
            plt.pcolormesh(XX.T, YY.T, permeability[:, :, i], cmap="jet")
            title = "Perm_Layer_" + str(i + 1)
            plt.title(title, fontsize=15)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (nx - 1), 0, (ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" K (mD)", fontsize=13)
            plt.clim(min(cp.ravel(permeability)), max(cp.ravel(permeability)))
            Add_marker2(plt, XX, YY, injectors, producers)
        plt.savefig(os.path.join(path_save, "properties_perm.png"))
        plt.clf()
        plt.close()

        plt.figure(figsize=(12, 12))
        for i in range(nz):
            plt.subplot(2, 3, i + 1)
            plt.pcolormesh(XX.T, YY.T, porosity[:, :, i], cmap="jet")
            title = "Perm_Layer_" + str(i + 1)
            plt.title(title, fontsize=15)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (nx - 1), 0, (ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" units", fontsize=13)
            plt.clim(min(cp.ravel(porosity)), max(cp.ravel(porosity)))
            Add_marker2(plt, XX, YY, injectors, producers)
        plt.savefig(os.path.join(path_save, "properties_porosity.png"))
        plt.clf()
        plt.close()


# def print_section_title(text: str) -> None:
#     print('\n# ----------------------------------------')
#     print(f'# {text.upper()}')
#     print('# ----------------------------------------')


def Plot_performance(
    trueF, nx, ny, nz, namet, itt, dt, MAXZ, steppi, injectors, producers
):
    lookf = trueF[itt, :, :]
    lookf_sat = trueF[itt + steppi, :, :]
    lookf_oil = 1 - lookf_sat

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, lookf, cmap="jet")
    plt.title("Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, lookf_sat, cmap="jet")
    plt.title("water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(2, 2, 3)
    plt.pcolormesh(XX.T, YY.T, lookf_oil, cmap="jet")
    plt.title("oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = namet + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()


def Plot_performance2(
    trueF, nx, ny, nz, namet, itt, dt, MAXZ, steppi, injectors, producers
):
    lookf = trueF[itt, :, :]
    lookf_sat = trueF[itt + steppi, :, :]
    lookf_oil = trueF[itt + 2 * steppi, :, :]
    lookf_gas = 1 - (lookf_sat + lookf_oil)

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, lookf, cmap="jet")
    plt.title("Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, lookf_sat, cmap="jet")
    plt.title("water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(2, 2, 3)
    plt.pcolormesh(XX.T, YY.T, lookf_oil, cmap="jet")
    plt.title("oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(2, 2, 4)
    plt.pcolormesh(XX.T, YY.T, lookf_gas, cmap="jet")
    plt.title("gas_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" gas sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = namet + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()


# Geostatistics module


def intial_ensemble(Nx, Ny, Nz, N, permx):
    """
    Geostatistics module
    Function to generate an initial ensemble of permeability fields using Multiple-Point Statistics (MPS)
    Parameters:
        Nx: an integer representing the number of grid cells in the x-direction
        Ny: an integer representing the number of grid cells in the y-direction
        Nz: an integer representing the number of grid cells in the z-direction
        N: an integer representing the number of realizations in the ensemble
        permx: a numpy array representing the permeability field TI

    Return:
        ensemble: a numpy array representing the ensemble of permeability fields
    """

    # import MPSlib
    O = mps.mpslib()

    # set the MPS method to 'mps_snesim_tree'
    O = mps.mpslib(method="mps_snesim_tree")

    # set the number of realizations to N
    O.par["n_real"] = N

    # set the permeability field TI
    k = permx
    kjenn = k
    O.ti = kjenn

    # set the simulation grid size
    O.par["simulation_grid_size"] = (Ny, Nx, Nz)

    # run MPS simulation in parallel
    O.run_parallel()

    # get the ensemble of realizations
    ensemble = O.sim

    # reformat the ensemble
    ens = []
    for kk in range(N):
        temp = np.reshape(ensemble[kk], (-1, 1), "F")
        ens.append(temp)
    ensemble = np.hstack(ens)

    # remove temporary files generated during MPS simulation
    from glob import glob

    for f3 in glob("thread*"):
        rmtree(f3)

    for f4 in glob("*mps_snesim_tree_*"):
        os.remove(f4)

    for f4 in glob("*ti_thread_*"):
        os.remove(f4)

    return ensemble


def initial_ensemble_gaussian(Nx, Ny, Nz, N, minn, maxx):
    """
    Function to generate an initial ensemble of permeability fields using Gaussian distribution
    Parameters:
        Nx: an integer representing the number of grid cells in the x-direction
        Ny: an integer representing the number of grid cells in the y-direction
        Nz: an integer representing the number of grid cells in the z-direction
        N: an integer representing the number of realizations in the ensemble
        minn: a float representing the minimum value of the permeability field
        maxx: a float representing the maximum value of the permeability field

    Return:
        fensemble: a numpy array representing the ensemble of permeability fields
    """

    shape = (Nx, Ny)
    distrib = "gaussian"

    fensemble = np.zeros((Nx * Ny * Nz, N))

    for k in range(N):
        fout = []

        # generate a 3D field
        for j in range(Nz):
            field = generate_field(distrib, Pkgen(3), shape)
            field = imresize(field, output_shape=shape)
            foo = np.reshape(field, (-1, 1), "F")
            fout.append(foo)

        fout = np.vstack(fout)

        # scale the field to the desired range
        clfy = MinMaxScaler(feature_range=(minn, maxx))
        (clfy.fit(fout))
        fout = clfy.transform(fout)

        fensemble[:, k] = np.ravel(fout)

    return fensemble


def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk


# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


def Peaceman_well(
    inn,
    ooutp,
    oouts,
    MAXZ,
    mazw,
    s1,
    DX,
    steppi,
    pini_alt,
    SWI,
    SWR,
    UW,
    BW,
    DZ,
    UO,
    BO,
    dt,
    N_inj,
    N_pr,
    nz,
    NecessaryI,
    NecessaryP,
):
    """
    Calculates the pressure and flow rates for an injection and production well using the Peaceman model.

    Args:
    - inn (dictionary): dictionary containing the input parameters (including permeability and injection/production rates)
    - ooutp (numpy array): 4D numpy array containing pressure values for each time step and grid cell
    - oouts (numpy array): 4D numpy array containing saturation values for each time step and grid cell
    - MAXZ (float): length of the reservoir in the z-direction
    - mazw (float): the injection/production well location in the z-direction
    - s1 (float): the length of the computational domain in the z-direction
    - LUB (float): the upper bound of the rescaled permeability
    - HUB (float): the lower bound of the rescaled permeability
    - aay (float): the upper bound of the original permeability
    - bby (float): the lower bound of the original permeability
    - DX (float): the cell size in the x-direction
    - steppi (int): number of time steps
    - pini_alt (float): the initial pressure
    - SWI (float): the initial water saturation
    - SWR (float): the residual water saturation
    - UW (float): the viscosity of water
    - BW (float): the formation volume factor of water
    - DZ (float): the cell thickness in the z-direction
    - rwell (float): the well radius
    - skin (float): the skin factor
    - UO (float): the viscosity of oil
    - BO (float): the formation volume factor of oil
    - pwf_producer (float): the desired pressure at the producer well
    - dt (float): the time step
    - N_inj (int): the number of injection wells
    - N_pr (int): the number of production wells
    - nz (int): the number of cells in the z-direction

    Returns:
    - overr (numpy array): an array containing the time and flow rates (in BHP, qoil, qwater, and wct) for each time step
    """

    # ct1[0,:,:] =  at1[:,:,0] # permeability
    # ct1[1,:,:] = quse1[:,:,0]#/UIR # Overall f
    # ct1[2,:,:] = A1[:,:,0]#/UIR# f for water injection
    # ct1[3,:,:] =  at2[:,:,0] # porosity
    if nz == 1:
        Injector_location = np.where(inn[0, 1, :, :].ravel() > 0)[0]
        producer_location = np.where(inn[0, 1, :, :].ravel() < 0)[0]
        PERM = np.reshape(inn[0, 0, :, :], (-1,), "F")
    else:
        Injector_location = np.where(inn[0, 1, :, :, :].ravel() > 0)[0]
        producer_location = np.where(inn[0, 1, :, :, :].ravel() < 0)[0]
        PERM = np.reshape(inn[0, 0, :, :, :], (-1,), "F")

    kuse_inj = PERM[Injector_location]

    kuse_prod = PERM[producer_location]

    RE = 0.2 * DX
    Baa = []
    Timz = []
    for kk in range(steppi):
        if nz == 1:
            Ptito = ooutp[:, kk, :, :]
            Stito = oouts[:, kk, :, :]
        else:
            Ptito = ooutp[:, kk, :, :, :]
            Stito = oouts[:, kk, :, :, :]

        # average_pressure = np.mean(Ptito.ravel()) * pini_alt
        average_pressure = Ptito.ravel()[producer_location]
        p_inj = Ptito.ravel()[Injector_location]
        # p_prod = (Ptito.ravel()[producer_location] ) * pini_alt

        S = Stito.ravel().reshape(-1, 1)

        Sout = (S - SWI) / (1 - SWI - SWR)
        Krw = Sout**2  # Water mobility
        Kro = (1 - Sout) ** 2  # Oil mobility
        krwuse = Krw.ravel()[Injector_location]
        krwusep = Krw.ravel()[producer_location]

        krouse = Kro.ravel()[producer_location]

        up = UW * BW
        down = 2 * np.pi * kuse_inj * krwuse * DZ
        if nz == 1:
            right = np.log(RE / NecessaryI[:, 0]) + NecessaryI[:, 1]
        else:
            right = np.log(
                RE
                / np.tile(
                    NecessaryI[:, 0],
                    int(Injector_location.size / NecessaryI[:, 0].size),
                )
            ) + np.tile(
                NecessaryI[:, 1], int(Injector_location.size / NecessaryI[:, 1].size)
            )
        temp = (up / down) * right
        # temp[temp ==-inf] = 0
        Pwf = p_inj + temp
        Pwf = np.abs(Pwf)
        BHP = np.sum(np.reshape(Pwf, (-1, N_inj), "C"), axis=0) / nz

        up = UO * BO
        down = 2 * np.pi * kuse_prod * krouse * DZ
        if nz == 1:
            right = np.log(RE / NecessaryP[:, 0]) + NecessaryP[:, 1]
        else:
            right = np.log(
                RE
                / np.tile(
                    NecessaryP[:, 0],
                    int(producer_location.size / NecessaryP[:, 0].size),
                )
            ) + np.tile(
                NecessaryP[:, 1], int(producer_location.size / NecessaryP[:, 1].size)
            )
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        if nz == 1:
            drawdown = average_pressure - NecessaryP[:, 2]
        else:
            drawdown = average_pressure - np.tile(
                NecessaryP[:, 2], int(producer_location.size / NecessaryP[:, 1].size)
            )
        qoil = np.abs(-(drawdown * J))
        qoil = np.sum(np.reshape(qoil, (-1, N_pr), "C"), axis=0) / nz

        up = UW * BW
        down = 2 * np.pi * kuse_prod * krwusep * DZ
        if nz == 1:
            right = np.log(RE / NecessaryP[:, 0]) + NecessaryP[:, 1]
        else:
            right = np.log(
                RE
                / np.tile(
                    NecessaryP[:, 0],
                    int(producer_location.size / NecessaryP[:, 0].size),
                )
            ) + np.tile(
                NecessaryP[:, 1], int(producer_location.size / NecessaryP[:, 1].size)
            )
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        if nz == 1:
            drawdown = average_pressure - NecessaryP[:, 2]
        else:
            drawdown = average_pressure - np.tile(
                NecessaryP[:, 2], int(producer_location.size / NecessaryP[:, 2].size)
            )
        qwater = np.abs(-(drawdown * J))
        qwater = np.sum(np.reshape(qwater, (-1, N_pr), "C"), axis=0) / nz
        # qwater[qwater==0] = 0

        # water cut
        wct = (qwater / (qwater + qoil)) * np.float32(100)

        timz = ((kk + 1) * dt) * MAXZ
        # timz = timz.reshape(1,1)
        qs = [BHP, qoil, qwater, wct]
        # print(qs.shape)
        qs = np.asarray(qs)
        qs = qs.reshape(1, -1)

        Baa.append(qs)
        Timz.append(timz)
    Baa = np.vstack(Baa)
    Timz = np.vstack(Timz)

    overr = np.hstack([Timz, Baa])

    return overr  # np.vstack(B)


def Peaceman_well2(
    inn,
    ooutp,
    oouts,
    ooutsoil,
    outg,
    MAXZ,
    mazw,
    s1,
    DX,
    steppi,
    pini_alt,
    SWI,
    SWR,
    UW,
    BW,
    DZ,
    UO,
    BO,
    UG,
    BG,
    dt,
    N_inj,
    N_pr,
    nz,
    NecessaryI,
    NecessaryP,
):
    # ct1[0,:,:] =  at1[:,:,0] # permeability
    # ct1[1,:,:] = quse1[:,:,0]#/UIR # Overall f
    # ct1[2,:,:] = A1[:,:,0]#/UIR# f for water injection
    # ct1[3,:,:] =  at2[:,:,0] # porosity
    if nz == 1:
        Injector_location = np.where(inn[0, 1, :, :].ravel() > 0)[0]
        producer_location = np.where(inn[0, 1, :, :].ravel() < 0)[0]
        PERM = np.reshape(inn[0, 0, :, :], (-1,), "F")
    else:
        Injector_location = np.where(inn[0, 1, :, :, :].ravel() > 0)[0]
        producer_location = np.where(inn[0, 1, :, :, :].ravel() < 0)[0]
        PERM = np.reshape(inn[0, 0, :, :, :], (-1,), "F")

    kuse_inj = PERM[Injector_location]

    kuse_prod = PERM[producer_location]

    RE = 0.2 * DX
    Baa = []
    Timz = []
    for kk in range(steppi):
        if nz == 1:
            Ptito = ooutp[:, kk, :, :]
            Stito = oouts[:, kk, :, :]
            # Stitooil = ooutsoil[:,kk,:,:]
            Stitogas = outg[:, kk, :, :]
        else:
            Ptito = ooutp[:, kk, :, :, :]
            Stito = oouts[:, kk, :, :, :]
            # Stitooil = ooutsoil[:,kk,:,:,:]
            Stitogas = outg[:, kk, :, :, :]

        # average_pressure = np.mean(Ptito.ravel()) * pini_alt
        average_pressure = Ptito.ravel()[producer_location]
        p_inj = Ptito.ravel()[Injector_location]
        # p_prod = (Ptito.ravel()[producer_location] ) * pini_alt

        S = Stito.ravel().reshape(-1, 1)
        # Soil = Stito.ravel().reshape(-1,1)
        Sg = Stitogas.ravel().reshape(-1, 1)

        # S = (Sa - SWI) / (1 - SWI - SWR)
        # #So = (1 - Sa - SGI - SGR) / (1 - SWI - SWR)
        # #Sg = 1 - S - So

        # Mw = (S ** 2) / (UW * BW)  # Water mobility
        # Mo = ((1 - S - Sg) ** 2) / (UO * BO)  # Oil mobility
        # Mg = (Sg ** 2) / (UG * BG)  # Gas mobility

        Sout = (S - SWI) / (1 - SWI - SWR)
        Krw = Sout**2
        Kro = (1 - Sout - Sg) ** 2
        Krg = Sg**2

        krwuse = Krw.ravel()[Injector_location]
        krwusep = Krw.ravel()[producer_location]
        krouse = Kro.ravel()[producer_location]
        krguse = Krg.ravel()[producer_location]

        up = UW * BW
        down = 2 * np.pi * kuse_inj * krwuse * DZ
        if nz == 1:
            right = np.log(RE / NecessaryI[:, 0]) + NecessaryI[:, 1]
        else:
            right = np.log(
                RE
                / np.tile(
                    NecessaryI[:, 0],
                    int(Injector_location.size / NecessaryI[:, 0].size),
                )
            ) + np.tile(
                NecessaryI[:, 1], int(Injector_location.size / NecessaryI[:, 1].size)
            )
        temp = (up / down) * right
        # temp[temp ==-inf] = 0
        Pwf = p_inj + temp
        Pwf = np.abs(Pwf)
        BHP = np.sum(np.reshape(Pwf, (-1, N_inj), "C"), axis=0) / nz

        up = UO * BO
        down = 2 * np.pi * kuse_prod * krouse * DZ
        if nz == 1:
            right = np.log(RE / NecessaryP[:, 0]) + NecessaryP[:, 1]
        else:
            right = np.log(
                RE
                / np.tile(
                    NecessaryP[:, 0],
                    int(producer_location.size / NecessaryP[:, 0].size),
                )
            ) + np.tile(
                NecessaryP[:, 1], int(producer_location.size / NecessaryP[:, 1].size)
            )
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        if nz == 1:
            drawdown = average_pressure - NecessaryP[:, 2]
        else:
            drawdown = average_pressure - np.tile(
                NecessaryP[:, 2], int(producer_location.size / NecessaryP[:, 2].size)
            )
        qoil = np.abs(-(drawdown * J))
        qoil = np.sum(np.reshape(qoil, (-1, N_pr), "C"), axis=0) / nz

        up = UW * BW
        down = 2 * np.pi * kuse_prod * krwusep * DZ
        if nz == 1:
            right = np.log(RE / NecessaryP[:, 0]) + NecessaryP[:, 1]
        else:
            right = np.log(
                RE
                / np.tile(
                    NecessaryP[:, 0],
                    int(producer_location.size / NecessaryP[:, 0].size),
                )
            ) + np.tile(
                NecessaryP[:, 1], int(producer_location.size / NecessaryP[:, 1].size)
            )
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        if nz == 1:
            drawdown = average_pressure - NecessaryP[:, 2]
        else:
            drawdown = average_pressure - np.tile(
                NecessaryP[:, 2], int(producer_location.size / NecessaryP[:, 2].size)
            )
        qwater = np.abs(-(drawdown * J))
        qwater = np.sum(np.reshape(qwater, (-1, N_pr), "C"), axis=0) / nz
        # qwater[qwater==0] = 0

        up = UG * BG
        up = up.get()
        down = 2 * np.pi * kuse_prod * krguse * DZ
        if nz == 1:
            right = np.log(RE / NecessaryP[:, 0]) + NecessaryP[:, 1]
        else:
            right = np.log(
                RE
                / np.tile(
                    NecessaryP[:, 0],
                    int(producer_location.size / NecessaryP[:, 0].size),
                )
            ) + np.tile(
                NecessaryP[:, 1], int(producer_location.size / NecessaryP[:, 1].size)
            )
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        if nz == 1:
            drawdown = average_pressure - NecessaryP[:, 2]
        else:
            drawdown = average_pressure - np.tile(
                NecessaryP[:, 2], int(producer_location.size / NecessaryP[:, 2].size)
            )
        qgas = np.abs(-(drawdown * J))
        qgas = np.sum(np.reshape(qgas, (-1, N_pr), "C"), axis=0) / nz

        # water cut
        wct = (qwater / (qwater + qoil)) * np.float32(100)

        timz = ((kk + 1) * dt) * MAXZ
        # timz = timz.reshape(1,1)
        qs = [BHP, qoil, qwater, qgas, wct]
        # print(qs.shape)
        qs = np.asarray(qs)
        qs = qs.reshape(1, -1)

        Baa.append(qs)
        Timz.append(timz)
    Baa = np.vstack(Baa)
    Timz = np.vstack(Timz)

    overr = np.hstack([Timz, Baa])

    return overr  # np.vstack(B)


# Points generation


def test_points_gen(n_test, nder, interval=(-1.0, 1.0), distrib="random", **kwargs):
    return {
        "random": lambda n_test, nder: (interval[1] - interval[0])
        * np.random.rand(n_test, nder)
        + interval[0],
        "lhs": lambda n_test, nder: (interval[1] - interval[0])
        * lhs(nder, samples=n_test, **kwargs)
        + interval[0],
    }[distrib.lower()](n_test, nder)


def smoothn(
    y,
    nS0=10,
    axis=None,
    smoothOrder=2.0,
    sd=None,
    verbose=False,
    s0=None,
    z0=None,
    isrobust=False,
    W=None,
    s=None,
    MaxIter=100,
    TolZ=1e-3,
    weightstr="bisquare",
):
    if type(y) == ma.core.MaskedArray:  # masked array
        # is_masked = True
        mask = y.mask
        y = np.array(y)
        y[mask] = 0.0
        if np.any(W is not None):
            W = np.array(W)
            W[mask] = 0.0
        if np.any(sd is not None):
            W = np.array(1.0 / sd**2)
            W[mask] = 0.0
            sd = None
        y[mask] = np.nan

    if np.any(sd is not None):
        sd_ = np.array(sd)
        mask = sd > 0.0
        W = np.zeros_like(sd_)
        W[mask] = 1.0 / sd_[mask] ** 2
        sd = None

    if np.any(W is not None):
        W = W / W.max()

    sizy = y.shape

    # sort axis
    if axis is None:
        axis = tuple(np.arange(y.ndim))

    noe = y.size  # number of elements
    if noe < 2:
        z = y
        exitflag = 0
        Wtot = 0
        return z, s, exitflag, Wtot
    # ---
    # Smoothness parameter and weights
    # if s != None:
    #  s = []
    if np.all(W is None):
        W = np.ones(sizy)

    # if z0 == None:
    #  z0 = y.copy()

    # ---
    # "Weighting function" criterion
    weightstr = weightstr.lower()
    # ---
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    IsFinite = np.array(np.isfinite(y)).astype(bool)
    nof = IsFinite.sum()  # number of finite elements
    W = W * IsFinite
    if any(W < 0):
        raise RuntimeError("smoothn:NegativeWeights", "Weights must all be >=0")
    else:
        # W = W/np.max(W)
        pass
    # ---
    # Weighted or missing data?
    isweighted = any(W != 1)
    # ---
    # Robust smoothing?
    # isrobust
    # ---
    # Automatic smoothing?
    isauto = not s
    # ---
    # DCTN and IDCTN are required
    try:
        from scipy.fftpack.realtransforms import dct, idct
    except:
        z = y
        exitflag = -1
        Wtot = 0
        return z, s, exitflag, Wtot

    ## Creation of the Lambda tensor
    # ---
    # Lambda contains the eingenvalues of the difference matrix used in this
    # penalized least squares process.
    axis = tuple(np.array(axis).flatten())
    d = y.ndim
    Lambda = np.zeros(sizy)
    for i in axis:
        # create a 1 x d array (so e.g. [1,1] for a 2D case
        siz0 = np.ones((1, y.ndim))[0].astype(int)
        siz0[i] = sizy[i]
        # cos(pi*(reshape(1:sizy(i),siz0)-1)/sizy(i)))
        # (arange(1,sizy[i]+1).reshape(siz0) - 1.)/sizy[i]
        Lambda = Lambda + (
            np.cos(np.pi * (np.arange(1, sizy[i] + 1) - 1.0) / sizy[i]).reshape(siz0)
        )
        # else:
        #  Lambda = Lambda + siz0
    Lambda = -2.0 * (len(axis) - Lambda)
    if not isauto:
        Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)

    ## Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter (Equation #12 in the
    # referenced CSDA paper).
    N = sum(np.array(sizy) != 1)
    # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    # (h/n)**2 = (1 + a)/( 2 a)
    # a = 1/(2 (h/n)**2 -1)
    # where a = sqrt(1 + 16 s)
    # (a**2 -1)/16
    try:
        sMinBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMax ** (2.0 / N))) / 4.0 / hMax ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
        sMaxBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMin ** (2.0 / N))) / 4.0 / hMin ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
    except:
        sMinBnd = None
        sMaxBnd = None
    ## Initialize before iterating
    # ---
    Wtot = W
    # --- Initial conditions for z
    if isweighted:
        # --- With weighted/missing data
        # An initial guess is provided to ensure faster convergence. For that
        # purpose, a nearest neighbor interpolation followed by a coarse
        # smoothing are performed.
        # ---
        if z0 is not None:  # an initial guess (z0) has been provided
            z = z0
        else:
            z = y  # InitialGuess(y,IsFinite);
            z[~IsFinite] = 0.0
    else:
        z = np.zeros(sizy)
    # ---
    z0 = z
    y[~IsFinite] = 0
    # arbitrary values for missing y-data
    # ---
    tol = 1.0
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    # --- Error on p. Smoothness parameter s = 10^p
    errp = 0.1
    # opt = optimset('TolX',errp);
    # --- Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75 * isweighted
    # ??
    ## Main iterative process
    # ---
    if isauto:
        try:
            xpost = np.array([(0.9 * np.log10(sMinBnd) + np.log10(sMaxBnd) * 0.1)])
        except:
            np.array([100.0])
    else:
        xpost = np.array([np.log10(s)])
    while RobustIterativeProcess:
        # --- "amount" of weights (see the function GCVscore)
        aow = sum(Wtot) / noe
        # 0 < aow <= 1
        # ---
        while tol > TolZ and nit < MaxIter:
            if verbose:
                print("tol", tol, "nit", nit)
            nit = nit + 1
            DCTy = dctND(Wtot * (y - z) + z, f=dct)
            if isauto and not np.remainder(np.log2(nit), 1):
                # ---
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter s that minimizes the GCV
                # score i.e. s = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when nit is a power of 2)
                # ---
                # errp in here somewhere

                # xpost,f,d = lbfgsb.fmin_l_bfgs_b(gcv,xpost,fprime=None,factr=10.,\
                #   approx_grad=True,bounds=[(log10(sMinBnd),log10(sMaxBnd))],\
                #   args=(Lambda,aow,DCTy,IsFinite,Wtot,y,nof,noe))

                # if we have no clue what value of s to use, better span the
                # possible range to get a reasonable starting point ...
                # only need to do it once though. nS0 is teh number of samples used
                if not s0:
                    ss = np.arange(nS0) * (1.0 / (nS0 - 1.0)) * (
                        np.log10(sMaxBnd) - np.log10(sMinBnd)
                    ) + np.log10(sMinBnd)
                    g = np.zeros_like(ss)
                    for i, p in enumerate(ss):
                        g[i] = gcv(
                            p,
                            Lambda,
                            aow,
                            DCTy,
                            IsFinite,
                            Wtot,
                            y,
                            nof,
                            noe,
                            smoothOrder,
                        )
                        # print 10**p,g[i]
                    xpost = [ss[g == g.min()]]
                    # print '==============='
                    # print nit,tol,g.min(),xpost[0],s
                    # print '==============='
                else:
                    xpost = [s0]
                xpost, f, d = lbfgsb.fmin_l_bfgs_b(
                    gcv,
                    xpost,
                    fprime=None,
                    factr=1e7,
                    approx_grad=True,
                    bounds=[(np.log10(sMinBnd), np.log10(sMaxBnd))],
                    args=(Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder),
                )
            s = 10 ** xpost[0]
            # update the value we use for the initial s estimate
            s0 = xpost[0]

            Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)

            z = RF * dctND(Gamma * DCTy, f=idct) + (1 - RF) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = isweighted * norm(z0 - z) / norm(z)

            z0 = z
            # re-initialization
        exitflag = nit < MaxIter

        if isrobust:  # -- Robust Smoothing: iteratively re-weighted process
            # --- average leverage
            h = np.sqrt(1 + 16.0 * s)
            h = np.sqrt(1 + h) / np.sqrt(2) / h
            h = h**N
            # --- take robust weights into account
            Wtot = W * RobustWeights(y - z, IsFinite, h, weightstr)
            # --- re-initialize for another iterative weighted process
            isweighted = True
            tol = 1
            nit = 0
            # ---
            RobustStep = RobustStep + 1
            RobustIterativeProcess = RobustStep < 3
            # 3 robust steps are enough.
        else:
            RobustIterativeProcess = False
            # stop the whole process

    ## Warning messages
    # ---
    if isauto:
        if abs(np.log10(s) - np.log10(sMinBnd)) < errp:
            warning(
                "MATLAB:smoothn:SLowerBound",
                [
                    "s = %.3f " % (s)
                    + ": the lower bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
        elif abs(np.log10(s) - np.log10(sMaxBnd)) < errp:
            warning(
                "MATLAB:smoothn:SUpperBound",
                [
                    "s = %.3f " % (s)
                    + ": the upper bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
    return z, s, exitflag, Wtot


def warning(s1, s2):
    print(s1)
    print(s2[0])


## GCV score
# ---
# function GCVscore = gcv(p)
def gcv(p, Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder):
    # Search the smoothing parameter s that minimizes the GCV score
    # ---
    s = 10**p
    Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)
    # --- RSS = Residual sum-of-squares
    if aow > 0.9:  # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        RSS = norm(DCTy * (Gamma - 1.0)) ** 2
    else:
        # take account of the weights to calculate RSS:
        yhat = dctND(Gamma * DCTy, f=idct)
        RSS = norm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite])) ** 2
    # ---
    TrH = sum(Gamma)
    GCVscore = RSS / float(nof) / (1.0 - TrH / float(noe)) ** 2
    return GCVscore


## Robust weights
# function W = RobustWeights(r,I,h,wstr)
def RobustWeights(r, I, h, wstr):
    # weights for robust smoothing.
    MAD = np.median(abs(r[I] - np.median(r[I])))
    # median absolute deviation
    u = abs(r / (1.4826 * MAD) / np.sqrt(1 - h))
    # studentized residuals
    if wstr == "cauchy":
        c = 2.385
        W = 1.0 / (1 + (u / c) ** 2)
        # Cauchy weights
    elif wstr == "talworth":
        c = 2.795
        W = u < c
        # Talworth weights
    else:
        c = 4.685
        W = (1 - (u / c) ** 2) ** 2.0 * ((u / c) < 1)
        # bisquare weights

    W[np.isnan(W)] = 0
    return W


## Initial Guess with weighted/missing data
# function z = InitialGuess(y,I)
def InitialGuess(y, I):
    # -- nearest neighbor interpolation (in case of missing values)
    if any(~I):
        try:
            from scipy.ndimage.morphology import distance_transform_edt

            # if license('test','image_toolbox')
            # [z,L] = bwdist(I);
            L = distance_transform_edt(1 - I)
            z = y
            z[~I] = y[L[~I]]
        except:
            # If BWDIST does not exist, NaN values are all replaced with the
            # same scalar. The initial guess is not optimal and a warning
            # message thus appears.
            z = y
            z[~I] = np.mean(y[I])
    else:
        z = y
    # coarse fast smoothing
    z = dctND(z, f=dct)
    k = np.array(z.shape)
    m = np.ceil(k / 10) + 1
    d = []
    for i in np.xrange(len(k)):
        d.append(np.arange(m[i], k[i]))
    d = np.array(d).astype(int)
    z[d] = 0.0
    z = dctND(z, f=idct)
    return z


def dctND(data, f=dct):
    nd = len(data.shape)
    if nd == 1:
        return f(data, norm="ortho", type=2)
    elif nd == 2:
        return f(f(data, norm="ortho", type=2).T, norm="ortho", type=2).T
    elif nd == 3:
        return f(
            f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
            norm="ortho",
            type=2,
            axis=2,
        )
    elif nd == 4:
        return f(
            f(
                f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
                norm="ortho",
                type=2,
                axis=2,
            ),
            norm="ortho",
            type=2,
            axis=3,
        )


def peaks(n):
    """
    Mimic basic of matlab peaks fn
    """
    xp = np.arange(n)
    [x, y] = np.meshgrid(xp, xp)
    z = np.zeros_like(x).astype(float)
    for i in np.xrange(n / 5):
        x0 = random() * n
        y0 = random() * n
        sdx = random() * n / 4.0
        sdy = sdx
        c = random() * 2 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - ((x - x0) / sdx) * ((y - y0) / sdy) * c
        )
        # f /= f.sum()
        f *= random()
        z += f
    return z


def Upstream_2PHASE(
    nx, ny, nz, S, UW, UO, BW, BO, SWI, SWR, Vol, qinn, V, Tt, porosity
):
    """
    This function solves a 2-phase flow reservoir simulation problem using an upstream scheme.
    Args:
    - nx: int, number of grid cells in the x-direction.
    - ny: int, number of grid cells in the y-direction.
    - nz: int, number of grid cells in the z-direction.
    - S: array, initial saturation field.
    - UW: float, water viscosity.
    - UO: float, oil viscosity.
    - BW: float, water formation volume factor.
    - BO: float, oil formation volume factor.
    - SWI: float, initial water saturation.
    - SWR: float, residual water saturation.
    - Vol: array, grid cell volumes.
    - qinn: array, inflow rate for each grid cell.
    - V: dict, containing arrays with the x, y, and z coordinates of the grid cell faces.
    - Tt: float, total time to simulate.
    - porosity: array, porosity values for each grid cell.
    Returns:
    - S: array, final saturation field.
    """

    Nx = nx
    Ny = ny
    Nz = nz
    N = Nx * Ny * Nz
    poro = cp.reshape(porosity, (N, 1), "F")
    pv = cp.multiply(Vol, poro)
    qinn = cp.reshape(qinn, (-1, 1), "F")
    fi = cp.maximum(qinn, 0)
    XP = cp.maximum(V["x"], 0)
    XN = cp.minimum(V["x"], 0)
    YP = cp.maximum(V["y"], 0)
    YN = cp.minimum(V["y"], 0)
    ZP = cp.maximum(V["z"], 0)
    ZN = cp.minimum(V["z"], 0)
    Vi = (
        XP[:Nx, :, :]
        + YP[:, :Ny, :]
        + ZP[:, :, :Nz]
        - XN[1 : Nx + 1, :, :]
        - YN[:, 1 : Ny + 1, :]
        - ZN[:, :, 1 : Nz + 1]
    )
    Vi = cp.reshape(Vi, (N, 1), "F")
    pm = min(pv / (Vi + fi))
    cfl = ((1 - SWR) / 3) * pm
    # Nts =#30
    Nts = math.ceil(Tt / cfl)
    # print(Nts)
    dtx = cp.divide(cp.divide(Tt, Nts), pv)
    # dtx = cp.divide(30*1e-4,pv)
    dtx = cp.ravel(dtx)
    A = GenA(nx, ny, nz, V, qinn)
    Afirst = A
    A = spdiags(dtx, 0, N, N, format="csr") @ Afirst
    # A=Afirst
    fitemp = cp.maximum(qinn, 0)
    fi = cp.multiply(fitemp, cp.reshape(dtx, (-1, 1), "F"))
    for t in range(Nts):
        # print(  '........' +str(t) + '/' +str (Nts))
        mw, mo, dMw, dMo = RelPerm2(S, UW, UO, BW, BO, SWI, SWR, nx, ny, nz)
        fw = cp.divide(mw, cp.add(mw, mo))
        Asa = A @ fw
        S = cp.add(S, cp.add(Asa, fi))
    return S


def Upstream_3PHASE(
    nx,
    ny,
    nz,
    S,
    Soil,
    UW,
    UO,
    UG,
    BW,
    BO,
    BG,
    RS,
    SWI,
    SWR,
    Vol,
    qinn,
    qinnoil,
    V,
    Tt,
    porosity,
):
    Nx = nx
    Ny = ny
    Nz = nz
    N = Nx * Ny * Nz
    poro = cp.reshape(porosity, (N, 1), "F")
    pv = cp.multiply(Vol, poro)
    qinn = cp.reshape(qinn, (-1, 1), "F")
    qinno = cp.reshape(qinnoil, (-1, 1), "F")
    fi = cp.maximum(qinn, 0)
    fio = cp.maximum(qinno, 0)
    XP = cp.maximum(V["x"], 0)
    XN = cp.minimum(V["x"], 0)
    YP = cp.maximum(V["y"], 0)
    YN = cp.minimum(V["y"], 0)
    ZP = cp.maximum(V["z"], 0)
    ZN = cp.minimum(V["z"], 0)
    Vi = (
        XP[:Nx, :, :]
        + YP[:, :Ny, :]
        + ZP[:, :, :Nz]
        - XN[1 : Nx + 1, :, :]
        - YN[:, 1 : Ny + 1, :]
        - ZN[:, :, 1 : Nz + 1]
    )
    Vi = cp.reshape(Vi, (N, 1), "F")

    pm = min(pv / (Vi + fi))
    pmo = min(pv / (Vi + fio))

    cfl = ((1 - SWR) / 3) * pm
    cflo = ((1 - SWR) / 3) * pmo

    # Nts =#30
    Nts = math.ceil(Tt / cfl)
    Ntso = math.ceil(Tt / cflo)

    # print(Nts)
    dtx = cp.divide(cp.divide(Tt, Nts), pv)
    dtxo = cp.divide(cp.divide(Tt, Ntso), pv)
    # dtx = cp.divide(30*1e-4,pv)
    dtx = cp.ravel(dtx)
    dtxo = cp.ravel(dtxo)

    A = GenA(nx, ny, nz, V, qinn)
    Ao = GenA(nx, ny, nz, V, qinno)

    Afirst = A
    Afirsto = Ao

    A = spdiags(dtx, 0, N, N, format="csr") @ Afirst
    Ao = spdiags(dtxo, 0, N, N, format="csr") @ Afirsto
    # A=Afirst
    fitemp = cp.maximum(qinn, 0)
    fitempo = cp.maximum(qinno, 0)

    fi = cp.multiply(fitemp, cp.reshape(dtx, (-1, 1), "F"))
    fio = cp.multiply(fitempo, cp.reshape(dtxo, (-1, 1), "F"))

    for t in range(Nts):
        # print(  '........' +str(t) + '/' +str (Nts))
        # mw,mo,dMw,dMo=RelPerm2(S,UW,UO,BW,BO,SWI,SWR)
        mw, mo, mg, _, _, _ = RelPerm3(
            S, Soil, UW, UO, UG, BW, BO, BG, SWI, SWR, nx, ny, nz
        )
        fw = cp.divide(mw, cp.add(cp.add(cp.add(mw, mo), mg), mo * RS))
        Asa = A @ fw
        S = cp.add(S, cp.add(Asa, fi))

        fwo = cp.divide(mg, cp.add(cp.add(cp.add(mw, mo), mg), mo * RS))
        Asa = Ao @ fwo
        Soil = cp.add(Soil, cp.add(Asa, fio))

    return S, Soil


def RelPerm2(Sa, UW, UO, BW, BO, SWI, SWR, nx, ny, nz):
    """
    Computes the relative permeability and its derivative w.r.t saturation S,
    based on Brooks and Corey model.

    Parameters
    ----------
    Sa : array_like
        Saturation value.
    UW : float
        Water viscosity.
    UO : float
        Oil viscosity.
    BW : float
        Water formation volume factor.
    BO : float
        Oil formation volume factor.
    SWI : float
        Initial water saturation.
    SWR : float
        Residual water saturation.
    nx, ny, nz : int
        The number of grid cells in x, y, and z directions.

    Returns
    -------
    Mw : array_like
        Water relative permeability.
    Mo : array_like
        Oil relative permeability.
    dMw : array_like
        Water relative permeability derivative w.r.t saturation.
    dMo : array_like
        Oil relative permeability derivative w.r.t saturation.
    """
    S = (Sa - SWI) / (1 - SWI - SWR)
    Mw = (S**2) / (UW * BW)  # Water mobility
    Mo = ((1 - S) ** 2) / (UO * BO)  # Oil mobility
    dMw = 2 * S / (UW * BW) / (1 - SWI - SWR)
    dMo = -2 * (1 - S) / (UO * BO) / (1 - SWI - SWR)

    return (
        cp.reshape(Mw, (-1, 1), "F"),
        cp.reshape(Mo, (-1, 1), "F"),
        cp.reshape(dMw, (-1, 1), "F"),
        cp.reshape(dMo, (-1, 1), "F"),
    )


def RelPerm3(Sa, Sg, UW, UO, UG, BW, BO, BG, SWI, SWR, nx, ny, nz):
    """
    Computes the relative permeability and its derivative w.r.t saturation S,
    based on Brooks and Corey model.

    Parameters
    ----------
    Sa : array_like
        Saturation value.
    UW : float
        Water viscosity.
    UO : float
        Oil viscosity.
    UG : float
        Gas viscosity.
    BW : float
        Water formation volume factor.
    BO : float
        Oil formation volume factor.
    BG : float
        Gas formation volume factor.
    SWI : float
        Initial water saturation.
    SWR : float
        Residual water saturation.
    SGI : float
        Initial gas saturation.
    SGR : float
        Residual gas saturation.
    nx, ny, nz : int
        The number of grid cells in x, y, and z directions.

    Returns
    -------
    Mw : array_like
        Water relative permeability.
    Mo : array_like
        Oil relative permeability.
    Mg : array_like
        Gas relative permeability.
    dMw : array_like
        Water relative permeability derivative w.r.t saturation.
    dMo : array_like
        Oil relative permeability derivative w.r.t saturation.
    dMg : array_like
        Gas relative permeability derivative w.r.t saturation.
    """
    S = (Sa - SWI) / (1 - SWI - SWR)
    # So = (1 - Sa - SGI - SGR) / (1 - SWI - SWR)
    # Sg = 1 - S - So

    Mw = (S**2) / (UW * BW)  # Water mobility
    Mo = ((1 - S - Sg) ** 2) / (UO * BO)  # Oil mobility
    Mg = (Sg**2) / (UG * BG)  # Gas mobility

    dMw = 2 * S / (UW * BW) / (1 - SWI - SWR)
    dMo = -2 * (1 - S - Sg) / (UO * BO) / (1 - SWI - SWR)
    dMg = 2 * Sg / (UG * BG) / (1 - SWI - SWR)

    return (
        cp.reshape(Mw, (-1, 1), "F"),
        cp.reshape(Mo, (-1, 1), "F"),
        cp.reshape(Mg, (-1, 1), "F"),
        cp.reshape(dMw, (-1, 1), "F"),
        cp.reshape(dMo, (-1, 1), "F"),
        cp.reshape(dMg, (-1, 1), "F"),
    )


def NewtRaph(
    nx, ny, nz, porosity, Vol, S, V, qinn, Tt, UW, UO, SWI, SWR, method2, BW, BO
):
    """
    Uses Newton-Raphson method to solve the two-phase flow problem in a reservoir.

    Args:
    nx (int): Number of grid blocks in x-direction.
    ny (int): Number of grid blocks in y-direction.
    nz (int): Number of grid blocks in z-direction.
    porosity (ndarray): Array of shape (nx,ny,nz) containing porosity values.
    Vol (ndarray): Array of shape (nx,ny,nz) containing grid block volumes.
    S (ndarray): Array of shape (nx,ny,nz) containing initial water saturation values.
    V (dict): Dictionary containing arrays of x,y, and z directions of grid block boundaries.
    qinn (ndarray): Array of shape (nx,ny,nz) containing injection rate values.
    Tt (float): Total time for simulation.
    UW (float): Water viscosity.
    UO (float): Oil viscosity.
    SWI (float): Initial water saturation.
    SWR (float): Residual water saturation.
    method2 (int): Specifies the linear solver to be used.
                    1: GMRES
                    2: Sparse LU factorization
                    3: Conjugate gradient
                    4: LSQR
                    5: Adaptive algebraic multigrid
                    6: Conjugate projected gradient method
                    7: AMGX
    typee (str): Specifies the AMGX solver to be used. For details see `AMGX_Inverse_problem` function.
    BW (float): Water formation volume factor.
    BO (float): Oil formation volume factor.

    Returns:
    S (ndarray): Array of shape (nx,ny,nz) containing water saturation values after simulation.
    """

    Nx = nx
    Ny = ny
    Nz = nz
    N = Nx * Ny * Nz
    A = GenA(nx, ny, nz, V, qinn)
    conv = 0
    IT = 0
    S00 = S
    while conv == 0:
        dt = Tt / (2**IT)
        poro = cp.reshape(porosity, (N, 1), "F")
        pv = cp.multiply(Vol, poro)
        dtx = cp.divide(dt, pv)  # dt/(pv)
        dtx = cp.nan_to_num(dtx, copy=True, nan=0.0)
        fi = cp.multiply(cp.maximum(qinn.reshape(-1, 1), 0), dtx.reshape(-1, 1))
        dtx = cp.ravel(dtx)
        B = spdiags(dtx, 0, N, N, format="csr") @ A
        S0 = S
        I = 0
        while I < 2**IT:
            S0 = S
            dsn = 1
            it = 0
            I = I + 1
            while (dsn > 0.001) and (it < 10):
                Mw, Mo, dMw, dMo = RelPerm2(S, UW, UO, BW, BO, SWI, SWR, nx, ny, nz)
                df = cp.divide(dMw, (Mw + Mo)) - cp.multiply(
                    cp.divide(Mw, ((Mw + Mo) ** (2))), (dMw + dMo)
                )  # dMw/(Mw + Mo)-Mw/(Mw+Mo)**(2)*(dMw+dMo)
                df = cp.ravel(df)
                dG = sparse.eye(N, dtype=cp.float32) - B @ spdiags(
                    df, 0, N, N, format="csr"
                )
                dG.data = cp.nan_to_num(dG.data, copy=True, nan=0.0)

                fw = Mw / (Mw + Mo)
                G = S - S0 - (cp.add(B @ fw, fi))
                G = cp.nan_to_num(G, copy=True, nan=0.0)
                # G = sm.eliminate_zeros()

                if method2 == 1:  # GMRES
                    M2 = spilu(-dG)

                    def M_x(x):
                        return M2.solve(x)

                    M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
                    ds, exitCode = gmres(
                        -dG, G, tol=1e-6, atol=0, restart=20, maxiter=100, M=M
                    )

                elif method2 == 2:
                    ds = spsolve(-dG, G)
                elif method2 == 3:
                    M2 = spilu(-dG)

                    def M_x(x):
                        return M2.solve(x)

                    M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
                    ds, exitCode = cg(-dG, G, tol=1e-6, atol=0, maxiter=100, M=M)
                elif method2 == 4:  # LSQR
                    G = cp.ravel(G)
                    ds, istop, itn, normr = lsqr(-dG, G)[:4]
                else:  # CPR
                    M2 = spilu(-dG)

                    def M_x(x):
                        return M2.solve(x)

                    M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
                    ds, exitCode = gmres(
                        -dG, G, tol=1e-6, atol=0, restart=20, maxiter=100, M=M
                    )

                # print(exitCode)
                ds = cp.reshape(ds, (-1, 1), "F")
                S = cp.add(S, ds)  # S+ds
                dsn = cp.linalg.norm(ds, 2)
                it = it + 1

            if dsn > 0.001:
                I = 2 ** (IT)
                S = S00

        if dsn < 0.001:
            conv = 1
        else:
            IT = IT + 1
    return S


def NewtRaph2(
    nx,
    ny,
    nz,
    porosity,
    Vol,
    S,
    Soil,
    V,
    qinn,
    qinnoil,
    Tt,
    UW,
    UO,
    UG,
    SWI,
    SWR,
    method2,
    BW,
    BO,
    BG,
    RS,
):
    Nx = nx
    Ny = ny
    Nz = nz
    N = Nx * Ny * Nz
    A = GenA(nx, ny, nz, V, qinn)
    Aoil = GenA(nx, ny, nz, V, qinnoil)
    conv = 0
    IT = 0
    S00 = S
    S00oil = Soil
    while conv == 0:
        dt = Tt / (2**IT)
        poro = cp.reshape(porosity, (N, 1), "F")
        pv = cp.multiply(Vol, poro)
        dtx = cp.divide(dt, pv)  # dt/(pv)
        dtx = cp.nan_to_num(dtx, copy=True, nan=0.0)
        fi = cp.multiply(cp.maximum(qinn.reshape(-1, 1), 0), dtx.reshape(-1, 1))
        fioil = cp.multiply(cp.maximum(qinnoil.reshape(-1, 1), 0), dtx.reshape(-1, 1))
        dtx = cp.ravel(dtx)
        B = spdiags(dtx, 0, N, N, format="csr") @ A
        Boil = spdiags(dtx, 0, N, N, format="csr") @ Aoil
        S0 = S
        S0oil = Soil
        I = 0
        while I < 2**IT:
            S0 = S
            S0oil = Soil
            dsn = 1
            it = 0
            I = I + 1
            while (dsn > 0.01) and (it < 5):
                Mw, Mo, Mg, dMw, dMo, dMg = RelPerm3(
                    S, Soil, UW, UO, UG, BW, BO, BG, SWI, SWR, nx, ny, nz
                )
                df = cp.divide(dMw, (Mw + Mo + Mg + RS * Mo)) - cp.multiply(
                    cp.divide(Mw, ((Mw + Mo + Mg + RS * Mo) ** (2))), (dMw + dMo + dMg)
                )  # dMw/(Mw + Mo)-Mw/(Mw+Mo)**(2)*(dMw+dMo)
                dfoil = cp.divide(dMg, (Mw + Mo + Mg + RS * Mo)) - cp.multiply(
                    cp.divide(Mg, ((Mw + Mo + Mg + RS * Mo) ** (2))), (dMw + dMo + dMg)
                )
                df = cp.ravel(df)
                dfoil = cp.ravel(dfoil)
                dG = sparse.eye(N, dtype=cp.float32) - B @ spdiags(
                    df, 0, N, N, format="csr"
                )
                dGoil = sparse.eye(N, dtype=cp.float32) - Boil @ spdiags(
                    dfoil, 0, N, N, format="csr"
                )
                dG.data = cp.nan_to_num(dG.data, copy=True, nan=0.0)
                dGoil.data = cp.nan_to_num(dGoil.data, copy=True, nan=0.0)

                fw = Mw / (Mw + Mo + Mg + RS * Mo)
                fwoil = Mg / (Mw + Mo + Mg + RS * Mo)

                G = S - S0 - (cp.add(B @ fw, fi))
                Goil = Soil - S0oil - (cp.add(Boil @ fwoil, fioil))

                G = cp.nan_to_num(G, copy=True, nan=0.0)
                Goil = cp.nan_to_num(Goil, copy=True, nan=0.0)
                # G = sm.eliminate_zeros()

                if method2 == 1:  # GMRES
                    M2 = spilu(-dG)

                    def M_x(x):
                        return M2.solve(x)

                    M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
                    ds, exitCode = gmres(
                        -dG, G, tol=1e-6, atol=0, restart=20, maxiter=100, M=M
                    )

                    M2oil = spilu(-dGoil)

                    def M_xoil(x):
                        return M2oil.solve(x)

                    Moil = LinearOperator((nx * ny * nz, nx * ny * nz), M_xoil)
                    dsoil, exitCodeoil = gmres(
                        -dGoil, Goil, tol=1e-6, atol=0, restart=20, maxiter=100, M=Moil
                    )

                elif method2 == 2:
                    ds = spsolve(-dG, G)
                    dsoil = spsolve(-dGoil, Goil)
                elif method2 == 3:
                    M2 = spilu(-dG)

                    def M_x(x):
                        return M2.solve(x)

                    M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
                    ds, exitCode = cg(-dG, G, tol=1e-6, atol=0, maxiter=100, M=M)

                    M2oil = spilu(-dGoil)

                    def M_xoil(x):
                        return M2oil.solve(x)

                    Moil = LinearOperator((nx * ny * nz, nx * ny * nz), M_xoil)
                    dsoil, exitCodeoil = cg(
                        -dGoil, Goil, tol=1e-6, atol=0, maxiter=100, M=Moil
                    )

                elif method2 == 4:  # LSQR
                    G = cp.ravel(G)
                    ds, istop, itn, normr = lsqr(-dG, G)[:4]

                    Goil = cp.ravel(Goil)
                    dsoil, istopo, itno, normro = lsqr(-dGoil, Goil)[:4]
                else:  # CPR
                    M2 = spilu(-dG)

                    def M_x(x):
                        return M2.solve(x)

                    M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
                    ds, exitCode = gmres(
                        -dG, G, tol=1e-6, atol=0, restart=20, maxiter=100, M=M
                    )

                    M2oil = spilu(-dGoil)

                    def M_xoil(x):
                        return M2oil.solve(x)

                    Moil = LinearOperator((nx * ny * nz, nx * ny * nz), M_xoil)
                    dsoil, exitCodeoil = gmres(
                        -dGoil, Goil, tol=1e-6, atol=0, restart=20, maxiter=100, M=Moil
                    )

                # print(exitCode)
                ds = cp.reshape(ds, (-1, 1), "F")
                dsoil = cp.reshape(dsoil, (-1, 1), "F")

                S = cp.add(S, ds)  # S+ds
                Soil = cp.add(Soil, dsoil)  # S+ds

                dsn = cp.linalg.norm(ds, 2)
                dsnoil = cp.linalg.norm(dsoil, 2)

                it = it + 1

            if (dsn > 0.01) or (dsnoil > 0.01):
                I = 2 ** (IT)
                S = S00
                Soil = S00oil

        if (dsn < 0.01) or (dsnoil < 0.01):
            conv = 1
        else:
            IT = IT + 1
    return S, Soil

    return S


def calc_mu_g(p):
    # Avergae reservoir pressure
    mu_g = 3e-10 * p**2 + 1e-6 * p + 0.0133
    return mu_g


def calc_rs(p_bub, p):
    # p=average reservoir pressure
    if p < p_bub:
        rs_factor = 1
    else:
        rs_factor = 0
    rs = 178.11**2 / 5.615 * ((p / p_bub) ** 1.3 * rs_factor + (1 - rs_factor))
    return rs


def calc_dp(p_bub, p_atm, p):
    if p < p_bub:
        dp = p_atm - p
    else:
        dp = p_atm - p_bub
    return dp


def calc_bg(p_bub, p_atm, p):
    # P is avergae reservoir pressure
    b_g = 1 / (cp.exp(1.7e-3 * calc_dp(p_bub, p_atm, p)))
    return b_g


def calc_bo(p_bub, p_atm, CFO, p):
    # p is average reservoir pressure
    if p < p_bub:
        b_o = 1 / cp.exp(-8e-5 * (p_atm - p))
    else:
        b_o = 1 / (cp.exp(-8e-5 * (p_atm - p_bub)) * cp.exp(-CFO * (p - p_bub)))
    return b_o


def GenA(nx, ny, nz, V, qsa):
    """
    Input:

    nx, ny, nz: integers representing the number of grid cells in the x, y, and z directions, respectively.
    V: a dictionary containing the coordinate information for the grid cells.
    qsa: an array of shape (nx, ny, nz) representing the source term.
    Output:

    A: a sparse CSR matrix of shape (NxNyNz, NxNyNz) representing the discretized differential operator.
    Description:
    This function generates a sparse CSR matrix A that represents the discretized
    differential operator for the given grid and source term using the finite
    volume method. The matrix A is generated based on the Upstream weighting scheme.
    The input V is a dictionary containing the coordinate information for each grid cell,
    and qsa is the source term. The output A is a sparse CSR matrix of shape (NxNyNz, NxNyNz)
    that can be used to solve the system of linear equations representing the flow of fluid through the porous media.
    """
    Nx = nx
    Ny = ny
    Nz = nz
    N = Nx * Ny * Nz
    fp = cp.minimum(qsa, 0)
    fp = cp.reshape(fp, (N), "F")
    XN = cp.minimum(V["x"], 0)
    x1 = cp.reshape(XN[:Nx, :, :], (N), "F")
    YN = cp.minimum(V["y"], 0)
    y1 = cp.reshape(YN[:, :Ny, :], (N), "F")
    ZN = cp.minimum(V["z"], 0)
    z1 = cp.reshape(ZN[:, :, :Nz], (N), "F")
    XP = cp.maximum(V["x"], 0)
    x2 = cp.reshape(XP[1 : Nx + 1, :, :], (N), "F")
    YP = cp.maximum(V["y"], 0)
    y2 = cp.reshape(YP[:, 1 : Ny + 1, :], (N), "F")
    ZP = cp.maximum(V["z"], 0)
    z2 = cp.reshape(ZP[:, :, 1 : Nz + 1], (N), "F")

    # print((fp+x1-x2+y1-y2+z1-z2).shape)
    tempzz = fp + x1 - x2 + y1 - y2 + z1 - z2
    tempzz = cp.ravel(tempzz)

    DiagVecs = [z2, y2, x2, tempzz, -x1, -y1, -z1]
    DiagIndx = [-Nx * Ny, -Nx, -1, 0, 1, Nx, Nx * Ny]
    A = spdiags(DiagVecs, DiagIndx, N, N, format="csr")
    return A


def ProgressBar(Total, Progress, BarLength=20, ProgressIcon="#", BarIcon="-"):
    try:
        # You can't have a progress bar with zero or negative length.
        if BarLength < 1:
            BarLength = 20
        # Use status variable for going to the next line after progress completion.
        Status = ""
        # Calcuting progress between 0 and 1 for percentage.
        Progress = float(Progress) / float(Total)
        # Doing this conditions at final progressing.
        if Progress >= 1.0:
            Progress = 1
            Status = "\r\n"  # Going to the next line
        # Calculating how many places should be filled
        Block = int(round(BarLength * Progress))
        # Show this
        Bar = "[{}] {:.0f}% {}".format(
            ProgressIcon * Block + BarIcon * (BarLength - Block),
            round(Progress * 100, 0),
            Status,
        )
        return Bar
    except:
        return "ERROR"


def ShowBar(Bar):
    sys.stdout.write(Bar)
    sys.stdout.flush()


def Equivalent_time(tim1, max_t1, tim2, max_t2):
    tk2 = tim1 / max_t1
    tc2 = np.arange(0.0, 1 + tk2, tk2)
    tc2[tc2 >= 1] = 1
    tc2 = tc2.reshape(-1, 1)  # reference scaled to 1

    tc2r = np.arange(0.0, max_t1 + tim1, tim1)
    tc2r[tc2r >= max_t1] = max_t1
    tc2r = tc2r.reshape(-1, 1)  # reference original
    func = interpolate.interp1d(tc2r.ravel(), tc2.ravel())

    tc2rr = np.arange(0.0, max_t2 + tim2, tim2)
    tc2rr[tc2rr >= max_t2] = max_t2
    tc2rr = tc2rr.reshape(-1, 1)  # reference original
    ynew = func(tc2rr.ravel())

    return ynew


##############################################################################
#         FINITE VOLUME RESERVOIR SIMULATOR
##############################################################################
def Reservoir_Simulator(
    Kuse,
    porosity,
    quse,
    quse_water,
    nx,
    ny,
    nz,
    factorr,
    max_t,
    Dx,
    Dy,
    Dz,
    BO,
    BW,
    CFL,
    timmee,
    MAXZ,
    PB,
    PATM,
    CFO,
    IWSw,
    method,
    steppi,
    SWI,
    SWR,
    UW,
    UO,
    step2,
    pini_alt,
):
    """
    Reservoir_Simulator function for 2 phase flow

    This function simulates the flow of fluids in a porous reservoir by solving the
    pressure and saturation equations using different numerical methods.
    The function takes the following parameters:

    Kuse: an array of shape (nx, ny, nz) representing the permeability values in the reservoir.

    porosity: an array of shape (nx, ny, nz) representing the porosity values in the reservoir.

    quse: an array of shape (nx, ny, nz) representing the source terms in the reservoir.

    quse_water: an array of shape (nx, ny, nz) representing the water source terms in the reservoir.

    nx, ny, nz: integers representing the number of grid cells in the x, y, and z directions, respectively.

    factorr: a float representing the anisotropy factor for the permeability tensor.

    max_t: a float representing the maximum simulation time.

    Dx, Dy, Dz: floats representing the dimensions of the grid cells in the x, y, and z directions, respectively.

    BO: a float representing the initial oil formation volume factor.

    BW: a float representing the initial water formation volume factor.

    CFL: an integer representing whether or not to use CFL condition for time step control.

    timmee: a float representing the total simulation time.

    MAXZ: a float representing the maximum depth of the reservoir.

    PB: a float representing the reservoir pressure at the bottom.

    PATM: a float representing the atmospheric pressure.

    CFO: a float representing the compressibility of the formation.

    IWSw: a float representing the initial water saturation in the reservoir.

    method: an integer representing the numerical method to use for solving the equations.

    steppi: an integer representing the number of time steps to take.

    SWI: a float representing the irreducible water saturation.

    SWR: a float representing the residual water saturation.

    UW: a float representing the water viscosity.

    UO: a float representing the oil viscosity.

    typee: an integer representing the solver type for AMGX.

    step2: an integer representing the number of sub-steps to use for implicit saturation calculations.

    pini_alt: a float representing the initial pressure in the reservoir.

    Returns:

    Big: a numpy array of shape (steppi, nx, ny, 2) representing the pressure and saturation fields over time.
    """
    text = """
                                                                                           
    NNNNNNNN        NNNNNNNVVVVVVVV           VVVVVVVRRRRRRRRRRRRRRRRR     SSSSSSSSSSSSSSS 
    N:::::::N       N::::::V::::::V           V::::::R::::::::::::::::R  SS:::::::::::::::S
    N::::::::N      N::::::V::::::V           V::::::R::::::RRRRRR:::::RS:::::SSSSSS::::::S
    N:::::::::N     N::::::V::::::V           V::::::RR:::::R     R:::::S:::::S     SSSSSSS
    N::::::::::N    N::::::NV:::::V           V:::::V  R::::R     R:::::S:::::S            
    N:::::::::::N   N::::::N V:::::V         V:::::V   R::::R     R:::::S:::::S            
    N:::::::N::::N  N::::::N  V:::::V       V:::::V    R::::RRRRRR:::::R S::::SSSS         
    N::::::N N::::N N::::::N   V:::::V     V:::::V     R:::::::::::::RR   SS::::::SSSSS    
    N::::::N  N::::N:::::::N    V:::::V   V:::::V      R::::RRRRRR:::::R    SSS::::::::SS  
    N::::::N   N:::::::::::N     V:::::V V:::::V       R::::R     R:::::R      SSSSSS::::S 
    N::::::N    N::::::::::N      V:::::V:::::V        R::::R     R:::::R           S:::::S
    N::::::N     N:::::::::N       V:::::::::V         R::::R     R:::::R           S:::::S
    N::::::N      N::::::::N        V:::::::V        RR:::::R     R:::::SSSSSSS     S:::::S
    N::::::N       N:::::::N         V:::::V         R::::::R     R:::::S::::::SSSSSS:::::S
    N::::::N        N::::::N          V:::V          R::::::R     R:::::S:::::::::::::::SS 
    NNNNNNNN         NNNNNNN           VVV           RRRRRRRR     RRRRRRRSSSSSSSSSSSSSSS  
    """
    print(text)
    # Compute transmissibilities by harmonic averaging using Two-point flux approimxation

    Nx = cp.int32(nx)
    Dx = cp.int32(Dx)
    hx = Dx / Nx
    Ny = cp.int32(ny)
    Dy = cp.int32(Dy)
    hy = Dy / Ny
    Nz = cp.int32(nz)
    Dz = cp.int32(Dz)
    hz = Dz / Nz
    N = Nx * Ny * Nz

    tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)

    dt = np.diff(tc2)[0]

    porosity = cp.asarray(porosity)
    Vol = hx * hy * hz
    S = IWSw * cp.ones((N, 1), dtype=cp.float32)  # (SWI*cp.ones((N,1)))
    Kq = cp.zeros((3, nx, ny, nz))
    datause = cp.asarray(Kuse)
    Qq = cp.asarray(quse)
    quse_water = cp.asarray(quse_water)

    # datause = Kuse#cp.reshape(Kuse,(nx,ny,nz),'F') #(dataa.reshape(nx,ny,1))
    Kq[0, :, :, :] = datause
    Kq[1, :, :, :] = datause
    Kq[2, :, :, :] = factorr * datause

    Runs = tc2.shape[0]
    ty = np.arange(1, Runs + 1)
    # print('-----------------------------FORWARDING---------------------------')

    St = dt
    Nx = nx
    Ny = ny
    Nz = nz

    N = Nx * Ny * Nz
    hx = 1 / Nx
    hy = 1 / Ny
    hz = 1 / Nz
    tx = 2 * hy * hz / hx
    ty = 2 * hx * hz / hy
    tz = 2 * hx * hy / hz

    if nz == 1:
        output_allp = cp.zeros((steppi, nx, ny))
        output_alls = cp.zeros((steppi, nx, ny))
    else:
        output_allp = cp.zeros((steppi, nx, ny, nz))
        output_alls = cp.zeros((steppi, nx, ny, nz))

    TX = cp.zeros((Nx + 1, Ny, Nz))
    TY = cp.zeros((Nx, Ny + 1, Nz))
    TZ = cp.zeros((Nx, Ny, Nz + 1))

    b = Qq
    for t in range(tc2.shape[0] - 1):
        # Mw,Mo,_,_= RelPerm(S,UW,UO,BW,BO,SWI,SWR)

        Sout = (S - SWI) / (1 - SWI - SWR)
        Mw = (Sout**2) / (UW * BW)  # Water mobility
        Mo = (1 - Sout) ** 2 / (UO * BO)  # Oil mobility
        Mt = cp.add(Mw, Mo)

        # RelPerm3(S,So, UW, UO, UG, BW, BO, BG, SWI, SWR, nx, ny, nz)

        atemp = cp.stack([Mt, Mt, Mt], axis=1)
        KM = cp.multiply(cp.reshape(atemp.T, (3, nx, ny, nz), "F"), Kq)

        Ll = 1.0 / KM

        TX[1:Nx, :, :] = tx / (Ll[0, : Nx - 1, :, :] + Ll[0, 1:Nx, :, :])
        TY[:, 1:Ny, :] = ty / (Ll[1, :, : Ny - 1, :] + Ll[1, :, 1:Ny, :])
        TZ[:, :, 1:Nz] = tz / (Ll[2, :, :, : Nz - 1] + Ll[2, :, :, 1:Nz])
        # Assemble TPFA discretization matrix.
        x1 = cp.reshape(TX[:Nx, :, :], (N), "F")
        x2 = cp.reshape(TX[1 : Nx + 2, :, :], (N), "F")
        y1 = cp.reshape(TY[:, :Ny, :], (N), "F")
        y2 = cp.reshape(TY[:, 1 : Ny + 2, :], (N), "F")
        z1 = cp.reshape(TZ[:, :, :Nz], (N), "F")
        z2 = cp.reshape(TZ[:, :, 1 : Nz + 2], (N), "F")
        DiagVecs = [-z2, -y2, -x2, x1 + x2 + y1 + y2 + z1 + z2, -x1, -y1, -z1]
        DiagIndx = [-Nx * Ny, -Nx, -1, 0, 1, Nx, Nx * Ny]
        A = spdiags(DiagVecs, DiagIndx, N, N, format="csr")
        A[0, 0] = A[0, 0] + cp.sum(Kq[:, 0, 0, 0])

        if method == 1:  # GMRES
            M2 = spilu(A)

            def M_x(x):
                return M2.solve(x)

            M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
            u, exitCode = gmres(A, b, tol=1e-6, atol=0, restart=20, maxiter=100, M=M)

        elif method == 2:  # SPSOLVE
            u = spsolve(A, b)

        elif method == 3:  # CONJUGATE GRADIENT
            M2 = spilu(A)

            def M_x(x):
                return M2.solve(x)

            M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
            u, exitCode = cg(A, b, tol=1e-6, atol=0, maxiter=100, M=M)

        elif method == 4:  # LSQR
            u, istop, itn, normr = lsqr(A, b)[:4]
        else:  # adaptive AMG
            u = v_cycle(
                A,
                b,
                x=cp.zeros_like(b),
                smoother="SOR",
                levels=2,
                tol=1e-6,
                smoothing_steps=2,
            )

        P = cp.reshape(u, (Nx, Ny, Nz), "F")
        V = {
            "x": cp.zeros((Nx + 1, Ny, Nz)),
            "y": cp.zeros((Nx, Ny + 1, Nz)),
            "z": cp.zeros((Nx, Ny, Nz + 1)),
        }
        V["x"][1:Nx, :, :] = (P[: Nx - 1, :, :] - P[1:Nx, :, :]) * TX[1:Nx, :, :]
        V["y"][:, 1:Ny, :] = (P[:, : Ny - 1, :] - P[:, 1:Ny, :]) * TY[:, 1:Ny, :]
        V["z"][:, :, 1:Nz] = (P[:, :, : Nz - 1] - P[:, :, 1:Nz]) * TZ[:, :, 1:Nz]

        if CFL == 1:
            S = Upstream_2PHASE(
                nx,
                ny,
                nz,
                S,
                UW,
                UO,
                BW,
                BO,
                SWI,
                SWR,
                Vol,
                quse_water,
                V,
                dt,
                porosity,
            )
        else:
            for ts in range(step2):
                S = NewtRaph(
                    nx,
                    ny,
                    nz,
                    porosity,
                    Vol,
                    S,
                    V,
                    quse_water,
                    St / float(step2),
                    UW,
                    UO,
                    SWI,
                    SWR,
                    method,
                    BW,
                    BO,
                )

        # print(msg)
        S = np.clip(S, SWI, 1)
        S2 = np.reshape(S, (Nx, Ny, Nz), "F")
        # soil = np.clip((1-S),SWI,1)
        pinii = cp.reshape(P, (-1, 1), "F")

        if nz == 1:
            output_allp[t, :, :] = P[:, :, 0]
            output_alls[t, :, :] = cp.asarray(S2[:, :, 0])
        else:
            output_allp[t, :, :, :] = P
            output_alls[t, :, :, :] = cp.asarray(S2)

        Ppz = cp.mean(pinii.reshape(-1, 1), axis=0)

        BO = cp.float32(np.ndarray.item(cp.asnumpy(calc_bo(PB, PATM, CFO, Ppz))))

    # """
    Big = cp.vstack([output_allp, output_alls])
    return cp.asnumpy(Big)


##############################################################################
#         FINITE VOLUME RESERVOIR SIMULATOR
##############################################################################
def Reservoir_Simulator2(
    Kuse,
    porosity,
    quse,
    quse_water,
    quse_oil,
    nx,
    ny,
    nz,
    factorr,
    max_t,
    Dx,
    Dy,
    Dz,
    BO,
    BW,
    BG,
    RS,
    CFL,
    timmee,
    MAXZ,
    PB,
    PATM,
    CFO,
    IWSw,
    IWSo,
    method,
    steppi,
    SWI,
    SWR,
    UW,
    UO,
    UG,
    step2,
    pini_alt,
):
    # Compute transmissibilities by harmonic averaging using Two-point flux approimxation

    Nx = cp.int32(nx)
    Dx = cp.int32(Dx)
    hx = Dx / Nx
    Ny = cp.int32(ny)
    Dy = cp.int32(Dy)
    hy = Dy / Ny
    Nz = cp.int32(nz)
    Dz = cp.int32(Dz)
    hz = Dz / Nz
    N = Nx * Ny * Nz

    tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)

    dt = np.diff(tc2)[0]

    porosity = cp.asarray(porosity)
    Vol = hx * hy * hz
    S = IWSw * cp.ones((N, 1), dtype=cp.float32)  # (SWI*cp.ones((N,1)))
    Soil = IWSo * cp.ones((N, 1), dtype=cp.float32)  # (SWI*cp.ones((N,1)))
    Kq = cp.zeros((3, nx, ny, nz))
    datause = cp.asarray(Kuse)
    Qq = cp.asarray(quse)
    quse_water = cp.asarray(quse_water)
    quse_oil = cp.asarray(quse_oil)
    # datause = Kuse#cp.reshape(Kuse,(nx,ny,nz),'F') #(dataa.reshape(nx,ny,1))
    Kq[0, :, :, :] = datause
    Kq[1, :, :, :] = datause
    Kq[2, :, :, :] = factorr * datause

    Runs = tc2.shape[0]
    ty = np.arange(1, Runs + 1)
    # print('-----------------------------FORWARDING---------------------------')

    St = dt
    Nx = nx
    Ny = ny
    Nz = nz

    N = Nx * Ny * Nz
    hx = 1 / Nx
    hy = 1 / Ny
    hz = 1 / Nz
    tx = 2 * hy * hz / hx
    ty = 2 * hx * hz / hy
    tz = 2 * hx * hy / hz

    if nz == 1:
        output_allp = cp.zeros((steppi, nx, ny))
        output_alls = cp.zeros((steppi, nx, ny))
        output_allsoil = cp.zeros((steppi, nx, ny))
        output_allsgas = cp.zeros((steppi, nx, ny))
    else:
        output_allp = cp.zeros((steppi, nx, ny, nz))
        output_alls = cp.zeros((steppi, nx, ny, nz))
        output_allsoil = cp.zeros((steppi, nx, ny, nz))
        output_allsgas = cp.zeros((steppi, nx, ny, nz))

    TX = cp.zeros((Nx + 1, Ny, Nz))
    TY = cp.zeros((Nx, Ny + 1, Nz))
    TZ = cp.zeros((Nx, Ny, Nz + 1))

    b = Qq
    for t in range(tc2.shape[0] - 1):
        # step = t

        Mw, Mo, Mg, _, _, _ = RelPerm3(
            S, Soil, UW, UO, UG, BW, BO, BG, SWI, SWR, nx, ny, nz
        )

        Mt = Mw + Mo + Mg + Mo * RS  # cp.add(cp.add(Mw,Mo),Mg)
        atemp = cp.stack([Mt, Mt, Mt], axis=1)
        KM = cp.multiply(cp.reshape(atemp.T, (3, nx, ny, nz), "F"), Kq)

        Ll = 1.0 / KM

        TX[1:Nx, :, :] = tx / (Ll[0, : Nx - 1, :, :] + Ll[0, 1:Nx, :, :])
        TY[:, 1:Ny, :] = ty / (Ll[1, :, : Ny - 1, :] + Ll[1, :, 1:Ny, :])
        TZ[:, :, 1:Nz] = tz / (Ll[2, :, :, : Nz - 1] + Ll[2, :, :, 1:Nz])
        # Assemble TPFA discretization matrix.
        x1 = cp.reshape(TX[:Nx, :, :], (N), "F")
        x2 = cp.reshape(TX[1 : Nx + 2, :, :], (N), "F")
        y1 = cp.reshape(TY[:, :Ny, :], (N), "F")
        y2 = cp.reshape(TY[:, 1 : Ny + 2, :], (N), "F")
        z1 = cp.reshape(TZ[:, :, :Nz], (N), "F")
        z2 = cp.reshape(TZ[:, :, 1 : Nz + 2], (N), "F")
        DiagVecs = [-z2, -y2, -x2, x1 + x2 + y1 + y2 + z1 + z2, -x1, -y1, -z1]
        DiagIndx = [-Nx * Ny, -Nx, -1, 0, 1, Nx, Nx * Ny]
        A = spdiags(DiagVecs, DiagIndx, N, N, format="csr")
        A[0, 0] = A[0, 0] + cp.sum(Kq[:, 0, 0, 0])

        if method == 1:  # GMRES
            M2 = spilu(A)

            def M_x(x):
                return M2.solve(x)

            M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
            u, exitCode = gmres(A, b, tol=1e-6, atol=0, restart=20, maxiter=100, M=M)

        elif method == 2:  # SPSOLVE
            u = spsolve(A, b)

        elif method == 3:  # CONJUGATE GRADIENT
            M2 = spilu(A)

            def M_x(x):
                return M2.solve(x)

            M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
            u, exitCode = cg(A, b, tol=1e-6, atol=0, maxiter=100, M=M)

        elif method == 4:  # LSQR
            u, istop, itn, normr = lsqr(A, b)[:4]
        else:  # adaptive AMG
            u = v_cycle(
                A,
                b,
                x=cp.zeros_like(b),
                smoother="SOR",
                levels=2,
                tol=1e-6,
                smoothing_steps=2,
            )

        P = cp.reshape(u, (Nx, Ny, Nz), "F")
        V = {
            "x": cp.zeros((Nx + 1, Ny, Nz)),
            "y": cp.zeros((Nx, Ny + 1, Nz)),
            "z": cp.zeros((Nx, Ny, Nz + 1)),
        }
        V["x"][1:Nx, :, :] = (P[: Nx - 1, :, :] - P[1:Nx, :, :]) * TX[1:Nx, :, :]
        V["y"][:, 1:Ny, :] = (P[:, : Ny - 1, :] - P[:, 1:Ny, :]) * TY[:, 1:Ny, :]
        V["z"][:, :, 1:Nz] = (P[:, :, : Nz - 1] - P[:, :, 1:Nz]) * TZ[:, :, 1:Nz]

        if CFL == 1:
            S, Soil = Upstream_3PHASE(
                nx,
                ny,
                nz,
                S,
                Soil,
                UW,
                UO,
                UG,
                BW,
                BO,
                BG,
                RS,
                SWI,
                SWR,
                Vol,
                quse_water,
                quse_oil,
                V,
                dt,
                porosity,
            )
        else:
            for ts in range(step2):
                S, Soil = NewtRaph2(
                    nx,
                    ny,
                    nz,
                    porosity,
                    Vol,
                    S,
                    Soil,
                    V,
                    quse_water,
                    quse_oil,
                    St / float(step2),
                    UW,
                    UO,
                    UG,
                    SWI,
                    SWR,
                    method,
                    BW,
                    BO,
                    BG,
                    RS,
                )

        # print(msg)
        S = np.clip(S, SWI, 1)
        # Soil = np.clip(Soil, 0., 1)

        S2 = np.reshape((S), (Nx, Ny, Nz), "F")  # it is water
        S2oil = np.reshape((Soil), (Nx, Ny, Nz), "F")  # it is gas
        S2gas = 1 - abs(S2 + S2oil)  # it is oil
        S2gas = np.clip(S2gas, SWI, 1)
        # soil = np.clip((1-S),SWI,1)
        pinii = cp.reshape(P, (-1, 1), "F")

        if nz == 1:
            output_allp[t, :, :] = P[:, :, 0]
            output_alls[t, :, :] = cp.asarray(S2[:, :, 0])
            output_allsoil[t, :, :] = cp.asarray(S2gas[:, :, 0])
            output_allsgas[t, :, :] = cp.asarray(S2oil[:, :, 0])
        else:
            output_allp[t, :, :, :] = P
            output_alls[t, :, :, :] = cp.asarray(S2)
            output_allsoil[t, :, :, :] = cp.asarray(S2gas)
            output_allsgas[t, :, :, :] = cp.asarray(S2oil)

        Ppz = cp.mean(pinii.reshape(-1, 1), axis=0)

        BO = cp.float32(np.ndarray.item(cp.asnumpy(calc_bo(PB, PATM, CFO, Ppz))))
        BG = cp.float32(np.ndarray.item(cp.asnumpy(calc_bg(PB, PATM, Ppz))))
        RS = cp.float32(np.ndarray.item(cp.asnumpy(calc_rs(PB, Ppz))))

        # if (BG <= 0.001):
        #     BG = 0.001

        # if (BG >= 0.01):
        #     BG = 0.01

        # if (RS <= 50.0):
        #     RS = 50.
        # if (RS >= 2000.0):
        #     RS = 2000.

    # """
    Big = cp.vstack([output_allp, output_alls, output_allsoil, output_allsgas])
    return cp.asnumpy(Big)


def compute_f(
    pressure, kuse, krouse, krwuse, rwell1, skin, pwf_producer1, UO, BO, DX, UW, BW, DZ
):
    RE = 0.2 * cp.asarray(DX)
    up = UO * BO

    # facc = tf.constant(10,dtype = tf.float64)

    DZ = cp.asarray(DZ)
    down = 2.0 * cp.pi * kuse * krouse * DZ
    # down = piit * pii * krouse * DZ1

    right = cp.log(RE / cp.asarray(rwell1)) + cp.asarray(skin)
    J = down / (up * right)
    drawdown = pressure - cp.asarray(pwf_producer1)
    qoil = -((drawdown) * J)
    aa = qoil * 1e-5
    # aa[aa<=0] = 0
    # print(aa)

    # water production
    up2 = UW * BW
    down = 2.0 * cp.pi * kuse * krwuse * DZ
    J = down / (up2 * right)
    drawdown = pressure - cp.asarray(pwf_producer1)
    qwater = -((drawdown) * J)
    aaw = qwater * 1e-5
    # aaw = (qwater)
    # aaw[aaw<=0] = 0
    # print(qwater)
    ouut = aa + aaw
    return -(ouut)  # outnew


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_numpy_pytorch(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_pytorch_numpy(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (maximum - minimum) / (new_max - new_min)
    b = minimum - m * new_min
    return m * array + b


def plot3d2static(arr_3d, nx, ny, nz, namet, titti, maxii, minii, injectors, producers):
    """
    Plot a 3D array with matplotlib and annotate specific points on the plot.

    Args:
    arr_3d (np.ndarray): 3D array to plot.
    nx (int): number of cells in the x direction.
    ny (int): number of cells in the y direction.
    nz (int): number of cells in the z direction.
    itt (int): current iteration number.
    dt (float): time step.
    MAXZ (int): maximum number of iterations in the z direction.
    namet (str): name of the file to save the plot.
    titti (str): title of the plot.
    maxii (float): maximum value of the colorbar.
    minii (float): minimum value of the colorbar.

    Returns:
    None.
    """
    fig = plt.figure(figsize=(15, 15), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Shift the coordinates to center the points at the voxel locations
    x, y, z = np.indices((arr_3d.shape))
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5

    # Set the colors of each voxel using a jet colormap
    colors = plt.cm.jet(arr_3d)
    norm = matplotlib.colors.Normalize(vmin=minii, vmax=maxii)

    # Plot each voxel and save the mappable object
    ax.voxels(arr_3d, facecolors=colors, alpha=0.5, edgecolor="none", shade=True)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

    if titti == "Pressure":
        plt.colorbar(m, fraction=0.02, pad=0.1, label="Pressure [psia]")
    elif titti == "water_sat":
        plt.colorbar(m, fraction=0.02, pad=0.1, label="water_sat [units]")
    else:
        plt.colorbar(m, fraction=0.02, pad=0.1, label="oil_sat [psia]")

    # Add a colorbar for the mappable object
    # plt.colorbar(mappable)
    # Set the axis labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    # ax.set_title(titti,fontsize= 14)

    # Set axis limits to reflect the extent of each axis of the matrix
    ax.set_xlim(0, arr_3d.shape[0])
    ax.set_ylim(0, arr_3d.shape[1])
    ax.set_zlim(0, arr_3d.shape[2])
    # ax.set_zlim(0, 60)

    # Remove the grid
    ax.grid(False)

    # Set lighting to bright
    ax.set_facecolor("white")
    # Set the aspect ratio of the plot

    ax.set_box_aspect([nx, ny, nz * 2])

    # Set the projection type to orthogonal
    ax.set_proj_type("ortho")

    # Remove the tick labels on each axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove the tick lines on each axis
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4

    # Set the azimuth and elevation to make the plot brighter
    ax.view_init(elev=30, azim=45)

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "black", linewidth=2)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="black",
            weight="bold",
            fontsize=16,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "r", linewidth=2)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="r",
            weight="bold",
            fontsize=16,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.suptitle(titti, fontsize=16)

    name = namet + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()


def plot3d2(
    arr_3d, nx, ny, nz, itt, dt, MAXZ, namet, titti, maxii, minii, injectors, producers
):
    """
    Plot a 3D array with matplotlib and annotate specific points on the plot.

    Args:
    arr_3d (np.ndarray): 3D array to plot.
    nx (int): number of cells in the x direction.
    ny (int): number of cells in the y direction.
    nz (int): number of cells in the z direction.
    itt (int): current iteration number.
    dt (float): time step.
    MAXZ (int): maximum number of iterations in the z direction.
    namet (str): name of the file to save the plot.
    titti (str): title of the plot.
    maxii (float): maximum value of the colorbar.
    minii (float): minimum value of the colorbar.

    Returns:
    None.
    """
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Shift the coordinates to center the points at the voxel locations
    x, y, z = np.indices((arr_3d.shape))
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5

    # Set the colors of each voxel using a jet colormap
    colors = plt.cm.jet(arr_3d)
    norm = matplotlib.colors.Normalize(vmin=minii, vmax=maxii)

    # Plot each voxel and save the mappable object
    ax.voxels(arr_3d, facecolors=colors, alpha=0.5, edgecolor="none", shade=True)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

    if titti == "Pressure":
        plt.colorbar(m, fraction=0.02, pad=0.1, label="Pressure [psia]")
    elif titti == "water_sat":
        plt.colorbar(m, fraction=0.02, pad=0.1, label="water_sat [units]")
    else:
        plt.colorbar(m, fraction=0.02, pad=0.1, label="oil_sat [psia]")

    # Add a colorbar for the mappable object
    # plt.colorbar(mappable)
    # Set the axis labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    # Set axis limits to reflect the extent of each axis of the matrix
    ax.set_xlim(0, arr_3d.shape[0])
    ax.set_ylim(0, arr_3d.shape[1])
    ax.set_zlim(0, arr_3d.shape[2])
    # ax.set_zlim(0, 10)

    # Remove the grid
    ax.grid(False)

    # Set lighting to bright
    ax.set_facecolor("white")
    # Set the aspect ratio of the plot

    ax.set_box_aspect([nx, ny, nz * 2])

    # Set the projection type to orthogonal
    ax.set_proj_type("ortho")

    # Remove the tick labels on each axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove the tick lines on each axis
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4

    # Set the azimuth and elevation to make the plot brighter
    ax.view_init(elev=30, azim=45)

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "black", linewidth=2)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="black",
            weight="bold",
            fontsize=16,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "r", linewidth=2)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="r",
            weight="bold",
            fontsize=16,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = str(titti) + "- Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = namet + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.close(fig)
