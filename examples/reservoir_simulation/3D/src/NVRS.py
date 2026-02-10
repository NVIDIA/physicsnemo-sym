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
NVIDIA Finite volume reservoir simulator with flexible solver

AMG to solve the pressure and saturation well possed inverse problem

Geostatistics packages are also provided

@Author: Clement Etienam

"""

print(".........................IMPORT SOME LIBRARIES.....................")
import os
import numpy as np


def is_available():
    """
    Check NVIDIA with nvidia-smi command
    Returning code 0 if no error, it means NVIDIA is installed
    Other codes mean not installed
    """
    code = os.system("nvidia-smi")
    return code


Yet = is_available()
if Yet == 0:
    print("GPU Available with CUDA")
    try:
        import pyamgx
    except:
        pyamgx = None

    import cupy as cp
    from numba import cuda

    print(cuda.detect())  # Print the GPU information
    import tensorflow as tf

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)
    # import pyamgx
    from cupyx.scipy.sparse import csr_matrix, spmatrix

    clementtt = 0
else:
    print("No GPU Available")
    import numpy as cp
    from scipy.sparse import csr_matrix

    clementtt = 1

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import os.path
import torch

# import yaml
from scipy import interpolate
import multiprocessing
import mpslib as mps
import numpy.matlib
from scipy.spatial.distance import cdist
from pyDOE import lhs
import matplotlib.colors
from matplotlib import cm
from shutil import rmtree
from kneed import KneeLocator
import numpy

# from PIL import Image
from scipy.fftpack import dct
import numpy.matlib
import matplotlib.lines as mlines

# os.environ['KERAS_BACKEND'] = 'tensorflow'
import os.path
import random
import os.path

# import dolfin as df
import sys
from numpy import *
import scipy.optimize.lbfgsb as lbfgsb
import numpy.linalg
from numpy.linalg import norm
from scipy.fftpack.realtransforms import idct
import numpy.ma as ma
import logging
import os
import matplotlib as mpl
from FyeldGenerator import generate_field
from imresize import *
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # I have just 1 GPU
from cpuinfo import get_cpu_info

# Prints a json string describing the cpu
s = get_cpu_info()
print("Cpu info")
for k, v in s.items():
    print(f"\t{k}: {v}")
cores = multiprocessing.cpu_count()

logger = logging.getLogger(__name__)
# numpy.random.seed(99)
print(" ")
print(" This computer has %d cores, which will all be utilised in parallel " % cores)
print(" ")
print("......................DEFINE SOME FUNCTIONS.....................")


def Reinvent(matt):
    nx, ny, nz = matt.shape[1], matt.shape[2], matt.shape[0]
    dess = np.zeros((nx, ny, nz))
    for i in range(nz):
        dess[:, :, i] = matt[i, :, :]
    return dess


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


def Plot_RSM_percentile(pertoutt, True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    P10 = pertoutt

    plt.figure(figsize=(40, 40))

    plt.subplot(4, 4, 1)
    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 1], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 2)
    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 2], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 3)
    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 3], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 4)
    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 4], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 5)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 5], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 6)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 6], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 7)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 7], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 8)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 8], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 9)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 9], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 10)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 10], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 11)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 11], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 12)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 12], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 13)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 13], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 14)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 14], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 15)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 15], color="blue", lw="2", label="PINO Model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(4, 4, 16)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 16], color="blue", lw="2", label="PINO Model")
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


def Plot_performance(
    PINN, PINN2, trueF, nx, ny, namet, UIR, itt, dt, MAXZ, pini_alt, steppi, wells
):
    look = (PINN[itt, :, :, :]) * pini_alt
    look_sat = PINN2[itt, :, :, :]
    look_oil = 1 - look_sat

    lookf = (trueF[itt, :, :, :]) * pini_alt

    # print(lookf.shape)
    lookf_sat = trueF[itt + steppi, :, :, :]
    lookf_oil = 1 - lookf_sat

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 3, 1)
    plt.pcolormesh(XX.T, YY.T, look[0, :, :], cmap="jet")
    plt.title(" Layer 1 - Pressure PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(
        np.min(np.reshape(lookf[:, :, 0], (-1,))),
        np.max(np.reshape(lookf[:, :, 0], (-1,))),
    )
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 4)
    plt.pcolormesh(XX.T, YY.T, look[1, :, :], cmap="jet")
    plt.title(" Layer 2 - Pressure PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(
        np.min(np.reshape(lookf[:, :, 1], (-1,))),
        np.max(np.reshape(lookf[:, :, 1], (-1,))),
    )
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 7)
    plt.pcolormesh(XX.T, YY.T, look[2, :, :], cmap="jet")
    plt.title(" Layer 3 - Pressure PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(
        np.min(np.reshape(lookf[:, :, 2], (-1,))),
        np.max(np.reshape(lookf[:, :, 2], (-1,))),
    )
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 2)
    plt.pcolormesh(XX.T, YY.T, lookf[:, :, 0], cmap="jet")
    plt.title(" Layer 1 - Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 5)
    plt.pcolormesh(XX.T, YY.T, lookf[:, :, 1], cmap="jet")
    plt.title(" Layer 2 - Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 8)
    plt.pcolormesh(XX.T, YY.T, lookf[:, :, 2], cmap="jet")
    plt.title(" Layer 3 - Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 3)
    plt.pcolormesh(XX.T, YY.T, abs(look[0, :, :] - lookf[:, :, 0]), cmap="jet")
    plt.title(" Layer 1 - Pressure (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 6)
    plt.pcolormesh(XX.T, YY.T, abs(look[1, :, :] - lookf[:, :, 1]), cmap="jet")
    plt.title(" Layer 2 - Pressure (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 9)
    plt.pcolormesh(XX.T, YY.T, abs(look[2, :, :] - lookf[:, :, 2]), cmap="jet")
    plt.title(" Layer 3 - Pressure (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = "pressure_evolution" + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 3, 1)
    plt.pcolormesh(XX.T, YY.T, look_sat[0, :, :], cmap="jet")
    plt.title("Layer 1 - water_sat PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(
        np.min(np.reshape(lookf_sat[:, :, 0], (-1,))),
        np.max(np.reshape(lookf_sat[:, :, 0], (-1,))),
    )
    cbar1.ax.set_ylabel(" water_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 4)
    plt.pcolormesh(XX.T, YY.T, look_sat[1, :, :], cmap="jet")
    plt.title("Layer 2 - water_sat PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(
        np.min(np.reshape(lookf_sat[:, :, 1], (-1,))),
        np.max(np.reshape(lookf_sat[:, :, 1], (-1,))),
    )
    cbar1.ax.set_ylabel(" water_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 7)
    plt.pcolormesh(XX.T, YY.T, look_sat[2, :, :], cmap="jet")
    plt.title("Layer 3 - water_sat PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(
        np.min(np.reshape(lookf_sat[:, :, 2], (-1,))),
        np.max(np.reshape(lookf_sat[:, :, 2], (-1,))),
    )
    cbar1.ax.set_ylabel(" water_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 2)
    plt.pcolormesh(XX.T, YY.T, lookf_sat[:, :, 0], cmap="jet")
    plt.title("Layer 1 - water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    # plt.clim(np.min(np.reshape(lookf_sat,(-1,))),np.max(np.reshape(lookf_sat,(-1,))))
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 5)
    plt.pcolormesh(XX.T, YY.T, lookf_sat[:, :, 1], cmap="jet")
    plt.title("Layer 2 - water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    # plt.clim(np.min(np.reshape(lookf_sat,(-1,))),np.max(np.reshape(lookf_sat,(-1,))))
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 8)
    plt.pcolormesh(XX.T, YY.T, lookf_sat[:, :, 2], cmap="jet")
    plt.title("Layer 3 - water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    # plt.clim(np.min(np.reshape(lookf_sat,(-1,))),np.max(np.reshape(lookf_sat,(-1,))))
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 3)
    plt.pcolormesh(XX.T, YY.T, abs(look_sat[0, :, :] - lookf_sat[:, :, 0]), cmap="jet")
    plt.title("Layer 1 - water_sat (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat ", fontsize=13)
    # plt.clim(np.min(np.reshape(lookf_sat,(-1,))),np.max(np.reshape(lookf_sat,(-1,))))
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 6)
    plt.pcolormesh(XX.T, YY.T, abs(look_sat[1, :, :] - lookf_sat[:, :, 1]), cmap="jet")
    plt.title("Layer 2 - water_sat (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat ", fontsize=13)
    # plt.clim(np.min(np.reshape(lookf_sat,(-1,))),np.max(np.reshape(lookf_sat,(-1,))))
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 9)
    plt.pcolormesh(XX.T, YY.T, abs(look_sat[2, :, :] - lookf_sat[:, :, 2]), cmap="jet")
    plt.title("Layer 3 - water_sat (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat ", fontsize=13)
    # plt.clim(np.min(np.reshape(lookf_sat,(-1,))),np.max(np.reshape(lookf_sat,(-1,))))
    Add_marker(plt, XX, YY, wells)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = "water_evolution" + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(15, 15))

    plt.subplot(3, 3, 1)
    plt.pcolormesh(XX.T, YY.T, look_oil[0, :, :], cmap="jet")
    plt.title("Layer 1- oil_sat PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(
        np.min(np.reshape(lookf_oil[:, :, 0], (-1,))),
        np.max(np.reshape(lookf_oil[:, :, 0], (-1,))),
    )
    cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 4)
    plt.pcolormesh(XX.T, YY.T, look_oil[1, :, :], cmap="jet")
    plt.title("Layer 2 - oil_sat PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(
        np.min(np.reshape(lookf_oil[:, :, 1], (-1,))),
        np.max(np.reshape(lookf_oil[:, :, 1], (-1,))),
    )
    cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 7)
    plt.pcolormesh(XX.T, YY.T, look_oil[2, :, :], cmap="jet")
    plt.title("Layer 3 - oil_sat PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(
        np.min(np.reshape(lookf_oil[:, :, 2], (-1,))),
        np.max(np.reshape(lookf_oil[:, :, 2], (-1,))),
    )
    cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 2)
    plt.pcolormesh(XX.T, YY.T, lookf_oil[:, :, 0], cmap="jet")
    plt.title(" Layer 1 - oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 5)
    plt.pcolormesh(XX.T, YY.T, lookf_oil[:, :, 1], cmap="jet")
    plt.title(" Layer 2 - oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 8)
    plt.pcolormesh(XX.T, YY.T, lookf_oil[:, :, 2], cmap="jet")
    plt.title(" Layer 3 - oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 3)
    plt.pcolormesh(XX.T, YY.T, abs(look_oil[0, :, :] - lookf_oil[:, :, 0]), cmap="jet")
    plt.title(" Layer 1 - oil_sat (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat ", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 6)
    plt.pcolormesh(XX.T, YY.T, abs(look_oil[1, :, :] - lookf_oil[:, :, 1]), cmap="jet")
    plt.title(" Layer 2 - oil_sat (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat ", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 9)
    plt.pcolormesh(XX.T, YY.T, abs(look_oil[2, :, :] - lookf_oil[:, :, 2]), cmap="jet")
    plt.title(" Layer 3 - oil_sat (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat ", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = "oil_evolution" + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()


def Plot_performance2(
    trueF, nx, ny, namet, UIR, itt, dt, MAXZ, pini_alt, steppi, wells
):
    lookf = (trueF[itt, :, :, :]) * pini_alt
    lookf_sat = trueF[itt + steppi, :, :, :]
    lookf_oil = 1 - lookf_sat

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 3, 1)
    plt.pcolormesh(XX.T, YY.T, lookf[:, :, 0], cmap="jet")
    plt.title(" Layer 1 - Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 4)
    plt.pcolormesh(XX.T, YY.T, lookf[:, :, 1], cmap="jet")
    plt.title(" Layer 2 - Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 7)
    plt.pcolormesh(XX.T, YY.T, lookf[:, :, 2], cmap="jet")
    plt.title(" Layer 3 - Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 2)
    plt.pcolormesh(XX.T, YY.T, lookf_sat[:, :, 0], cmap="jet")
    plt.title("Layer 1 - water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    # plt.clim(np.min(np.reshape(lookf_sat,(-1,))),np.max(np.reshape(lookf_sat,(-1,))))
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 5)
    plt.pcolormesh(XX.T, YY.T, lookf_sat[:, :, 1], cmap="jet")
    plt.title("Layer 2 - water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    # plt.clim(np.min(np.reshape(lookf_sat,(-1,))),np.max(np.reshape(lookf_sat,(-1,))))
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 8)
    plt.pcolormesh(XX.T, YY.T, lookf_sat[:, :, 2], cmap="jet")
    plt.title("Layer 3 - water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    # plt.clim(np.min(np.reshape(lookf_sat,(-1,))),np.max(np.reshape(lookf_sat,(-1,))))
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 3)
    plt.pcolormesh(XX.T, YY.T, lookf_oil[:, :, 0], cmap="jet")
    plt.title(" Layer 1 - oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 6)
    plt.pcolormesh(XX.T, YY.T, lookf_oil[:, :, 1], cmap="jet")
    plt.title(" Layer 2 - oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 9)
    plt.pcolormesh(XX.T, YY.T, lookf_oil[:, :, 2], cmap="jet")
    plt.title(" Layer 3 - oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = "evolution" + str(int(itt)) + ".png"
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
    LUB,
    HUB,
    aay,
    bby,
    DX,
    steppi,
    pini_alt,
    SWI,
    SWR,
    UW,
    BW,
    DZ,
    rwell,
    skin,
    UO,
    BO,
    pwf_producer,
    dt,
    N_inj,
    N_pr,
    CFD,
    nz,
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

    matI = Reinvent(inn["Qw"][0, 0, :, :, :].detach().cpu().numpy())
    matP = Reinvent(inn["Q"][0, 0, :, :, :].detach().cpu().numpy())

    Injector_location = np.where(matI.ravel() > 0)[0]
    producer_location = np.where(matP.ravel() < 0)[0]

    PERM = Reinvent(inn["perm"][0, 0, :, :, :].detach().cpu().numpy())
    PERM = rescale_linear_pytorch_numpy(
        np.reshape(PERM, (-1,), "F"), LUB, HUB, aay, bby
    )

    kuse_inj = PERM[Injector_location]

    kuse_prod = PERM[producer_location]

    RE = 0.2 * DX
    Baa = []
    Timz = []

    # print(ooutp.shape)
    for kk in range(steppi):
        Ptito = ooutp[0, kk, :, :, :]
        Stito = oouts[0, kk, :, :, :]

        if CFD == 0:
            Ptito = Reinvent(Ptito)
            Stito = Reinvent(Stito)
        else:
            pass

        average_pressure = (
            (Ptito.ravel()[producer_location]) * pini_alt
        )  # np.mean(Ptito.ravel()) * pini_alt #* (1/aug)
        p_inj = (Ptito.ravel()[Injector_location]) * pini_alt  # *(1/aug)
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
        right = np.log(RE / rwell) + skin
        temp = (up / down) * right
        # temp[temp ==-inf] = 0
        Pwf = p_inj + temp
        Pwf = np.abs(Pwf)
        BHP = np.sum(np.reshape(Pwf, (-1, N_inj), "C"), axis=0) / nz

        up = UO * BO
        down = 2 * np.pi * kuse_prod * krouse * DZ
        right = np.log(RE / rwell) + skin
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        drawdown = average_pressure - pwf_producer
        qoil = np.abs(-(drawdown * J))
        qoil = np.sum(np.reshape(qoil, (-1, N_pr), "C"), axis=0) / nz

        up = UW * BW
        down = 2 * np.pi * kuse_prod * krwusep * DZ
        right = np.log(RE / rwell) + skin
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        drawdown = average_pressure - pwf_producer
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


def getoptimumk(X):
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    myarray = np.array(distortions)

    knn = KneeLocator(
        Kss, myarray, curve="convex", direction="decreasing", interp_method="interp1d"
    )
    kuse = knn.knee

    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, "bx-")
    plt.xlabel("cluster size")
    plt.ylabel("Distortion")
    plt.title("optimal n_clusters for machine")
    plt.savefig("machine_elbow.png")
    plt.clf()
    return kuse


class LpLoss(object):
    """
    loss function with rel/abs Lp loss
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def H(y, t0=0):
    """
    Step fn with step at t0
    """
    np.zeros_like(y)
    tuple([slice(0, y.shape[i]) for i in y.ndim])


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
    Mo = (1 - S) ** 2 / (UO * BO)  # Oil mobility
    dMw = 2 * S / (UW * BW) / (1 - SWI - SWR)
    dMo = -2 * (1 - S) / (UO * BO) / (1 - SWI - SWR)

    return (
        cp.reshape(Mw, (-1, 1), "F"),
        cp.reshape(Mo, (-1, 1), "F"),
        cp.reshape(dMw, (-1, 1), "F"),
        cp.reshape(dMo, (-1, 1), "F"),
    )


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


def compute_metrics(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    TSS = np.sum((y_true - y_true_mean) ** 2)
    RSS = np.sum((y_true - y_pred) ** 2)

    R2 = 1 - (RSS / TSS)
    L2_accuracy = 1 - np.sqrt(RSS) / np.sqrt(TSS)

    return R2, L2_accuracy


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


def Plot_2D(XX, YY, plt, nx, ny, nz, Truee, N_injw, N_pr, varii, injectors, producers):
    Pressz = np.reshape(Truee, (nx, ny, nz), "F")
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())

    avg_2d = np.mean(Pressz, axis=2)

    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs
    # XX, YY = np.meshgrid(np.arange(nx),np.arange(ny))
    # plt.subplot(224)

    plt.pcolormesh(XX.T, YY.T, avg_2d, cmap="jet")
    cbar = plt.colorbar()

    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=11)
        plt.title("Permeability Field with well locations", fontsize=11, weight="bold")
    elif varii == "water PhysicsNeMo":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation -PhysicsNeMo", fontsize=11, weight="bold")
    elif varii == "water Numerical":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation - Numerical", fontsize=11, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=11)
        plt.title(
            "water saturation - (Numerical(GPU) -PhysicsNeMo)",
            fontsize=11,
            weight="bold",
        )

    elif varii == "oil PhysicsNeMo":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation -PhysicsNeMo", fontsize=11, weight="bold")

    elif varii == "oil Numerical":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation - Numerical", fontsize=11, weight="bold")

    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=11)
        plt.title(
            "oil saturation - (Numerical(GPU) -PhysicsNeMo)", fontsize=11, weight="bold"
        )

    elif varii == "pressure PhysicsNeMo":
        cbar.set_label("pressure(psia)", fontsize=11)
        plt.title("Pressure -PhysicsNeMo", fontsize=11, weight="bold")

    elif varii == "pressure Numerical":
        cbar.set_label("pressure(psia)", fontsize=11)
        plt.title("Pressure -Numerical", fontsize=11, weight="bold")

    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=11)
        plt.title(
            "Pressure - (Numerical(GPU) -PhysicsNeMo)", fontsize=11, weight="bold"
        )

    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=11)
        plt.title("Porosity Field", fontsize=11, weight="bold")
    cbar.mappable.set_clim(minii, maxii)

    plt.ylabel("Y", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers)


def Plot_PhysicsNeMo(ax, nx, ny, nz, Truee, N_injw, N_pr, varii, injectors, producers):
    # matplotlib.use('Agg')
    Pressz = np.reshape(Truee, (nx, ny, nz), "F")

    avg_2d = np.mean(Pressz, axis=2)

    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs

    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii

    masked_Pressz = Pressz
    colors = plt.cm.jet(masked_Pressz)
    # colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
    # alpha = np.where(np.isnan(Pressz), 0.0, 0.8) # set alpha to 0 for NaN values
    norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)

    arr_3d = Pressz
    # fig = plt.figure(figsize=(20, 20), dpi = 200)
    # ax = fig.add_subplot(221, projection='3d')

    # Shift the coordinates to center the points at the voxel locations
    x, y, z = np.indices((arr_3d.shape))
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5

    # Set the colors of each voxel using a jet colormap
    # colors = plt.cm.jet(arr_3d)
    # norm = matplotlib.colors.Normalize(vmin=minii, vmax=maxii)

    # Plot each voxel and save the mappable object
    ax.voxels(arr_3d, facecolors=colors, alpha=0.5, edgecolor="none", shade=True)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

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
    # ax.set_facecolor('white')
    # Set the aspect ratio of the plot

    ax.set_box_aspect([nx, ny, nz])

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

    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 7)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 7], "blue", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="blue",
            weight="bold",
            fontsize=5,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 5)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 5], "r", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="g",
            weight="bold",
            fontsize=5,
        )

    blue_line = mlines.Line2D([], [], color="blue", linewidth=2, label="water injector")
    green_line = mlines.Line2D([], [], color="red", linewidth=2, label="oil Producer")

    # Add the legend to the plot
    ax.legend(handles=[blue_line, green_line], loc="lower left", fontsize=9)

    # Add a horizontal colorbar to the plot
    cbar = plt.colorbar(m, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title(
            "Permeability Field with well locations", fontsize=12, weight="bold"
        )
    elif varii == "water PhysicsNeMo":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation -PhysicsNeMo", fontsize=12, weight="bold")
    elif varii == "water Numerical":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation - Numerical(GPU)", fontsize=12, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "water saturation - (Numerical(GPU) -PhysicsNeMo))",
            fontsize=12,
            weight="bold",
        )

    elif varii == "oil PhysicsNeMo":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation -PhysicsNeMo", fontsize=12, weight="bold")

    elif varii == "oil Numerical":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation - Numerical(GPU)", fontsize=12, weight="bold")

    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "oil saturation - (Numerical(GPU) -PhysicsNeMo))",
            fontsize=12,
            weight="bold",
        )

    elif varii == "pressure PhysicsNeMo":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -PhysicsNeMo", fontsize=12, weight="bold")

    elif varii == "pressure Numerical":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -Numerical(GPU)", fontsize=12, weight="bold")

    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "Pressure - (Numerical(GPU) -PhysicsNeMo))", fontsize=12, weight="bold"
        )

    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=12)
        ax.set_title("Porosity Field", fontsize=12, weight="bold")
    cbar.mappable.set_clim(minii, maxii)


def Plot_Models(True_mat):
    colors = ["r", "b", "k"]
    linestyles = ["-", "--", "--"]
    markers = ["o", "v", "*"]

    timezz = True_mat[0][:, 0].reshape(-1, 1)

    plt.figure(figsize=(40, 40))

    plt.subplot(4, 4, 1)
    plt.plot(
        timezz,
        True_mat[0][:, 1],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 1],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 1],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("BHP(Psia)", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("I1", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 2)
    plt.plot(
        timezz,
        True_mat[0][:, 2],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 2],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 2],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("BHP(Psia)", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("I2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 3)
    plt.plot(
        timezz,
        True_mat[0][:, 3],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 3],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 3],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("BHP(Psia)", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("I3", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 4)
    plt.plot(
        timezz,
        True_mat[0][:, 4],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 4],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 4],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("BHP(Psia)", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("I4", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 5)
    plt.plot(
        timezz,
        True_mat[0][:, 5],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 5],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 5],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$Q_{oil}(bbl/day)$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P1", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 6)
    plt.plot(
        timezz,
        True_mat[0][:, 6],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 6],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 6],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$Q_{oil}(bbl/day)$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P2", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 7)
    plt.plot(
        timezz,
        True_mat[0][:, 7],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 7],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 7],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$Q_{oil}(bbl/day)$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P3", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 8)
    plt.plot(
        timezz,
        True_mat[0][:, 8],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 8],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 8],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$Q_{oil}(bbl/day)$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P4", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 9)
    plt.plot(
        timezz,
        True_mat[0][:, 9],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 9],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 9],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$Q_{water}(bbl/day)$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P1", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 10)
    plt.plot(
        timezz,
        True_mat[0][:, 10],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 10],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 10],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$Q_{water}(bbl/day)$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P2", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 11)
    plt.plot(
        timezz,
        True_mat[0][:, 11],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 11],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 11],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$Q_{water}(bbl/day)$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P3", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 12)
    plt.plot(
        timezz,
        True_mat[0][:, 12],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 12],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 12],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$Q_{water}(bbl/day)$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P4", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 13)
    plt.plot(
        timezz,
        True_mat[0][:, 13],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 13],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 13],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$WWCT{%}$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P1", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 14)
    plt.plot(
        timezz,
        True_mat[0][:, 14],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 14],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 14],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$WWCT{%}$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P2", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 15)
    plt.plot(
        timezz,
        True_mat[0][:, 15],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 15],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 15],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$WWCT{%}$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P1", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    plt.subplot(4, 4, 16)
    plt.plot(
        timezz,
        True_mat[0][:, 16],
        linestyle=linestyles[0],
        marker=markers[0],
        markersize=1,
        color=colors[0],
        lw="2",
        label="Numerical Model",
    )
    plt.plot(
        timezz,
        True_mat[1][:, 16],
        linestyle=linestyles[1],
        marker=markers[1],
        markersize=1,
        color=colors[1],
        lw="2",
        label="FNO",
    )
    plt.plot(
        timezz,
        True_mat[2][:, 16],
        linestyle=linestyles[2],
        marker=markers[2],
        markersize=1,
        color=colors[2],
        lw="2",
        label="PINO",
    )
    plt.xlabel("Time (days)", weight="bold", fontsize=14)
    plt.ylabel("$WWCT{%}$", weight="bold", fontsize=14)
    # plt.ylim((0,25000))
    plt.title("P1", weight="bold", fontsize=14)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    # plt.legend()
    legend = plt.legend(fontsize="large", title_fontsize="large")
    for text in legend.get_texts():
        text.set_weight("bold")

    # os.chdir('RESULTS')
    plt.savefig(
        "Compare_models.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_bar(True_mat):
    a1 = rmsee(True_mat[1][:, 1:].ravel(), True_mat[0][:, 1:].ravel())
    a2 = rmsee(True_mat[2][:, 1:].ravel(), True_mat[0][:, 1:].ravel())

    models = ["FNO", "PINO"]

    rmse_values = [a1, a2]
    colors = ["red", "blue"]

    # Create a bar chart
    plt.figure(figsize=(10, 10))
    plt.bar(models, rmse_values, color=colors)

    # Add a title and labels
    plt.title("RMSE accuracy", weight="bold", fontsize=16)

    # Add x and y labels with bold and bigger font
    plt.xlabel("Surrogate Models", weight="bold", fontsize=14)
    plt.ylabel("RMSE", weight="bold", fontsize=14)
    plt.savefig(
        "Bar_chat.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def rmsee(predictions, targets):
    noww = predictions.reshape(-1, 1)
    measurment = targets.reshape(-1, 1)
    rmse_val = (np.sum(((noww - measurment) ** 2))) ** (0.5) / (measurment.shape[0])
    # the square root of the mean of the squared differences
    return rmse_val
