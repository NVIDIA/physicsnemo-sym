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
    from cupyx.scipy.sparse import spdiags
    from numba import cuda

    print(cuda.detect())  # Print the GPU information
    import tensorflow as tf

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)
    # import pyamgx
    from cupyx.scipy.sparse.linalg import (
        gmres,
        cg,
        spsolve,
        lsqr,
        LinearOperator,
        spilu,
    )
    from cupyx.scipy.sparse import csr_matrix, spmatrix
    from cupy import sparse

    clementtt = 0
else:
    print("No GPU Available")
    import numpy as cp
    from scipy.sparse import csr_matrix
    from scipy.sparse import spdiags
    from scipy.sparse.linalg import gmres, spsolve, cg
    from scipy import sparse

    clementtt = 1

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import os.path
import torch
from scipy import interpolate
import multiprocessing
import mpslib as mps
from shutil import rmtree
import numpy.matlib
from scipy.spatial.distance import cdist
from pyDOE import lhs
import matplotlib.colors
from matplotlib import cm
from kneed import KneeLocator
import numpy

# from PIL import Image
from scipy.fftpack import dct
import numpy.matlib

# os.environ['KERAS_BACKEND'] = 'tensorflow'
import os.path
import time
import random

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
import math

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


def v_cycle(A, b, x, levels, tol, smoothing_steps=2):
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
    for level in range(levels):
        if level == levels - 1:
            # solve exactly on finest grid
            x = spsolve(A, b)

            if ((cp.sum(b - A @ x)) / b.shape[0]) < tol:
                break
            return x
        else:
            # perform smoothing
            for i in range(smoothing_steps):
                r = residual(A, b, x)
                x += spsolve(A, r)
            # restrict residual to coarser grid
            # coarse_r = restriction(r, coarse_grid_size=A.shape[0]//2)
            coarse_A, coarse_r = restriction(A, r)
            # solve exactly on coarser grid
            coarse_x = v_cycle(
                coarse_A, coarse_r, cp.zeros_like(coarse_r), levels - 1, tol
            )
            # interpolate solution back to fine grid
            x += prolongation(coarse_x, fine_grid_size=A.shape[0])
    return x


def restriction(A, f):
    """
    Coarsen the matrix A and vector f.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        The coefficient matrix.
    f : ndarray, shape (N,)
        The right-hand side vector.

    Returns
    -------
    A_coarse : ndarray, shape (N_coarse, N_coarse)
        The coarsened coefficient matrix.
    f_coarse : ndarray, shape (N_coarse,)
        The coarsened right-hand side vector.
    """
    N = A.shape[0]
    N_coarse = N // 2

    # Create the aggregation matrix P
    P = cp.sparse.eye(N, N_coarse, format="csr")

    # Coarsen the matrix and vector

    A_coarse = P.T @ A @ P
    f_coarse = P.T @ f

    return A_coarse, f_coarse


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
        mw, mo, dMw, dMo = RelPerm2(S, UW, UO, BW, BO, SWI, SWR)
        fw = cp.divide(mw, cp.add(mw, mo))
        Asa = A @ fw
        S = cp.add(S, cp.add(Asa, fi))
    return S


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


def NewtRaph(
    nx, ny, nz, porosity, Vol, S, V, qinn, Tt, UW, UO, SWI, SWR, method2, typee, BW, BO
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
                elif method2 == 5:  # adaptive AMG
                    # print(G.shape)
                    # print(dG.shape)
                    G = cp.ravel(G)
                    ds = v_cycle(
                        -dG,
                        G,
                        x=cp.zeros_like(G),
                        levels=2,
                        tol=1e-5,
                        smoothing_steps=2,
                    )

                if method2 == 6:  # CPR
                    M2 = spilu(-dG)

                    def M_x(x):
                        return M2.solve(x)

                    M = LinearOperator((nx * ny * nz, nx * ny * nz), M_x)
                    ds, exitCode = gmres(
                        -dG, G, tol=1e-6, atol=0, restart=20, maxiter=100, M=M
                    )

                else:  # AMGX
                    # Upload the Jacobian, L.H.S and solution vector
                    # sol = Pmatrix
                    jaco = -dG
                    rhs = G
                    sol = cp.zeros(nx * ny * nz, dtype=np.float64)
                    u = AMGX_Inverse_problem(jaco, rhs, sol, typee)
                    ds = cp.asarray(u)

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
    typee,
    step2,
    pini_alt,
):
    """
    Reservoir_Simulator function

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
    Kuse = cp.asarray(Kuse)
    Qq = cp.asarray(quse)
    quse_water = cp.asarray(quse_water)
    datause = Kuse  # cp.reshape(Kuse,(nx,ny,nz),'F') #(dataa.reshape(nx,ny,1))
    Kq[0, :, :, :] = datause
    Kq[1, :, :, :] = datause
    Kq[2, :, :, :] = factorr * datause

    if method == 7:
        pyamgx.initialize()

    Runs = tc2.shape[0]
    ty = np.arange(1, Runs + 1)
    print("-----------------------------FORWARDING---------------------------")

    St = dt
    output_allp = cp.zeros((steppi, nx, ny, nz))
    output_alls = cp.zeros((steppi, nx, ny, nz))
    for t in range(tc2.shape[0] - 1):
        # step = t
        progressBar = "\rSimulation Progress: " + ProgressBar(Runs - 1, t, Runs - 1)
        ShowBar(progressBar)
        time.sleep(1)

        # Mw,Mo,_,_= RelPerm(S,UW,UO,BW,BO,SWI,SWR)

        Sout = (S - SWI) / (1 - SWI - SWR)
        Mw = (Sout**2) / (UW * BW)  # Water mobility
        Mo = (1 - Sout) ** 2 / (UO * BO)  # Oil mobility

        Mt = cp.add(Mw, Mo)
        atemp = cp.stack([Mt, Mt, Mt], axis=1)
        KM = cp.multiply(cp.reshape(atemp.T, (3, nx, ny, nz), "F"), Kq)

        Nx = nx
        Ny = ny
        Nz = nz
        N = Nx * Ny * Nz
        hx = 1 / Nx
        hy = 1 / Ny
        hz = 1 / Nz
        Ll = 1.0 / KM
        tx = 2 * hy * hz / hx
        TX = cp.zeros((Nx + 1, Ny, Nz))
        ty = 2 * hx * hz / hy
        TY = cp.zeros((Nx, Ny + 1, Nz))
        tz = 2 * hx * hy / hz
        TZ = cp.zeros((Nx, Ny, Nz + 1))
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
        b = Qq

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
        elif method == 5:  # adaptive AMG
            u = v_cycle(A, b, x=cp.zeros_like(b), levels=2, tol=1e-5, smoothing_steps=2)
        elif method == 6:  # CPR
            u = v_cycle(A, b, x=cp.zeros_like(b), levels=2, tol=1e-5, smoothing_steps=2)

        else:  # AMGX
            # Upload the Jacobian, L.H.S and solution vector
            # sol = Pmatrix
            # pyamgx.initialize()
            jaco = A
            rhs = b
            sol = cp.zeros(nx * ny * nz, dtype=np.float64)
            u = AMGX_Inverse_problem(jaco, rhs, sol, typee)  # """
            # pyamgx.finalize()
            u = cp.asarray(u)

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
                    typee,
                    BW,
                    BO,
                )

        # print(msg)
        S = np.clip(S, SWI, 1)
        S2 = np.reshape(S, (Nx, Ny, Nz), "F")
        # soil = np.clip((1-S),SWI,1)
        pinii = cp.reshape(P, (-1, 1), "F")

        output_allp[t, :, :, :] = P / pini_alt
        output_alls[t, :, :, :] = cp.asarray(S2)

        Ppz = cp.mean(pinii.reshape(-1, 1), axis=0)

        BO = cp.float32(np.ndarray.item(cp.asnumpy(calc_bo(PB, PATM, CFO, Ppz))))
    progressBar = "\rSimulation Progress: " + ProgressBar(Runs - 1, t + 1, Runs - 1)
    ShowBar(progressBar)
    time.sleep(1)

    if method == 7:  # """
        pyamgx.finalize()
    # """
    Big = cp.vstack([output_allp, output_alls])
    return cp.asnumpy(Big)


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


def Get_actual_few(
    ini,
    nx,
    ny,
    nz,
    max_t,
    Dx,
    Dy,
    Dz,
    BO,
    BW,
    CFL,
    timmee,
    MAXZ,
    factorr,
    steppi,
    LIR,
    UIR,
    LUB,
    HUB,
    aay,
    bby,
    mpor,
    hpor,
    dt,
    IWSw,
    PB,
    PATM,
    CFO,
    method,
    SWI,
    SWR,
    UW,
    UO,
    typee,
    step2,
    pini_alt,
    input_channel,
    pena,
):
    paramss = ini
    Ne = paramss.shape[1]

    ct = np.zeros((Ne, input_channel, nx, ny, nz), dtype=np.float32)
    cP = np.zeros((Ne, 2 * steppi, nx, ny, nz), dtype=np.float32)

    kka = np.random.randint(LIR, UIR + 1, (1, Ne))

    A = np.zeros((nx, ny, nz))
    A1 = np.zeros((nx, ny, nz))

    for kk in range(Ne):
        ct1 = np.zeros((input_channel, nx, ny, nz), dtype=np.float32)
        print(str(kk + 1) + " | " + str(Ne))
        points = np.reshape(
            np.random.randint(1, nx, 16), (-1, 2), "F"
        )  # 16 is total number of wells

        Injcl = points[:4, :]
        prodcl = points[4:, :]

        a = np.reshape(paramss[:, kk], (nx, ny, nz), "F")
        inj_rate = kka[:, kk]

        at1 = paramss[:, kk]
        at1 = rescale_linear_numpy_pytorch(at1, LUB, HUB, aay, bby)

        at2 = paramss[:, kk]
        at2 = rescale_linear(at2, mpor, hpor)

        at1 = np.reshape(at1, (nx, ny, nz), "F")
        at2 = np.reshape(at2, (nx, ny, nz), "F")

        if pena == 1:
            for jj in range(nz):
                for m in range(prodcl.shape[0]):
                    A[prodcl[m, :][0], prodcl[m, :][1], jj] = -50

                for m in range(Injcl.shape[0]):
                    A[Injcl[m, :][0], Injcl[m, :][1], jj] = inj_rate

                for m in range(Injcl.shape[0]):
                    A1[Injcl[m, :][0], Injcl[m, :][1], jj] = inj_rate
        else:
            for jj in range(nz):
                A[1, 24, jj] = inj_rate
                A[1, 1, jj] = inj_rate
                A[31, 1, jj] = inj_rate
                A[31, 31, jj] = inj_rate
                A[7, 9, jj] = -50
                A[14, 12, jj] = -50
                A[28, 19, jj] = -50
                A[14, 27, jj] = -50

                A1[1, 24, jj] = inj_rate
                A1[1, 1, jj] = inj_rate
                A1[31, 1, jj] = inj_rate
                A1[31, 31, jj] = inj_rate

        quse1 = A

        quse = np.reshape(quse1, (-1, 1), "F")
        quse_water = np.reshape(A1, (-1, 1), "F")
        # print(A)
        A_out = Reservoir_Simulator(
            a,
            at2,
            quse.ravel(),
            quse_water.ravel(),
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
            typee,
            step2,
            pini_alt,
        )

        ct1[0, :, :, :] = at1  # permeability
        ct1[1, :, :, :] = quse1 / UIR  # Overall f
        ct1[2, :, :, :] = A1 / UIR  # f for water injection
        ct1[3, :, :, :] = at2  # porosity
        ct1[4, :, :, :] = dt * np.ones((nx, ny, nz))
        ct1[5, :, :, :] = np.ones((nx, ny, nz))  # Initial pressure
        ct1[6, :, :, :] = IWSw * np.ones((nx, ny, nz))  # Initial water saturation

        cP[kk, :, :, :, :] = A_out
        ct[kk, :, :, :, :] = ct1
    return ct, cP


def Get_actual_few_parallel(
    ini,
    nx,
    ny,
    nz,
    max_t,
    Dx,
    Dy,
    Dz,
    BO,
    BW,
    CFL,
    timmee,
    MAXZ,
    factorr,
    steppi,
    LIR,
    UIR,
    LUB,
    HUB,
    aay,
    bby,
    mpor,
    hpor,
    dt,
    IWSw,
    PB,
    PATM,
    CFO,
    method,
    SWI,
    SWR,
    UW,
    UO,
    typee,
    step2,
    pini_alt,
    input_channel,
    pena,
):
    paramss = ini
    Ne = paramss.shape[1]

    ct = np.zeros((Ne, input_channel, nx, ny, nz), dtype=np.float32)
    cP = np.zeros((Ne, 2 * steppi, nx, ny, nz), dtype=np.float32)

    kka = np.random.randint(LIR, UIR + 1, (1, Ne))

    A = np.zeros((nx, ny, nz))
    A1 = np.zeros((nx, ny, nz))

    for kk in range(Ne):
        ct1 = np.zeros((input_channel, nx, ny, nz), dtype=np.float32)
        print(str(kk + 1) + " | " + str(Ne))
        points = np.reshape(
            np.random.randint(1, nx, 16), (-1, 2), "F"
        )  # 16 is total number of wells

        Injcl = points[:4, :]
        prodcl = points[4:, :]

        a = np.reshape(paramss[:, kk], (nx, ny, nz), "F")
        inj_rate = kka[:, kk]

        at1 = paramss[:, kk]
        at1 = rescale_linear_numpy_pytorch(at1, LUB, HUB, aay, bby)

        at2 = paramss[:, kk]
        at2 = rescale_linear(at2, mpor, hpor)

        at1 = np.reshape(at1, (nx, ny, nz), "F")
        at2 = np.reshape(at2, (nx, ny, nz), "F")

        if pena == 1:
            for jj in range(nz):
                for m in range(prodcl.shape[0]):
                    A[prodcl[m, :][0], prodcl[m, :][1], jj] = -50

                for m in range(Injcl.shape[0]):
                    A[Injcl[m, :][0], Injcl[m, :][1], jj] = inj_rate

                for m in range(Injcl.shape[0]):
                    A1[Injcl[m, :][0], Injcl[m, :][1], jj] = inj_rate
        else:
            for jj in range(nz):
                A[1, 24, jj] = inj_rate
                A[1, 1, jj] = inj_rate
                A[31, 1, jj] = inj_rate
                A[31, 31, jj] = inj_rate
                A[7, 9, jj] = -50
                A[14, 12, jj] = -50
                A[28, 19, jj] = -50
                A[14, 27, jj] = -50

                A1[1, 24, jj] = inj_rate
                A1[1, 1, jj] = inj_rate
                A1[31, 1, jj] = inj_rate
                A1[31, 31, jj] = inj_rate

        quse1 = A

        quse = np.reshape(quse1, (-1, 1), "F")
        quse_water = np.reshape(A1, (-1, 1), "F")
        # print(A)
        A_out = Reservoir_Simulator(
            a,
            at2,
            quse.ravel(),
            quse_water.ravel(),
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
            typee,
            step2,
            pini_alt,
        )

        ct1[0, :, :, :] = at1  # permeability
        ct1[1, :, :, :] = quse1 / UIR  # Overall f
        ct1[2, :, :, :] = A1 / UIR  # f for water injection
        ct1[3, :, :, :] = at2  # porosity
        ct1[4, :, :, :] = dt * np.ones((nx, ny, nz))
        ct1[5, :, :, :] = np.ones((nx, ny, nz))  # Initial pressure
        ct1[6, :, :, :] = IWSw * np.ones((nx, ny, nz))  # Initial water saturation

        cP[kk, :, :, :, :] = A_out
        ct[kk, :, :, :, :] = ct1
    return ct, cP


def No_Sim(
    ini,
    nx,
    ny,
    nz,
    max_t,
    Dx,
    Dy,
    Dz,
    BO,
    BW,
    CFL,
    timmee,
    MAXZ,
    factorr,
    steppi,
    LIR,
    UIR,
    LUB,
    HUB,
    aay,
    bby,
    mpor,
    hpor,
    dt,
    IWSw,
    PB,
    PATM,
    CFO,
    method,
    SWI,
    SWR,
    UW,
    UO,
    typee,
    step2,
    pini_alt,
    input_channel,
    pena,
):
    paramss = ini
    Ne = paramss.shape[1]

    ct = np.zeros((Ne, input_channel, nx, ny, nz), dtype=np.float32)
    kka = np.random.randint(LIR, UIR + 1, (1, Ne))

    A = np.zeros((nx, ny, nz))
    A1 = np.zeros((nx, ny, nz))
    for kk in range(Ne):
        ct1 = np.zeros((input_channel, nx, ny, nz), dtype=np.float32)
        print(str(kk + 1) + " | " + str(Ne))
        # a = np.reshape(paramss[:,kk],(nx,ny,nz),'F')
        points = np.reshape(
            np.random.randint(1, nx, 16), (-1, 2), "F"
        )  # 16 is total number of wells

        Injcl = points[:4, :]
        prodcl = points[4:, :]

        inj_rate = kka[:, kk]

        at1 = paramss[:, kk]
        at1 = rescale_linear_numpy_pytorch(at1, LUB, HUB, aay, bby)

        at2 = paramss[:, kk]
        at2 = rescale_linear(at2, mpor, hpor)

        at1 = np.reshape(at1, (nx, ny, nz), "F")
        at2 = np.reshape(at2, (nx, ny, nz), "F")

        atemp = np.zeros((nx, ny, nz))
        atemp[:, :, 0] = at1[:, :, 0]

        if pena == 1:
            for jj in range(nz):
                for m in range(prodcl.shape[0]):
                    A[prodcl[m, :][0], prodcl[m, :][1], jj] = -50

                for m in range(Injcl.shape[0]):
                    A[Injcl[m, :][0], Injcl[m, :][1], jj] = inj_rate

                for m in range(Injcl.shape[0]):
                    A1[Injcl[m, :][0], Injcl[m, :][1], jj] = inj_rate
        else:
            for jj in range(nz):
                A[1, 24, jj] = inj_rate
                A[1, 1, jj] = inj_rate
                A[31, 1, jj] = inj_rate
                A[31, 31, jj] = inj_rate
                A[7, 9, jj] = -50
                A[14, 12, jj] = -50
                A[28, 19, jj] = -50
                A[14, 27, jj] = -50

                A1[1, 24, jj] = inj_rate
                A1[1, 1, jj] = inj_rate
                A1[31, 1, jj] = inj_rate
                A1[31, 31, jj] = inj_rate

        quse1 = A

        ct1[0, :, :, :] = at1  # permeability
        ct1[1, :, :, :] = quse1 / UIR  # Overall f
        ct1[2, :, :, :] = A1 / UIR  # f for water injection
        ct1[3, :, :, :] = at2  # porosity
        ct1[4, :, :, :] = dt * np.ones((nx, ny, nz))
        ct1[5, :, :, :] = np.ones((nx, ny, nz))  # Initial pressure
        ct1[6, :, :, :] = IWSw * np.ones((nx, ny, nz))  # Initial water saturation

        ct[kk, :, :, :, :] = ct1
    return ct


def inference_single(
    ini,
    inip,
    nx,
    ny,
    nz,
    max_t,
    Dx,
    Dy,
    Dz,
    BO,
    BW,
    CFL,
    timmee,
    MAXZ,
    factorr,
    steppi,
    LIR,
    UIR,
    LUB,
    HUB,
    aay,
    bby,
    mpor,
    hpor,
    dt,
    IWSw,
    PB,
    PATM,
    CFO,
    method,
    SWI,
    SWR,
    UW,
    UO,
    typee,
    step2,
    pini_alt,
    input_channel,
    kka,
):
    paramss = ini
    Ne = paramss.shape[1]

    ct = np.zeros((Ne, input_channel, nx, ny, nz), dtype=np.float32)
    cP = np.zeros((Ne, 2 * steppi, nx, ny, nz), dtype=np.float32)

    A = np.zeros((nx, ny, nz))
    A1 = np.zeros((nx, ny, nz))
    for kk in range(Ne):
        ct1 = np.zeros((input_channel, nx, ny, nz), dtype=np.float32)
        print(str(kk + 1) + " | " + str(Ne))
        a = np.reshape(paramss[:, kk], (nx, ny, nz), "F")
        inj_rate = kka[:, kk]

        at1 = ini[:, kk]
        at1 = rescale_linear_numpy_pytorch(at1, LUB, HUB, aay, bby)

        at2 = inip[:, kk]
        # at2 = rescale_linear(at2, mpor, hpor)

        at1 = np.reshape(at1, (nx, ny, nz), "F")
        at2 = np.reshape(at2, (nx, ny, nz), "F")

        for jj in range(nz):
            A[1, 24, jj] = inj_rate
            A[1, 1, jj] = inj_rate
            A[31, 1, jj] = inj_rate
            A[31, 31, jj] = inj_rate
            A[7, 9, jj] = -50
            A[14, 12, jj] = -50
            A[28, 19, jj] = -50
            A[14, 27, jj] = -50

            A1[1, 24, jj] = inj_rate
            A1[1, 1, jj] = inj_rate
            A1[31, 1, jj] = inj_rate
            A1[31, 31, jj] = inj_rate

        quse1 = A

        quse = np.reshape(quse1, (-1, 1), "F")
        quse_water = np.reshape(A1, (-1, 1), "F")
        # print(A)
        A_out = Reservoir_Simulator(
            a,
            at2,
            quse.ravel(),
            quse_water.ravel(),
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
            typee,
            step2,
            pini_alt,
        )

        ct1[0, :, :, :] = at1  # permeability
        ct1[1, :, :, :] = quse1 / UIR  # Overall f
        ct1[2, :, :, :] = A1 / UIR  # f for water injection
        ct1[3, :, :, :] = at2  # porosity
        ct1[4, :, :, :] = dt * np.ones((nx, ny, nz))
        ct1[5, :, :, :] = np.ones((nx, ny, nz))  # Initial pressure
        ct1[6, :, :, :] = IWSw * np.ones((nx, ny, nz))  # Initial water saturation

        cP[kk, :, :, :, :] = A_out
        ct[kk, :, :, :, :] = ct1

        return ct, cP


def AMGX_Inverse_problem(M, rhs, sol, typee):
    # Initialize config and resources:

    if typee == 1:
        cfg = pyamgx.Config().create_from_dict(
            {
                "config_version": 2,
                "determinism_flag": 1,
                "exception_handling": 1,
                "solver": {
                    "monitor_residual": 1,
                    "solver": "BICGSTAB",
                    "convergence": "RELATIVE_INI_CORE",
                    "preconditioner": {"solver": "NOSOLVER"},
                },
            }
        )

    elif typee == 2:
        cfg = pyamgx.Config()
        cfg.create_from_dict(
            {
                "config_version": 2,
                "solver": {
                    "print_grid_stats": 1,
                    "store_res_history": 1,
                    "solver": "FGMRES",
                    "print_solve_stats": 1,
                    "obtain_timings": 1,
                    "preconditioner": {
                        "interpolator": "D2",
                        "solver": "AMG",
                        "print_grid_stats": 1,
                        "aggressive_levels": 1,
                        "interp_max_elements": 4,
                        "smoother": {
                            "relaxation_factor": 1,
                            "scope": "jacobi",
                            "solver": "JACOBI_L1",
                        },
                        "presweeps": 2,
                        "selector": "PMIS",
                        "coarsest_sweeps": 2,
                        "coarse_solver": "NOSOLVER",
                        "max_iters": 1,
                        "max_row_sum": 0.9,
                        "strength_threshold": 0.25,
                        "min_coarse_rows": 2,
                        "scope": "amg_solver",
                        "max_levels": 24,
                        "cycle": "V",
                        "postsweeps": 2,
                    },
                    "max_iters": 100,
                    "monitor_residual": 1,
                    "gmres_n_restart": 10,
                    "convergence": "RELATIVE_INI_CORE",
                    "tolerance": 1e-06,
                    "norm": "L2",
                },
            }
        )
    elif typee == 3:  # DILU
        cfg = pyamgx.Config()
        cfg.create_from_dict(
            {
                "config_version": 2,
                "solver": {
                    "matrix_coloring_scheme": "MIN_MAX",
                    "max_uncolored_percentage": 0.15,
                    "algorithm": "AGGREGATION",
                    "obtain_timings": 1,
                    "solver": "AMG",
                    "smoother": "MULTICOLOR_DILU",
                    "print_solve_stats": 1,
                    "presweeps": 1,
                    "selector": "SIZE_2",
                    "coarsest_sweeps": 2,
                    "max_iters": 1000,
                    "monitor_residual": 1,
                    "scope": "main",
                    "max_levels": 1000,
                    "postsweeps": 1,
                    "tolerance": 0.1,
                    "print_grid_stats": 1,
                    "norm": "L1",
                    "cycle": "V",
                },
            }
        )

    elif typee == 4:  # AGGREGATON_DS
        cfg = pyamgx.Config()
        cfg.create_from_dict(
            {
                "config_version": 2,
                "determinism_flag": 1,
                "solver": {
                    "print_grid_stats": 1,
                    "max_uncolored_percentage": 0.15,
                    "algorithm": "AGGREGATION",
                    "obtain_timings": 1,
                    "solver": "AMG",
                    "smoother": "MULTICOLOR_GS",
                    "print_solve_stats": 1,
                    "presweeps": 1,
                    "symmetric_GS": 1,
                    "selector": "SIZE_2",
                    "coarsest_sweeps": 2,
                    "max_iters": 1000,
                    "monitor_residual": 1,
                    "postsweeps": 1,
                    "scope": "main",
                    "max_levels": 1000,
                    "matrix_coloring_scheme": "MIN_MAX",
                    "tolerance": 0.1,
                    "norm": "L1",
                    "cycle": "V",
                },
            }
        )

    elif typee == 5:  # AGGREGATON_JACOBI
        cfg = pyamgx.Config()
        cfg.create_from_dict(
            {
                "config_version": 2,
                "determinism_flag": 1,
                "solver": {
                    "print_grid_stats": 1,
                    "algorithm": "AGGREGATION",
                    "obtain_timings": 1,
                    "solver": "AMG",
                    "smoother": "BLOCK_JACOBI",
                    "print_solve_stats": 1,
                    "presweeps": 2,
                    "selector": "SIZE_2",
                    "convergence": "RELATIVE_MAX_CORE",
                    "coarsest_sweeps": 2,
                    "max_iters": 100,
                    "monitor_residual": 1,
                    "min_coarse_rows": 2,
                    "relaxation_factor": 0.75,
                    "scope": "main",
                    "max_levels": 1000,
                    "postsweeps": 2,
                    "tolerance": 0.1,
                    "norm": "L1",
                    "cycle": "V",
                },
            }
        )

    elif typee == 6:  # AGGREGATON_THRUST
        cfg = pyamgx.Config()
        cfg.create_from_dict(
            {
                "config_version": 2,
                "solver": {
                    "print_grid_stats": 1,
                    "algorithm": "AGGREGATION",
                    "coarseAgenerator": "THRUST",
                    "solver": "AMG",
                    "smoother": "BLOCK_JACOBI",
                    "print_solve_stats": 1,
                    "presweeps": 1,
                    "selector": "SIZE_2",
                    "obtain_timings": 1,
                    "max_iters": 100,
                    "monitor_residual": 1,
                    "scope": "main",
                    "postsweeps": 1,
                    "tolerance": 0.1,
                    "cycle": "V",
                },
            }
        )

    elif typee == 7:  # AGGREGATON_CG
        cfg = pyamgx.Config()
        cfg.create_from_dict(
            {
                "config_version": 2,
                "solver": {
                    "print_grid_stats": 1,
                    "solver": "AMG",
                    "algorithm": "AGGREGATION",
                    "selector": "SIZE_4",
                    "print_solve_stats": 1,
                    "smoother": "JACOBI_L1",
                    "presweeps": 0,
                    "postsweeps": 3,
                    "obtain_timings": 1,
                    "max_iters": 100,
                    "monitor_residual": 1,
                    "convergence": "ABSOLUTE",
                    "scope": "main",
                    "max_levels": 100,
                    "cycle": "CG",
                    "tolerance": 1e-06,
                    "norm": "L2",
                },
            }
        )

    elif typee == 8:  # PBICGSTAB_AGGREGATION_W_JACOBI
        cfg = pyamgx.Config()
        cfg.create_from_dict(
            {
                "config_version": 2,
                "solver": {
                    "preconditioner": {
                        "error_scaling": 2,
                        "algorithm": "AGGREGATION",
                        "solver": "AMG",
                        "smoother": {
                            "relaxation_factor": 0.9,
                            "scope": "amg_smoother",
                            "solver": "BLOCK_JACOBI",
                        },
                        "presweeps": 1,
                        "selector": "SIZE_2",
                        "max_iters": 1,
                        "monitor_residual": 1,
                        "convergence": "RELATIVE_INI_CORE",
                        "scope": "amg",
                        "cycle": "W",
                        "norm": "L1",
                        "postsweeps": 2,
                    },
                    "solver": "PBICGSTAB",
                    "print_solve_stats": 1,
                    "obtain_timings": 1,
                    "max_iters": 20,
                    "monitor_residual": 1,
                    "scope": "main",
                    "tolerance": 0.001,
                    "norm": "L2",
                },
            }
        )

    elif typee == 9:  # CG_DILU
        cfg = pyamgx.Config()
        cfg.create_from_dict(
            {
                "config_version": 2,
                "solver": {
                    "preconditioner": {"scope": "precond", "solver": "MULTICOLOR_DILU"},
                    "solver": "CG",
                    "print_solve_stats": 1,
                    "obtain_timings": 1,
                    "max_iters": 20,
                    "monitor_residual": 1,
                    "scope": "main",
                    "tolerance": 1e-06,
                    "norm": "L2",
                },
            }
        )

    elif typee == 10:  # GMRES_AMG
        cfg = pyamgx.Config()
        cfg.create_from_dict(
            {
                "config_version": 2,
                "determinism_flag": 1,
                "exception_handling": 1,
                "solver": {
                    "print_grid_stats": 1,
                    "store_res_history": 1,
                    "solver": "GMRES",
                    "print_solve_stats": 1,
                    "obtain_timings": 1,
                    "preconditioner": {
                        "interpolator": "D2",
                        "print_grid_stats": 1,
                        "solver": "AMG",
                        "smoother": "JACOBI_L1",
                        "presweeps": 2,
                        "selector": "PMIS",
                        "coarsest_sweeps": 2,
                        "coarse_solver": "NOSOLVER",
                        "max_iters": 1,
                        "interp_max_elements": 4,
                        "min_coarse_rows": 2,
                        "scope": "amg_solver",
                        "max_levels": 24,
                        "cycle": "V",
                        "postsweeps": 2,
                    },
                    "max_iters": 100,
                    "monitor_residual": 1,
                    "gmres_n_restart": 10,
                    "convergence": "RELATIVE_INI_CORE",
                    "tolerance": 1e-06,
                    "norm": "L2",
                },
            }
        )
    rsc = pyamgx.Resources().create_simple(cfg)

    # Create matrices and vectors:
    A = pyamgx.Matrix().create(rsc)
    b = pyamgx.Vector().create(rsc)
    x = pyamgx.Vector().create(rsc)

    # Create solver:
    solver = pyamgx.Solver().create(rsc, cfg)

    A.upload_CSR(M)
    b.upload(rhs)
    x.upload(sol)

    solver.setup(A)
    solver.solve(b, x)

    solution = x.download()

    A.destroy()
    x.destroy()
    b.destroy()
    solver.destroy()
    rsc.destroy()
    cfg.destroy()
    return solution


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


def plot3d2(arr_3d, nx, ny, nz, itt, dt, MAXZ, namet, titti, maxii, minii):
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
    ax.set_title(titti, fontsize=14)

    # Set axis limits to reflect the extent of each axis of the matrix
    ax.set_xlim(0, arr_3d.shape[0])
    ax.set_ylim(0, arr_3d.shape[1])
    ax.set_zlim(0, arr_3d.shape[2])

    # Remove the grid
    ax.grid(False)

    # Set lighting to bright
    ax.set_facecolor("white")
    # Set the aspect ratio of the plot

    ax.set_box_aspect([nx, ny, 5])

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

    # Define the coordinates of the voxel
    voxel_loc = (1, 24, 0)
    # Define the direction of the line
    line_dir = (0, 0, 5)
    # Define the coordinates of the line end
    x_line_end = voxel_loc[0] + line_dir[0]
    y_line_end = voxel_loc[1] + line_dir[1]
    z_line_end = voxel_loc[2] + line_dir[2]
    ax.plot([1, 1], [24, 24], [0, 5], "black", linewidth=2)
    ax.text(x_line_end, y_line_end, z_line_end, "I1", color="black", fontsize=16)

    # Define the coordinates of the voxel
    voxel_loc = (1, 1, 0)
    # Define the direction of the line
    line_dir = (0, 0, 10)
    # Define the coordinates of the line end
    x_line_end = voxel_loc[0] + line_dir[0]
    y_line_end = voxel_loc[1] + line_dir[1]
    z_line_end = voxel_loc[2] + line_dir[2]
    ax.plot([1, 1], [1, 1], [0, 10], "black", linewidth=2)
    ax.text(x_line_end, y_line_end, z_line_end, "I2", color="black", fontsize=16)

    # Define the coordinates of the voxel
    voxel_loc = (31, 1, 0)
    # Define the direction of the line
    line_dir = (0, 0, 7)
    # Define the coordinates of the line end
    x_line_end = voxel_loc[0] + line_dir[0]
    y_line_end = voxel_loc[1] + line_dir[1]
    z_line_end = voxel_loc[2] + line_dir[2]
    ax.plot([31, 31], [1, 1], [0, 7], "black", linewidth=2)
    ax.text(x_line_end, y_line_end, z_line_end, "I3", color="black", fontsize=16)

    # Define the coordinates of the voxel
    voxel_loc = (31, 31, 0)
    # Define the direction of the line
    line_dir = (0, 0, 8)
    # Define the coordinates of the line end
    x_line_end = voxel_loc[0] + line_dir[0]
    y_line_end = voxel_loc[1] + line_dir[1]
    z_line_end = voxel_loc[2] + line_dir[2]
    ax.plot([31, 31], [31, 31], [0, 8], "black", linewidth=2)
    ax.text(x_line_end, y_line_end, z_line_end, "I4", color="black", fontsize=20)

    # Define the coordinates of the voxel
    voxel_loc = (7, 9, 0)
    # Define the direction of the line
    line_dir = (0, 0, 8)
    # Define the coordinates of the line end
    x_line_end = voxel_loc[0] + line_dir[0]
    y_line_end = voxel_loc[1] + line_dir[1]
    z_line_end = voxel_loc[2] + line_dir[2]
    ax.plot([7, 7], [9, 9], [0, 8], "r", linewidth=2)
    ax.text(x_line_end, y_line_end, z_line_end, "P1", color="r", fontsize=16)

    # Define the coordinates of the voxel
    voxel_loc = (14, 12, 0)
    # Define the direction of the line
    line_dir = (0, 0, 10)
    # Define the coordinates of the line end
    x_line_end = voxel_loc[0] + line_dir[0]
    y_line_end = voxel_loc[1] + line_dir[1]
    z_line_end = voxel_loc[2] + line_dir[2]
    ax.plot([14, 14], [12, 12], [0, 10], "r", linewidth=2)
    ax.text(x_line_end, y_line_end, z_line_end, "P2", color="r", fontsize=16)

    # Define the coordinates of the voxel
    voxel_loc = (28, 19, 0)
    # Define the direction of the line
    line_dir = (0, 0, 15)
    # Define the coordinates of the line end
    x_line_end = voxel_loc[0] + line_dir[0]
    y_line_end = voxel_loc[1] + line_dir[1]
    z_line_end = voxel_loc[2] + line_dir[2]
    ax.plot([28, 28], [19, 19], [0, 15], "r", linewidth=2)
    ax.text(x_line_end, y_line_end, z_line_end, "P3", color="r", fontsize=16)

    # Define the coordinates of the voxel
    voxel_loc = (14, 27, 0)
    # Define the direction of the line
    line_dir = (0, 0, 15)
    # Define the coordinates of the line end
    x_line_end = voxel_loc[0] + line_dir[0]
    y_line_end = voxel_loc[1] + line_dir[1]
    z_line_end = voxel_loc[2] + line_dir[2]
    ax.plot([14, 14], [27, 27], [0, 15], "r", linewidth=2)
    ax.text(x_line_end, y_line_end, z_line_end, "P4", color="r", fontsize=16)
    # plt.show()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = namet + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()
