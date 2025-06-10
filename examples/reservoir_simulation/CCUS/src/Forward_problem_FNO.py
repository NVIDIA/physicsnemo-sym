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

    import cupy as cp
    from numba import cuda

    print(cuda.detect())  # Print the GPU information
    import tensorflow as tf

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)

    clementtt = 0
else:
    print("No GPU Available")
    import numpy as cp

    clementtt = 1

from sklearn.preprocessing import MinMaxScaler
import os.path
import torch

torch.set_default_dtype(torch.float32)
from joblib import Parallel, delayed
import multiprocessing
import mpslib as mps
from shutil import rmtree
import numpy.matlib
import re
from pyDOE import lhs
import pickle
import numpy
from scipy.stats import norm as nrm
from gstools import SRF, Gaussian
from gstools.random import MasterRNG
from scipy.fftpack import dct
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import numpy.matlib
import os.path
import math
import random
import os.path
import pandas as pd
from skimage.transform import resize as rzz
import h5py
from physicsnemo.sym.hydra import to_absolute_path
import scipy.io as sio
import yaml
from typing import Dict
from struct import unpack
import fnmatch
import os
import gzip

try:
    import gdown
except:
    gdown = None

import scipy.io
import shutil
import sys
from numpy import *
import scipy.optimize.lbfgsb as lbfgsb
import scipy
import numpy.linalg
from scipy.fftpack.realtransforms import idct
import numpy.ma as ma
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # I have just 1 GPU
from cpuinfo import get_cpu_info

s = get_cpu_info()
print("Cpu info")
for k, v in s.items():
    print(f"\t{k}: {v}")
cores = multiprocessing.cpu_count()
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float32)
import physicsnemo
import gc
from physicsnemo.sym.hydra import PhysicsNeMoConfig

# from physicsnemo.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from physicsnemo.sym.models.fully_connected import *
from physicsnemo.sym.models.fno import *
from physicsnemo.sym.loss.loss import Loss
from physicsnemo.sym.models.activation import Activation
import requests
from typing import Union, Tuple
from pathlib import Path
import torch
import warnings
from torch import Tensor

print(" ")
print(" This computer has %d cores, which will all be utilised in parallel " % cores)
print(" ")
print("......................DEFINE SOME FUNCTIONS.....................")


def replace_nans_and_infs_with_mean(tensor):
    # Create masks for NaN and Inf values
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    # Combine masks to identify all invalid (NaN or Inf) positions
    invalid_mask = nan_mask | inf_mask

    # Calculate the mean of valid (finite) elements in the tensor
    valid_elements = tensor[~invalid_mask]  # Elements that are not NaN or Inf
    if valid_elements.numel() > 0:  # Ensure there are valid elements to calculate mean
        mean_val = valid_elements.mean()
    else:
        mean_val = torch.tensor(
            1e-4, device=tensor.device
        )  # Fallback value if no valid elements

    # Repeat mean_val to match the shape of invalid elements
    repeated_mean_val = mean_val.expand_as(tensor)

    # Create a new tensor with values replaced with mean where necessary
    new_tensor = (
        tensor.clone()
    )  # Clone the original tensor to avoid inplace modification
    new_tensor[invalid_mask] = repeated_mean_val[
        invalid_mask
    ]  # Replace NaN and Inf values with mean

    return new_tensor


def process_tensor(tensor):
    """
    Processes the input tensor to:
    - Replace NaNs and infinities with zero.
    - Ensure the tensor is of torch.float32 precision.
    """
    tensor = torch.where(
        torch.isnan(tensor) | torch.isinf(tensor),
        torch.tensor(1e-6, dtype=torch.float32),
        tensor,
    )
    tensor = tensor.to(dtype=torch.float32)
    return tensor


class PointwiseLossNormC(Loss):
    """
    L-p loss function for pointwise data
    Computes the p-th order loss of each output tensor

    Parameters
    ----------
    ord : int, optional
        Order of the loss. For example, `ord=2` would be the L2 loss. Default is 2.
    """

    def __init__(self, ord: int = 2):
        super().__init__()
        self.ord = ord

    @staticmethod
    def _loss(
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
        ord: float,  # Now ord can be passed dynamically
    ) -> Dict[str, Tensor]:
        losses = {}

        # for key in pred_outvar.keys():
        #     l = lambda_weighting[key] * torch.abs(pred_outvar[key] - true_outvar[key]).pow(ord)
        #     if "area" in invar.keys():
        #         l *= invar["area"]
        #     losses[key] = l.sum()
        # return losses

        l1 = lambda_weighting["pressure"] * torch.abs(
            pred_outvar["pressure"] - true_outvar["pressure"]
        ).pow(2)
        # If area is a factor, apply it
        if "area" in invar.keys():
            l1 *= invar["area"]
        losses["pressure"] = l1.sum()

        l = lambda_weighting["water_sat"] * torch.abs(
            pred_outvar["water_sat"] - true_outvar["water_sat"]
        ).pow(2)
        if "area" in invar.keys():
            l *= invar["area"]
        losses["water_sat"] = l.sum()

        l = lambda_weighting["gas_sat"] * torch.abs(
            pred_outvar["gas_sat"] - true_outvar["gas_sat"]
        ).pow(2)
        if "area" in invar.keys():
            l *= invar["area"]
        losses["gas_sat"] = l.sum()

        return losses

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
        ord: int = None,  # Optional override for ord
    ) -> Dict[str, Tensor]:
        if ord is None:
            ord = self.ord  # Use the instance's ord if not overridden
        return PointwiseLossNormC._loss(
            invar, pred_outvar, true_outvar, lambda_weighting, step, ord
        )


def print_header(title, num_breaks=1):
    print("\n" * num_breaks + "=" * 50)
    print(" " * 20 + title)
    print("=" * 50 + "\n")


def get_shape(t):
    shape = []
    while isinstance(t, tuple):
        shape.append(len(t))
        t = t[0]
    return tuple(shape)


def read_until_line(file_path, line_num, skip=0, sep="\s+", header=None):
    """
    Reads a CSV file up to a specific line and returns it as a numpy array.

    Parameters:
    - file_path (str): Path to the CSV file.
    - line_num (int): The line number up to which the file should be read.
    - skip (int, optional): Number of rows to skip at the start. Default is 0.
    - sep (str, optional): Delimiter to use. Default is '\s+' (whitespace).
    - header (str or None, optional): Row number to use as the column names. Default is None.

    Returns:
    - np.array: A numpy array containing the read data.
    """

    # Calculate number of rows to read
    nrows_to_read = line_num - skip

    # Read the file using pandas
    df = pd.read_csv(
        file_path, skiprows=skip, nrows=nrows_to_read, sep=sep, header=header
    )

    # Convert DataFrame to numpy array and return
    return df.values


def Reinvent(matt):
    nx, ny, nz = matt.shape[1], matt.shape[2], matt.shape[0]
    dess = np.zeros((nx, ny, nz))
    for i in range(nz):
        dess[:, :, i] = matt[i, :, :]
    return dess


def initial_ensemble_gaussian(Nx, Ny, Nz, N, minn, maxx, minnp, maxxp):
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

    fensemble = np.zeros((Nx * Ny * Nz, N))
    ensemblep = np.zeros((Nx * Ny * Nz, N))
    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)
    model = Gaussian(dim=3, var=200, len_scale=200)  # Variance and lenght scale
    srf = SRF(model)
    seed = MasterRNG(20170519)
    for k in range(N):
        # fout=[]
        aoutt = srf.structured([x, y, z], seed=seed())
        foo = np.reshape(aoutt, (-1, 1), "F")

        clfy = MinMaxScaler(feature_range=(minn, maxx))
        (clfy.fit(foo))
        fout = clfy.transform(foo)
        fensemble[:, k] = np.ravel(fout)

        clfy1 = MinMaxScaler(feature_range=(minnp, maxxp))
        (clfy1.fit(foo))
        fout1 = clfy1.transform(foo)
        ensemblep[:, k] = np.ravel(fout1)

    return fensemble, ensemblep


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


def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk


# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


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


def round_array_to_4dp(arr):
    """
    Rounds the values of an input array to 4 decimal places.

    Args:
    - arr (numpy.ndarray or array-like): The input array.

    Returns:
    - numpy.ndarray: Array with values rounded to 4 decimal places.
    """
    try:
        arr = np.asarray(arr)  # Convert input to a numpy array if it's not already
        return np.around(arr, 4)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None  # You can choose to return None or handle the error differently


def H(y, t0=0):
    """
    Step fn with step at t0
    """
    # h = np.zeros_like(y)
    # args = tuple([slice(0,y.shape[i]) for i in y.ndim])


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
        if np.any(W != None):
            W = np.array(W)
            W[mask] = 0.0
        if np.any(sd != None):
            W = np.array(1.0 / sd**2)
            W[mask] = 0.0
            sd = None
        y[mask] = np.nan

    if np.any(sd != None):
        sd_ = np.array(sd)
        mask = sd > 0.0
        W = np.zeros_like(sd_)
        W[mask] = 1.0 / sd_[mask] ** 2
        sd = None

    if np.any(W != None):
        W = W / W.max()

    sizy = y.shape

    # sort axis
    if axis == None:
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
    if np.all(W == None):
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
        if z0 != None:  # an initial guess (z0) has been provided
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
            tol = isweighted * nrm(z0 - z) / nrm(z)

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
        RSS = nrm(DCTy * (Gamma - 1.0)) ** 2
    else:
        # take account of the weights to calculate RSS:
        yhat = dctND(Gamma * DCTy, f=idct)
        RSS = nrm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite])) ** 2
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
            - (((x - x0) / sdx)) * ((y - y0) / sdy) * c
        )
        # f /= f.sum()
        f *= random()
        z += f
    return z


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


def Get_Time(nx, ny, nz, N):

    Timee = []
    for k in range(N):
        check = np.ones((nx, ny, nz), dtype=np.float16)
        second_column = []  # List to hold the second column values

        with open("CO2STORE_GASWAT.RSM", "r") as f:
            # Skip the first 9 lines
            for _ in range(10):
                next(f)

            # Process the remaining lines
            for line in f:
                stripped_line = line.strip()  # Remove leading and trailing spaces
                words = stripped_line.split()  # Split line into words
                if len(words) > 1 and words[1].replace(".", "", 1).isdigit():
                    # If the second word is a number (integer or float), append it to second_column
                    second_column.append(float(words[1]))
                else:
                    # If the second word is not a number, it might be a header line or the end of the relevant data
                    break

        # Convert list to numpy array
        np_array2 = np.array(second_column)[
            :-1
        ]  # No need to remove the last item if it's valid
        np_array2 = np_array2
        unie = []
        for zz in range(len(np_array2)):
            aa = np_array2[zz] * check
            unie.append(aa)
        Time = np.stack(unie, axis=0)
        Timee.append(Time)

    Timee = np.stack(Timee, axis=0)
    return Timee


def Get_source_sink(N, nx, ny, nz, steppi):
    Qw = np.zeros((1, steppi, nx, ny, nz), dtype=np.float32)
    Qg = np.zeros((1, steppi, nx, ny, nz), dtype=np.float32)

    QG = np.zeros((steppi, nx, ny, nz), dtype=np.float32)

    for k in range(steppi):

        QG[k, 24, 24, -1] = 1000

    Qg[0, :, :, :, :] = QG

    return Qw, Qg


def fit_clement(tensor, target_min, target_max, tensor_min, tensor_max):
    # Rescale between target min and target max
    rescaled_tensor = tensor / tensor_max
    return rescaled_tensor


SUPPORTED_DATA_TYPES = {
    "INTE": (4, "i", 1000),
    "REAL": (4, "f", 1000),
    "LOGI": (4, "i", 1000),
    "DOUB": (8, "d", 1000),
    "CHAR": (8, "8s", 105),
    "MESS": (8, "8s", 105),
    "C008": (8, "8s", 105),
}


def parse_egrid(path_to_result):

    egrid_path = path_to_result
    attrs = ("GRIDHEAD", "ACTNUM")
    egrid = _parse_ech_bin(egrid_path, attrs)

    return egrid


def parse_unrst(path_to_result):

    unrst_path = path_to_result
    attrs = ("PRESSURE", "SGAS", "SWAT")
    states = _parse_ech_bin(unrst_path, attrs)
    return states


def _check_and_fetch_type_info(data_type):
    """Returns element size, format and element skip for the given data type.

    Parameters
    ----------
    data_type: str
        Should be a key from the SUPPORTED_DATA_TYPES

    Returns
    -------
    type_info: tuple
    """
    try:
        return SUPPORTED_DATA_TYPES[data_type]
    except KeyError as exc:
        raise ValueError("Unknown datatype %s." % data_type) from exc


def _check_and_fetch_file(path, pattern, return_relative=False):

    found = []
    reg_expr = re.compile(fnmatch.translate(pattern), re.IGNORECASE)

    # Listing files in the specified directory
    for f in os.listdir(path):
        # Check if the file matches the pattern
        if re.match(reg_expr, f):
            f_path = os.path.join(path, f)
            if return_relative:
                found.append(os.path.relpath(f_path, start=path))
            else:
                found.append(f_path)

    return found


def _parse_keywords(path, attrs=None):

    sections_counter = {} if attrs is None else {attr: 0 for attr in attrs}

    with open(path, "rb") as f:
        header = f.read(4)
        sections = dict()
        while True:
            try:
                section_name = (
                    unpack("8s", f.read(8))[0].decode("ascii").strip().upper()
                )
            except:
                break
            n_elements = unpack(">i", f.read(4))[0]
            data_type = unpack("4s", f.read(4))[0].decode("ascii")
            f.read(8)
            element_size, fmt, element_skip = _check_and_fetch_type_info(data_type)
            f.seek(f.tell() - 24)
            binary_data = f.read(
                24
                + element_size * n_elements
                + 8 * (math.floor((n_elements - 1) / element_skip) + 1)
            )
            if (attrs is None) or (section_name in attrs):
                sections_counter[section_name] = (
                    sections_counter.get(section_name, 0) + 1
                )
                if section_name not in sections:
                    sections[section_name] = []
                section = (
                    n_elements,
                    data_type,
                    element_size,
                    fmt,
                    element_skip,
                    binary_data,
                )
                section = _fetch_keyword_data(section)
                sections[section_name].append(section)

    return header, sections


def _parse_ech_bin(path, attrs=None):

    if attrs is None:
        raise ValueError("Keyword attribute cannot be empty")

    if isinstance(attrs, str):
        attrs = [attrs]

    attrs = [attr.strip().upper() for attr in attrs]
    _, sections = _parse_keywords(path, attrs)

    return sections


def _fetch_keyword_data(section):

    n_elements, data_type, element_size, fmt, element_skip, binary_data = section

    n_skip = math.floor((n_elements - 1) / element_skip)
    skip_elements = 8 // element_size
    skip_elements_total = n_skip * skip_elements
    data_format = fmt * (n_elements + skip_elements_total)
    data_size = element_size * (n_elements + skip_elements_total)
    if data_type in ["INTE", "REAL", "LOGI", "DOUB"]:
        data_format = ">" + data_format
    decoded_section = list(unpack(data_format, binary_data[24 : 24 + data_size]))
    del_ind = np.repeat(np.arange(1, 1 + n_skip) * element_skip, skip_elements)
    del_ind += np.arange(len(del_ind))
    decoded_section = np.delete(decoded_section, del_ind)
    if data_type in ["CHAR", "C008"]:
        decoded_section = np.char.decode(decoded_section, encoding="ascii")
    return decoded_section


def Geta_all(folder, nx, ny, nz, oldfolder, check, steppi):

    os.chdir(folder)

    # os.system(string_Jesus)

    check = np.ones((nx, ny, nz), dtype=np.float32)
    years_column = []

    with open("CO2STORE_GASWAT.RSM", "r") as f:
        # Skip the header lines to reach the data table
        for _ in range(10):
            next(f)

        # Read each line of the data section
        for line in f:
            stripped_line = line.strip()  # Remove leading and trailing spaces
            words = stripped_line.split()  # Split line into words based on whitespace
            if (
                len(words) > 1
            ):  # Ensure there's at least two elements (for "TIME" and "YEARS")
                try:
                    years_value = float(
                        words[1]
                    )  # Convert the second element (YEARS) to float
                    years_column.append(years_value)
                except ValueError:
                    # Handle the case where conversion to float fails (e.g., not a number)
                    break  # Exit the loop if we encounter data that doesn't fit the expected pattern

    # Convert the list to a numpy array
    years_array = np.array(years_column)

    # Use the years data as needed
    # Example: Multiply each year value by the 'check' array and stack the results
    unie = []
    for zz in range(
        min(steppi, len(years_array))
    ):  # Ensure we don't exceed the bounds of 'years_array'
        aa = years_array[zz] * check
        unie.append(aa)

    Time = np.stack(unie, axis=0)

    # 'Time' now contains the processed "YEARS" data as specified in the user's code logic

    pressure = []
    swat = []
    sgas = []

    attrs = ("GRIDHEAD", "ACTNUM")
    egrid = _parse_ech_bin("CO2STORE_GASWAT.EGRID", attrs)
    nx, ny, nz = egrid["GRIDHEAD"][0][1:4]
    actnum = egrid["ACTNUM"][0]  # numpy array of size nx * ny * nz

    states = parse_unrst("CO2STORE_GASWAT.UNRST")
    pressuree = states["PRESSURE"]
    swatt = states["SWAT"]
    sgass = states["SGAS"]

    active_index_array = np.where(actnum == 1)[0]
    len_act_indx = len(active_index_array)

    # Filter the slices of interest before the loop
    filtered_pressure = pressuree
    filtered_swat = swatt
    filtered_sgas = sgass

    # Active index array and its length
    active_index_array = np.where(actnum == 1)[0]
    len_act_indx = len(active_index_array)

    # Iterate over the filtered slices
    for pr_slice, sw_slice, sg_slice in zip(
        filtered_pressure, filtered_swat, filtered_sgas
    ):
        for state_var, all_slices in zip(
            [pr_slice, sw_slice, sg_slice], [pressure, swat, sgas]
        ):
            # Initialize an array of zeros
            resize_state_var = np.zeros((nx * ny * nz, 1))

            # Resize and update the array at active indices
            resize_state_var[active_index_array] = rzz(
                state_var.reshape(-1, 1), (len_act_indx,), order=1, preserve_range=True
            )

            # Reshape to 3D grid
            resize_state_var = np.reshape(resize_state_var, (nx, ny, nz), "F")

            # Append to the corresponding list
            all_slices.append(resize_state_var)

    # Stack the lists to create 3D arrays for each variable
    sgas = np.stack(sgas, axis=0)
    pressure = np.stack(pressure, axis=0)
    swat = np.stack(swat, axis=0)

    sgas = sgas[1:, :, :, :]
    swat = swat[1:, :, :, :]
    pressure = pressure[1:, :, :, :]

    os.chdir(oldfolder)
    return pressure, swat, sgas, Time


def copy_files(source_dir, dest_dir):
    files = os.listdir(source_dir)
    for file in files:
        shutil.copy(os.path.join(source_dir, file), dest_dir)


def save_files(perm, poro, dest_dir, oldfolder):
    os.chdir(dest_dir)
    filename1 = "permx" + ".dat"
    np.savetxt(
        filename1,
        perm,
        fmt="%.4f",
        delimiter=" \t",
        newline="\n",
        header="PERMX",
        footer="/",
        comments="",
    )

    filename2 = "porosity" + ".dat"
    np.savetxt(
        filename2,
        poro,
        fmt="%.4f",
        delimiter=" \t",
        newline="\n",
        header="PORO",
        footer="/",
        comments="",
    )

    os.chdir(oldfolder)


def Run_simulator(dest_dir, oldfolder, string_jesus2):
    os.chdir(dest_dir)
    os.system(string_jesus2)
    os.chdir(oldfolder)


def pytorch_cupy(tensor):
    dxa = to_dlpack(tensor)
    cx = cp.fromDlpack(dxa)
    return cx


def cupy_pytorch(cx):
    tx2 = from_dlpack(cx.toDlpack())
    return tx2


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    # If min_val and max_val are equal, just set the tensor to min_val
    # if min_val == max_val:
    #     original_tensor = np.full(rescaled_tensor.shape, min_val)
    # else:
    #     original_tensor = min_val + (rescaled_tensor - target_min) * (max_val - min_val) / (target_max - target_min)

    return rescaled_tensor * max_val


def replace_nans_and_infs(tensor, value=1e-6):
    tensor[torch.isnan(tensor) | torch.isinf(tensor)] = value
    return tensor


def scale_clement(tensor, target_min, target_max):
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0

    min_val = np.min(tensor)
    max_val = np.max(tensor)

    # # Check for tensor_max equal to tensor_min to avoid division by zero
    # if max_val == min_val:
    #     rescaled_tensor = np.full(tensor.shape, (target_max + target_min) / 2)  # Set the tensor to the average of target_min and target_max
    # else:
    rescaled_tensor = tensor / max_val

    return min_val, max_val, rescaled_tensor


# Define the threshold as the largest finite representable number for np.float32
threshold = np.finfo(np.float32).max


def replace_large_and_invalid_values(arr, placeholder=1e-6):
    """
    Replaces large values, infinities, and NaNs in a numpy array with a placeholder.

    Parameters:
    - arr: numpy array to be modified.
    - placeholder: value to replace invalid entries with.

    Returns:
    - Modified numpy array.
    """
    invalid_indices = (np.isnan(arr)) | (np.isinf(arr)) | (np.abs(arr) > threshold)
    arr[invalid_indices] = placeholder
    return arr


def clean_dict_arrays(data_dict):
    """
    Replaces large numbers, infinities, and NaNs in numpy arrays in a dictionary with a placeholder.

    Parameters:
    - data_dict: dictionary containing numpy arrays.

    Returns:
    - Dictionary with cleaned numpy arrays.
    """
    for key in data_dict:
        data_dict[key] = replace_large_and_invalid_values(data_dict[key])
    return data_dict


def clip_and_convert_to_float32(array):
    """
    Clips the values in the input array to the representable range of np.float32
    and then converts the array to dtype np.float32.

    Parameters:
    - array: numpy array to be clipped and converted.

    Returns:
    - Clipped and converted numpy array.
    """
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min

    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def clip_and_convert_to_float3(array):
    """
    Clips the values in the input array to the representable range of np.float32
    and then converts the array to dtype np.float32.

    Parameters:
    - array: numpy array to be clipped and converted.

    Returns:
    - Clipped and converted numpy array.
    """
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min

    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def Make_correct(array):
    """
    This function rearranges the dimensions of a 5D numpy array.

    Given an array of shape (a, b, c, d, e), it returns an array
    of shape (a, b, d, e, c). The function swaps the third and
    fifth dimensions.

    Parameters:
    - array (numpy.ndarray): A 5D numpy array to be rearranged.

    Returns:
    - numpy.ndarray: A 5D numpy array with rearranged dimensions.
    """

    # Initialize a new 5D array with swapped axes 2 and 4
    new_array = np.zeros(
        (array.shape[0], array.shape[1], array.shape[3], array.shape[4], array.shape[2])
    )

    # Loop through the first dimension
    for kk in range(array.shape[0]):
        # Initialize a 4D array for temporary storage
        perm_big = np.zeros(
            (array.shape[1], array.shape[3], array.shape[4], array.shape[2])
        )

        # Loop through the second dimension
        for mum in range(array.shape[1]):
            # Initialize a 3D array for deeper storage
            mum1 = np.zeros((array.shape[3], array.shape[4], array.shape[2]))

            # Loop through the third (original) dimension
            for i in range(array.shape[2]):
                # Rearrange the data from the original array
                mum1[:, :, i] = array[kk, :, :, :, :][mum, :, :, :][i, :, :]

            # Update the 4D array
            perm_big[mum, :, :, :] = mum1

        # Update the new 5D array
        new_array[kk, :, :, :, :] = perm_big

    return new_array


def Split_Matrix(matrix, sizee):
    x_split = np.split(matrix, sizee, axis=0)
    return x_split


def Remove_folder(N_ens, straa):
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)


def linear_interp(x, xp, fp):
    contiguous_xp = xp.contiguous()
    left_indices = torch.clamp(
        torch.searchsorted(contiguous_xp, x) - 1, 0, len(contiguous_xp) - 2
    )

    # Calculate denominators and handle zero case
    denominators = contiguous_xp[left_indices + 1] - contiguous_xp[left_indices]
    close_to_zero = denominators.abs() < 1e-6
    denominators[close_to_zero] = 1.0  # or any non-zero value to avoid NaN

    interpolated_value = (
        ((fp[left_indices + 1] - fp[left_indices]) / denominators)
        * (x - contiguous_xp[left_indices])
    ) + fp[left_indices]
    return interpolated_value


def replace_nan_with_zero(tensor):
    # Create masks for NaN and Inf values
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    # Combine masks to identify all invalid (NaN or Inf) positions
    invalid_mask = nan_mask | inf_mask

    # Calculate the mean of valid (finite) elements in the tensor
    valid_elements = tensor[~invalid_mask]  # Elements that are not NaN or Inf
    if valid_elements.numel() > 0:  # Ensure there are valid elements to calculate mean
        mean_value = valid_elements.mean()
    else:
        mean_value = torch.tensor(
            1e-6, device=tensor.device
        )  # Fallback value if no valid elements

    # Replace NaN and Inf values with the calculated mean
    # Note: This line combines the original tensor where values are valid, with the mean value where they are not
    return torch.where(invalid_mask, mean_value, tensor)


def interp_torch(cuda, reference_matrix1, reference_matrix2, tensor1):
    chunk_size = 1

    chunks = torch.chunk(tensor1, chunks=chunk_size, dim=0)
    processed_chunks = []
    for start_idx in range(chunk_size):
        interpolated_chunk = linear_interp(
            chunks[start_idx], reference_matrix1, reference_matrix2
        )
        processed_chunks.append(interpolated_chunk)

    torch.cuda.empty_cache()
    return processed_chunks


def sort_key(s):
    """Extract the number from the filename for sorting."""
    return int(re.search(r"\d+", s).group())


def load_FNO_dataset(path, input_keys, output_keys, n_examples=None):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar, output_keys)]:
        for k in keys:

            # get data
            x = data[k]  # N, C, H, W
            x = x.astype(np.float16)

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar)


def load_FNO_dataset2(path, input_keys, output_keys1, n_examples=None):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar1 = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar1, output_keys1)]:
        for k in keys:

            # get data
            x = data[k]  # N, C, H, W
            x = x.astype(np.float16)

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar1)


def load_FNO_dataset2d(path, input_keys, output_keys1, n_examples=None):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar1 = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar1, output_keys1)]:
        for k in keys:

            # get data
            x = data[k]  # N, C, H, W
            x = x.astype(np.float16)

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar1)


def load_FNO_dataset2a(path, input_keys, output_keys1, n_examples=None):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar1 = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar1, output_keys1)]:
        for k in keys:

            # get data
            x = data[k]  # N, C, H, W
            x = x.astype(np.float16)

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar1)


def _download_file_from_google_drive(id, path):
    "Downloads a file from google drive"

    # use gdown library to download file
    gdown.download(id=id, output=path)


def preprocess_FNO_mat2(path):
    "Convert a FNO .gz file to a hdf5 file, adding extra dimension to data arrays"

    assert path.endswith(".gz")
    # data = scipy.io.loadmat(path)
    with gzip.open(path, "rb") as f1:
        data = pickle.load(f1)

    ks = [k for k in data.keys() if not k.startswith("__")]
    with h5py.File(path[:-4] + ".hdf5", "w") as f:
        for k in ks:
            # x = np.expand_dims(data[k], axis=1)  # N, C, H, W
            x = data[k]
            f.create_dataset(
                k, data=x, dtype="float16", compression="gzip", compression_opts=9
            )  # note h5 files larger than .mat because no compression used


def preprocess_FNO_mat(path):
    "Convert a FNO .mat file to a hdf5 file, adding extra dimension to data arrays"

    assert path.endswith(".mat")
    data = scipy.io.loadmat(path)
    ks = [k for k in data.keys() if not k.startswith("__")]
    with h5py.File(path[:-4] + ".hdf5", "w") as f:
        for k in ks:
            # x = np.expand_dims(data[k], axis=1)  # N, C, H, W
            x = data[k]
            x = x.astype(np.float16)
            f.create_dataset(
                k, data=x, dtype="float16", compression="gzip", compression_opts=9
            )  # note h5 files larger than .mat because no compression used


def to_absolute_path_and_create(
    *args: Union[str, Path]
) -> Union[Path, str, Tuple[Union[Path, str]]]:
    """Converts file path to absolute path based on the current working directory and creates the subfolders."""

    out = ()
    base = Path(os.getcwd())
    for path in args:
        p = Path(path)

        if p.is_absolute():
            ret = p
        else:
            ret = base / p

        # Create the directory/subfolders
        ret.mkdir(parents=True, exist_ok=True)

        if isinstance(path, str):
            out = out + (str(ret),)
        else:
            out = out + (ret,)

    if len(args) == 1:
        out = out[0]
    return out


torch.set_default_dtype(torch.float32)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, "confirm": 1}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        # params = { 'id' : id, 'confirm' : 1 }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def improve(tensor, maxK):
    # Replace NaN values with maxK
    tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, maxK), tensor)

    # Replace positive and negative infinity with maxK
    tensor = torch.where(torch.isinf(tensor), torch.full_like(tensor, maxK), tensor)

    # Ensure no values exceed maxK
    tensor = torch.where(tensor > maxK, torch.full_like(tensor, maxK), tensor)
    return tensor


# Function to replace NaN and Inf values with the mean of valid values in a tensor
def replace_with_mean(tensor):
    # Ensure the input tensor is of type torch.float32
    tensor = tensor.to(torch.float32)

    valid_elements = tensor[torch.isfinite(tensor)]  # Filter out NaN and Inf values
    if valid_elements.numel() > 0:  # Check if there are any valid elements
        mean_value = valid_elements.mean()  # Calculate mean of valid elements
        # Add a slight perturbation to the mean value and ensure it's float32
        perturbed_mean_value = (
            mean_value
            + torch.normal(mean=0.0, std=0.01, size=(1,), device=tensor.device)
        ).to(torch.float32)
    else:
        # Fallback value if no valid elements, ensuring it's float32
        perturbed_mean_value = torch.tensor(
            1e-4, device=tensor.device, dtype=torch.float32
        )

    # Replace NaN and Inf values with the slightly perturbed mean value
    ouut = torch.where(
        torch.isnan(tensor) | torch.isinf(tensor), perturbed_mean_value, tensor
    )

    # ouut = (ouut * 10**6).round() / 10**6
    ouut = torch.abs(ouut)
    return ouut


def normalize_tensors_adjusted(tensor_dict):
    """
    Normalize each tensor in the dictionary to have values between a non-negative perturbed value around 0.1 and 1
    based on their own min and max values, ensuring tensors are of type torch.float32.

    Parameters:
    - tensor_dict: A dictionary with tensor values.

    Returns:
    A dictionary with the same keys as tensor_dict but with all tensors normalized between a non-negative perturbed 0.1 and 1, and ensuring torch.float32 precision.
    """
    normalized_dict = {}
    for key, tensor in tensor_dict.items():
        tensor = tensor.to(torch.float32)  # Ensure tensor is float32
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        if max_val - min_val > 0:
            # Normalize between 0 and 1
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
            # Generate perturbation
            perturbation = torch.normal(
                mean=0.1, std=0.01, size=normalized_tensor.size(), device=tensor.device
            )
            # Ensure perturbation does not go below 0.1 to avoid negative values
            perturbation_clipped = torch.clamp(perturbation, min=0.1)
            # Adjust to be between a non-negative perturbed value around 0.1 and 1
            adjusted_tensor = normalized_tensor * 0.9 + perturbation_clipped
        else:
            # Set to a perturbed value around 0.1 if the tensor has the same value for all elements, ensuring no negative values
            perturbed_value = torch.normal(
                mean=0.1, std=0.01, size=tensor.size(), device=tensor.device
            )
            perturbed_value_clipped = torch.clamp(perturbed_value, min=0.1)
            adjusted_tensor = (
                torch.full_like(tensor, 0).float() + perturbed_value_clipped
            )
        normalized_dict[key] = adjusted_tensor
    return normalized_dict


@physicsnemo.sym.main(config_path="conf", config_name="config_FNO")
def run(cfg: PhysicsNeMoConfig) -> None:
    text = """
                                                              dddddddd                                                         
    MMMMMMMM               MMMMMMMM                           d::::::d                lllllll                                  
    M:::::::M             M:::::::M                           d::::::d                l:::::l                                  
    M::::::::M           M::::::::M                           d::::::d                l:::::l                                  
    M:::::::::M         M:::::::::M                           d:::::d                 l:::::l                                  
    M::::::::::M       M::::::::::M  ooooooooooo      ddddddddd:::::duuuuuu    uuuuuu  l::::luuuuuu    uuuuuu     ssssssssss   
    M:::::::::::M     M:::::::::::Moo:::::::::::oo  dd::::::::::::::du::::u    u::::u  l::::lu::::u    u::::u   ss::::::::::s  
    M:::::::M::::M   M::::M:::::::o:::::::::::::::od::::::::::::::::du::::u    u::::u  l::::lu::::u    u::::u ss:::::::::::::s 
    M::::::M M::::M M::::M M::::::o:::::ooooo:::::d:::::::ddddd:::::du::::u    u::::u  l::::lu::::u    u::::u s::::::ssss:::::s
    M::::::M  M::::M::::M  M::::::o::::o     o::::d::::::d    d:::::du::::u    u::::u  l::::lu::::u    u::::u  s:::::s  ssssss 
    M::::::M   M:::::::M   M::::::o::::o     o::::d:::::d     d:::::du::::u    u::::u  l::::lu::::u    u::::u    s::::::s      
    M::::::M    M:::::M    M::::::o::::o     o::::d:::::d     d:::::du::::u    u::::u  l::::lu::::u    u::::u       s::::::s   
    M::::::M     MMMMM     M::::::o::::o     o::::d:::::d     d:::::du:::::uuuu:::::u  l::::lu:::::uuuu:::::u ssssss   s:::::s 
    M::::::M               M::::::o:::::ooooo:::::d::::::ddddd::::::du:::::::::::::::ul::::::u:::::::::::::::us:::::ssss::::::s
    M::::::M               M::::::o:::::::::::::::od:::::::::::::::::du:::::::::::::::l::::::lu:::::::::::::::s::::::::::::::s 
    M::::::M               M::::::Moo:::::::::::oo  d:::::::::ddd::::d uu::::::::uu:::l::::::l uu::::::::uu:::us:::::::::::ss  
    MMMMMMMM               MMMMMMMM  ooooooooooo     ddddddddd   ddddd   uuuuuuuu  uuullllllll   uuuuuuuu  uuuu sssssssssss   
    """
    print(text)
    print("")
    textaa = """
            CCCCCCCCCCCCC      CCCCCCCCCCCCUUUUUUUU     UUUUUUUU  SSSSSSSSSSSSSSS 
         CCC::::::::::::C   CCC::::::::::::U::::::U     U::::::USS:::::::::::::::S
       CC:::::::::::::::C CC:::::::::::::::U::::::U     U::::::S:::::SSSSSS::::::S
      C:::::CCCCCCCC::::CC:::::CCCCCCCC::::UU:::::U     U:::::US:::::S     SSSSSSS
     C:::::C       CCCCCC:::::C       CCCCCCU:::::U     U:::::US:::::S            
    C:::::C            C:::::C              U:::::D     D:::::US:::::S            
    C:::::C            C:::::C              U:::::D     D:::::U S::::SSSS         
    C:::::C            C:::::C              U:::::D     D:::::U  SS::::::SSSSS    
    C:::::C            C:::::C              U:::::D     D:::::U    SSS::::::::SS  
    C:::::C            C:::::C              U:::::D     D:::::U       SSSSSS::::S 
    C:::::C            C:::::C              U:::::D     D:::::U            S:::::S
     C:::::C       CCCCCC:::::C       CCCCCCU::::::U   U::::::U            S:::::S
      C:::::CCCCCCCC::::CC:::::CCCCCCCC::::CU:::::::UUU:::::::USSSSSSS     S:::::S
       CC:::::::::::::::C CC:::::::::::::::C UU:::::::::::::UU S::::::SSSSSS:::::S
         CCC::::::::::::C   CCC::::::::::::C   UU:::::::::UU   S:::::::::::::::SS 
            CCCCCCCCCCCCC      CCCCCCCCCCCCC     UUUUUUUUU      SSSSSSSSSSSSSSS   
                                                                                  
    """
    print(textaa)
    print("")
    print("------------------------------------------------------------------")
    print("")
    print("\n")
    print("|-----------------------------------------------------------------|")
    print("|                 TRAIN THE MODEL USING AN FNO APPROACH:        |")
    print("|-----------------------------------------------------------------|")
    print("")
    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    DEFAULT = None
    while True:
        DEFAULT = int(input("Use best default options:\n1=Yes\n2=No\n"))
        if (DEFAULT > 2) or (DEFAULT < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:

            break
    print("")

    learn_pde = "FNO"

    if DEFAULT == 1:
        interest = 2
    else:
        interest = None
        while True:
            interest = int(
                input(
                    "Select 1 = Run the Flow simulation to generate samples | 2 = Use saved data \n"
                )
            )
            if (interest > 2) or (interest < 1):
                # raise SyntaxError('please select value between 1-2')
                print("")
                print("please try again and select value between 1-2")
            else:

                break

    if not os.path.exists(to_absolute_path("../PACKETS")):
        os.makedirs(to_absolute_path("../PACKETS"))
    else:
        pass

    if interest == 1:
        # bb = os.path.isfile(to_absolute_path('../PACKETS/conversions.mat'))
        if os.path.isfile(to_absolute_path("../PACKETS/conversions.mat")) == True:
            os.remove(to_absolute_path("../PACKETS/conversions.mat"))
        if not os.path.exists(to_absolute_path("../RUNS")):
            os.makedirs(to_absolute_path("../RUNS"))
        else:
            pass

    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("No GPU found. Please run on a system with a GPU.") 
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:  # Choose GPU 1 (index 1)
            device = torch.device(f"cuda:0")
        else:  # If there's only one GPU or no GPUs, choose the first one (index 0)
            device = torch.device(f"cuda:0")
    else:  # If CUDA is not available, use the CPU
        raise RuntimeError("No GPU found. Please run on a system with a GPU.")
    torch.cuda.set_device(device)

    # Varaibles needed for NVRS

    nx = cfg.custom.PROPS.nx
    ny = cfg.custom.PROPS.ny
    nz = cfg.custom.PROPS.nz

    # training

    pini_alt = 200

    bb = os.path.isfile(to_absolute_path("../PACKETS/conversions.mat"))
    if bb == True:
        mat = sio.loadmat(to_absolute_path("../PACKETS/conversions.mat"))
        steppi = int(mat["steppi"])

        N_ens = int(mat["N_ens"])
        # print(N_ens)
        # print(steppi)
    else:
        steppi = 10

        N_ens = None
        while True:
            N_ens = int(input("Enter the ensemble size between 2-500\n"))
            if (N_ens > 500) or (N_ens < 2):
                # raise SyntaxError('please select value between 1-2')
                print("")
                print("please try again and select value between 2-500")
            else:

                break

    # print(steppi)
    # steppi = 246
    """
    input_channel = 4 #[Perm,Phi,initial_pressure, initial_water_sat] 
    output_channel = 3 #[Pressure, Sw,Sg]
    """

    oldfolder2 = os.getcwd()

    check = np.ones((nx, ny, nz), dtype=np.float32)

    string_Jesus2 = "flow CO2STORE_GASWAT.DATA"

    # N_ens = 2
    njobs = 3
    # njobs = int((multiprocessing.cpu_count() // 4) - 1)
    num_cores = njobs

    source_dir = to_absolute_path("../Necessaryy")
    # dest_dir = 'path_to_folder_B'

    minn = float(cfg.custom.PROPS.minn)
    maxx = float(cfg.custom.PROPS.maxx)
    minnp = float(cfg.custom.PROPS.minnp)
    maxxp = float(cfg.custom.PROPS.maxxp)

    if interest == 1:
        minn = float(cfg.custom.PROPS.minn)
        maxx = float(cfg.custom.PROPS.maxx)
        minnp = float(cfg.custom.PROPS.minnp)
        maxxp = float(cfg.custom.PROPS.maxxp)

        perm_ensemble, poro_ensemble = initial_ensemble_gaussian(
            nx, ny, nz, N_ens, minn, maxx, minnp, maxxp
        )

        X_ensemble = {"ensemble": perm_ensemble, "ensemblep": poro_ensemble}
        with gzip.open(to_absolute_path("../PACKETS/static.pkl.gz"), "wb") as f1:
            pickle.dump(X_ensemble, f1)
    else:
        with gzip.open(to_absolute_path("../PACKETS/static.pkl.gz"), "rb") as f2:
            mat = pickle.load(f2)
        X_data1 = mat
        for key, value in X_data1.items():
            print(f"For key '{key}':")
            print("\tContains inf:", np.isinf(value).any())
            print("\tContains -inf:", np.isinf(-value).any())
            print("\tContains NaN:", np.isnan(value).any())

        perm_ensemble = X_data1["ensemble"]
        poro_ensemble = X_data1["ensemblep"]

    perm_ensemble = clip_and_convert_to_float32(perm_ensemble)
    poro_ensemble = clip_and_convert_to_float32(poro_ensemble)

    if interest == 1:

        for kk in range(N_ens):
            path_out = to_absolute_path("../RUNS/Realisation" + str(kk))
            os.makedirs(path_out, exist_ok=True)

        Parallel(n_jobs=njobs, backend="loky", verbose=10)(
            delayed(copy_files)(
                source_dir, to_absolute_path("../RUNS/Realisation" + str(kk))
            )
            for kk in range(N_ens)
        )

        Parallel(n_jobs=njobs, backend="loky", verbose=10)(
            delayed(save_files)(
                perm_ensemble[:, kk],
                poro_ensemble[:, kk],
                to_absolute_path("../RUNS/Realisation" + str(kk)),
                oldfolder,
            )
            for kk in range(N_ens)
        )

        print("")
        print("---------------------------------------------------------------------")
        print("")
        print("\n")
        print("|-----------------------------------------------------------------|")
        print("|                 RUN FLOW SIMULATOR FOR ENSEMBLE                  |")
        print("|-----------------------------------------------------------------|")
        print("")

        Parallel(n_jobs=njobs, backend="loky", verbose=10)(
            delayed(Run_simulator)(
                to_absolute_path("../RUNS/Realisation" + str(kk)),
                oldfolder2,
                string_Jesus2,
            )
            for kk in range(N_ens)
        )

        print("|-----------------------------------------------------------------|")
        print("|                 EXECUTED RUN of  FLOW SIMULATION FOR ENSEMBLE   |")
        print("|-----------------------------------------------------------------|")

        print("|-----------------------------------------------------------------|")
        print("|                 DATA CURRATION IN PROCESS                       |")
        print("|-----------------------------------------------------------------|")
        N = N_ens
        pressure = []
        Sgas = []
        Swater = []
        Time = []

        permeability = np.zeros((N, 1, nx, ny, nz))
        porosity = np.zeros((N, 1, nx, ny, nz))

        for i in range(N):
            folder = to_absolute_path("../RUNS/Realisation" + str(i))
            Pr, sw, sg, tt = Geta_all(folder, nx, ny, nz, oldfolder, check, steppi)

            Pr = round_array_to_4dp(clip_and_convert_to_float32(Pr))
            sw = round_array_to_4dp(clip_and_convert_to_float32(sw))
            sg = round_array_to_4dp(clip_and_convert_to_float32(sg))
            tt = round_array_to_4dp(clip_and_convert_to_float32(tt))

            pressure.append(Pr)
            Sgas.append(sg)
            Swater.append(sw)
            Time.append(tt)

            permeability[i, 0, :, :, :] = np.reshape(
                perm_ensemble[:, i], (nx, ny, nz), "F"
            )
            porosity[i, 0, :, :, :] = np.reshape(poro_ensemble[:, i], (nx, ny, nz), "F")

            del Pr
            gc.collect()
            del sw
            gc.collect()
            del sg
            gc.collect()
            del tt
            gc.collect()

        pressure = np.stack(pressure, axis=0)
        Sgas = np.stack(Sgas, axis=0)
        Swater = np.stack(Swater, axis=0)
        Time = np.stack(Time, axis=0)
        ini_pressure = pini_alt * np.ones((N, 1, nx, ny, nz), dtype=np.float32)
        ini_sat = np.ones((N, 1, nx, ny, nz), dtype=np.float32)

        Qw, Qg = Get_source_sink(N, nx, ny, nz, steppi)
        Q = Qw + Qg

        print("|-----------------------------------------------------------------|")
        print("|                 DATA CURRATED                                   |")
        print("|-----------------------------------------------------------------|")

        target_min = 0.01
        target_max = 1.0

        permeability[np.isnan(permeability)] = 0.0
        Time[np.isnan(Time)] = 0.0
        pressure[np.isnan(pressure)] = 0.0
        Qw[np.isnan(Qw)] = 0.0
        Qg[np.isnan(Qg)] = 0.0
        Q[np.isnan(Q)] = 0.0

        permeability[np.isinf(permeability)] = 0.0
        Time[np.isinf(Time)] = 0.0
        pressure[np.isinf(pressure)] = 0.0
        Qw[np.isinf(Qw)] = 0.0
        Qg[np.isinf(Qg)] = 0.0
        Q[np.isinf(Q)] = 0.0

        minK, maxK, permeabilityx = scale_clement(
            permeability, target_min, target_max
        )  # Permeability
        minT, maxT, Timex = scale_clement(Time, target_min, target_max)  # Time
        minP, maxP, pressurex = scale_clement(
            pressure, target_min, target_max
        )  # pressure
        minQw, maxQw, Qwx = scale_clement(Qw, target_min, target_max)  # Qw
        minQg, maxQg, Qgx = scale_clement(Qg, target_min, target_max)  # Qg
        minQ, maxQ, Qx = scale_clement(Q, target_min, target_max)  # Q

        permeabilityx[np.isnan(permeabilityx)] = 0.0
        Timex[np.isnan(Timex)] = 0.0
        pressurex[np.isnan(pressurex)] = 0.0
        Qwx[np.isnan(Qwx)] = 0.0
        Qgx[np.isnan(Qgx)] = 0.0
        Qx[np.isnan(Qx)] = 0.0

        permeabilityx[np.isinf(permeabilityx)] = 0.0
        Timex[np.isinf(Timex)] = 0.0
        pressurex[np.isinf(pressurex)] = 0.0
        Qwx[np.isinf(Qwx)] = 0.0
        Qgx[np.isinf(Qgx)] = 0.0
        Qx[np.isinf(Qx)] = 0.0

        ini_pressure[np.isnan(ini_pressure)] = 0.0
        ini_pressurex = ini_pressure / maxP

        ini_pressurex = clip_and_convert_to_float32(ini_pressurex)

        ini_pressurex[np.isnan(ini_pressurex)] = 0.0
        porosity[np.isnan(porosity)] = 0.0
        Swater[np.isnan(Swater)] = 0.0
        Sgas[np.isnan(Sgas)] = 0.0
        ini_sat[np.isnan(ini_sat)] = 0.0

        ini_pressurex[np.isinf(ini_pressurex)] = 0.0
        porosity[np.isinf(porosity)] = 0.0
        Swater[np.isinf(Swater)] = 0.0
        Sgas[np.isinf(Sgas)] = 0.0
        ini_sat[np.isinf(ini_sat)] = 0.0

        X_data1 = {
            "permeability": permeabilityx,
            "porosity": porosity,
            "Pressure": pressurex,
            "Water_saturation": Swater,
            "Time": Timex,
            "Gas_saturation": Sgas,
            "Pini": ini_pressurex,
            "Qw": Qwx,
            "Qg": Qgx,
            "Q": Qx,
            "Sini": ini_sat,
        }

        X_data1 = clean_dict_arrays(X_data1)

        del permeabilityx
        gc.collect()
        del permeability
        gc.collect()
        del porosity
        gc.collect()
        del pressurex
        gc.collect()
        del Swater
        gc.collect()
        del Timex
        gc.collect()
        del Sgas
        gc.collect()
        del ini_pressurex
        gc.collect()
        del Qwx
        gc.collect()
        del Qgx
        gc.collect()
        del Qx
        gc.collect()
        del ini_sat
        gc.collect()

        del pressure
        gc.collect()
        del Time
        gc.collect()
        del ini_pressure
        gc.collect()

        # Save compressed dictionary
        with gzip.open(to_absolute_path("../PACKETS/data_train.pkl.gz"), "wb") as f1:
            pickle.dump(X_data1, f1)

        with gzip.open(to_absolute_path("../PACKETS/data_test.pkl.gz"), "wb") as f2:
            pickle.dump(X_data1, f2)

        sio.savemat(
            to_absolute_path("../PACKETS/conversions.mat"),
            {
                "minK": minK,
                "maxK": maxK,
                "minP": minP,
                "maxP": maxP,
                "steppi": steppi,
                "N_ens": N_ens,
                "learn_pde": learn_pde,
            },
            do_compression=True,
        )

        print("|-----------------------------------------------------------------|")
        print("|                 DATA SAVED                                      |")
        print("|-----------------------------------------------------------------|")

        print("|-----------------------------------------------------------------|")
        print("|                 REMOVE FOLDERS USED FOR THE RUN                 |")
        print("|-----------------------------------------------------------------|")

        for jj in range(N_ens):
            folderr = to_absolute_path("../RUNS/Realisation" + str(jj))
            rmtree(folderr)
        rmtree(to_absolute_path("../RUNS"))
    else:
        pass

    mat = sio.loadmat(to_absolute_path("../PACKETS/conversions.mat"))
    minK = mat["minK"]
    maxK = mat["maxK"]
    minP = mat["minP"]
    maxP = mat["maxP"]

    target_min = 0.01
    target_max = 1
    print("These are the values:")
    print("minK value is:", minK)
    print("maxK value is:", maxK)
    print("minP value is:", minP)
    print("maxP value is:", maxP)
    print("target_min value is:", target_min)
    print("target_max value is:", target_max)

    minKx = torch.from_numpy(minK).to(device)
    maxKx = torch.from_numpy(maxK).to(device)
    minPx = torch.from_numpy(minP).to(device)
    maxPx = torch.from_numpy(maxP).to(device)

    del mat
    gc.collect()

    print("Load simulated labelled training data")
    with gzip.open(to_absolute_path("../PACKETS/data_train.pkl.gz"), "rb") as f2:
        mat = pickle.load(f2)
    X_data1 = mat
    for key, value in X_data1.items():
        print(f"For key '{key}':")
        print("\tContains inf:", np.isinf(value).any())
        print("\tContains -inf:", np.isinf(-value).any())
        print("\tContains NaN:", np.isnan(value).any())

    del mat
    gc.collect()

    for key in X_data1.keys():
        # Convert NaN and infinity values to 0
        X_data1[key][np.isnan(X_data1[key])] = 0.0
        X_data1[key][np.isinf(X_data1[key])] = 0.0
        # X_data1[key] = np.clip(X_data1[key], target_min, target_max)
        X_data1[key] = clip_and_convert_to_float32(X_data1[key])

    cPerm = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Permeability
    cPhi = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Porosity
    cPini = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Initial pressure
    cSini = np.zeros(
        (N_ens, 1, nz, nx, ny), dtype=np.float32
    )  # Initial water saturation
    cQ = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cQw = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cQg = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cTime = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault

    cPress = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)  # Pressure
    cSat = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)  # Water saturation
    cSatg = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)  # gas saturation

    # print(X_data1['Q'].shape)

    X_data1["Q"][X_data1["Q"] <= 0] = 0
    X_data1["Qw"][X_data1["Qw"] <= 0] = 0
    X_data1["Qg"][X_data1["Qg"] <= 0] = 0

    for i in range(nz):
        X_data1["Q"][0, :, :, :, i] = np.where(
            X_data1["Q"][0, :, :, :, i] < 0, 0, X_data1["Q"][0, :, :, :, i]
        )
        X_data1["Qw"][0, :, :, :, i] = np.where(
            X_data1["Qw"][0, :, :, :, i] < 0, 0, X_data1["Qw"][0, :, :, :, i]
        )
        X_data1["Qg"][0, :, :, :, i] = np.where(
            X_data1["Qg"][0, :, :, :, i] < 0, 0, X_data1["Qg"][0, :, :, :, i]
        )

        cQ[0, :, i, :, :] = X_data1["Q"][0, :, :, :, i]
        cQw[0, :, i, :, :] = X_data1["Qw"][0, :, :, :, i]
        cQg[0, :, i, :, :] = X_data1["Qg"][0, :, :, :, i]
        cTime[0, :, i, :, :] = X_data1["Time"][0, :, :, :, i]

    neededM = {
        "Q": torch.from_numpy(cQ).to(device, torch.float32),
        "Qw": torch.from_numpy(cQw).to(device, dtype=torch.float32),
        "Qg": torch.from_numpy(cQg).to(device, dtype=torch.float32),
        "Time": torch.from_numpy(cTime).to(device, dtype=torch.float32),
    }

    for key in neededM:
        neededM[key] = replace_nans_and_infs(neededM[key])

    for kk in range(N_ens):

        # INPUTS
        for i in range(nz):
            cPerm[kk, 0, i, :, :] = clip_and_convert_to_float3(
                X_data1["permeability"][kk, 0, :, :, i]
            )
            cPhi[kk, 0, i, :, :] = clip_and_convert_to_float3(
                X_data1["porosity"][kk, 0, :, :, i]
            )
            cPini[kk, 0, i, :, :] = (
                clip_and_convert_to_float3(X_data1["Pini"][kk, 0, :, :, i]) / maxP
            )
            cSini[kk, 0, i, :, :] = clip_and_convert_to_float3(
                X_data1["Sini"][kk, 0, :, :, i]
            )

        # OUTPUTS
        for mum in range(steppi):
            for i in range(nz):
                cPress[kk, mum, i, :, :] = clip_and_convert_to_float3(
                    X_data1["Pressure"][kk, mum, :, :, i]
                )
                cSat[kk, mum, i, :, :] = clip_and_convert_to_float3(
                    X_data1["Water_saturation"][kk, mum, :, :, i]
                )
                cSatg[kk, mum, i, :, :] = clip_and_convert_to_float3(
                    X_data1["Gas_saturation"][kk, mum, :, :, i]
                )

    del X_data1
    gc.collect()
    data = {
        "perm": cPerm,
        "Phi": cPhi,
        "Pini": cPini,
        "Swini": cSini,
        "pressure": cPress,
        "water_sat": cSat,
        "gas_sat": cSatg,
    }

    with gzip.open(to_absolute_path("../PACKETS/simulations_train.pkl.gz"), "wb") as f4:
        pickle.dump(data, f4)

    preprocess_FNO_mat2(to_absolute_path("../PACKETS/simulations_train.pkl.gz"))
    os.remove(to_absolute_path("../PACKETS/simulations_train.pkl.gz"))
    del data
    gc.collect()
    del cPerm
    gc.collect()
    del cQ
    gc.collect()
    del cQw
    gc.collect()
    del cQg
    gc.collect()
    del cPhi
    gc.collect()
    del cTime
    gc.collect()
    del cPini
    gc.collect()
    del cSini
    gc.collect()
    del cPress
    gc.collect()
    del cSat
    gc.collect()
    del cSatg
    gc.collect()

    print("Load simulated labelled test data from .gz file")
    with gzip.open(to_absolute_path("../PACKETS/data_test.pkl.gz"), "rb") as f:
        mat = pickle.load(f)
    X_data1t = mat
    for key, value in X_data1t.items():
        print(f"For key '{key}':")
        print("\tContains inf:", np.isinf(value).any())
        print("\tContains -inf:", np.isinf(-value).any())
        print("\tContains NaN:", np.isnan(value).any())

    del mat
    gc.collect()

    for key in X_data1t.keys():
        # Convert NaN and infinity values to 0
        X_data1t[key][np.isnan(X_data1t[key])] = 0
        X_data1t[key][np.isinf(X_data1t[key])] = 0
        # X_data1t[key] = np.clip(X_data1t[key], target_min, target_max)
        X_data1t[key] = clip_and_convert_to_float32(X_data1t[key])

    cPerm = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Permeability
    cPhi = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Porosity
    cPini = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Initial pressure
    cSini = np.zeros(
        (N_ens, 1, nz, nx, ny), dtype=np.float32
    )  # Initial water saturation
    cQ = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cQw = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cQg = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cTime = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault

    cPress = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)  # Pressure
    cSat = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)  # Water saturation
    cSatg = np.zeros((N_ens, steppi, nz, nx, ny), dtype=np.float32)  # gas saturation

    for kk in range(N_ens):

        # INPUTS
        for i in range(nz):
            cPerm[kk, 0, i, :, :] = X_data1t["permeability"][kk, 0, :, :, i]
            cPhi[kk, 0, i, :, :] = X_data1t["porosity"][kk, 0, :, :, i]
            cPini[kk, 0, i, :, :] = X_data1t["Pini"][kk, 0, :, :, i] / maxP
            cSini[kk, 0, i, :, :] = X_data1t["Sini"][kk, 0, :, :, i]

        # OUTPUTS
        for mum in range(steppi):
            for i in range(nz):
                cPress[kk, mum, i, :, :] = X_data1t["Pressure"][kk, mum, :, :, i]
                cSat[kk, mum, i, :, :] = X_data1t["Water_saturation"][kk, mum, :, :, i]
                cSatg[kk, mum, i, :, :] = X_data1t["Gas_saturation"][kk, mum, :, :, i]
    del X_data1t
    gc.collect()
    data_test = {
        "perm": cPerm,
        "Phi": cPhi,
        "Pini": cPini,
        "Swini": cSini,
        "pressure": cPress,
        "water_sat": cSat,
        "gas_sat": cSatg,
    }

    with gzip.open(to_absolute_path("../PACKETS/simulations_test.pkl.gz"), "wb") as f4:
        pickle.dump(data_test, f4)

    preprocess_FNO_mat2(to_absolute_path("../PACKETS/simulations_test.pkl.gz"))
    os.remove(to_absolute_path("../PACKETS/simulations_test.pkl.gz"))

    del cPerm
    gc.collect()
    del cQ
    gc.collect()
    del cQw
    gc.collect()
    del cQg
    gc.collect()
    del cPhi
    gc.collect()
    del cTime
    gc.collect()
    del cPini
    gc.collect()
    del cSini
    gc.collect()
    del cPress
    gc.collect()
    del cSat
    gc.collect()
    del cSatg
    gc.collect()

    del data_test
    gc.collect()

    # load training/ test data for Numerical simulation model
    input_keys = [
        Key("perm"),
        Key("Phi"),
        Key("Pini"),
        Key("Swini"),
    ]

    output_keys = [
        Key("pressure"),
        Key("water_sat"),
        Key("gas_sat"),
    ]

    invar_train, outvar_train = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulations_train.pk.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=N_ens,
    )
    os.remove(to_absolute_path("../PACKETS/simulations_train.pk.hdf5"))
    # os.remove(to_absolute_path("../PACKETS/simulations_train.mat"))

    for key in invar_train.keys():
        invar_train[key][np.isnan(invar_train[key])] = 0  # Convert NaN to 0
        invar_train[key][np.isinf(invar_train[key])] = 0  # Convert infinity to 0
        invar_train[key] = clip_and_convert_to_float32(invar_train[key])

    for key, value in invar_train.items():
        print(f"For key '{key}':")
        print("\tContains inf:", np.isinf(value).any())
        print("\tContains -inf:", np.isinf(-value).any())
        print("\tContains NaN:", np.isnan(value).any())
        print("\tSize = :", value.shape)

    for key in outvar_train.keys():
        outvar_train[key][np.isnan(outvar_train[key])] = 0  # Convert NaN to 0
        outvar_train[key][np.isinf(outvar_train[key])] = 0  # Convert infinity to 0
        outvar_train[key] = clip_and_convert_to_float32(outvar_train[key])
    for key, value in outvar_train.items():
        print(f"For key '{key}':")
        print("\tContains inf:", np.isinf(value).any())
        print("\tContains -inf:", np.isinf(-value).any())
        print("\tContains NaN:", np.isnan(value).any())
        print("\tSize = :", value.shape)

    invar_test, outvar_test = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulations_test.pk.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=cfg.custom.ntest,
    )
    os.remove(to_absolute_path("../PACKETS/simulations_test.pk.hdf5"))
    # os.remove(to_absolute_path("../PACKETS/simulations_test.mat"))

    for key in invar_test.keys():
        invar_test[key][np.isnan(invar_test[key])] = 0  # Convert NaN to 0
        invar_test[key][np.isinf(invar_test[key])] = 0  # Convert infinity to 0
        invar_test[key] = clip_and_convert_to_float32(invar_test[key])
    for key, value in invar_test.items():
        print(f"For key '{key}':")
        print("\tContains inf:", np.isinf(value).any())
        print("\tContains -inf:", np.isinf(-value).any())
        print("\tContains NaN:", np.isnan(value).any())
        print("\tSize = :", value.shape)

    for key in outvar_test.keys():
        outvar_test[key][np.isnan(outvar_test[key])] = 0  # Convert NaN to 0
        outvar_test[key][np.isinf(outvar_test[key])] = 0  # Convert infinity to 0
        outvar_test[key] = clip_and_convert_to_float32(outvar_test[key])
    for key, value in outvar_test.items():
        print(f"For key '{key}':")
        print("\tContains inf:", np.isinf(value).any())
        print("\tContains -inf:", np.isinf(-value).any())
        print("\tContains NaN:", np.isnan(value).any())
        print("\tSize of = :", value.shape)

    train_dataset = DictGridDataset(invar_train, outvar_train)

    test_dataset = DictGridDataset(invar_test, outvar_test)

    # Define FNO model
    # Pressure
    decoder = ConvFullyConnectedArch(
        [Key("z", size=32)],
        [
            Key("pressure", size=steppi),
            Key("water_sat", size=steppi),
            Key("gas_sat", size=steppi),
        ],
    )

    fno_supervised = FNOArch(
        [
            Key("perm", size=1),
            Key("Phi", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        dimension=3,
        decoder_net=decoder,
    )

    nodes = [fno_supervised.make_node("fno_forward_model")]

    # make domain
    domain = Domain()

    supervised_dynamic = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=1,
    )
    domain.add_constraint(supervised_dynamic, "supervised_dynamic")

    # [constraint]
    # add validator

    test_dynamic = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.test,
        requires_grad=False,
    )
    domain.add_validator(test_dynamic, "test_dynamic")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
