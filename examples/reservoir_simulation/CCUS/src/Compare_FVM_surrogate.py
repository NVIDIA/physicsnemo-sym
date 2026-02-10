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

from __future__ import print_function

print(__doc__)
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

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os.path
import torch

torch.set_default_dtype(torch.float32)
from struct import unpack
import fnmatch
from joblib import Parallel, delayed
from scipy import interpolate
import multiprocessing
import mpslib as mps
from shutil import rmtree
import numpy.matlib
import re
from numpy import linalg as LA
from pyDOE import lhs
import matplotlib.colors
from matplotlib import cm
import pickle
from physicsnemo.sym.models.activation import Activation
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fno import *
from PIL import Image
import requests
import sys
import numpy
from scipy.stats import rankdata, norm

# from PIL import Image
from scipy.fftpack import dct
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import numpy.matlib

# from matplotlib import pyplot
from skimage.metrics import structural_similarity as ssim

# os.environ['KERAS_BACKEND'] = 'tensorflow'
import os.path
import math
import time
import random
from matplotlib.font_manager import FontProperties
import os.path
from datetime import timedelta
from skimage.transform import resize as rzz
import h5py
import scipy.io as sio
import yaml
import matplotlib as mpl
import matplotlib.lines as mlines
import os

# import zipfile
import gzip

try:
    import gdown
except:
    gdown = None

import scipy.io

import shutil
from numpy import *
import scipy.optimize.lbfgsb as lbfgsb
import numpy.linalg
from scipy.fftpack.realtransforms import idct
import numpy.ma as ma
import logging
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


def get_shape(t):
    shape = []
    while isinstance(t, tuple):
        shape.append(len(t))
        t = t[0]
    return tuple(shape)


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


def MyLossClement(a, b):
    loss = torch.sum(torch.abs(a - b) / a.shape[0])

    # loss = ((a-b)**2).mean()
    return loss


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


def Plot_2D(
    XX,
    YY,
    plt,
    nx,
    ny,
    nz,
    Truee,
    N_injw,
    N_pr,
    N_injg,
    varii,
    injectors,
    producers,
    gass,
):
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
    elif varii == "water FLOW":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation - FLOW", fontsize=11, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("water saturation - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")

    elif varii == "gas PhysicsNeMo":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -PhysicsNeMo", fontsize=11, weight="bold")

    elif varii == "gas FLOW":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -FLOW", fontsize=11, weight="bold")

    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("gas saturation - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")

    elif varii == "pressure PhysicsNeMo":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -PhysicsNeMo", fontsize=11, weight="bold")

    elif varii == "pressure FLOW":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -FLOW", fontsize=11, weight="bold")

    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("Pressure - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")

    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=11)
        plt.title("Porosity Field", fontsize=11, weight="bold")
    cbar.mappable.set_clim(minii, maxii)

    plt.ylabel("Y", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gass)


def Plot_all_layesr(nx, ny, nz, see, injectors, producers, gass, varii):
    see[see == 0] = np.nan  # Convert zeros to NaNs
    plt.figure(figsize=(20, 20), dpi=300)
    Pressz = np.reshape(see, (nx, ny, nz), "F")
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    for i in range(nz):
        plt.subplot(5, 5, i + 1)

        plt.pcolormesh(XX.T, YY.T, Pressz[:, :, i], cmap="jet")
        cbar = plt.colorbar()

        if varii == "perm":
            cbar.ax.set_ylabel("Log K(mD)", fontsize=9)
            plt.title(
                "Permeability Field Layer_" + str(i + 1), fontsize=11, weight="bold"
            )
        if varii == "porosity":
            cbar.ax.set_ylabel("porosity", fontsize=9)
            plt.title("porosity Field Layer_" + str(i + 1), weight="bold", fontsize=11)
        if varii == "water":
            cbar.ax.set_ylabel("Water saturation", fontsize=9)
            plt.title(
                "Water saturation Field Layer_" + str(i + 1), weight="bold", fontsize=11
            )

        plt.ylabel("Y", fontsize=11)
        plt.xlabel("X", fontsize=11)
        plt.axis([0, (nx - 1), 0, (ny - 1)])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        Add_marker3(plt, XX, YY, injectors, producers, gass)

    if varii == "perm":
        plt.suptitle(
            r"Permeability NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "water":
        plt.suptitle(
            r"Water Saturation NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "porosity":
        plt.suptitle(
            r"Porosity NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )

    #
    plt.savefig("All.png")
    plt.clf()


def Add_marker2(plt, XX, YY, injectors, producers, gass):
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

    n_injg = len(gass)  # Number of gas injectors

    for mm in range(n_injg):
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


def Add_marker3(plt, XX, YY, injectors, producers, gass):
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

    n_injg = len(gass)  # Number of gas injectors

    for mm in range(n_injg):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])

        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
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
            fontsize=9,
        )


def fit_clement(tensor, target_min, target_max, tensor_min, tensor_max):
    # Rescale between target min and target max
    rescaled_tensor = tensor / tensor_max
    return rescaled_tensor


def ensemble_pytorch(
    param_perm,
    param_poro,
    nx,
    ny,
    nz,
    Ne,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minP,
    maxP,
    steppi,
    device,
):
    ini_ensemble1 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)  # Permeability
    ini_ensemble2 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)  # Porosity
    ini_ensemble9 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    ini_ensemble10 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)

    ini_ensemble9 = 200 * np.ones((Ne, 1, nz, nx, ny))
    ini_ensemble10 = np.ones((Ne, 1, nz, nx, ny))

    for kk in range(Ne):
        a = np.reshape(param_perm[:, kk], (nx, ny, nz), "F")
        a1 = np.reshape(param_poro[:, kk], (nx, ny, nz), "F")

        for my in range(nz):
            ini_ensemble1[kk, 0, my, :, :] = a[:, :, my]  # Permeability
            ini_ensemble2[kk, 0, my, :, :] = a1[:, :, my]  # Porosity

    # Initial_pressure
    ini_ensemble9 = fit_clement(ini_ensemble9, target_min, target_max, minP, maxP)

    # Permeability
    ini_ensemble1 = fit_clement(ini_ensemble1, target_min, target_max, minK, maxK)

    # ini_ensemble = torch.from_numpy(ini_ensemble).to(device, dtype=torch.float32)
    inn = {
        "perm": torch.from_numpy(ini_ensemble1).to(device, torch.float32),
        "Phi": torch.from_numpy(ini_ensemble2).to(device, dtype=torch.float32),
        "Pini": torch.from_numpy(ini_ensemble9).to(device, dtype=torch.float32),
        "Swini": torch.from_numpy(ini_ensemble10).to(device, dtype=torch.float32),
    }
    return inn


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
    dx = to_dlpack(tensor)
    cx = cp.fromDlpack(dx)
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


def replace_nans_and_infs(tensor, value=0.0):
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


def replace_large_and_invalid_values(arr, placeholder=0.0):
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


def Forward_model_ensemble(
    N,
    x_true,
    steppi,
    target_min,
    target_max,
    minK,
    maxK,
    minP,
    maxP,
    modelP1,
    device,
    num_cores,
    oldfolder,
):
    #### ===================================================================== ####
    #                     RESERVOIR CCUS SIMULATOR WITH MODULUS
    #
    #### ===================================================================== ####
    """
    Parameters:
    -----------
    N : int
        Number of ensemble samples.

    x_true : dict
        A dictionary containing the ground truth data for perm, Phi, fault, Pini, and Swini.

    steppi : int
        Number of time steps.

    min_inn_fcn, max_inn_fcn : float
        Minimum and maximum values for input scaling.

    target_min, target_max : float
        Minimum and maximum target values for scaling.

    minK, maxK, minT, maxT, minP, maxP : float
        Min-max values for permeability, time, and pressure.

    modelP1, modelW1, modelG1: torch.nn.Module
        Trained PyTorch models for predicting pressure, water saturation, gas saturation respectively.

    device : torch.device
        The device (CPU or GPU) where the PyTorch models will run.



    Returns:
    --------
    ouut_p : array-like
        Predicted outputs based on the input ensemble using the trained models.

    Description:
    ------------
    The function starts by predicting pressure, water saturation, and gas
    saturation using the input x_true and the trained models modelP1,
    modelW1, and modelG1. Then, the function processes the predictions,
    and scales and converts them back to their original values.
    """
    texta = """
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
    print(texta)
    pressure = []
    Swater = []
    Sgas = []

    for clem in range(N):
        temp = {
            "perm": x_true["perm"][clem, :, :, :, :][None, :, :, :, :],
            "Phi": x_true["Phi"][clem, :, :, :, :][None, :, :, :, :],
            "Pini": x_true["Pini"][clem, :, :, :, :][None, :, :, :, :],
            "Swini": x_true["Swini"][clem, :, :, :, :][None, :, :, :, :],
        }

        with torch.no_grad():
            ouut_p1 = modelP1(temp)["pressure"]
            ouut_s1 = modelP1(temp)["water_sat"]
            ouut_sg1 = modelP1(temp)["gas_sat"]

        pressure.append(ouut_p1)
        Swater.append(ouut_s1)
        Sgas.append(ouut_sg1)

        del temp
        torch.cuda.empty_cache()

    pressure = torch.vstack(pressure).detach().cpu().numpy()
    pressure = Make_correct(pressure)
    Swater = torch.vstack(Swater).detach().cpu().numpy()
    Swater = Make_correct(Swater)
    Sgas = torch.vstack(Sgas).detach().cpu().numpy()
    Sgas = Make_correct(Sgas)
    pressure = convert_back(pressure, target_min, target_max, minP, maxP)

    return pressure, Swater, Sgas


def Split_Matrix(matrix, sizee):
    x_split = np.split(matrix, sizee, axis=0)
    return x_split


def Remove_folder(N_ens, straa):
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)


def replace_nan_with_zero(tensor):
    nan_mask = torch.isnan(tensor)
    return tensor * (~nan_mask).float() + nan_mask.float() * 0.0


def compute_metrics(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    TSS = np.sum((y_true - y_true_mean) ** 2)
    RSS = np.sum((y_true - y_pred) ** 2)

    R2 = 1 - (RSS / TSS)
    L2_accuracy = 1 - np.sqrt(RSS) / np.sqrt(TSS)

    return R2, L2_accuracy


def compute_metrics_2(y_true1, y_pred_fno1, y_pred_pino1, steppi, name):
    big_out_r2 = np.zeros((steppi, 3))  # Adjusted for step index, R2fno, R2pino
    big_out_l2 = np.zeros((steppi, 3))  # Adjusted for step index, L2fno, L2pino
    big_out_ssim = np.zeros((steppi, 3))
    big_out_comb = np.zeros((steppi, 3))

    for i in range(steppi):
        y_true = y_true1[0, i, :, :, :].ravel()
        y_pred_fno = y_pred_fno1[0, i, :, :, :].ravel()
        y_pred_pino = y_pred_pino1[0, i, :, :, :].ravel()

        TSS = np.sum((y_true - np.mean(y_true)) ** 2)
        RSS_fno = np.sum((y_true - y_pred_fno) ** 2)
        RSS_pino = np.sum((y_true - y_pred_pino) ** 2)

        R2fno = 1 - (RSS_fno / TSS)
        R2pino = 1 - (RSS_pino / TSS)

        l2fno = np.linalg.norm(y_true - y_pred_fno)
        l2pino = np.linalg.norm(y_true - y_pred_pino)

        big_out_r2[i, 0] = 50 * (i + 1)
        big_out_r2[i, 1] = R2fno * 100
        big_out_r2[i, 2] = R2pino * 100

        big_out_l2[i, 0] = 50 * (i + 1)
        big_out_l2[i, 1] = l2fno
        big_out_l2[i, 2] = l2pino

        big_out_ssim[i, 0] = 50 * (i + 1)
        big_out_ssim[i, 1] = calculate_ssim_3d(
            y_true1[0, i, :, :, :], y_pred_fno1[0, i, :, :, :]
        )
        big_out_ssim[i, 2] = calculate_ssim_3d(
            y_true1[0, i, :, :, :], y_pred_pino1[0, i, :, :, :]
        )

        a1 = calculate_ssim_3d(y_true1[0, i, :, :, :], y_pred_fno1[0, i, :, :, :])
        a1 = abs(1 - a1)

        a2 = calculate_ssim_3d(y_true1[0, i, :, :, :], y_pred_pino1[0, i, :, :, :])
        a2 = abs(1 - a2)

        big_out_comb[i, 0] = 50 * (i + 1)
        big_out_comb[i, 1] = (1 - R2fno) + l2fno + a1
        big_out_comb[i, 2] = (1 - R2pino) + l2pino + a2

    # Creating the 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # R2 accuracy plot
    bar_width = 0.35
    index = np.arange(big_out_r2.shape[0])
    axs[0, 0].bar(index, big_out_r2[:, 1], bar_width, label="FNO", color="b")
    axs[0, 0].bar(
        index + bar_width, big_out_r2[:, 2], bar_width, label="PINO", color="r"
    )
    axs[0, 0].set_xlabel("years")
    axs[0, 0].set_ylabel("R2 Score (%)")
    axs[0, 0].set_title("R2 Score for FNO and PINO")
    axs[0, 0].set_xticks(index + bar_width / 2)
    axs[0, 0].set_xticklabels([f"{int(val)}" for val in big_out_r2[:, 0]])
    axs[0, 0].legend()

    # L2 norm plot
    axs[0, 1].bar(index, big_out_l2[:, 1], bar_width, label="FNO", color="b")
    axs[0, 1].bar(
        index + bar_width, big_out_l2[:, 2], bar_width, label="PINO", color="r"
    )
    axs[0, 1].set_xlabel("years")
    axs[0, 1].set_ylabel("L2 norm ")
    axs[0, 1].set_title("L2 norm for FNO and PINO")
    axs[0, 1].set_xticks(index + bar_width / 2)
    axs[0, 1].set_xticklabels([f"{int(val)}" for val in big_out_l2[:, 0]])
    axs[0, 1].legend()

    # SSIM norm plot
    axs[1, 0].bar(index, big_out_ssim[:, 1], bar_width, label="FNO", color="b")
    axs[1, 0].bar(
        index + bar_width, big_out_ssim[:, 2], bar_width, label="PINO", color="r"
    )
    axs[1, 0].set_xlabel("years")
    axs[1, 0].set_ylabel("SSIM value")
    axs[1, 0].set_title("SSIM index for FNO and PINO")
    axs[1, 0].set_xticks(index + bar_width / 2)
    axs[1, 0].set_xticklabels([f"{int(val)}" for val in big_out_ssim[:, 0]])
    axs[1, 0].legend()

    # overall norm plot
    axs[1, 1].bar(index, big_out_comb[:, 1], bar_width, label="FNO", color="b")
    axs[1, 1].bar(
        index + bar_width, big_out_comb[:, 2], bar_width, label="PINO", color="r"
    )
    axs[1, 1].set_xlabel("years")
    axs[1, 1].set_ylabel("Combined loss value")
    axs[1, 1].set_title("Combined loss for FNO and PINO")
    axs[1, 1].set_xticks(index + bar_width / 2)
    axs[1, 1].set_xticklabels([f"{int(val)}" for val in big_out_comb[:, 0]])
    axs[1, 1].legend()

    plt.tight_layout()
    namez = "R2L2_" + name + ".png"
    plt.savefig(namez)
    plt.clf()
    plt.close()


def calculate_ssim_3d(image1_array, image2_array):
    # Assuming the third dimension is slices or some non-color feature
    num_slices = image1_array.shape[2]
    ssim_values = np.zeros(num_slices)

    for i in range(num_slices):
        slice1 = image1_array[:, :, i]
        slice2 = image2_array[:, :, i]

        # Ensure slices are in the correct data type
        slice1 = slice1.astype(np.uint8)
        slice2 = slice2.astype(np.uint8)

        # Calculate SSIM for each slice
        ssim_values[i] = ssim(slice1, slice2)

    # Aggregate SSIM values across slices if necessary
    average_ssim = np.mean(ssim_values)
    return average_ssim


def Plot_PhysicsNeMo(ax, nx, ny, nz, Truee, N_injg, varii, gass):
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

    for mm in range(N_injg):
        usethis = gass[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 5)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "red", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="red",
            weight="bold",
            fontsize=5,
        )

    red_line = mlines.Line2D([], [], color="red", linewidth=2, label="gas injectors")

    # Add the legend to the plot
    ax.legend(handles=[red_line], loc="lower left", fontsize=9)

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
        ax.set_title("water saturation - Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "water saturation - (Numerical(Flow) -PhysicsNeMo))",
            fontsize=12,
            weight="bold",
        )

    elif varii == "gas PhysicsNeMo":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation -PhysicsNeMo", fontsize=12, weight="bold")

    elif varii == "gas Numerical":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation - Numerical(Flow)", fontsize=12, weight="bold")

    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "gas saturation - (Numerical(Flow) -PhysicsNeMo))",
            fontsize=12,
            weight="bold",
        )

    elif varii == "pressure PhysicsNeMo":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -PhysicsNeMo", fontsize=12, weight="bold")

    elif varii == "pressure Numerical":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -Numerical(Flow)", fontsize=12, weight="bold")

    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "Pressure - (Numerical(Flow) -PhysicsNeMo))", fontsize=12, weight="bold"
        )

    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=12)
        ax.set_title("Porosity Field", fontsize=12, weight="bold")
    cbar.mappable.set_clim(minii, maxii)


# list of FNO dataset url ids on drive: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-


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


def load_FNO_dataset2(
    path, input_keys, output_keys1, output_keys2, output_keys3, n_examples=None
):
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
    invar, outvar1, outvar2, outvar3 = dict(), dict(), dict(), dict()
    for d, keys in [
        (invar, input_keys),
        (outvar1, output_keys1),
        (outvar2, output_keys2),
        (outvar3, output_keys3),
    ]:
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

    return (invar, outvar1, outvar2, outvar3)


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


def interpolatebetween(xtrain, cdftrain, xnew):
    numrows1 = len(xnew)
    numcols = len(xnew[0])
    norm_cdftest2 = np.zeros((numrows1, numcols))
    for i in range(numcols):
        f = interpolate.interp1d((xtrain[:, i]), cdftrain[:, i], kind="linear")
        cdftest = f(xnew[:, i])
        norm_cdftest2[:, i] = np.ravel(cdftest)
    return norm_cdftest2


def gaussianizeit(input1):
    numrows1 = len(input1)
    numcols = len(input1[0])
    newbig = np.zeros((numrows1, numcols))
    for i in range(numcols):
        input11 = input1[:, i]
        newX = norm.ppf(rankdata(input11) / (len(input11) + 1))
        newbig[:, i] = newX.T
    return newbig


def best_fit(X, Y):
    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)
    n = len(X)  # or len(Y)
    numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar

    print("best fit line:\ny = {:.2f} + {:.2f}x".format(a, b))
    return a, b


from copy import copy


def Performance_plot_cost(CCR, Trued, stringg, training_master, oldfolder):
    CoDview = np.zeros((1, Trued.shape[1]))
    R2view = np.zeros((1, Trued.shape[1]))

    plt.figure(figsize=(40, 40))

    for jj in range(Trued.shape[1]):
        print(" Compute L2 and R2 for the machine _" + str(jj + 1))

        clementanswer2 = np.reshape(CCR[:, jj], (-1, 1))
        outputtest2 = np.reshape(Trued[:, jj], (-1, 1))
        numrowstest = len(outputtest2)
        outputtest2 = np.reshape(outputtest2, (-1, 1))
        Lerrorsparse = (
            LA.norm(outputtest2 - clementanswer2) / LA.norm(outputtest2)
        ) ** 0.5
        L_22 = 1 - (Lerrorsparse**2)
        # Coefficient of determination
        outputreq = np.zeros((numrowstest, 1))
        for i in range(numrowstest):
            outputreq[i, :] = outputtest2[i, :] - np.mean(outputtest2)
        CoDspa = 1 - (LA.norm(outputtest2 - clementanswer2) / LA.norm(outputreq))
        CoD2 = 1 - (1 - CoDspa) ** 2
        print("")

        CoDview[:, jj] = CoD2
        R2view[:, jj] = L_22

        jk = jj + 1
        plt.subplot(9, 9, jk)
        palette = copy(plt.get_cmap("inferno_r"))
        palette.set_under("white")  # 1.0 represents not transparent
        palette.set_over("black")  # 1.0 represents not transparent
        vmin = min(np.ravel(outputtest2))
        vmax = max(np.ravel(outputtest2))
        sc = plt.scatter(
            np.ravel(clementanswer2),
            np.ravel(outputtest2),
            c=np.ravel(outputtest2),
            vmin=vmin,
            vmax=vmax,
            s=35,
            cmap=palette,
        )
        plt.colorbar(sc)
        plt.title("Energy_" + str(jj), fontsize=9)
        plt.ylabel("Machine", fontsize=9)
        plt.xlabel("True data", fontsize=9)
        a, b = best_fit(
            np.ravel(clementanswer2),
            np.ravel(outputtest2),
        )
        yfit = [a + b * xi for xi in np.ravel(clementanswer2)]
        plt.plot(np.ravel(clementanswer2), yfit, color="r")
        plt.annotate(
            "R2= %.3f" % CoD2,
            (0.8, 0.2),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size=9,
        )

    CoDoverall = (np.sum(CoDview, axis=1)) / Trued.shape[1]
    R2overall = (np.sum(R2view, axis=1)) / Trued.shape[1]
    os.chdir(training_master)
    plt.savefig("%s.jpg" % stringg)
    os.chdir(oldfolder)
    return CoDoverall, R2overall, CoDview, R2view


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, "confirm": 1}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
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


from contextlib import contextmanager


@contextmanager
def change_dir(destination):
    current_dir = os.getcwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(current_dir)


def process_step(
    kk,
    steppi,
    dt,
    pressure,
    pressure_true,
    Swater,
    Swater_true,
    Sgas,
    Sgas_true,
    nx,
    ny,
    nz,
    N_injg,
    gass,
    fold,
    fold1,
):
    os.chdir(fold)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    current_time = dt[kk]
    # Time_vector[kk] = current_time

    f_3 = plt.figure(figsize=(20, 20), dpi=200)

    look = pressure[0, kk, :, :, :]  # [:, :, ::-1]

    lookf = pressure_true[0, kk, :, :, :]  # [:, :, ::-1]
    # lookf = lookf * pini_alt
    diff1 = abs(look - lookf)  # [:, :, ::-1]

    ax1 = f_3.add_subplot(3, 3, 1, projection="3d")
    Plot_PhysicsNeMo(ax1, nx, ny, nz, look, N_injg, "pressure PhysicsNeMo", gass)
    ax2 = f_3.add_subplot(3, 3, 2, projection="3d")
    Plot_PhysicsNeMo(ax2, nx, ny, nz, lookf, N_injg, "pressure Numerical", gass)
    ax3 = f_3.add_subplot(3, 3, 3, projection="3d")
    Plot_PhysicsNeMo(ax3, nx, ny, nz, diff1, N_injg, "pressure diff", gass)
    R2p, L2p = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_presure[kk, 0] = R2p
    Accuracy_presure[kk, 1] = L2p

    look = Swater[0, kk, :, :, :]  # [:, :, ::-1]
    lookf = Swater_true[0, kk, :, :, :]  # [:, :, ::-1]
    diff1 = abs(look - lookf)  # [:, :, ::-1]
    ax1 = f_3.add_subplot(3, 3, 4, projection="3d")
    Plot_PhysicsNeMo(ax1, nx, ny, nz, look, N_injg, "water PhysicsNeMo", gass)
    ax2 = f_3.add_subplot(3, 3, 5, projection="3d")
    Plot_PhysicsNeMo(ax2, nx, ny, nz, lookf, N_injg, "water Numerical", gass)
    ax3 = f_3.add_subplot(3, 3, 6, projection="3d")
    Plot_PhysicsNeMo(ax3, nx, ny, nz, diff1, N_injg, "water diff", gass)
    R2w, L2w = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_water[kk, 0] = R2w
    Accuracy_water[kk, 1] = L2w

    look = Sgas[0, kk, :, :, :]  # [:, :, ::-1]
    lookf = Sgas_true[0, kk, :, :, :]  # [:, :, ::-1]
    diff1 = abs(look - lookf)  # [:, :, ::-1]
    ax1 = f_3.add_subplot(3, 3, 7, projection="3d")
    Plot_PhysicsNeMo(ax1, nx, ny, nz, look, N_injg, "gas PhysicsNeMo", gass)
    ax2 = f_3.add_subplot(3, 3, 8, projection="3d")
    Plot_PhysicsNeMo(ax2, nx, ny, nz, lookf, N_injg, "gas Numerical", gass)
    ax3 = f_3.add_subplot(3, 3, 9, projection="3d")
    Plot_PhysicsNeMo(ax3, nx, ny, nz, diff1, N_injg, "gas diff", gass)
    R2g, L2g = compute_metrics(look.ravel(), lookf.ravel())
    Accuracy_gas[kk, 0] = R2g
    Accuracy_gas[kk, 1] = L2g

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(current_time) + " years"
    plt.suptitle(tita, fontsize=16)
    # plt.savefig('Dynamic' + str(int(kk)))
    plt.savefig("Dynamic" + str(int(kk)))
    plt.clf()
    plt.close()
    return R2p, L2p, R2w, L2w, R2g, L2g
    os.chdir(fold1)


oldfolder = os.getcwd()
os.chdir(oldfolder)

if not os.path.exists("../COMPARE_RESULTS"):
    os.makedirs("../COMPARE_RESULTS")

print("")

mat = sio.loadmat("../PACKETS/conversions.mat")
learn_pde = mat["learn_pde"]


if not os.path.exists("../COMPARE_RESULTS/PINO/"):
    os.makedirs("../COMPARE_RESULTS/PINO/")

folderr = os.path.join(oldfolder, "..", "COMPARE_RESULTS", "PINO")

print("")

# num_cores = 6
num_cores = multiprocessing.cpu_count()
njobs = (num_cores // 4) - 1
num_cores = njobs
# num_cores = 2
# njobs= 2
print("")


fname = "conf/config_PINO.yaml"


mat = sio.loadmat("../PACKETS/conversions.mat")
minK = mat["minK"]
maxK = mat["maxK"]
minP = mat["minP"]
maxP = mat["maxP"]
steppi = int(mat["steppi"])
# print(steppi)

target_min = 0.01
target_max = 1


plan = read_yaml(fname)
nx = plan["custom"]["PROPS"]["nx"]
ny = plan["custom"]["PROPS"]["ny"]
nz = plan["custom"]["PROPS"]["nz"]
Ne = 1

with gzip.open(("../PACKETS/static.pkl.gz"), "rb") as f2:
    mat = pickle.load(f2)
X_data1 = mat
for key, value in X_data1.items():
    print(f"For key '{key}':")
    print("\tContains inf:", np.isinf(value).any())
    print("\tContains -inf:", np.isinf(-value).any())
    print("\tContains NaN:", np.isnan(value).any())

perm_ensemble = X_data1["ensemble"]
poro_ensemble = X_data1["ensemblep"]

# index = np.random.choice(perm_ensemble.shape[1], Ne, \
#                          replace=False)
index = 20
perm_use = perm_ensemble[:, index].reshape(-1, 1)
poro_use = poro_ensemble[:, index].reshape(-1, 1)


gass = plan["custom"]["WELLSPECS"]["gas_injector_wells"]


N_injg = len(
    plan["custom"]["WELLSPECS"]["gas_injector_wells"]
)  # Number of gas injectors
string_Jesus2 = "flow CO2STORE_GASWAT.DATA"
oldfolder2 = os.getcwd()
path_out = "../True_Flow"
os.makedirs(path_out, exist_ok=True)
copy_files("../Necessaryy", path_out)
save_files(perm_use, poro_use, path_out, oldfolder2)

print("")
print("---------------------------------------------------------------------")
print("")
print("\n")
print("|-----------------------------------------------------------------|")
print("|                 RUN FLOW SIMULATOR                              |")
print("|-----------------------------------------------------------------|")
print("")
start_time_plots1 = time.time()
Run_simulator(path_out, oldfolder2, string_Jesus2)
elapsed_time_secs = (time.time() - start_time_plots1) / 2
msg = "Reservoir simulation with FLOW  took: %s secs (Wall clock time)" % timedelta(
    seconds=round(elapsed_time_secs)
)
print(msg)
print("")
print("Finished FLOW NUMERICAL simulations")

print("|-----------------------------------------------------------------|")
print("|                 DATA CURRATION IN PROCESS                       |")
print("|-----------------------------------------------------------------|")
N = Ne
# steppi = 246
check = np.ones((nx, ny, nz), dtype=np.float16)
pressure = []
Sgas = []
Swater = []
Time = []

permeability = np.zeros((N, 1, nx, ny, nz))
porosity = np.zeros((N, 1, nx, ny, nz))
actnumm = np.zeros((N, 1, nx, ny, nz))


folder = path_out
Pr, sw, sg, tt = Geta_all(folder, nx, ny, nz, oldfolder2, check, steppi)
pressure.append(Pr)
Sgas.append(sg)
Swater.append(sw)
Time.append(tt)

permeability[0, 0, :, :, :] = np.reshape(perm_ensemble[:, index], (nx, ny, nz), "F")
porosity[0, 0, :, :, :] = np.reshape(poro_ensemble[:, index], (nx, ny, nz), "F")


pressure_true = np.stack(pressure, axis=0)
Sgas_true = np.stack(Sgas, axis=0)
Swater_true = np.stack(Swater, axis=0)
Soil_true = 1 - (Sgas_true + Swater_true)
Time = np.stack(Time, axis=0)


folderrr = "../True_Flow"
rmtree(folderrr)

print("")
print("---------------------------------------------------------------------")
print("")
print("\n")
print("|-----------------------------------------------------------------|")
print("|          RUN  NVIDIA MODULUS RESERVOIR SIMULATION SURROGATE     |")
print("|-----------------------------------------------------------------|")
print("")


if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:  # Choose GPU 1 (index 1)
        device = torch.device("cuda:0")
    else:  # If there's only one GPU or no GPUs, choose the first one (index 0)
        device = torch.device("cuda:0")
else:  # If CUDA is not available, use the CPU
    raise RuntimeError("No GPU found. Please run on a system with a GPU.")
torch.cuda.set_device(device)

inn = ensemble_pytorch(
    perm_use,
    poro_use,
    nx,
    ny,
    nz,
    Ne,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minP,
    maxP,
    steppi,
    device,
)


print("")
print("Finished constructing Pytorch inputs")


print("*******************Load the trained Forward models (PINO)*******************")

decoder1 = ConvFullyConnectedArch(
    [Key("z", size=32)],
    [
        Key("pressure", size=steppi),
        Key("water_sat", size=steppi),
        Key("gas_sat", size=steppi),
    ],
    activation_fn=Activation.RELU,
)

fno_supervised1 = FNOArch(
    [
        Key("perm", size=1),
        Key("Phi", size=1),
        Key("Pini", size=1),
        Key("Swini", size=1),
    ],
    dimension=3,
    decoder_net=decoder1,
    activation_fn=Activation.RELU,
)


os.chdir("outputs/Forward_problem_PINO/CCUS")
print(" Surrogate model learned with PINO for dynamic properties - Gas model")
fno_supervised1.load_state_dict(torch.load("fno_forward_model.0.pth"))
fno_supervised1 = fno_supervised1.to(device)
fno_supervised1.eval()
os.chdir(oldfolder)

print("********************Model Loaded*************************************")

texta = """
PPPPPPPPPPPPPPPPP  IIIIIIIIINNNNNNNN        NNNNNNNN    OOOOOOOOO     
P::::::::::::::::P I::::::::N:::::::N       N::::::N  OO:::::::::OO   
P::::::PPPPPP:::::PI::::::::N::::::::N      N::::::NOO:::::::::::::OO 
PP:::::P     P:::::II::::::IN:::::::::N     N::::::O:::::::OOO:::::::O
  P::::P     P:::::P I::::I N::::::::::N    N::::::O::::::O   O::::::O
  P::::P     P:::::P I::::I N:::::::::::N   N::::::O:::::O     O:::::O
  P::::PPPPPP:::::P  I::::I N:::::::N::::N  N::::::O:::::O     O:::::O
  P:::::::::::::PP   I::::I N::::::N N::::N N::::::O:::::O     O:::::O
  P::::PPPPPPPPP     I::::I N::::::N  N::::N:::::::O:::::O     O:::::O
  P::::P             I::::I N::::::N   N:::::::::::O:::::O     O:::::O
  P::::P             I::::I N::::::N    N::::::::::O:::::O     O:::::O
  P::::P             I::::I N::::::N     N:::::::::O::::::O   O::::::O
PP::::::PP         II::::::IN::::::N      N::::::::O:::::::OOO:::::::O
P::::::::P         I::::::::N::::::N       N:::::::NOO:::::::::::::OO 
P::::::::P         I::::::::N::::::N        N::::::N  OO:::::::::OO   
PPPPPPPPPP         IIIIIIIIINNNNNNNN         NNNNNNN    OOOOOOOOO                                                                                                                                                

"""

print(texta)
start_time_plots2 = time.time()


pressure, Swater, Sgas = Forward_model_ensemble(
    Ne,
    inn,
    steppi,
    target_min,
    target_max,
    minK,
    maxK,
    minP,
    maxP,
    fno_supervised1,
    device,
    num_cores,
    oldfolder,
)

elapsed_time_secs2 = time.time() - start_time_plots2
msg = (
    "Reservoir simulation with NVIDIA PhysicsNeMo took: %s secs (Wall clock time)"
    % timedelta(seconds=round(elapsed_time_secs2))
)
print(msg)
print("")


physicsnemo_time = elapsed_time_secs2
flow_time = elapsed_time_secs


# Determine faster and slower times
if physicsnemo_time < flow_time:
    slower_time = physicsnemo_time
    faster_time = flow_time
    slower = "NVIDIA PhysicsNeMo Surrogate"
    faster = "Flow Reservoir Simulator"
else:
    slower_time = flow_time
    faster_time = physicsnemo_time
    slower = "Flow Reservoir Simulator"
    faster = "NVIDIA PhysicsNeMo Surrogate"

# Calculate speedup
speedup = math.ceil(faster_time / slower_time)

# Navigate to folder for saving the plot
os.chdir(folderr)

# Data for plotting
tasks = ["Flow", "PhysicsNeMo"]
times = [flow_time, physicsnemo_time]
colors = ["green", "red"]

# Create the plot
plt.figure(figsize=(10, 8))
bars = plt.bar(tasks, times, color=colors)
plt.ylabel(
    "Time (seconds)", fontweight="bold", fontsize=16
)  # Increased font size for ylabel
plt.title(
    "Execution Time Comparison for PhysicsNeMo vs. Flow", fontweight="bold", fontsize=15
)  # Increased font size for title
plt.ylim(0, max(times) * 1.5)  # Increase the upper limit to 150% of the maximum time

# Annotate bars with their values
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,
        f"{round(yval, 2)}s",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=15,
    )  # Optional: Adjusted font size for annotations


# plt.show()
# Indicate speedup on the chart
plt.text(
    0.5,
    max(times) + 2,
    f"Speedup: {speedup}x",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="blue",
)  # Adjusted font size for speedup text

# Save and close the plot
namez = "Compare_time.png"
plt.savefig(namez)
plt.clf()
plt.close()

# Navigate back to the original directory
# os.chdir(oldfolder)


message = (
    f"{slower} execution took: {slower_time} seconds\n"
    f"{faster} execution took: {faster_time} seconds\n"
    f"Speedup =  {speedup}X  "
)

print(message)


Time_unie = np.zeros((steppi))
for i in range(steppi):
    Time_unie[i] = Time[0, i, 0, 0, 0]


dt = Time_unie
print("")
print("Plotting outputs")


Runs = steppi
ty = np.arange(1, Runs + 1)
Time_vector = Time_unie
Accuracy_presure = np.zeros((steppi, 2))
Accuracy_oil = np.zeros((steppi, 2))
Accuracy_water = np.zeros((steppi, 2))
Accuracy_gas = np.zeros((steppi, 2))


os.chdir(oldfolder)


results = Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
    delayed(process_step)(
        kk,
        steppi,
        dt,
        pressure,
        pressure_true,
        Swater,
        Swater_true,
        Sgas,
        Sgas_true,
        nx,
        ny,
        nz,
        N_injg,
        gass,
        folderr,
        oldfolder,
    )
    for kk in range(steppi)
)

progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, steppi - 1, steppi - 1)
ShowBar(progressBar)
time.sleep(1)

os.chdir(oldfolder)
os.chdir(folderr)
# Aggregating results
for kk, (R2p, L2p, R2w, L2w, R2g, L2g) in enumerate(results):
    Accuracy_presure[kk, 0] = R2p
    Accuracy_presure[kk, 1] = L2p
    Accuracy_water[kk, 0] = R2w
    Accuracy_water[kk, 1] = L2w
    Accuracy_gas[kk, 0] = R2g
    Accuracy_gas[kk, 1] = L2g

# os.chdir(folderr)
fig4 = plt.figure(figsize=(20, 20), dpi=100)
font = FontProperties()
font.set_family("Helvetica")
font.set_weight("bold")

fig4.text(
    0.5,
    0.98,
    "R2(%) Accuracy - PhysicsNeMo/Numerical",
    ha="center",
    va="center",
    fontproperties=font,
    fontsize=11,
)
fig4.text(
    0.5,
    0.49,
    "L2(%) Accuracy - PhysicsNeMo/Numerical",
    ha="center",
    va="center",
    fontproperties=font,
    fontsize=11,
)

# Plot R2 accuracies
plt.subplot(2, 3, 1)
plt.plot(
    Time_vector,
    Accuracy_presure[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("Pressure", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

plt.subplot(2, 3, 2)
plt.plot(
    Time_vector,
    Accuracy_water[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("water saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

plt.subplot(2, 3, 3)
plt.plot(
    Time_vector,
    Accuracy_gas[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("gas saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

# Plot L2 accuracies
plt.subplot(2, 3, 4)
plt.plot(
    Time_vector,
    Accuracy_presure[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("Pressure", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)

plt.subplot(2, 3, 5)
plt.plot(
    Time_vector,
    Accuracy_water[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("water saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)


plt.subplot(2, 3, 6)
plt.plot(
    Time_vector,
    Accuracy_gas[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("gas saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)


plt.tight_layout(rect=[0, 0.05, 1, 0.93])
namez = "R2L2.png"
plt.savefig(namez)
plt.clf()
plt.close()


print("")
print("Now - Creating GIF")
import glob


frames = []
imgs = sorted(glob.glob("*Dynamic*"), key=sort_key)

for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "Evolution.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0,
)


from glob import glob

for f3 in glob("*Dynamic*"):
    os.remove(f3)

os.chdir(oldfolder)

del folderr


if not os.path.exists("../COMPARE_RESULTS/FNO/"):
    os.makedirs("../COMPARE_RESULTS/FNO/")

folderr1 = os.path.join(oldfolder, "..", "COMPARE_RESULTS", "FNO")

print("*******************Load the trained Forward model(FNO)*******************")

decoder_fno = ConvFullyConnectedArch(
    [Key("z", size=32)],
    [
        Key("pressure", size=steppi),
        Key("water_sat", size=steppi),
        Key("gas_sat", size=steppi),
    ],
)

fno_supervised2 = FNOArch(
    [
        Key("perm", size=1),
        Key("Phi", size=1),
        Key("Pini", size=1),
        Key("Swini", size=1),
    ],
    dimension=3,
    decoder_net=decoder_fno,
)


os.chdir("outputs/Forward_problem_FNO/CCUS")
print(" Surrogate model learned with PINO for dynamic properties - Gas model")
fno_supervised2.load_state_dict(torch.load("fno_forward_model.0.pth"))
fno_supervised2 = fno_supervised2.to(device)
fno_supervised2.eval()
os.chdir(oldfolder)

print("********************Model Loaded*************************************")

texta = """
FFFFFFFFFFFFFFFFFFFFFNNNNNNNN        NNNNNNNN    OOOOOOOOO     
F::::::::::::::::::::N:::::::N       N::::::N  OO:::::::::OO   
F::::::::::::::::::::N::::::::N      N::::::NOO:::::::::::::OO 
FF::::::FFFFFFFFF::::N:::::::::N     N::::::O:::::::OOO:::::::O
  F:::::F       FFFFFN::::::::::N    N::::::O::::::O   O::::::O
  F:::::F            N:::::::::::N   N::::::O:::::O     O:::::O
  F::::::FFFFFFFFFF  N:::::::N::::N  N::::::O:::::O     O:::::O
  F:::::::::::::::F  N::::::N N::::N N::::::O:::::O     O:::::O
  F:::::::::::::::F  N::::::N  N::::N:::::::O:::::O     O:::::O
  F::::::FFFFFFFFFF  N::::::N   N:::::::::::O:::::O     O:::::O
  F:::::F            N::::::N    N::::::::::O:::::O     O:::::O
  F:::::F            N::::::N     N:::::::::O::::::O   O::::::O
FF:::::::FF          N::::::N      N::::::::O:::::::OOO:::::::O
F::::::::FF          N::::::N       N:::::::NOO:::::::::::::OO 
F::::::::FF          N::::::N        N::::::N  OO:::::::::OO   
FFFFFFFFFFF          NNNNNNNN         NNNNNNN    OOOOOOOOO   
"""
print(texta)
start_time_plots2 = time.time()


pressuref, Swaterf, Sgasf = Forward_model_ensemble(
    Ne,
    inn,
    steppi,
    target_min,
    target_max,
    minK,
    maxK,
    minP,
    maxP,
    fno_supervised2,
    device,
    num_cores,
    oldfolder,
)

elapsed_time_secs2 = time.time() - start_time_plots2
msg = (
    "Reservoir simulation with NVIDIA PhysicsNeMo (FNO) took: %s secs (Wall clock time)"
    % timedelta(seconds=round(elapsed_time_secs2))
)
print(msg)
print("")


physicsnemo_time = elapsed_time_secs2
flow_time = elapsed_time_secs


# Determine faster and slower times
if physicsnemo_time < flow_time:
    slower_time = physicsnemo_time
    faster_time = flow_time
    slower = "NVIDIA PhysicsNeMo Surrogate"
    faster = "Flow Reservoir Simulator"
else:
    slower_time = flow_time
    faster_time = physicsnemo_time
    slower = "Flow Reservoir Simulator"
    faster = "NVIDIA PhysicsNeMo Surrogate"

# Calculate speedup
speedup = math.ceil(faster_time / slower_time)

# Navigate to folder for saving the plot
os.chdir(folderr1)

# Data for plotting
tasks = ["Flow", "PhysicsNeMo"]
times = [flow_time, physicsnemo_time]
colors = ["green", "red"]

# Create the plot
plt.figure(figsize=(10, 8))
bars = plt.bar(tasks, times, color=colors)
plt.ylabel(
    "Time (seconds)", fontweight="bold", fontsize=16
)  # Increased font size for ylabel
plt.title(
    "Execution Time Comparison for PhysicsNeMo vs. Flow", fontweight="bold", fontsize=15
)  # Increased font size for title
plt.ylim(0, max(times) * 1.5)  # Increase the upper limit to 150% of the maximum time

# Annotate bars with their values
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,
        f"{round(yval, 2)}s",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=15,
    )  # Optional: Adjusted font size for annotations


# plt.show()
# Indicate speedup on the chart
plt.text(
    0.5,
    max(times) + 2,
    f"Speedup: {speedup}x",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="blue",
)  # Adjusted font size for speedup text

# Save and close the plot
namez = "Compare_time.png"
plt.savefig(namez)
plt.clf()
plt.close()

# Navigate back to the original directory
# os.chdir(oldfolder)


message = (
    f"{slower} execution took: {slower_time} seconds\n"
    f"{faster} execution took: {faster_time} seconds\n"
    f"Speedup =  {speedup}X  "
)

print(message)


Time_unie = np.zeros((steppi))
for i in range(steppi):
    Time_unie[i] = Time[0, i, 0, 0, 0]


dt = Time_unie
print("")
print("Plotting outputs")


Runs = steppi
ty = np.arange(1, Runs + 1)
Time_vector = Time_unie
Accuracy_presure = np.zeros((steppi, 2))
Accuracy_oil = np.zeros((steppi, 2))
Accuracy_water = np.zeros((steppi, 2))
Accuracy_gas = np.zeros((steppi, 2))
os.chdir(oldfolder)

# os.chdir(folderr1)

results = Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
    delayed(process_step)(
        kk,
        steppi,
        dt,
        pressuref,
        pressure_true,
        Swaterf,
        Swater_true,
        Sgasf,
        Sgas_true,
        nx,
        ny,
        nz,
        N_injg,
        gass,
        folderr1,
        oldfolder,
    )
    for kk in range(steppi)
)


progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, steppi - 1, steppi - 1)
ShowBar(progressBar)
time.sleep(1)

os.chdir(oldfolder)
# Aggregating results
for kk, (R2p, L2p, R2w, L2w, R2g, L2g) in enumerate(results):
    Accuracy_presure[kk, 0] = R2p
    Accuracy_presure[kk, 1] = L2p
    Accuracy_water[kk, 0] = R2w
    Accuracy_water[kk, 1] = L2w
    Accuracy_gas[kk, 0] = R2g
    Accuracy_gas[kk, 1] = L2g

os.chdir(folderr1)
fig4 = plt.figure(figsize=(20, 20), dpi=100)
font = FontProperties()
font.set_family("Helvetica")
font.set_weight("bold")

fig4.text(
    0.5,
    0.98,
    "R2(%) Accuracy - PhysicsNeMo/Numerical",
    ha="center",
    va="center",
    fontproperties=font,
    fontsize=11,
)
fig4.text(
    0.5,
    0.49,
    "L2(%) Accuracy - PhysicsNeMo/Numerical",
    ha="center",
    va="center",
    fontproperties=font,
    fontsize=11,
)

# Plot R2 accuracies
plt.subplot(2, 3, 1)
plt.plot(
    Time_vector,
    Accuracy_presure[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("Pressure", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

plt.subplot(2, 3, 2)
plt.plot(
    Time_vector,
    Accuracy_water[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("water saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

plt.subplot(2, 3, 3)
plt.plot(
    Time_vector,
    Accuracy_gas[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("gas saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("R2(%)", fontproperties=font)

# Plot L2 accuracies
plt.subplot(2, 3, 4)
plt.plot(
    Time_vector,
    Accuracy_presure[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("Pressure", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)

plt.subplot(2, 3, 5)
plt.plot(
    Time_vector,
    Accuracy_water[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("water saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)


plt.subplot(2, 3, 6)
plt.plot(
    Time_vector,
    Accuracy_gas[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("gas saturation", fontproperties=font)
plt.xlabel("Time (days)", fontproperties=font)
plt.ylabel("L2(%)", fontproperties=font)


plt.tight_layout(rect=[0, 0.05, 1, 0.93])
namez = "R2L2.png"
plt.savefig(namez)
plt.clf()
plt.close()


print("")
print("Now - Creating GIF")
import glob


frames = []
imgs = sorted(glob.glob("*Dynamic*"), key=sort_key)

for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "Evolution.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0,
)


from glob import glob

for f3 in glob("*Dynamic*"):
    os.remove(f3)


os.chdir(oldfolder)

os.chdir("../COMPARE_RESULTS")
compute_metrics_2(pressure_true, pressuref, pressure, steppi, "pressure")
compute_metrics_2(Sgas_true, Sgasf, Sgas, steppi, "Gas_saturation")
compute_metrics_2(Swater_true, Swaterf, Swater, steppi, "Brine_saturation")
os.chdir(oldfolder)

print("")
print("----------------------------------------------------------------------")
print("-------------------PROGRAM EXECUTED-----------------------------------")
