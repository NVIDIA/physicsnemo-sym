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

Reservoir Modelling with a GPU accelerated  Black oil simulator.

AMG to solve the pressure and saturation well possed inverse problem

Solve an Inverse Problem:
    
@Parametrisation Methods:
    1)KSVD/OMP
    2)DCT
    3)DENOISING AUTOENCODER
    5)AUTOENCODER
    6)SVD
    7)MoE/CCR
    8)GAN
    9) DIFFUSION MODEL
@Data Assimilation Methods: Weighted Adaptive REKI - Adaptive Regularised Ensemble Kalman\
Inversion

    
16 Measurements to be matched: 4 WBHP and 4 WWCT, 4 WOPR , 4 WWPR
The Field has 4 producers and 4 Injectors

@Author : Clement Etienam
"""

from __future__ import print_function

print(__doc__)
print(".........................IMPORT SOME LIBRARIES.....................")
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)
from NVR import *
import shutil
import pandas as pd
import scipy.io as sio
from shutil import rmtree
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from loky import get_reusable_executor

oldfolder = os.getcwd()
os.chdir(oldfolder)
# import numpy as np
# executor = get_reusable_executor()


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


def is_available():
    """
    Check NVIDIA with nvidia-smi command
    Returning code 0 if no error, it means NVIDIA is installed
    Other codes mean not installed
    """
    code = os.system("nvidia-smi")
    return code


# import matplotlib.pyplot as plt
import time
from datetime import timedelta
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import MiniBatchKMeans
import os.path
from scipy.fftpack import dct
from scipy.fftpack.realtransforms import idct
import multiprocessing

from sklearn.model_selection import train_test_split
import numpy.matlib
from pyDOE import lhs
from matplotlib.colors import LinearSegmentedColormap

# rcParams['font.family'] = 'sans-serif'
# cParams['font.sans-serif'] = ['Helvetica']
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import (
    BatchNormalization,
    ZeroPadding2D,
    Dense,
    Input,
    LeakyReLU,
    UpSampling2D,
    Conv2D,
    MaxPooling2D,
)
from numpy.linalg import inv
from math import sqrt
import numpy
import numpy.matlib

# os.environ['KERAS_BACKEND'] = 'tensorflow'
import torch

torch.cuda.empty_cache()
import random as ra

# import torch.nn.functional as F
import scipy.linalg as sla
from torch.utils.data import DataLoader, TensorDataset

# from torch.autograd import Variable

# import dolfin as df
from numpy import *
import scipy.optimize.lbfgsb as lbfgsb
import numpy.linalg
from numpy.linalg import norm
import numpy.ma as ma
import logging
import os

# from joblib import Parallel, delayed
from FyeldGenerator import generate_field


colors = [
    (0, 0, 0),
    (0.3, 0.15, 0.75),
    (0.6, 0.2, 0.50),
    (1, 0.25, 0.15),
    (0.9, 0.5, 0),
    (0.9, 0.9, 0.5),
    (1, 1, 1),
]
n_bins = 7  # Discretizes the interpolation into bins
cmap_name = "my_list"
cmm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
import numpy
import numpy.matlib

# import sys
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # I have just 1 GPU
from cpuinfo import get_cpu_info
import requests

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

torch.cuda.empty_cache()
torch.set_default_dtype(torch.float32)


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


def load_data_numpy(inn, batch_size):
    x_data = inn
    print(f"x_data: {x_data.shape}")
    data_tuple = (torch.FloatTensor(x_data),)
    data_loader = DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk


# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


class RMS:
    """Compute RMS error & dev."""

    def __init__(self, truth, ensemble):
        mean = ensemble.mean(axis=0)
        err = truth - mean
        dev = ensemble - mean
        self.rmse = norm(err)
        self.rmsd = norm(dev)

    def __str__(self):
        return "%6.4f (rmse),  %6.4f (std)" % (self.rmse, self.rmsd)


def RMS_all(series, vs):
    """RMS for each item in series."""
    for k in series:
        if k != vs:
            print(f"{k:8}:", str(RMS(series[vs], series[k])))


def svd0(A):
    """Similar to Matlab's svd(A,0).

    Compute the

     - full    svd if nrows > ncols
     - reduced svd otherwise.

    As in Matlab: svd(A,0),
    except that the input and output are transposed

    .. seealso:: tsvd() for rank (and threshold) truncation.
    """
    M, N = A.shape
    if M > N:
        return sla.svd(A, full_matrices=True)
    return sla.svd(A, full_matrices=False)


def pad0(ss, N):
    """Pad ss with zeros so that len(ss)==N."""
    out = np.zeros(N)
    out[: len(ss)] = ss
    return out


def center(E, axis=0, rescale=False):
    """Center ensemble.

    Makes use of np features: keepdims and broadcasting.

    - rescale: Inflate to compensate for reduction in the expected variance.
    """
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N / (N - 1))

    x = x.squeeze()

    return X, x


def mean0(E, axis=0, rescale=True):
    """Same as: center(E,rescale=True)[0]"""
    return center(E, axis=axis, rescale=rescale)[0]


def inflate_ens(E, factor):
    """Inflate the ensemble (center, inflate, re-combine)."""
    if factor == 1:
        return E
    X, x = center(E)
    return x + X * factor


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


# import random


def honour2(sgsim2, DupdateK, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_p):
    if Yet == 0:
        sgsim2 = cp.asarray(sgsim2)
        DupdateK = cp.asarray(DupdateK)
    else:
        pass

    output = cp.zeros((nx * ny * nz, N_ens))
    outputporo = cp.zeros((nx * ny * nz, N_ens))

    output = DupdateK
    outputporo = sgsim2

    output[output >= High_K] = High_K  # highest value in true permeability
    output[output <= Low_K] = Low_K

    outputporo[outputporo >= High_P] = High_P
    outputporo[outputporo <= Low_P] = Low_P
    if Yet == 0:
        return cp.asnumpy(output), cp.asnumpy(outputporo)
    else:
        return output, outputporo


def funcGetDataMismatch(simData, measurment):
    """


    Parameters
    ----------
    simData : Simulated data

    measurment : True Measurement
        DESCRIPTION.

    Returns
    -------
    obj : Root mean squared error

    objStd : standard deviation

    objReal : Mean

    """

    ne = simData.shape[1]
    measurment = measurment.reshape(-1, 1)
    objReal = np.zeros((ne, 1))
    for j in range(ne):
        noww = simData[:, j].reshape(-1, 1)
        objReal[j] = (np.sum(((noww - measurment) ** 2))) ** (0.5) / (
            measurment.shape[0]
        )
    obj = np.mean(objReal)

    objStd = np.std(objReal)
    return obj, objStd, objReal


def pinvmatt(A, tol=0):
    """


    Parameters
    ----------
    A : Input Matrix to invert

    tol : Tolerance level : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    V : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    U : TYPE
        DESCRIPTION.

    """
    V, S1, U = cp.linalg.svd(A, full_matrices=0)

    # Calculate the default value for tolerance if no tolerance is specified
    if tol == 0:
        tol = cp.amax((A.size) * cp.spacing(cp.linalg.norm(S1, cp.inf)))

    r1 = sum(S1 > tol) + 1
    v = V[:, : r1 - 1]
    U1 = U.T
    u = U1[:, : r1 - 1]
    S11 = S1[: r1 - 1]
    s = S11[:]
    S = 1 / s[:]
    X = (u * S.T).dot(v.T)

    return (V, X, U)


def pinvmat(A, tol=0):
    """


    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    tol : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    X : TYPE
        DESCRIPTION.

    """
    V, S1, U = cp.linalg.svd(A, full_matrices=0)

    # Calculate the default value for tolerance if no tolerance is specified
    if tol == 0:
        tol = cp.amax((A.size) * cp.spacing(cp.linalg.norm(S1, cp.inf)))

    r1 = cp.sum(S1 > tol) + 1
    v = V[:, : r1 - 1]
    U1 = U.T
    u = U1[:, : r1 - 1]
    S11 = S1[: r1 - 1]
    s = S11[:]
    S = 1 / s[:]
    X = (u * S).dot(v.T)
    return X


class MinMaxScalerVectorized(object):
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, tensor):
        """Fit features

        Parameters
        ----------
        stacked_features : tuple, list
            List of stacked features.

        Returns
        -------
        tensor
            A tensor with scaled features using requested preprocessor.
        """

        tensor = torch.stack(tensor)

        # Feature range
        a, b = self.feature_range

        dist = tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
        tensor.mul_(b - a).add_(a)

        return tensor


def load_data_numpy_2(inn, out, ndata, batch_size):
    x_data = inn
    y_data = out
    print(f"xtrain_data: {x_data.shape}")
    print(f"ytrain_data: {y_data.shape}")
    data_tuple = (torch.FloatTensor(x_data), torch.FloatTensor(y_data))
    data_loader = DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def Getporosity_ensemble(ini_ensemble, machine_map, N_ens):
    ini_ensemblep = []
    for ja in range(N_ens):
        usek = np.reshape(ini_ensemble[:, ja], (-1, 1), "F")

        ini_ensemblep.append(usek)

    ini_ensemble = np.vstack(ini_ensemblep)
    ini_ensemblep = machine_map.predict(ini_ensemble)

    ini_ensemblee = np.split(ini_ensemblep, N_ens, axis=0)

    ini_ensemble = []
    for ky in range(N_ens):
        aa = ini_ensemblee[ky]
        aa = np.reshape(aa, (-1, 1), "F")
        ini_ensemble.append(aa)
    ini_ensemble = np.hstack(ini_ensemble)
    return ini_ensemble


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
    step2,
    pini_alt,
    INJ,
    PROD,
    opennI,
    opennP,
):
    paramss = ini
    Ne = 1  # paramss.shape[1]

    if nz == 1:
        ct = np.zeros((Ne, 4, nx, ny), dtype=np.float32)
        cP = np.zeros((Ne, 2 * steppi, nx, ny), dtype=np.float32)
    else:
        ct = np.zeros((Ne, 4, nx, ny, nz), dtype=np.float32)
        cP = np.zeros((Ne, 2 * steppi, nx, ny, nz), dtype=np.float32)

    Ainj = np.zeros((nx, ny, nz))
    Aprod = np.zeros((nx, ny, nz))

    n_inj = len(INJ)  # Number of injectors
    n_prod = len(PROD)  # Number of producers
    for kk in range(Ne):
        if nz == 1:
            ct1 = np.zeros((4, nx, ny), dtype=np.float32)
        else:
            ct1 = np.zeros((4, nx, ny, nz), dtype=np.float32)
        # print(str(kk + 1) + ' | ' +  str (Ne))
        a = np.reshape(paramss, (nx, ny, nz), "F")

        at1 = ini
        # at1 = rescale_linear_numpy_pytorch(at1, LUB, HUB,aay,bby)

        at2 = inip
        # at2 = rescale_linear(at2, mpor, hpor)

        at1 = np.reshape(at1, (nx, ny, nz), "F")
        at2 = np.reshape(at2, (nx, ny, nz), "F")

        if opennI == "all":
            for ay in range(n_inj):
                use = np.array(INJ[ay])
                for jj in range(nz):
                    xloc = int(use[0])
                    yloc = int(use[1])
                    Ainj[xloc, yloc, jj] = float(use[3])

        else:
            for ay in range(n_inj):
                use = np.array(INJ[ay])
                xloc = int(use[0])
                yloc = int(use[1])
                Ainj[xloc, yloc, -1] = float(use[3])

        if opennP == "all":
            for ay in range(n_prod):
                use = np.array(PROD[ay])
                for jj in range(nz):
                    xloc = int(use[0])
                    yloc = int(use[1])
                    Aprod[xloc, yloc, jj] = float(use[3])
        else:
            for ay in range(n_prod):
                use = np.array(PROD[ay])
                xloc = int(use[0])
                yloc = int(use[1])
                Aprod[xloc, yloc, -1] = float(use[3])

        quse1 = Ainj + Aprod
        quse = np.reshape(quse1, (-1, 1), "F")
        quse_water = np.reshape(Ainj, (-1, 1), "F")

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
            step2,
            pini_alt,
        )
        if nz == 1:
            ct1[0, :, :] = at1[:, :, 0]  # permeability
            ct1[1, :, :] = quse1[:, :, 0]  # /UIR # Overall f
            ct1[2, :, :] = Ainj[:, :, 0]  # /UIR# f for water injection
            ct1[3, :, :] = at2[:, :, 0]  # porosity
        else:
            ct1[0, :, :, :] = at1  # permeability
            ct1[1, :, :, :] = quse1  # /UIR # Overall f
            ct1[2, :, :, :] = Ainj  # /UIR# f for water injection
            ct1[3, :, :, :] = at2  # porosity

        if nz == 1:
            cP[kk, :, :, :] = A_out
            ct[kk, :, :, :] = ct1
        else:
            cP[kk, :, :, :, :] = A_out
            ct[kk, :, :, :, :] = ct1

        return ct, cP


def Remove_folder(N_ens, straa):
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)


def Get_Latent(ini_ensemble, N_ens, nx, ny, nz, High_K):
    X_unie = np.zeros((N_ens, nx, ny, nz))
    for i in range(N_ens):
        X_unie[i, :, :, :] = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")
    ax = X_unie / High_K
    ouut = np.zeros((20 * 20 * 4, Ne))
    decoded_imgs2 = load_model("PACKETS/encoder.h5").predict(ax)
    for i in range(N_ens):
        jj = decoded_imgs2[i]  # .reshape(20,20,4)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def Get_Latentp(ini_ensemble, N_ens, nx, ny, nz, High_P):
    X_unie = np.zeros((N_ens, nx, ny, nz))
    for i in range(N_ens):
        X_unie[i, :, :, :] = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")
    # ax=X_unie/High_P
    ax = X_unie / High_P
    ouut = np.zeros((20 * 20 * 4, Ne))
    decoded_imgs2 = load_model("PACKETS/encoderp.h5").predict(ax)
    for i in range(N_ens):
        jj = decoded_imgs2[i]  # .reshape(20,20,4)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def Recover_image(x, Ne, nx, ny, nz, High_K):
    X_unie = np.zeros((Ne, 20, 20, 4))
    for i in range(Ne):
        X_unie[i, :, :, :] = np.reshape(x[:, i], (20, 20, 4), "F")
    decoded_imgs2 = (load_model("PACKETS/decoder.h5").predict(X_unie)) * High_K
    # print(decoded_imgs2.shape)
    ouut = np.zeros((nx * ny * nz, Ne))
    for i in range(Ne):
        jj = decoded_imgs2[i]  # .reshape(nx,ny,nz)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def Recover_imagep(x, Ne, nx, ny, nz, High_P):
    X_unie = np.zeros((Ne, 20, 20, 4))
    for i in range(Ne):
        X_unie[i, :, :, :] = np.reshape(x[:, i], (20, 20, 4), "F")
    decoded_imgs2 = load_model("PACKETS/decoderp.h5").predict(X_unie)
    # print(decoded_imgs2.shape)
    ouut = np.zeros((nx * ny * nz, Ne))
    for i in range(Ne):
        jj = decoded_imgs2[i]  # .reshape(nx,ny,nz)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def use_denoising(ensemblee, nx, ny, nz, N_ens, High_K):
    X_unie = np.zeros((N_ens, nx, ny, nz))
    for i in range(N_ens):
        X_unie[i, :, :, :] = np.reshape(ensemblee[:, i], (nx, ny, nz), "F")
    ax = X_unie / High_K
    ouut = np.zeros((nx * ny * nz, Ne))
    decoded_imgs2 = (load_model("PACKETS/denosingautoencoder.h5").predict(ax)) * High_K
    for i in range(N_ens):
        jj = decoded_imgs2[i]  # .reshape(nx,ny,nz)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def use_denoisingp(ensemblee, nx, ny, nz, N_ens, High_P):
    # N_ens = Ne
    # ensemblee = ensemblep
    X_unie = np.zeros((N_ens, nx, ny, nz))
    for i in range(N_ens):
        X_unie[i, :, :, :] = np.reshape(ensemblee[:, i], (nx, ny, nz), "F")
    ax = X_unie
    # ax = High_P
    ouut = np.zeros((nx * ny * nz, Ne))
    decoded_imgs2 = load_model("PACKETS/denosingautoencoderp.h5").predict(ax)
    for i in range(N_ens):
        jj = decoded_imgs2[i]  # .reshape(nx,ny,nz)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def DenosingAutoencoder(nx, ny, nz, High_K, Low_K):
    """
    Trains  Denosing Autoencoder for the permeability field

    Parameters
    ----------
    nx : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.
    nz : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    filename = "PACKETS/Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble = mat["Z"]
    ini_ensemble = Trim_ensemble(ini_ensemble)

    clfye = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfye.fit(ini_ensemble))
    ini_ensemble = clfye.transform(ini_ensemble)

    X_unie = np.zeros((ini_ensemble.shape[1], nx, ny, nz))
    for i in range(ini_ensemble.shape[1]):
        X_unie[i, :, :, :] = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")

    # x_train=X_unie/500
    ax = X_unie / High_K

    x_train, x_test, y_train, y_test = train_test_split(
        ax, ax, test_size=0.01, random_state=42
    )
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_train.shape
    )
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = x_test + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_test.shape
    )
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

    # nx,ny,nz=40,40,3
    input_image = Input(shape=(nx, ny, nz))
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_image)
    x = MaxPooling2D((1, 1), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((1, 1), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((1, 1), padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)  # reduces it by this value

    encoder = Model(input_image, encoded)
    encoder.summary()

    decoder_input = Input(shape=(20, 20, 32))
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(decoder_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((1, 1))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((1, 1))(x)
    decoded = Conv2D(nz, (3, 3), activation="tanh", padding="same")(x)

    decoder = Model(decoder_input, decoded)
    decoder.summary()

    autoencoder_input = Input(shape=(nx, ny, nz))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100)
    mc = ModelCheckpoint(
        "PACKETS/denosingautoencoder.h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    autoencoder.fit(
        x_train_noisy,
        x_train,
        epochs=5000,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
        callbacks=[es, mc],
    )


def DenosingAutoencoderp(nx, ny, nz, N_ens, High_P, High_K, Low_K):
    """
    Trains  Denosing Autoencoder for porosity field

    Parameters
    ----------
    nx : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.
    nz : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    filename = "PACKETS/Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble = mat["Z"]
    ini_ensemble = Trim_ensemble(ini_ensemble)
    clfye = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfye.fit(ini_ensemble))
    ini_ensemble = clfye.transform(ini_ensemble)

    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ini_ensemble))
    ini_ensemble = clfye.transform(ini_ensemble)

    # ini_ensemble=Getporosity_ensemble(ini_ensemble,machine_map,N_ens)

    X_unie = np.zeros((ini_ensemble.shape[1], nx, ny, nz))
    for i in range(ini_ensemble.shape[1]):
        X_unie[i, :, :, :] = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")

    # x_train=X_unie/500
    ax = X_unie
    # ax = High_P

    x_train, x_test, y_train, y_test = train_test_split(
        ax, ax, test_size=0.01, random_state=42
    )
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_train.shape
    )
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = x_test + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=x_test.shape
    )
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

    # nx,ny,nz=40,40,3
    input_image = Input(shape=(nx, ny, nz))
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_image)
    x = MaxPooling2D((1, 1), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((1, 1), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((1, 1), padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)  # reduces it by this value

    encoder = Model(input_image, encoded)
    encoder.summary()

    decoder_input = Input(shape=(20, 20, 32))
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(decoder_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((1, 1))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((1, 1))(x)
    decoded = Conv2D(nz, (3, 3), activation="tanh", padding="same")(x)

    decoder = Model(decoder_input, decoded)
    decoder.summary()

    autoencoder_input = Input(shape=(nx, ny, nz))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100)
    mc = ModelCheckpoint(
        "PACKETS/denosingautoencoderp.h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    autoencoder.fit(
        x_train_noisy,
        x_train,
        epochs=5000,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
        callbacks=[es, mc],
    )


def Autoencoder2(nx, ny, nz, High_K, Low_K):
    """
    Trains  convolution Autoencoder for permeability field

    Parameters
    ----------
    nx : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.
    nz : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    filename = "PACKETS/Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble = mat["Z"]
    ini_ensemble = Trim_ensemble(ini_ensemble)
    clfye = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfye.fit(ini_ensemble))
    ini_ensemble = clfye.transform(ini_ensemble)
    X_unie = np.zeros((ini_ensemble.shape[1], nx, ny, nz))
    for i in range(ini_ensemble.shape[1]):
        X_unie[i, :, :, :] = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")

    ax = X_unie / High_K

    x_train, x_test, y_train, y_test = train_test_split(
        ax, ax, test_size=0.1, random_state=42
    )

    # nx,ny,nz=40,40,3
    input_image = Input(shape=(nx, ny, nz))
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(input_image)
    x = MaxPooling2D((1, 1), padding="same")(x)
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((1, 1), padding="same")(x)
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((1, 1), padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)  # reduces it by this value

    encoder = Model(input_image, encoded)
    encoder.summary()

    decoder_input = Input(shape=(20, 20, 4))
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(decoder_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((1, 1))(x)
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((1, 1))(x)
    decoded = Conv2D(nz, (3, 3), activation="tanh", padding="same")(x)

    decoder = Model(decoder_input, decoded)
    decoder.summary()

    autoencoder_input = Input(shape=(nx, ny, nz))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100)
    mc = ModelCheckpoint(
        "PACKETS/autoencoder.h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    autoencoder.fit(
        x_train,
        x_train,
        epochs=5000,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[es, mc],
    )

    # os.chdir(dirr)
    encoder.save("PACKETS/encoder.h5")
    decoder.save("PACKETS/decoder.h5")
    os.chdir(oldfolder)


def Autoencoder2p(nx, ny, nz, High_P, High_K, Low_K):
    """
    Trains  convolution Autoencoder for porosity field

    Parameters
    ----------
    nx : TYPE
        DESCRIPTION.
    ny : TYPE
        DESCRIPTION.
    nz : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    filename = "PACKETS/Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble = mat["Z"]
    clfye = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfye.fit(ini_ensemble))
    ini_ensemble = clfye.transform(ini_ensemble)

    clfyee = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfyee.fit(ini_ensemble))
    ini_ensemble = clfyee.transform(ini_ensemble)
    # ini_ensemble=Getporosity_ensemble(ini_ensemble,machine_map,N_ens)
    X_unie = np.zeros((ini_ensemble.shape[1], nx, ny, nz))
    for i in range(ini_ensemble.shape[1]):
        X_unie[i, :, :, :] = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")

    # ax=X_unie/High_P
    ax = X_unie

    x_train, x_test, y_train, y_test = train_test_split(
        ax, ax, test_size=0.1, random_state=42
    )

    # nx,ny,nz=40,40,3
    input_image = Input(shape=(nx, ny, nz))
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(input_image)
    x = MaxPooling2D((1, 1), padding="same")(x)
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((1, 1), padding="same")(x)
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((1, 1), padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)  # reduces it by this value

    encoder = Model(input_image, encoded)
    encoder.summary()

    decoder_input = Input(shape=(20, 20, 4))
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(decoder_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((1, 1))(x)
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((1, 1))(x)
    decoded = Conv2D(nz, (3, 3), activation="tanh", padding="same")(x)

    decoder = Model(decoder_input, decoded)
    decoder.summary()

    autoencoder_input = Input(shape=(nx, ny, nz))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(autoencoder_input, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100)
    mc = ModelCheckpoint(
        "PACKETS/autoencoderp.h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    autoencoder.fit(
        x_train,
        x_train,
        epochs=5000,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[es, mc],
    )

    # os.chdir(dirr)
    encoder.save("PACKETS/encoderp.h5")
    decoder.save("PACKETS/decoderp.h5")
    os.chdir(oldfolder)


def RSM_extract(Ne, folder, oldfolder, scalei, scalei2):
    simout = []
    simplot = []
    for i in range(Ne):
        os.chdir(folder + str(i))
        True_measurement = pd.read_csv("RSM_FVM.csv")
        True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
        True_mat = True_measurement
        Presz = True_mat[:, 1:5] / scalei3
        Oilz = True_mat[:, 5:9] / scalei
        Watsz = True_mat[:, 9:13] / scalei2
        wctz = True_mat[:, 13:]
        True_data = np.hstack([Presz, Oilz, Watsz, wctz])
        True_data = np.reshape(True_data, (-1, 1), "F")
        os.chdir(oldfolder)
        simout.append(True_data)
        simplot.append(True_measurement)
    simout = np.hstack(simout)
    return simout, simplot


def RSM_extract1(Ne, folder, oldfolder, scalei, scalei2):
    simout = []
    simplot = []
    for i in range(Ne):
        os.chdir(folder)
        True_measurement = pd.read_csv("RSM_FVM.csv")
        True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
        True_mat = True_measurement
        Presz = True_mat[:, 1:5] / scalei3
        Oilz = True_mat[:, 5:9] / scalei
        Watsz = True_mat[:, 9:13] / scalei2
        wctz = True_mat[:, 13:]
        True_data = np.hstack([Presz, Oilz, Watsz, wctz])
        True_data = np.reshape(True_data, (-1, 1), "F")
        os.chdir(oldfolder)
        simout.append(True_data)
        simplot.append(True_measurement)
    simout = np.hstack(simout)
    return simout, simplot


def Select_TI(oldfolder, ressimmaster, N_small, nx, ny, nz, True_data, Low_K, High_K):
    valueTI = np.zeros((5, 1))

    N_enss = N_small
    os.chdir(ressimmaster)
    print("")
    print("--------------------------------------------------------------")
    print("TI = 1")
    k = np.genfromtxt("iglesias2.out", skip_header=0, dtype="float")
    os.chdir(oldfolder)
    k = k.reshape(-1, 1)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfy.fit(k))
    k = clfy.transform(k)
    k = np.reshape(k, (nx, ny, nz), "F")
    k = k[:, :, 0]
    kjenn = k
    TI1 = kjenn
    see = intial_ensemble(nx, ny, nz, N_small, kjenn)

    ini_ensemble1 = see
    clfy = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfy.fit(ini_ensemble1))
    ensemblep = clfy.transform(ini_ensemble1)

    for kk in range(N_small):
        path_out = "TI" + str(kk)
        os.makedirs(path_out, exist_ok=True)
    start_time_peaceman = time.time()
    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ini_ensemble1[:, kk], ensemblep[:, kk], "TI" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # free up memory
    # executor.shutdown(wait=True)
    elapsed_time_secs = time.time() - start_time_peaceman
    msg = (
        "Simulation of "
        + str(Ne)
        + " ensemble memebers took "
        + str(timedelta(seconds=round(elapsed_time_secs)))
        + "seconds"
    )
    print(msg)

    simDatafinal, predMatrix = RSM_extract(Ne, "TI", oldfolder, scalei, scalei2)

    Remove_folder(N_small, "TI")

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    cc = cc.reshape(-1, 1)
    clem1 = np.mean(cc, axis=0)  # mean best
    cle = np.argmin(cc)
    clem1a = cc[cle, :]  # best
    valueTI[0, :] = (clem1 + clem1a) / 2
    print("Error = " + str((clem1 + clem1a) / 2))
    yett = 1

    print("")
    print("--------------------------------------------------------------")
    print("TI = 2")
    os.chdir(ressimmaster)
    k = np.genfromtxt("TI_3.out", skip_header=3, dtype="float")
    os.chdir(oldfolder)
    k = k.reshape(-1, 1)
    # os.chdir(oldfolder)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfy.fit(k))
    k = clfy.transform(k)
    # k=k.T
    k = np.reshape(k, (768, 243), "F")
    kjenn = k
    # shape1 = (768, 243)
    TI2 = kjenn
    # kti2=
    see = intial_ensemble(nx, ny, nz, N_small, kjenn)

    ini_ensemble2 = see

    clfy = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfy.fit(ini_ensemble2))
    ensemblep = clfy.transform(ini_ensemble2)

    for kk in range(N_small):
        path_out = "TI" + str(kk)
        os.makedirs(path_out, exist_ok=True)
    start_time_peaceman = time.time()
    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ini_ensemble2[:, kk], ensemblep[:, kk], "TI" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # free up memory
    # executor.shutdown(wait=True)
    elapsed_time_secs = time.time() - start_time_peaceman
    msg = (
        "Simulation of "
        + str(Ne)
        + " ensemble memebers took "
        + str(timedelta(seconds=round(elapsed_time_secs)))
        + "seconds"
    )
    print(msg)

    simDatafinal, predMatrix = RSM_extract(Ne, "TI", oldfolder, scalei, scalei2)

    Remove_folder(N_small, "TI")

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    cc = cc.reshape(-1, 1)
    clem2 = np.mean(cc, axis=0)
    cle = np.argmin(cc)
    clem2a = cc[cle, :]  # best
    valueTI[1, :] = (clem2 + clem2a) / 2
    print("Error = " + str((clem2 + clem2a) / 2))
    yett = 2

    print("")
    print("--------------------------------------------------------------")
    print("TI = 3 ")
    os.chdir(ressimmaster)
    k = np.genfromtxt("TI_2.out", skip_header=3, dtype="float")
    k = k.reshape(-1, 1)
    os.chdir(oldfolder)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfy.fit(k))
    k = clfy.transform(k)
    k = np.reshape(k, (250, 250), "F")
    kjenn = k.T
    TI3 = kjenn
    see = intial_ensemble(nx, ny, nz, N_small, kjenn)

    ini_ensemble3 = see

    clfy = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfy.fit(ini_ensemble3))
    ensemblep = clfy.transform(ini_ensemble3)

    for kk in range(N_small):
        path_out = "TI" + str(kk)
        os.makedirs(path_out, exist_ok=True)
    start_time_peaceman = time.time()
    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ini_ensemble3[:, kk], ensemblep[:, kk], "TI" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    elapsed_time_secs = time.time() - start_time_peaceman
    msg = (
        "Simulation of "
        + str(Ne)
        + " ensemble memebers took "
        + str(timedelta(seconds=round(elapsed_time_secs)))
        + "seconds"
    )
    print(msg)

    simDatafinal, predMatrix = RSM_extract(Ne, "TI", oldfolder, scalei, scalei2)

    Remove_folder(N_small, "TI")
    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    cc = cc.reshape(-1, 1)
    clem3 = np.mean(cc, axis=0)
    cle = np.argmin(cc)
    clem3a = cc[cle, :]  # best
    valueTI[2, :] = (clem3 + clem3a) / 2
    print("Error = " + str((clem3 + clem3a) / 2))
    yett = 3

    print("")
    print("--------------------------------------------------------------")
    print("TI = 4")
    os.chdir(ressimmaster)
    k = np.loadtxt("TI_4.out", skiprows=4, dtype="float")
    kuse = k[:, 1]
    k = kuse.reshape(-1, 1)
    os.chdir(oldfolder)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfy.fit(k))
    k = clfy.transform(k)
    # k=k.T
    k = np.reshape(k, (100, 100, 2), "F")
    kjenn = k[:, :, 0]
    # shape1 = (100, 100)
    TI4 = kjenn
    see = intial_ensemble(nx, ny, nz, N_small, kjenn)

    ini_ensemble4 = see

    clfy = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfy.fit(ini_ensemble4))
    ensemblep = clfy.transform(ini_ensemble4)

    for kk in range(N_small):
        path_out = "TI" + str(kk)
        os.makedirs(path_out, exist_ok=True)
    start_time_peaceman = time.time()
    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ini_ensemble4[:, kk], ensemblep[:, kk], "TI" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    elapsed_time_secs = time.time() - start_time_peaceman
    msg = (
        "Simulation of "
        + str(Ne)
        + " ensemble memebers took "
        + str(timedelta(seconds=round(elapsed_time_secs)))
        + "seconds"
    )
    print(msg)

    simDatafinal, predMatrix = RSM_extract(Ne, "TI", oldfolder, scalei, scalei2)

    Remove_folder(N_small, "TI")
    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    cc = cc.reshape(-1, 1)
    clem4 = np.mean(cc, axis=0)
    cle = np.argmin(cc)
    clem4a = cc[cle, :]  # best
    valueTI[3, :] = (clem4 + clem4a) / 2
    print("Error = " + str((clem4 + clem4a) / 2))
    yett = 4

    print("")
    print("--------------------------------------------------------------")
    print("TI = 5")
    os.chdir(ressimmaster)
    k = np.genfromtxt("TI_1.out", skip_header=3, dtype="float")
    k = k.reshape(-1, 1)
    os.chdir(oldfolder)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfy.fit(k))
    k = clfy.transform(k)
    # k=k.T
    k = np.reshape(k, (400, 400), "F")
    kjenn = k
    # shape1 = (260, 300)
    TI5 = kjenn
    see = intial_ensemble(nx, ny, nz, N_small, kjenn)
    # kout = kjenn

    ini_ensemble5 = see

    clfy = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfy.fit(ini_ensemble5))
    ensemblep = clfy.transform(ini_ensemble5)
    # ensemblep=Getporosity_ensemble(ini_ensemble5,machine_map,N_enss)

    for kk in range(N_small):
        path_out = "TI" + str(kk)
        os.makedirs(path_out, exist_ok=True)
    start_time_peaceman = time.time()
    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ini_ensemble5[:, kk], ensemblep[:, kk], "TI" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    elapsed_time_secs = time.time() - start_time_peaceman
    msg = (
        "Simulation of "
        + str(Ne)
        + " ensemble memebers took "
        + str(timedelta(seconds=round(elapsed_time_secs)))
        + "seconds"
    )
    print(msg)

    simDatafinal, predMatrix = RSM_extract(Ne, "TI", oldfolder, scalei, scalei2)

    Remove_folder(N_small, "TI")
    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    cc = cc.reshape(-1, 1)
    clem5 = np.mean(cc, axis=0)
    cle = np.argmin(cc)
    clem5a = cc[cle, :]  # best
    valueTI[4, :] = (clem5 + clem5a) / 2
    print("Error = " + str((clem5 + clem5a) / 2))
    yett = 5

    TIs = {"TI1": TI1, "TI2": TI2, "TI3": TI3, "TI4": TI4, "TI5": TI5}

    clem = np.argmin(valueTI)
    valueM = valueTI[clem, :]

    print("")
    print("--------------------------------------------------------------")
    print(" Gaussian Random Field Simulation")
    ini_ensembleG = initial_ensemble_gaussian(nx, ny, nz, N_enss, Low_K, High_K)

    clfy = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfy.fit(ini_ensembleG))
    EnsemblepG = clfy.transform(ini_ensembleG)

    for kk in range(N_enss):
        path_out = "TI" + str(kk)
        os.makedirs(path_out, exist_ok=True)
    start_time_peaceman = time.time()
    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ini_ensembleG[:, kk], EnsemblepG[:, kk], "TI" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    elapsed_time_secs = time.time() - start_time_peaceman
    msg = (
        "Simulation of "
        + str(Ne)
        + " ensemble memebers took "
        + str(timedelta(seconds=round(elapsed_time_secs)))
        + "seconds"
    )
    print(msg)

    simDatafinal, predMatrix = RSM_extract(Ne, "TI", oldfolder, scalei, scalei2)

    Remove_folder(N_small, "TI")
    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    cc = cc.reshape(-1, 1)
    cle = np.argmin(cc)
    clem1a = cc[cle, :]  # best
    valueG = (np.mean(cc, axis=0) + clem1a) / 2
    print("Error = " + str(valueG))

    plt.figure(figsize=(16, 16))
    plt.subplot(2, 3, 1)
    plt.imshow(TI1)
    # plt.gray()
    plt.title("TI1 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)

    plt.subplot(2, 3, 2)
    plt.imshow(TI2)
    # plt.gray()
    plt.title("TI2 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)

    plt.subplot(2, 3, 3)
    plt.imshow(TI3)
    # plt.gray()
    plt.title("TI3 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)

    plt.subplot(2, 3, 4)
    plt.imshow(TI4)
    # plt.gray()
    plt.title("TI4 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)

    plt.subplot(2, 3, 5)
    plt.imshow(TI5)
    # plt.gray()
    plt.title("TI5 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)

    os.chdir(ressimmaster)
    plt.savefig("TISS.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()
    clemm = clem + 1
    if valueG < valueM:
        print("Gaussian distribution better suited")
        mum = 2
        permxx = 0
        yett = 6
    else:
        if (valueM < valueG) and (clemm == 1):
            print("MPS distribution better suited")
            mum = 1
            permxx = TI1
            yett = 1
        if (valueM < valueG) and (clemm == 2):
            mum = 1
            permxx = TI2
            yett = 2
        if (valueM < valueG) and (clemm == 3):
            mum = 1
            permxx = TI3
            yett = 3
        if (valueM < valueG) and (clemm == 4):
            mum = 1
            permxx = TI4
            yett = 4
        if (valueM < valueG) and (clemm == 5):
            mum = 1
            permxx = TI5
            yett = 5
    return permxx, mum, TIs, yett


def parad2_TI(X_train, y_traind, namezz):
    namezz = "PACKETS/" + namezz + ".h5"
    # np.random.seed(7)
    modelDNN = Sequential()
    modelDNN.add(Dense(200, activation="relu", input_dim=X_train.shape[1]))
    modelDNN.add(Dense(units=820, activation="relu"))
    modelDNN.add(Dense(units=220, activation="relu"))
    modelDNN.add(Dense(units=21, activation="relu"))
    modelDNN.add(Dense(units=1))
    modelDNN.compile(loss="mean_squared_error", optimizer="Adam", metrics=["mse"])
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
    mc = ModelCheckpoint(
        namezz, monitor="val_loss", mode="min", verbose=1, save_best_only=True
    )
    a0 = X_train
    a0 = np.reshape(a0, (-1, X_train.shape[1]), "F")

    b0 = y_traind
    b0 = np.reshape(b0, (-1, y_traind.shape[1]), "F")
    gff = len(a0) // 100
    if gff < 1:
        gff = 1
    modelDNN.fit(
        a0, b0, validation_split=0.1, batch_size=gff, epochs=500, callbacks=[es, mc]
    )


def Select_TI_2(
    oldfolder, ressimmaster, nx, ny, nz, perm_high, perm_low, poro_high, poro_low, i
):
    os.chdir(ressimmaster)
    print("")
    print("--------------------------------------------------------------")
    if i == 1:
        print("TI = 1")
        k = np.genfromtxt("iglesias2.out", skip_header=0, dtype="float")
        k = k.reshape(-1, 1)
        os.chdir(oldfolder)
        # k=k.T
        clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
        (clfy.fit(k))
        k = clfy.transform(k)
        k = np.reshape(k, (nx, ny, nz), "F")
        k = k[:, :, 0]
        # k=k.T
        # k=k.reshape(ny,nx)
        kjenn = k
        TrueK = np.reshape(kjenn, (-1, 1), "F")
        clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
        (clfy.fit(TrueK))
        y_train = clfy.transform(TrueK)
        parad2_TI(TrueK, y_train, "MAP1")

    elif i == 2:
        print("")
        print("--------------------------------------------------------------")
        print("TI = 2")
        os.chdir(ressimmaster)
        k = np.genfromtxt("TI_3.out", skip_header=3, dtype="float")
        os.chdir(oldfolder)
        k = k.reshape(-1, 1)
        # os.chdir(oldfolder)
        clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
        (clfy.fit(k))
        k = clfy.transform(k)
        # k=k.T
        k = np.reshape(k, (768, 243), "F")
        kjenn = k
        TrueK = np.reshape(kjenn, (-1, 1), "F")
        clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
        (clfy.fit(TrueK))
        y_train = clfy.transform(TrueK)
        parad2_TI(TrueK, y_train, "MAP2")

    elif i == 3:
        print("")
        print("--------------------------------------------------------------")
        print("TI = 3 ")
        os.chdir(ressimmaster)
        k = np.genfromtxt("TI_2.out", skip_header=3, dtype="float")
        k = k.reshape(-1, 1)
        os.chdir(oldfolder)
        clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
        (clfy.fit(k))
        k = clfy.transform(k)
        # k=k.T
        k = np.reshape(k, (250, 250), "F")
        kjenn = k.T
        TrueK = np.reshape(kjenn, (-1, 1), "F")
        clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
        (clfy.fit(TrueK))
        y_train = clfy.transform(TrueK)
        parad2_TI(TrueK, y_train, "MAP3")

    elif i == 4:
        print("")
        print("--------------------------------------------------------------")
        print("TI = 4")
        os.chdir(ressimmaster)
        k = np.loadtxt("TI_4.out", skiprows=4, dtype="float")
        kuse = k[:, 1]
        k = kuse.reshape(-1, 1)
        os.chdir(oldfolder)
        clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
        (clfy.fit(k))
        k = clfy.transform(k)
        # k=k.T
        k = np.reshape(k, (100, 100, 2), "F")
        kjenn = k[:, :, 0]
        TrueK = np.reshape(kjenn, (-1, 1), "F")
        clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
        (clfy.fit(TrueK))
        y_train = clfy.transform(TrueK)
        parad2_TI(TrueK, y_train, "MAP4")

    elif i == 5:
        print("")
        print("--------------------------------------------------------------")
        print("TI = 5")
        os.chdir(ressimmaster)
        k = np.genfromtxt("TI_1.out", skip_header=3, dtype="float")
        k = k.reshape(-1, 1)
        os.chdir(oldfolder)
        clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
        (clfy.fit(k))
        k = clfy.transform(k)
        # k=k.T
        k = np.reshape(k, (400, 400), "F")
        kjenn = k
        TrueK = np.reshape(kjenn, (-1, 1), "F")
        clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
        (clfy.fit(TrueK))
        y_train = clfy.transform(TrueK)
        parad2_TI(TrueK, y_train, "MAP5")
    else:
        print(" Gaussian Random Field Simulation")

        fout = []
        shape = (nx, ny)
        for j in range(nz):
            field = generate_field(distrib, Pkgen(3), shape)
            field = imresize(field, output_shape=shape)
            foo = np.reshape(field, (-1, 1), "F")
            fout.append(foo)
        fout = np.vstack(fout)
        clfy = MinMaxScaler(feature_range=(perm_low, perm_high))
        (clfy.fit(fout))
        k = clfy.transform(fout)
        kjenn = k
        TrueK = np.reshape(kjenn, (-1, 1), "F")
        clfy = MinMaxScaler(feature_range=(poro_low, poro_high))
        (clfy.fit(TrueK))
        y_train = clfy.transform(TrueK)
        parad2_TI(TrueK, y_train, "MAP6")
    return kjenn


def De_correlate_ensemble(nx, ny, nz, Ne, High_K, Low_K):
    filename = "PACKETS/Ganensemble.mat"  # Ensemble generated offline
    mat = sio.loadmat(filename)
    ini_ensemblef = mat["Z"]
    ini_ensemblef = cp.asarray(ini_ensemblef)

    beta = int(cp.ceil(int(ini_ensemblef.shape[0] / Ne)))

    V, S1, U = cp.linalg.svd(ini_ensemblef, full_matrices=1)
    v = V[:, :Ne]
    U1 = U.T
    u = U1[:, :Ne]
    S11 = S1[:Ne]
    s = S11[:]
    S = (1 / ((beta) ** (0.5))) * s
    # S=s
    X = (v * S).dot(u.T)
    if Yet == 0:
        X = cp.asnumpy(X)
        ini_ensemblef = cp.asnumpy(ini_ensemblef)
    else:
        pass
    X[X <= Low_K] = Low_K
    X[X >= High_K] = High_K
    return X[:, :Ne]


def Remove_folder(N_ens, straa):
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)  # remove everything


def whiten(X, method="zca"):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ["zca", "pca", "cholesky"]:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == "zca":
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method == "pca":
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == "cholesky":
            W = np.linalg.cholesky(
                np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))
            ).T
    elif method in ["zca_cor", "pca_cor"]:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == "zca_cor":
            W = np.dot(
                np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)),
                np.linalg.inv(V_sqrt),
            )
        elif method == "pca_cor":
            W = np.dot(
                np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt)
            )
    else:
        raise Exception("Whitening method not found.")

    return np.dot(X_centered, W.T)


def Plot_Histogram_now(N, E11, E12, mean_cost, best_cost):
    mean_cost = np.vstack(mean_cost).reshape(-1, 1)
    best_cost = np.vstack(best_cost).reshape(-1, 1)
    reali = np.arange(1, N + 1)
    timezz = np.arange(1, mean_cost.shape[0] + 1)
    plttotalerror = np.reshape(E11, (N))  # Initial
    plttotalerror2 = np.reshape(E12, (N))  # Final
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.bar(reali, plttotalerror, color="c")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.ylim(ymin=0)
    plt.title("Initial Cost function")

    plt.scatter(reali, plttotalerror, s=1, color="k")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.xlim([1, (N - 1)])

    plt.subplot(2, 2, 2)
    plt.bar(reali, plttotalerror2, color="c")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.ylim(ymin=0)
    plt.ylim(ymax=max(plttotalerror))
    plt.title("Final Cost function")

    plt.scatter(reali, plttotalerror2, s=1, color="k")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.xlim([1, (N - 1)])

    plt.subplot(2, 2, 3)
    plt.plot(timezz, mean_cost, color="green", lw="2", label="mean_model_cost")
    plt.plot(timezz, best_cost, color="blue", lw="2", label="best_model_cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function Value")
    # plt.ylim((0,25000))
    plt.title("Cost Evolution")
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    os.chdir("HM_RESULTS")
    plt.savefig("Cost_Function.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()


def pytorch_to_inverse(Ne, val, nx, ny, nz):
    X_unie = np.zeros((nz * nx * ny, Ne))
    for i in range(Ne):
        tempp = val[i, :, :, :]
        aa = np.zeros((nx, ny, nz))
        for kk in range(nz):
            newy = imresize(tempp[kk, :, :], output_shape=(nx, ny))
            aa[:, :, kk] = newy
        X_unie[:, i] = np.reshape(aa, (-1,), "F")
    return X_unie


def REKI_ASSIMILATION_NORMAL_SCORE(
    sgsim,
    ensemblepi,
    simDatafinal,
    alpha,
    True_data,
    Ne,
    pertubations,
    Yet,
    nx,
    ny,
    nz,
    High_K,
    Low_K,
    High_P,
    Low_P,
    CDd,
):
    sizeclem = cp.asarray(nx * ny * nz)

    quantileperm = QuantileTransformer(
        n_quantiles=nx * ny * nz, output_distribution="normal", random_state=0
    )
    quantileporo = QuantileTransformer(
        n_quantiles=nx * ny * nz, output_distribution="normal", random_state=0
    )

    # quantiledata = QuantileTransformer(n_quantiles=True_data.shape[0],\
    #                                    output_distribution='normal',random_state=0)

    quantileperm.fit(sgsim)
    quantileporo.fit(ensemblepi)

    sgsimm = quantileperm.transform(sgsim)

    # (quantileporo.fit(ensemblep))
    ensemblepp = quantileporo.transform(ensemblepi)

    overall = cp.vstack([cp.asarray(sgsimm), cp.asarray(ensemblepp)])

    Y = overall

    Sim1 = cp.asarray(simDatafinal)

    if weighting == 1:
        weights = Get_weighting(simDatafinal, True_dataa)
        weight1 = cp.asarray(weights)

        Sp = cp.zeros((Sim1.shape[0], Ne))
        yp = cp.zeros((Y.shape[0], Y.shape[1]))

        for jc in range(Ne):
            Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
            yp[:, jc] = (overall[:, jc]) * weight1[jc]

        M = cp.mean(Sp, axis=1)

        M2 = cp.mean(yp, axis=1)
    if weighting == 2:
        weights = Get_weighting(simDatafinal, True_dataa)
        weight1 = cp.asarray(weights)

        Sp = cp.zeros((Sim1.shape[0], Ne))
        yp = cp.zeros((Y.shape[0], Y.shape[1]))

        for jc in range(Ne):
            Sp[:, jc] = Sim1[:, jc]
            yp[:, jc] = (overall[:, jc]) * weight1[jc]

        M = cp.mean(Sp, axis=1)

        M2 = cp.mean(yp, axis=1)
    if weighting == 3:
        M = cp.mean(Sim1, axis=1)

        M2 = cp.mean(overall, axis=1)

    S = cp.zeros((Sim1.shape[0], Ne))
    yprime = cp.zeros((Y.shape[0], Y.shape[1]))

    for jc in range(Ne):
        S[:, jc] = Sim1[:, jc] - M
        yprime[:, jc] = overall[:, jc] - M2
    Cydd = (yprime) / cp.sqrt(Ne - 1)
    Cdd = (S) / cp.sqrt(Ne - 1)

    GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
    inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
    Cdd = GDT.T @ GDT
    Cyd = Cydd @ GDT
    Usig, Sig, Vsig = cp.linalg.svd(
        (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
    )
    Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
    valuesig = Bsig[-1]  # last element
    valuesig = valuesig * 0.9999
    indices = (Bsig >= valuesig).ravel().nonzero()
    toluse = Sig[indices]
    tol = toluse[0]

    (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)
    innovation = (
        cp.tile(cp.asarray(True_data), Ne)
        + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
    ) - Sim1
    # if Yet==0:
    #     quantiledata.fit(cp.asnumpy(innovation))
    # else:
    #     quantiledata.fit((innovation))
    # innovation=cp.asarray(quantiledata.transform(cp.asnumpy(innovation)))
    update_term = (Cyd @ (X)) @ (inv_CDd) @ innovation

    Ynew = Y + update_term

    if Yet == 0:
        updated_ensemble = cp.asnumpy(Ynew[:sizeclem, :])
        updated_ensemblep = cp.asnumpy(Ynew[sizeclem : 2 * sizeclem, :])

    else:
        updated_ensemble = Ynew[:sizeclem, :]
        updated_ensemblep = Ynew[sizeclem : 2 * sizeclem, :]

    updated_yesK = quantileperm.inverse_transform(updated_ensemble)
    updated_yesP = quantileporo.inverse_transform(updated_ensemblep)

    # Honour well pixel values
    ensemblee, ensembleep = honour2(
        updated_yesP, updated_yesK, nx, ny, nz, Ne, High_K, Low_K, High_P, Low_P
    )

    return ensemblee, ensembleep


def Remove_True(enss):
    # enss is the very large ensemble
    # Used the last one as true model
    return np.reshape(enss[:, locc - 1], (-1, 1), "F")


def Trim_ensemble(enss):
    return np.delete(enss, locc - 1, 1)  # delete last column


class WGAN:
    def __init__(self, latent_dim, nx, ny, nz):
        self.img_rows = nx
        self.img_cols = ny
        self.channels = nz
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(
            loss=self.wasserstein_loss, optimizer=optimizer, metrics=["accuracy"]
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(
            loss=self.wasserstein_loss, optimizer=optimizer, metrics=["accuracy"]
        )

    def wasserstein_loss(self, y_true, y_pred):
        return tensorflow.keras.backend.mean(y_true * y_pred)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(128 * 20 * 20, activation="relu", input_dim=self.latent_dim))
        model.add(tensorflow.keras.layers.Reshape((20, 20, 128)))
        model.add(UpSampling2D(size=(2, 2)))
        # model.add(Conv2D(128, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(UpSampling2D(size=(1, 1)))
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(tensorflow.keras.layers.Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(tensorflow.keras.layers.Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):
        model = Sequential()

        model.add(
            Conv2D(
                16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(tensorflow.keras.layers.Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(tensorflow.keras.layers.Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(tensorflow.keras.layers.Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(tensorflow.keras.layers.Dropout(0.25))
        model.add(tensorflow.keras.layers.Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, dataa, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        X_train = dataa
        # Rescale -1 to 1
        X_train = (
            X_train / (High_K / 2)
        ) - 1  # 127.5 - 1.# (X_train.astype(np.float32) - 127.5) / 127.5

        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [
                        np.clip(w, -self.clip_value, self.clip_value) for w in weights
                    ]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print(
                "%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0])
            )

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = ((self.generator.predict(noise)) + 1) * (High_K / 2)
        # gen_imgs = self.generator.predict(noise)
        # gen_imgs = 0.5 * gen_imgs + 0.5
        plt.figure(figsize=(20, 20))
        XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
        for i in range(3):
            jj = gen_imgs[i, :, :, :]
            ii = i + 1
            # aa=i

            iiuse = (3 * i) + 0 + 1
            plt.subplot(r, c, iiuse)
            # permmean=(np.reshape(jj,(nx,ny,nz),'F'))
            permmean = jj
            plt.pcolormesh(XX.T, YY.T, permmean[:, :, 0], cmap="jet")
            Tittle = "Synthetic_" + str(ii)
            plt.title(Tittle, fontsize=9)
            plt.ylabel("Y", fontsize=9)
            plt.xlabel("X", fontsize=9)
            plt.axis([0, (nx - 1), 0, (ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" K (mD)", fontsize=9)

        plt.savefig("/images/%d_perm.png" % epoch)
        plt.close()
        plt.clf()

    def save_model(self):
        def save(model, model_name):
            name = "PACKETS/" + model_name + ".h5"
            model.save(name)

        save(self.generator, "generator")


class WGANP:
    def __init__(self, latent_dim, nx, ny, nz):
        self.img_rows = nx
        self.img_cols = ny
        self.channels = nz
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(
            loss=self.wasserstein_loss, optimizer=optimizer, metrics=["accuracy"]
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(
            loss=self.wasserstein_loss, optimizer=optimizer, metrics=["accuracy"]
        )

    def wasserstein_loss(self, y_true, y_pred):
        return tensorflow.keras.backend.mean(y_true * y_pred)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(128 * 20 * 20, activation="relu", input_dim=self.latent_dim))
        model.add(tensorflow.keras.layers.Reshape((20, 20, 128)))
        model.add(UpSampling2D(size=(2, 2)))
        # model.add(Conv2D(128, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(UpSampling2D(size=(1, 1)))
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(tensorflow.keras.layers.Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(tensorflow.keras.layers.Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):
        model = Sequential()

        model.add(
            Conv2D(
                16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(tensorflow.keras.layers.Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(tensorflow.keras.layers.Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(tensorflow.keras.layers.Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(tensorflow.keras.layers.Dropout(0.25))
        model.add(tensorflow.keras.layers.Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, dataa, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        X_train = dataa
        # Rescale -1 to 1
        X_train = (
            X_train / (High_P / 2)
        ) - 1  # 127.5 - 1.# (X_train.astype(np.float32) - 127.5) / 127.5

        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [
                        np.clip(w, -self.clip_value, self.clip_value) for w in weights
                    ]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print(
                "%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0])
            )

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 3, 3
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = ((self.generator.predict(noise)) + 1) * (High_P / 2)
        # gen_imgs = self.generator.predict(noise)
        # gen_imgs = 0.5 * gen_imgs + 0.5
        plt.figure(figsize=(20, 20))
        XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
        for i in range(3):
            jj = gen_imgs[i, :, :, :]
            ii = i + 1
            # aa=i
            # for j in range(3):
            iiuse = (3 * i) + 0 + 1
            plt.subplot(r, c, iiuse)
            # permmean=(np.reshape(jj,(nx,ny,nz),'F'))
            permmean = jj
            plt.pcolormesh(XX.T, YY.T, permmean[:, :, 0], cmap="jet")
            Tittle = "Synthetic_" + str(ii)
            plt.title(Tittle, fontsize=9)
            plt.ylabel("Y", fontsize=9)
            plt.xlabel("X", fontsize=9)
            plt.axis([0, (nx - 1), 0, (ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" K (mD)", fontsize=9)

        plt.savefig("/images/%d_porosity.png" % epoch)
        plt.close()
        plt.clf()

    def save_model(self):
        def save(model, model_name):
            name = "PACKETS/" + model_name + ".h5"
            model.save(name)

        save(self.generator, "generatorp")


def WGAN_LEARNING(High_K, Low_K, High_P, Low_P, nx, ny, nz):
    filename = "PACKETS/Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble = mat["Z"]

    clfye = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfye.fit(ini_ensemble))
    ini_ensemble = clfye.transform(ini_ensemble)

    N_enss = ini_ensemble.shape[1]
    dataas = ini_ensemble

    N_enss = ini_ensemble.shape[1]
    dataasp = ini_ensemble
    scaler1ap = MinMaxScaler(feature_range=(Low_P, High_P))
    (scaler1ap.fit(dataasp))
    dataasp = scaler1ap.transform(dataasp)

    X_unie = np.zeros((N_enss, nx, ny, nz))
    for i in range(N_enss):
        X_unie[i, :, :, :] = np.reshape(dataas[:, i], (nx, ny, nz), "F")
    latent_dim = 20 * 20 * 4
    wgan = WGAN(latent_dim, nx, ny, nz)
    wgan.train(epochs=4000, dataa=X_unie, batch_size=128, sample_interval=100)
    wgan.save_model()

    X_unie = np.zeros((N_enss, nx, ny, nz))
    for i in range(N_enss):
        X_unie[i, :, :, :] = np.reshape(dataasp[:, i], (nx, ny, nz), "F")
    latent_dim = 20 * 20 * 4
    wganp = WGANP(latent_dim, nx, ny, nz)
    wganp.train(epochs=4000, dataa=X_unie, batch_size=128, sample_interval=100)
    wganp.save_model()


def WGAN_LEARNING_PERM(High_K, Low_K, nx, ny, nz):
    filename = "PACKETS/Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble = mat["Z"]

    clfye = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfye.fit(ini_ensemble))
    ini_ensemble = clfye.transform(ini_ensemble)

    N_enss = ini_ensemble.shape[1]
    dataas = ini_ensemble

    X_unie = np.zeros((N_enss, nx, ny, nz))
    for i in range(N_enss):
        X_unie[i, :, :, :] = np.reshape(dataas[:, i], (nx, ny, nz), "F")
    latent_dim = 20 * 20 * 4
    wgan = WGAN(latent_dim, nx, ny, nz)
    wgan.train(epochs=4000, dataa=X_unie, batch_size=128, sample_interval=100)
    wgan.save_model()


def AE_GAN_PERM(spitt, Ne, nx, ny, nz, generator, High_K):
    noise = spitt

    gen_imgs = ((generator.predict(noise.T)) + 1) * (High_K / 2)

    X_unie = gen_imgs
    ouut = np.zeros((nx * ny * nz, Ne))
    for i in range(Ne):
        jud = []
        inn = X_unie[i, :, :, :]
        for k in range(nz):
            jj = inn[:, :, k]
            jj = np.ravel(np.reshape(jj, (-1, 1), "F"))
            jud.append(jj)
        jud = np.vstack(jud)
        ouut[:, i] = np.ravel(jud)

    return ouut


def AE_GAN(spitt, Ne, nx, ny, nz, generatora, High_K):
    noise = spitt
    gen_imgs = ((generatora.predict(noise.T)) + 1) * (High_K / 2)

    X_unie = gen_imgs
    ouut = np.zeros((nx * ny * nz, Ne))
    for i in range(Ne):
        jud = []
        inn = X_unie[i, :, :, :]
        for k in range(nz):
            jj = inn[:, :, k]
            jj = np.ravel(np.reshape(jj, (-1, 1), "F"))
            jud.append(jj)
        jud = np.vstack(jud)
        ouut[:, i] = np.ravel(jud)

    return ouut


def AE_GANP(spittp, Ne, nx, ny, nz, generatorb, High_P):
    noisep = spittp
    # gen_imgs = generator.predict(noisep.T)
    gen_imgs = ((generatorb.predict(noisep.T)) + 1) * (High_P / 2)

    X_unie = gen_imgs
    ouut = np.zeros((nx * ny * nz, Ne))
    for i in range(Ne):
        jud = []
        inn = X_unie[i, :, :, :]
        for k in range(nz):
            jj = inn[:, :, k]
            jj = np.ravel(np.reshape(jj, (-1, 1), "F"))
            jud.append(jj)
        jud = np.vstack(jud)
        ouut[:, i] = np.ravel(jud)

    return ouut


def Make_Bimodal(matrix, nclusters, High_K, Low_K):
    ddy = []
    for kk in range(matrix.shape[1]):
        kmeans = MiniBatchKMeans(n_clusters=nclusters, max_iter=2000).fit(
            matrix[:, kk].reshape(-1, 1)
        )
        dd = kmeans.labels_
        dd = dd.T
        dd = np.reshape(dd, (-1, 1))

        ax = kmeans.cluster_centers_
        if ax[0, :] > ax[1, :]:
            dd[dd == 1] = 50
            dd[dd == 0] = High_K
            dd[dd == 50] = Low_K
        else:
            dd[dd == 0] = 50
            dd[dd == 1] = High_K
            dd[dd == 50] = Low_K
        ddy.append(dd)
    ddy = np.hstack(ddy)
    return ddy


def Split_Matrix(matrix, sizee):
    x_split = np.split(matrix, sizee, axis=0)
    return x_split


def Recover_imageV(x, Ne, nx, ny, nz, latent_dim, vae, High_K, mem):
    X_unie = np.zeros((Ne, latent_dim))
    for i in range(Ne):
        X_unie[i, :] = np.reshape(x[:, i], (latent_dim,), "F")
    if mem == 1:
        decoded_imgs2 = (vae.decoder.predict(X_unie)) * High_K
    else:
        decoded_imgs2 = (vae.decoder.predict(X_unie)) * 1
    # print(decoded_imgs2.shape)
    ouut = np.zeros((nx * ny * nz, Ne))
    for i in range(Ne):
        jj = decoded_imgs2[i]  # .reshape(nx,ny,nz)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


class Sampling(tensorflow.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def Variational_Autoencoder(latent_dim, nx, ny, nz, High_K, Low_K, mem):
    filename = "PACKETS/Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensemble = mat["Z"]

    clfye = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfye.fit(ini_ensemble))
    ini_ensemble = clfye.transform(ini_ensemble)

    X_unie = np.zeros((ini_ensemble.shape[1], nx, ny, nz))
    for i in range(ini_ensemble.shape[1]):
        X_unie[i, :, :, :] = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")

    if mem == 1:
        ax = X_unie / High_K
    else:
        ax = X_unie

    x_train, x_test, y_train, y_test = train_test_split(
        ax, ax, test_size=0.1, random_state=42
    )

    # latent_dim = 2

    encoder_inputs = keras.Input(shape=(nx, ny, nz))
    x = tensorflow.keras.layers.Conv2D(
        32, 3, activation="gelu", strides=2, padding="same"
    )(encoder_inputs)
    x = tensorflow.keras.layers.Conv2D(
        64, 3, activation="gelu", strides=2, padding="same"
    )(x)
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(200, activation="gelu")(x)
    z_mean = tensorflow.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tensorflow.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = tensorflow.keras.layers.Dense(10 * 10 * 64, activation="gelu")(latent_inputs)
    x = tensorflow.keras.layers.Reshape((10, 10, 64))(x)
    x = tensorflow.keras.layers.Conv2DTranspose(
        64, 3, activation="gelu", strides=2, padding="same"
    )(x)
    x = tensorflow.keras.layers.Conv2DTranspose(
        32, 3, activation="gelu", strides=2, padding="same"
    )(x)
    decoder_outputs = tensorflow.keras.layers.Conv2DTranspose(
        nz, 3, activation="linear", padding="same"
    )(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder)

    vae.compile(optimizer="adam", loss="mse")

    vae.fit(ax, epochs=3000, batch_size=128)

    return vae


def Get_LatentV(ini_ensemble, N_ens, nx, ny, nz, latent_dim, vae, High_K, mem):
    X_unie = np.zeros((N_ens, nx, ny, nz))
    for i in range(N_ens):
        X_unie[i, :, :, :] = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")
    if mem == 1:
        ax = X_unie / High_K
    else:
        ax = X_unie
    ouut = np.zeros((latent_dim, Ne))
    # _,_,decoded_imgs2=(vae.encoder.predict(ax))
    decoded_imgs2, _, _ = vae.encoder.predict(ax)
    for i in range(N_ens):
        jj = decoded_imgs2[i]  # .reshape(20,20,4)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def shuffle(x, axis=0):
    n_axis = len(x.shape)
    t = np.arange(n_axis)
    t[0] = axis
    t[axis] = 0
    xt = np.transpose(x.copy(), t)
    np.random.shuffle(xt)
    shuffled_x = np.transpose(xt, t)
    return shuffled_x


# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm="ortho").T, norm="ortho")


def dct22(a, Ne, nx, ny, nz, size1, size2):
    ouut = np.zeros((size1 * size2 * nz, Ne))
    for i in range(Ne):
        origi = np.reshape(a[:, i], (nx, ny, nz), "F")
        outt = []
        for j in range(nz):
            mike = origi[:, :, j]
            dctco = dct(dct(mike.T, norm="ortho").T, norm="ortho")
            subb = dctco[:size1, :size2]
            subb = np.reshape(subb, (-1, 1), "F")
            outt.append(subb)
        outt = np.vstack(outt)
        ouut[:, i] = np.ravel(outt)
    return ouut


def inference_single2(
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
    BG,
    RS,
    CFL,
    timmee,
    MAXZ,
    factorr,
    steppi,
    aay,
    bby,
    mpor,
    hpor,
    dt,
    IWSw,
    IWSg,
    PB,
    PATM,
    CFO,
    method,
    SWI,
    SWR,
    UW,
    UO,
    UG,
    step2,
    pini_alt,
    INJ,
    PROD,
    opennI,
    opennP,
):
    paramss = ini
    Ne = 1  # paramss.shape[1]

    if nz == 1:
        ct = np.zeros((Ne, 4, nx, ny), dtype=np.float32)
        cP = np.zeros((Ne, 4 * steppi, nx, ny), dtype=np.float32)
    else:
        ct = np.zeros((Ne, 4, nx, ny, nz), dtype=np.float32)
        cP = np.zeros((Ne, 4 * steppi, nx, ny, nz), dtype=np.float32)

    Ainj = np.zeros((nx, ny, nz))
    Aprod = np.zeros((nx, ny, nz))

    n_inj = len(INJ)  # Number of injectors
    n_prod = len(PROD)  # Number of producers
    for kk in range(Ne):
        if nz == 1:
            ct1 = np.zeros((4, nx, ny), dtype=np.float32)
        else:
            ct1 = np.zeros((4, nx, ny, nz), dtype=np.float32)
        # print(str(kk + 1) + ' | ' +  str (Ne))
        a = np.reshape(paramss, (nx, ny, nz), "F")

        at1 = ini
        # at1 = rescale_linear_numpy_pytorch(at1, LUB, HUB,aay,bby)

        at2 = inip
        # at2 = rescale_linear(at2, mpor, hpor)

        at1 = np.reshape(at1, (nx, ny, nz), "F")
        at2 = np.reshape(at2, (nx, ny, nz), "F")

        if opennI == "all":
            for ay in range(n_inj):
                use = np.array(INJ[ay])
                for jj in range(nz):
                    xloc = int(use[0])
                    yloc = int(use[1])
                    Ainj[xloc, yloc, jj] = float(use[3])

        else:
            for ay in range(n_inj):
                use = np.array(INJ[ay])
                xloc = int(use[0])
                yloc = int(use[1])
                Ainj[xloc, yloc, -1] = float(use[3])

        if opennP == "all":
            for ay in range(n_prod):
                use = np.array(PROD[ay])
                for jj in range(nz):
                    xloc = int(use[0])
                    yloc = int(use[1])
                    Aprod[xloc, yloc, jj] = float(use[3])
        else:
            for ay in range(n_prod):
                use = np.array(PROD[ay])
                xloc = int(use[0])
                yloc = int(use[1])
                Aprod[xloc, yloc, -1] = float(use[3])

        quse1 = Ainj + Aprod

        quse = np.reshape(quse1, (-1, 1), "F")
        quse_water = np.reshape(Ainj, (-1, 1), "F")
        quse_oil = np.reshape(Aprod, (-1, 1), "F")
        # print(A)
        A_out = Reservoir_Simulator2(
            a,
            at2,
            quse.ravel(),
            quse_water.ravel(),
            quse_oil.ravel(),
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
            IWSg,
            method,
            steppi,
            SWI,
            SWR,
            UW,
            UO,
            UG,
            step2,
            pini_alt,
        )
        if nz == 1:
            ct1[0, :, :] = at1[:, :, 0]  # permeability
            ct1[1, :, :] = quse1[:, :, 0]  # /UIR # Overall f
            ct1[2, :, :] = Ainj[:, :, 0]  # /UIR# f for water injection
            ct1[3, :, :] = at2[:, :, 0]  # porosity
        else:
            ct1[0, :, :, :] = at1  # permeability
            ct1[1, :, :, :] = quse1  # /UIR # Overall f
            ct1[2, :, :, :] = Ainj  # /UIR# f for water injection
            ct1[3, :, :, :] = at2  # porosity

        if nz == 1:
            cP[kk, :, :, :] = A_out
            ct[kk, :, :, :] = ct1
        else:
            cP[kk, :, :, :, :] = A_out
            ct[kk, :, :, :, :] = ct1

        return ct, cP


def Simulator(plan, ini, inip, path_out):
    N_components = int(plan["N_components"])
    nx = int(plan["DIMENS"]["nx"])
    ny = int(plan["DIMENS"]["ny"])
    nz = int(plan["DIMENS"]["nz"])
    BO = float(plan["PROPS"]["BO"])
    BW = float(plan["PROPS"]["BW"])
    UW = float(plan["PROPS"]["UW"])
    UO = float(plan["PROPS"]["UO"])
    DX = float(plan["GRID"]["DX"])
    DY = float(plan["GRID"]["DY"])
    DZ = float(plan["GRID"]["DZ"])
    opennI = str(plan["SCHEDULE"]["opennI"])
    opennP = str(plan["SCHEDULE"]["opennP"])

    DX = cp.float32(DX)
    DY = cp.float32(DY)
    UW = cp.float32(UW)
    UO = cp.float32(UO)
    SWI = cp.float32(plan["PROPS"]["SWI"])
    SWR = cp.float32(plan["PROPS"]["SWR"])
    CFO = cp.float32(plan["PROPS"]["CFO"])
    IWSw = float(plan["PROPS"]["S1"])
    # IWSo = float(plan['PROPS']['SO1'])
    IWSg = float(plan["PROPS"]["SG1"])
    pini_alt = float(plan["PROPS"]["P1"])
    P1 = cp.float32(pini_alt)
    step2 = int(plan["NSTACK"]["n_substep"])
    PB = P1
    mpor, hpor = float(plan["MPOR"]), float(plan["HPOR"])
    aay, bby = float(plan["aay"]), float(plan["bby"])
    # Low_K, High_K = aay,bby
    BW = cp.float32(BW)
    BO = cp.float32(BO)
    PATM = cp.float32(float(plan["PROPS"]["PATM"]))

    method = int(plan["SOLVER"]["solution_method"])
    make_animation = int(plan["SUMMARY"]["make_animation"])
    make_csv = int(plan["SUMMARY"]["make_csv"])

    # SG1 = cp.float32(0)  # Initial gas saturation

    UG = cp.float32(calc_mu_g(P1))  # gas viscosity in cP
    RS = cp.float32(calc_rs(PB, P1))  # Solution GOR
    BG = calc_bg(PB, PATM, P1)  # gas formation volume factor

    # path_out = 'OUTPUTS' + str(kkz)
    # #print(f'Create {path_out}')
    # os.makedirs(path_out, exist_ok=True)

    timmee = float(plan["REPORTING"]["time_step"])
    max_t = float(plan["REPORTING"]["time_max"])
    MAXZ = float(plan["REPORTING"]["time_limit"])
    steppi = int(max_t / timmee)

    factorr = float(plan["MULTIPLY"]["z_factor"])

    N_inj = len(plan["WELLSPECS"]["injector_wells"])  # Number of injectors
    N_pr = len(plan["WELLSPECS"]["producer_wells"])  # Number of producers

    # dt = timmee/max_t
    # print('Check the condition  CFL number')
    ratioo = timmee / DX
    if (ratioo <= 1) or (N_components == 3) or (nx * ny * nz) >= 10000:
        # print('Solution is unconditionally stable with time step, use IMPES method')
        CFL = 1  # IMPES
    else:
        # print('Solution is conditionally stable with time step, Using Fully-Implicit method')
        CFL = 2  # IMPLICT

    CFL = 1

    tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
    dt = np.diff(tc2)[0]  # Time-step

    injectors = plan["WELLSPECS"]["injector_wells"]
    producers = plan["WELLSPECS"]["producer_wells"]

    wellsI = np.array(np.vstack(injectors)[:, :3], dtype=float)
    wellsI[:, 2] = 1
    wellsP = np.array(np.vstack(producers)[:, :3], dtype=float)
    wellsP[:, 2] = 2
    np.vstack([wellsI, wellsP])

    # NecessaryI = np.vstack(injectors)[:,4:7]
    # NecessaryP = np.vstack(producers)[:,4:7]

    NecessaryI = np.array(np.vstack(injectors)[:, 4:7], dtype=float)
    NecessaryP = np.array(np.vstack(producers)[:, 4:7], dtype=float)

    os.chdir(path_out)
    ini = rescale_linear(ini, aay, bby)
    Pressz = np.reshape(ini, (nx, ny, nz), "F")
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii
    plot3d2static(
        Pressz,
        nx,
        ny,
        nz,
        "Permeability",
        "Permeability",
        maxii,
        minii,
        injectors,
        producers,
    )
    os.chdir(oldfolder)

    start_time_plots1 = time.time()

    if N_components == 2:
        X_data1, data_use1 = inference_single(
            ini,
            inip,
            nx,
            ny,
            nz,
            max_t,
            DX,
            DY,
            DZ,
            BO,
            BW,
            CFL,
            timmee,
            MAXZ,
            factorr,
            steppi,
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
            step2,
            pini_alt,
            injectors,
            producers,
            opennI,
            opennP,
        )
        os.chdir(path_out)
        sio.savemat(
            "UNRST.mat",
            {
                "permeability": X_data1[0, 0, :, :][:, :, None],
                "porosity": X_data1[0, 3, :, :][:, :, None],
                "Pressure": data_use1[0, :steppi, :, :][:, :, None],
                "Water_saturation": data_use1[0, steppi:, :, :][:, :, None],
                "Oil_saturation": 1 - data_use1[0, steppi:, :, :][:, :, None],
            },
            do_compression=True,
        )
        os.chdir(oldfolder)
    else:
        X_data1, data_use1 = inference_single2(
            ini,
            inip,
            nx,
            ny,
            nz,
            max_t,
            DX,
            DY,
            DZ,
            BO,
            BW,
            BG,
            RS,
            CFL,
            timmee,
            MAXZ,
            factorr,
            steppi,
            aay,
            bby,
            mpor,
            hpor,
            dt,
            IWSw,
            IWSg,
            PB,
            PATM,
            CFO,
            method,
            SWI,
            SWR,
            UW,
            UO,
            UG,
            step2,
            pini_alt,
            injectors,
            producers,
            opennI,
            opennP,
        )
        os.chdir(path_out)
        sio.savemat(
            "UNRST.mat",
            {
                "permeability": X_data1[0, 0, :, :, :],
                "porosity": X_data1[0, 3, :, :, :],
                "Pressure": data_use1[0, :steppi, :, :, :],
                "Water_saturation": data_use1[0, steppi : 2 * steppi, :, :, :],
                "Oil_saturation": data_use1[0, 2 * steppi : 3 * steppi, :, :, :],
                "Gas_saturation": data_use1[0, 3 * steppi :, :, :, :],
            },
            do_compression=True,
        )
        os.chdir(oldfolder)

    elapsed_time_secs = time.time() - start_time_plots1
    "Reservoir simulation  took: %s secs (Wall clock time)" % timedelta(
        seconds=round(elapsed_time_secs)
    )

    if N_components == 2:
        if make_animation:
            start_time_plots = time.time()
            # print('')
            # print('Plotting outputs')
            os.chdir(path_out)
            Runs = steppi
            np.arange(1, Runs + 1)

            if nz == 1:
                for kk in range(steppi):
                    Plot_performance(
                        data_use1[0, :, :, :],
                        nx,
                        ny,
                        nz,
                        "new",
                        kk,
                        dt,
                        MAXZ,
                        steppi,
                        injectors,
                        producers,
                    )

                    Pressz = data_use1[0, kk, :, :][:, :, None]

                    maxii = max(Pressz.ravel())
                    minii = min(Pressz.ravel())
                    Pressz = Pressz / maxii
                    plot3d2(
                        Pressz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_pressure",
                        "Pressure",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    watsz = data_use1[0, kk + steppi, :, :][:, :, None]
                    maxii = max(watsz.ravel())
                    minii = min(watsz.ravel())
                    plot3d2(
                        watsz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_water",
                        "water_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    oilsz = 1 - data_use1[0, kk + steppi, :, :][:, :, None]
                    maxii = max(oilsz.ravel())
                    minii = min(oilsz.ravel())
                    plot3d2(
                        oilsz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_oil",
                        "oil_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                print("")
                print("Now - Creating GIF")
                import glob

                frames1 = []
                imgs = sorted(glob.glob("*new*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames1.append(new_frame)

                frames1[0].save(
                    "Evolution2.gif",
                    format="GIF",
                    append_images=frames1[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_pressure*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_pressure_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_water*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_water_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_oil*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_oil_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                from glob import glob

                for f3 in glob("*PINN_model_PyTorch*"):
                    os.remove(f3)

                for f4 in glob("*new*"):
                    os.remove(f4)

                for f4 in glob("*unie_pressure*"):
                    os.remove(f4)

                for f4 in glob("*unie_water*"):
                    os.remove(f4)

                for f4 in glob("*unie_oil*"):
                    os.remove(f4)
            else:
                for kk in range(steppi):
                    Pressz = data_use1[0, kk, :, :, :][:, :, ::-1]
                    maxii = max(Pressz.ravel())
                    minii = min(Pressz.ravel())
                    Pressz = Pressz / maxii
                    plot3d2(
                        Pressz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_pressure",
                        "Pressure",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    watsz = data_use1[0, kk + steppi, :, :, :][:, :, ::-1]
                    maxii = max(watsz.ravel())
                    minii = min(watsz.ravel())
                    plot3d2(
                        watsz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_water",
                        "water_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    oilsz = 1 - data_use1[0, kk + steppi, :, :, :][:, :, ::-1]
                    maxii = max(oilsz.ravel())
                    minii = min(oilsz.ravel())
                    plot3d2(
                        oilsz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_oil",
                        "oil_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                import glob

                frames = []
                imgs = sorted(glob.glob("*unie_pressure*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_pressure_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_water*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_water_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_oil*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_oil_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                from glob import glob

                for f4 in glob("*unie_pressure*"):
                    os.remove(f4)

                for f4 in glob("*unie_water*"):
                    os.remove(f4)

                for f4 in glob("*unie_oil*"):
                    os.remove(f4)
            os.chdir(oldfolder)
            elapsed_time_secs = time.time() - start_time_plots
            (
                "pressure and saturation plots took: %s secs (Wall clock time)"
                % timedelta(seconds=round(elapsed_time_secs))
            )
        # print(msg)

        if make_csv:
            start_time_peaceman = time.time()
            os.chdir(path_out)
            spittsbig = [
                "Time(DAY)",
                "I1 - WBHP(PSIA)",
                "I2 - WBHP (PSIA)",
                "I3 - WBHP(PSIA)",
                "I4 - WBHP(PSIA)",
                "P1 - WOPR(BBL/DAY)",
                "P2 - WOPR(BBL/DAY)",
                "P3 - WOPR(BBL/DAY)",
                "P4 - WOPR(BBL/DAY)",
                "P1 - WWPR(BBL/DAY)",
                "P2 - WWPR(BBL/DAY)",
                "P3 - WWPR(BBL/DAY)",
                "P4 - WWPR(BBL/DAY)",
                "P1 - WWCT(%)",
                "P2 - WWCT(%)",
                "P3 - WWCT(%)",
                "P4 - WWCT(%)",
            ]

            if nz == 1:
                seeTrue = Peaceman_well(
                    X_data1,
                    data_use1[:, :steppi, :, :],
                    data_use1[:, steppi:, :, :],
                    MAXZ,
                    0,
                    1e1,
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
                )
            else:
                seeTrue = Peaceman_well(
                    X_data1,
                    data_use1[:, :steppi, :, :, :],
                    data_use1[:, steppi:, :, :, :],
                    MAXZ,
                    0,
                    1e1,
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
                )

            seeuset = pd.DataFrame(seeTrue)
            seeuset.to_csv("RSM_FVM.csv", header=spittsbig, sep=",")
            seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

            Plot_RSM_percentile(seeTrue, "Compare.png")

            os.chdir(oldfolder)
            elapsed_time_secs = time.time() - start_time_peaceman
            (
                "Well rates and pressure computation took: %s secs (Wall clock time)"
                % timedelta(seconds=round(elapsed_time_secs))
            )

    else:  # 3phase flow
        if make_animation:
            start_time_plots = time.time()

            os.chdir(path_out)
            Runs = steppi
            np.arange(1, Runs + 1)

            if nz == 1:
                for kk in range(steppi):
                    Plot_performance2(
                        data_use1[0, :, :, :],
                        nx,
                        ny,
                        nz,
                        "new",
                        kk,
                        dt,
                        MAXZ,
                        steppi,
                        injectors,
                        producers,
                    )

                    Pressz = data_use1[0, kk, :, :][:, :, None]
                    maxii = max(Pressz.ravel())
                    minii = min(Pressz.ravel())
                    Pressz = Pressz / maxii
                    plot3d2(
                        Pressz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_pressure",
                        "Pressure",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    watsz = data_use1[0, kk + steppi, :, :][:, :, None]
                    maxii = max(watsz.ravel())
                    minii = min(watsz.ravel())
                    plot3d2(
                        watsz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_water",
                        "water_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    oilsz = data_use1[0, kk + 2 * steppi, :, :][:, :, None]
                    maxii = max(oilsz.ravel())
                    minii = min(oilsz.ravel())
                    plot3d2(
                        oilsz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_oil",
                        "oil_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    gasz = data_use1[0, kk + 3 * steppi, :, :][:, :, None]
                    maxii = max(gasz.ravel())
                    minii = min(gasz.ravel())
                    plot3d2(
                        gasz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_gas",
                        "gas_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                import glob

                frames1 = []
                imgs = sorted(glob.glob("*new*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames1.append(new_frame)

                frames1[0].save(
                    "Evolution2.gif",
                    format="GIF",
                    append_images=frames1[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_pressure*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_pressure_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_water*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_water_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_oil*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_oil_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_gas*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_gas_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                from glob import glob

                for f3 in glob("*PINN_model_PyTorch*"):
                    os.remove(f3)

                for f4 in glob("*new*"):
                    os.remove(f4)

                for f4 in glob("*unie_pressure*"):
                    os.remove(f4)

                for f4 in glob("*unie_water*"):
                    os.remove(f4)

                for f4 in glob("*unie_oil*"):
                    os.remove(f4)

                for f4 in glob("*unie_gas*"):
                    os.remove(f4)

            else:
                for kk in range(steppi):
                    Pressz = data_use1[0, kk, :, :, :][:, :, ::-1]
                    maxii = max(Pressz.ravel())
                    minii = min(Pressz.ravel())
                    Pressz = Pressz / maxii
                    plot3d2(
                        Pressz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_pressure",
                        "Pressure",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    watsz = data_use1[0, kk + steppi, :, :, :][:, :, ::-1]
                    maxii = max(watsz.ravel())
                    minii = min(watsz.ravel())
                    plot3d2(
                        watsz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_water",
                        "water_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    oilsz = data_use1[0, kk + 2 * steppi, :, :, :][:, :, ::-1]
                    maxii = max(oilsz.ravel())
                    minii = min(oilsz.ravel())
                    plot3d2(
                        oilsz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_oil",
                        "oil_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                    gasz = data_use1[0, kk + 3 * steppi, :, :, :][:, :, ::-1]
                    maxii = max(gasz.ravel())
                    minii = min(gasz.ravel())
                    plot3d2(
                        gasz,
                        nx,
                        ny,
                        nz,
                        kk,
                        dt,
                        MAXZ,
                        "unie_gas",
                        "gas_sat",
                        maxii,
                        minii,
                        injectors,
                        producers,
                    )

                import glob

                frames = []
                imgs = sorted(glob.glob("*unie_pressure*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_pressure_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_water*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_water_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_oil*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_oil_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                frames = []
                imgs = sorted(glob.glob("*unie_gas*"), key=os.path.getmtime)
                for i in imgs:
                    new_frame = Image.open(i)
                    frames.append(new_frame)

                frames[0].save(
                    "Evolution_gas_3D.gif",
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=500,
                    loop=0,
                )

                from glob import glob

                for f4 in glob("*unie_pressure*"):
                    os.remove(f4)

                for f4 in glob("*unie_water*"):
                    os.remove(f4)

                for f4 in glob("*unie_oil*"):
                    os.remove(f4)

                for f4 in glob("*unie_gas*"):
                    os.remove(f4)
            os.chdir(oldfolder)
            elapsed_time_secs = time.time() - start_time_plots
            (
                "pressure and saturation plots took: %s secs (Wall clock time)"
                % timedelta(seconds=round(elapsed_time_secs))
            )
        # print(msg)

        if make_csv:
            start_time_peaceman = time.time()
            os.chdir(path_out)

            spittsbig = [
                "Time(DAY)",
                "I1 - WBHP(PSIA)",
                "I2 - WBHP (PSIA)",
                "I3 - WBHP(PSIA)",
                "I4 - WBHP(PSIA)",
                "P1 - WOPR(BBL/DAY)",
                "P2 - WOPR(BBL/DAY)",
                "P3 - WOPR(BBL/DAY)",
                "P4 - WOPR(BBL/DAY)",
                "P1 - WWPR(BBL/DAY)",
                "P2 - WWPR(BBL/DAY)",
                "P3 - WWPR(BBL/DAY)",
                "P4 - WWPR(BBL/DAY)",
                "P1 - WGPR(SCF/DAY)",
                "P2 - WGPR(SCF/DAY)",
                "P3 - WGPR(SCF/DAY)",
                "P4 - WGPR(SCF/DAY)",
                "P1 - WWCT(%)",
                "P2 - WWCT(%)",
                "P3 - WWCT(%)",
                "P4 - WWCT(%)",
            ]

            if nz == 1:
                seeTrue = Peaceman_well2(
                    X_data1,
                    data_use1[:, :steppi, :, :],
                    data_use1[:, steppi : 2 * steppi, :, :],
                    data_use1[:, 2 * steppi : 3 * steppi, :, :],
                    data_use1[:, 3 * steppi : 4 * steppi, :, :],
                    MAXZ,
                    0,
                    1e1,
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
                )

            else:
                seeTrue = Peaceman_well2(
                    X_data1,
                    data_use1[:, :steppi, :, :, :],
                    data_use1[:, steppi : 2 * steppi, :, :, :],
                    data_use1[:, 2 * steppi : 3 * steppi, :, :, :],
                    data_use1[:, 3 * steppi : 4 * steppi, :, :, :],
                    MAXZ,
                    0,
                    1e1,
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
                )

            seeuset = pd.DataFrame(seeTrue)
            seeuset.to_csv("RSM_FVM.csv", header=spittsbig, sep=",")
            seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

            Plot_RSM_percentile2(seeTrue, "Compare.png")

            os.chdir(oldfolder)
            elapsed_time_secs = time.time() - start_time_peaceman
            (
                "Well rates and pressure computation took: %s secs (Wall clock time)"
                % timedelta(seconds=round(elapsed_time_secs))
            )
            # print(msg)
    return seeTrue


def Plot_performance_model(PINN, PINN2, nx, ny, namet, UIR, itt, dt, MAXZ, pini_alt):
    look = (PINN[itt, :, :, :]) * pini_alt
    look_sat = PINN2[itt, :, :, :]
    look_oil = 1 - look_sat

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(20, 20))

    plt.subplot(3, 3, 1)
    plt.pcolormesh(XX.T, YY.T, look[0, :, :], cmap="jet")
    plt.title("Layer 1 - Presure", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    # plt.clim(np.min(np.reshape(lookf,(-1,))),np.max(np.reshape(lookf,(-1,))))
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 3, 4)
    plt.pcolormesh(XX.T, YY.T, look[1, :, :], cmap="jet")
    plt.title("Layer 2 - Presure", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    # plt.clim(np.min(np.reshape(lookf,(-1,))),np.max(np.reshape(lookf,(-1,))))
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 3, 7)
    plt.pcolormesh(XX.T, YY.T, look[2, :, :], cmap="jet")
    plt.title("Layer 3 - Presure", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    # plt.clim(np.min(np.reshape(lookf,(-1,))),np.max(np.reshape(lookf,(-1,))))
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 3, 2)
    plt.pcolormesh(XX.T, YY.T, look_sat[0, :, :], cmap="jet")
    plt.title("Later 1- water_sat", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(0, 1)
    cbar1.ax.set_ylabel(" water_sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 3, 5)
    plt.pcolormesh(XX.T, YY.T, look_sat[1, :, :], cmap="jet")
    plt.title("Later 2- water_sat", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(0, 1)
    cbar1.ax.set_ylabel(" water_sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 3, 8)
    plt.pcolormesh(XX.T, YY.T, look_sat[2, :, :], cmap="jet")
    plt.title("Later 3 - water_sat", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(0, 1)
    cbar1.ax.set_ylabel(" water_sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 3, 3)
    plt.pcolormesh(XX.T, YY.T, look_oil[0, :, :], cmap="jet")
    plt.title(" Layer 1 - oil_sat", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(0, 1)
    cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 3, 6)
    plt.pcolormesh(XX.T, YY.T, look_oil[1, :, :], cmap="jet")
    plt.title(" Layer 2 - oil_sat", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(0, 1)
    cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.subplot(3, 3, 9)
    plt.pcolormesh(XX.T, YY.T, look_oil[2, :, :], cmap="jet")
    plt.title(" Layer 3 - oil_sat", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(0, 1)
    cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
    Add_marker2(plt, XX, YY, injectors, producers)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(int((itt + 1) * dt * MAXZ)) + " days"

    plt.suptitle(tita, fontsize=16)

    name = namet + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()


# implement 2D IDCT
def idct22(a, Ne, nx, ny, nz, size1, size2):
    # a=updated_ensembledct
    ouut = np.zeros((nx * ny * nz, Ne))
    for ix in range(Ne):
        # i=0
        subbj = a[:, ix]
        subbj = np.reshape(subbj, (size1, size2, nz), "F")
        neww = np.zeros((nx, ny))
        outt = []
        for jg in range(nz):
            # j=0
            usee = subbj[:, :, jg]
            neww[:size1, :size2] = usee
            aa = idct(idct(neww.T, norm="ortho").T, norm="ortho")
            subbout = np.reshape(aa, (-1, 1), "F")
            outt.append(subbout)
        outt = np.vstack(outt)
        ouut[:, ix] = np.ravel(outt)
    return ouut


def Logitt(Low_K, High_K, matrixx, Low_precision):
    jj = (matrixx - Low_K) / (High_K - Low_K)
    jj[jj <= 0] = Low_precision
    jj[jj >= 1] = 1 - Low_precision
    LogS = np.log(jj / (1 - jj))
    return LogS


def Get_new_K(Low_K, High_K, LogS1):
    newK = (High_K * LogS1) + (1 - LogS1) * Low_K
    return newK


def Get_weighting(simData, measurment):
    ne = simData.shape[1]
    measurment = measurment.reshape(-1, 1)
    objReal = np.zeros((ne, 1))
    temp = np.zeros((ne, 1))
    for j in range(ne):
        a = np.sum(simData[:, j] - measurment) ** 2
        b = np.sum((simData[:, j]) ** 2) + np.sum((measurment) ** 2)
        weight = a / b
        temp[j] = weight
    tempbig = np.sum(temp)
    right = ne - tempbig

    for j in range(ne):
        a = np.sum(simData[:, j] - measurment) ** 2
        b = np.sum((simData[:, j]) ** 2) + np.sum((measurment) ** 2)
        objReal[j] = (1 - (a / b)) / right

    return objReal


##############################################################################
# configs
##############################################################################

oldfolder = os.getcwd()
os.chdir(oldfolder)
cur_dir = oldfolder

print("------------------Download pre-trained models------------------------")
if not os.path.exists("PACKETS"):
    os.makedirs("PACKETS")
else:
    pass

DEFAULT = None
while True:
    DEFAULT = int(input("Use best default options:\n1=Yes\n2=No\n"))
    if (DEFAULT > 2) or (DEFAULT < 1):
        # raise SyntaxError('please select value between 1-2')
        print("")
        print("please try again and select value between 1-2")
    else:
        break

sizq = 1e4
scalei, scalei2, scalei3 = 1e7, 1e6, 1e4
Nsmall = 20
degg = 3  # Degree Polynomial
nclusters = 2  # sand and shale

fname = "Run_ensemble.yaml"
plan = read_yaml(fname)

fname2 = "Run_single.yaml"
plan2 = read_yaml(fname2)

N_components = int(plan["N_components"])
nx = int(plan["DIMENS"]["nx"])
ny = int(plan["DIMENS"]["ny"])
nz = int(plan["DIMENS"]["nz"])
BO = float(plan["PROPS"]["BO"])
BW = float(plan["PROPS"]["BW"])
UW = float(plan["PROPS"]["UW"])
UO = float(plan["PROPS"]["UO"])
DX = float(plan["GRID"]["DX"])
DY = float(plan["GRID"]["DY"])
DZ = float(plan["GRID"]["DZ"])

opennI = str(plan["SCHEDULE"]["opennI"])
opennP = str(plan["SCHEDULE"]["opennP"])

DX = cp.float32(DX)
DY = cp.float32(DY)
UW = cp.float32(UW)
UO = cp.float32(UO)
DZtempp = cp.float32(DZ)
XBLOCKS = cp.int32(nx)  # no of grid blocks in x direction
YBLOCKS = cp.int32(ny)  # no of grid blocks in x direction
ZBLOCKS = cp.int32(nz)  # no of grid blocks in z direction
SWI = cp.float32(plan["PROPS"]["SWI"])
SWR = cp.float32(plan["PROPS"]["SWR"])
CFO = cp.float32(plan["PROPS"]["CFO"])
IWSw = float(plan["PROPS"]["S1"])
IWSo = float(plan["PROPS"]["SO1"])
IWSg = float(plan["PROPS"]["SG1"])
pini_alt = float(plan["PROPS"]["P1"])
P1 = cp.float32(pini_alt)
step2 = int(plan["NSTACK"]["n_substep"])
PB = P1
mpor, hpor = float(plan["MPOR"]), float(plan["HPOR"])
aay, bby = float(plan["aay"]), float(plan["bby"])
Low_K, High_K = aay, bby
BW = cp.float32(BW)
BO = cp.float32(BO)
PATM = cp.float32(float(plan["PROPS"]["PATM"]))
method = int(plan["SOLVER"]["solution_method"])
make_animation = int(plan["SUMMARY"]["make_animation"])
make_csv = int(plan["SUMMARY"]["make_csv"])
if N_components == 3:
    SG1 = cp.float32(0)  # Initial gas saturation
if N_components == 3:
    UG = cp.float32(calc_mu_g(P1))  # gas viscosity in cP
    RS = cp.float32(calc_rs(PB, P1))  # Solution GOR
    BG = calc_bg(PB, PATM, P1)  # gas formation volume factor

    # Ug = 0.02 # Gas viscosity
    # Bg = 0.001 # Gas formation volume factor

num_cores = multiprocessing.cpu_count()


timmee = float(plan["REPORTING"]["time_step"])
max_t = float(plan["REPORTING"]["time_max"])
MAXZ = float(plan["REPORTING"]["time_limit"])
steppi = int(max_t / timmee)


factorr = float(plan["MULTIPLY"]["z_factor"])
RE = 0.2 * DX


N_inj = len(plan["WELLSPECS"]["injector_wells"])  # Number of injectors
N_pr = len(plan["WELLSPECS"]["producer_wells"])  # Number of producers

# dt = timmee/max_t
# print('Check the condition  CFL number')
ratioo = timmee / DX
if (ratioo <= 1) or (N_components == 3) or (nx * ny * nz) >= 10000:
    # print('Solution is unconditionally stable with time step, use IMPES method')
    CFL = 1  # IMPES
else:
    # print('Solution is conditionally stable with time step, Using Fully-Implicit method')
    CFL = 2  # IMPLICT

CFL = 1

tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
dt = np.diff(tc2)[0]  # Time-step

# CFL = 1
n_inj = len(plan["WELLSPECS"]["injector_wells"])  # Number of injectors
n_prod = len(plan["WELLSPECS"]["producer_wells"])  # Number of producers

N_inj = n_inj  # 4 injector wells
N_pr = n_prod  # 4 producer wells

injectors = plan["WELLSPECS"]["injector_wells"]
producers = plan["WELLSPECS"]["producer_wells"]


wellsI = np.array(np.vstack(injectors)[:, :3], dtype=float)
wellsI[:, 2] = 1
wellsP = np.array(np.vstack(producers)[:, :3], dtype=float)
wellsP[:, 2] = 2
wells = np.vstack([wellsI, wellsP])

# NecessaryI = np.vstack(injectors)[:,4:7]
# NecessaryP = np.vstack(producers)[:,4:7]

NecessaryI = np.array(np.vstack(injectors)[:, 4:7], dtype=float)
NecessaryP = np.array(np.vstack(producers)[:, 4:7], dtype=float)


Low_P = mpor
High_P = hpor
BlockTotal = XBLOCKS * YBLOCKS * ZBLOCKS  # Total Number of GridBlocks
LBLOCKS = cp.int32(XBLOCKS * YBLOCKS)  # GridBlocks per layer
seed = 1  # 1 is the best
print("Random Seed: ", seed)
ra.seed(seed)
torch.manual_seed(seed)

cuda = 0
device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

# True model
bba = os.path.isfile("PACKETS/iglesias3.out")
if not bba:
    print("....Downloading Please hold.........")
    download_file_from_google_drive(
        "1VSy2m3ocUkZnhCsorbkhcJB5ADrPxzIp", "PACKETS/iglesias3.out"
    )
    print("...Downlaod completed.......")


Truee = np.genfromtxt("PACKETS/iglesias3.out", dtype="float")
Truee = np.reshape(Truee, (-1,))
Low_K1, High_K1 = aay, bby
perm_high = bby  # np.asscalar(np.amax(perm,axis=0)) + 300
perm_low = aay  # np.asscalar(np.amin(perm,axis=0))-50
poro_high = High_P  # np.asscalar(np.amax(poro,axis=0))+0.3
poro_low = Low_P  # np.asscalar(np.amin(poro,axis=0))-0.1

High_K, Low_K, High_P, Low_P = perm_high, perm_low, poro_high, poro_low
if not os.path.exists("HM_RESULTS"):
    os.makedirs("HM_RESULTS")
else:
    shutil.rmtree("HM_RESULTS")
    os.makedirs("HM_RESULTS")

print("")
if DEFAULT == 1:
    weighting = 3
else:
    weighting = None
    while True:
        weighting = int(
            input(
                "Enter the weighting scheme for model realisations:\n\
1 = weigthing\n\
2 = partial-weighting\n\
3 = Non-weighting\n\
        "
            )
        )
        if (weighting > 3) or (weighting < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break

if DEFAULT == 1:
    use_pretrained = 1
else:
    use_pretrained = None
    while True:
        use_pretrained = int(
            input(
                "Use pretrained models:\n\
        1 = Yes\n\
        2 = No\n\
        "
            )
        )
        if (use_pretrained > 2) or (use_pretrained < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break


if not os.path.exists("TRUE"):
    os.makedirs("TRUE")
    kq = np.genfromtxt("data/iglesias3.out", skip_header=0, dtype="float")
    kq = kq.reshape(-1, 1)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    clfy2 = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfy.fit(kq))
    (clfy2.fit(kq))
    kq = (clfy.transform(kq)).reshape(-1, 1)
    kq2 = (clfy2.transform(kq)).reshape(-1, 1)
    start_time_peaceman = time.time()
    Simulator(plan, kq, kq2, "TRUE")
    elapsed_time_secs = time.time() - start_time_peaceman
    msg = (
        "Simulation took "
        + str(timedelta(seconds=round(elapsed_time_secs)))
        + "seconds"
    )
    print(msg)

    # simDatafinal,predMatrix = RSM_extract (1,'TRUE',oldfolder,scalei,scalei2)
    # True_data = simDatafinal
    # True_mat = predMatrix
    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])
    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    # Remove_folder(1,'TRUE')
else:
    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Plot_RSM_singleT(True_mat, "Historical.png")
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])
    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)


True_yet = True_data
sdjesus = np.std(True_data, axis=0)
sdjesus = np.reshape(sdjesus, (1, -1), "F")

menjesus = np.mean(True_data, axis=0)
menjesus = np.reshape(menjesus, (1, -1), "F")


True_dataTI = True_data
True_dataTI = np.reshape(True_dataTI, (-1, 1), "F")
True_data = np.reshape(True_data, (-1, 1), "F")
Nop = True_dataTI.shape[0]


print("")
print("-----------------------Select Good prior-----------------------------")
# """
N_small = 20  # Size of realisations for initial selection of TI
ressimmastertest = os.path.join(oldfolder, "Training_Images")

# permx,Jesus,TIs,yett = Select_TI(oldfolder,ressimmastertest,N_small,\
#                                  nx,ny,nz,True_data,Low_K,High_K)
# """

os.chdir(oldfolder)
TII = 3
if TII == 3:
    print("TI = 3")
    os.chdir("Training_Images")
    kq = np.genfromtxt("iglesias3.out", skip_header=0, dtype="float")
    os.chdir(oldfolder)
    kq = kq.reshape(-1, 1)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfy.fit(kq))
    kq = np.reshape(kq, (nx, ny, nz), "F")
    kq2 = kq
    kq = kq[:, :, 0]
    kjennq = kq
    permx = kjennq

else:
    os.chdir("Training_Images")
    kq = np.genfromtxt("TI_2.out", skip_header=3, dtype="float")
    kq = kq.reshape(-1, 1)
    os.chdir(oldfolder)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfy.fit(kq))
    kq = clfy.transform(kq)
    kq = np.reshape(kq, (250, 250), "F")
    kjennq = kq.T
    permx = kjennq

True_K = np.reshape(kq2, (-1, 1), "F")
clfyy = MinMaxScaler(feature_range=(Low_P, High_P))
(clfyy.fit(True_K))
True_P = clfyy.transform(True_K)

yett = 3
# Jesus=2 #1=MPS, 2=SGSIM
if DEFAULT == 1:
    Jesus = 1
else:
    Jesus = None
    while True:
        Jesus = int(input("Input Geostatistics type:\n1=MPS\n2=SGSIM\n"))
        if (Jesus > 2) or (Jesus < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break

if Jesus == 1:
    print(" Multiple point statistics chosen")
    print("Plot the TI selected")
    plt.figure(figsize=(10, 10))
    plt.imshow(permx)
    # plt.gray()
    plt.title("Training_Image Selected ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    os.chdir("HM_RESULTS")
    plt.savefig("TI.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()

else:
    print(" Gaussian distribution chosen")

path = os.getcwd()

Geostats = int(Jesus)
os.chdir(oldfolder)

print("-------------------------------- Pior Selected------------------------")
print("---------------------------------------------------------------------")
print("")
print("---------------------------------------------------------------------")

# noise_level=float(input('Enter the masurement data noise level in % (1%-5%): '))
noise_level = 1
print("")
print("---------------------------------------------------------------------")
noise_level = noise_level / 100
print("")
if Geostats == 2:
    choice = 2
else:
    pass

sizeclem = nx * ny * nz
print("")
print("-------------------Decorrelate the ensemble---------------------------")
# """
if DEFAULT == 1:
    Deccorr = 2
else:
    Deccor = None
    while True:
        Deccor = int(input("De-correlate the ensemble:\n1=Yes\n2=No\n"))
        if (Deccor > 2) or (Deccor < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break

# """
# Deccor=2

print("")
print("-----------------------Alpha Parameter-------------------------------")

if DEFAULT == 1:
    DE_alpha = 1
else:
    De_alpha = None
    while True:
        De_alpha = int(input("Use recommended alpha:\n1=Yes\n2=No\n"))
        if (De_alpha > 2) or (De_alpha < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break
print("")
print("---------------------------------------------------------------------")

if DEFAULT == 1:
    afresh = 2
else:
    afresh = None
    while True:
        afresh = int(
            input(
                "Generate ensemble afresh  or random?:\n1=afresh\n\
2=random from Library\n"
            )
        )
        if (afresh > 2) or (afresh < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break
print("")
print("---------------------------------------------------------------------")
print("")
print("\n")
print("|-----------------------------------------------------------------|")
print("|                 SOLVE INVERSE PROBLEM WITH WEIGHTED aREKI:        |")
print("|-----------------------------------------------------------------|")
print("")

print("--------------------------ADAPTIVE-REKI----------------------------------")
print("History Matching using the Adaptive Regularised Ensemble Kalman Inversion")
print("Novel Implementation by Clement Etienam, SA-NVIDIA: SA-ML/A.I/Energy")

if DEFAULT == 1:
    Technique_REKI = 2
    print(
        "Default method is the Weighted  Adaptive Ensemble Kalman Inversion + Convolution Autoencoder "
    )
else:
    Technique_REKI = None
    while True:
        Technique_REKI = int(
            input(
                "Enter optimisation with the Adaptive Ensemble Kalman Inversion: \n\
        1 = ADAPT_REKI (Vanilla Adaptive Ensemble Kalman Inversion)\n\
        2 = ADAPT_REKI_AE  (Adaptive Ensemble Kalman Inversion + Convolution Autoencoder)\n\
        3 = ADAPT_REKI_DA  (Adaptive Ensemble Kalman Inversion + Denoising Autoencoder)\n\
        4 = ADAPT_REKI_NORMAL_SCORE  (Adaptive Ensemble Kalman Inversion + Normal Score)\n\
        5 = ADAPT_REKI_DE_GAN  (Adaptive Ensemble Kalman Inversion + Convolution Autoencoder + GAN)\n\
        6 = ADAPT_REKI_GAN  (Adaptive Ensemble Kalman Inversion + GAN on the permeability field alone:\
This method is optimal for multiple point statistics alone (MPS))\n\
        7 = ADAPT_REKI_KMEANS  (Adaptive Ensemble Kalman Inversion + KMEANS on the permeability field alone:\
This method is optimal for multiple point statistics alone (MPS))\n\
        8 = ADAPT_REKI_VAE  (Adaptive Ensemble Kalman Inversion + variational\
convolutional autoencoder on the permeability field alone:\
This method is optimal for multiple point statistics alone (MPS))\n\
        9 = ADAPT_REKI_DCT  (Adaptive Ensemble Kalman Inversion with DCT parametrisation\n\
        10 = ADAPT_REKI with Logit transformation\n"
            )
        )
        if (Technique_REKI > 10) or (Technique_REKI < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-10")
        else:
            break

if DEFAULT == 1:
    Termm = 20
else:
    Termm = None
    while True:
        Termm = int(input("Enter number of iterations (5-20)\n"))
        if (Termm > 20) or (Termm < 5):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 6-20")
        else:
            break

if DEFAULT == 1:
    bb = os.path.isfile("PACKETS/Training4.mat")
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1wYyREUcpp0qLhbRItG5RMPeRMxVtntDi", "PACKETS/Training4.mat"
        )
        print("...Downlaod completed.......")

        print("Load simulated labelled training data from MAT file")
        matt = sio.loadmat("PACKETS/Training4.mat")
        X_data1 = matt["INPUT"]
    else:
        print("Load simulated labelled training data from MAT file")
        matt = sio.loadmat("PACKETS/Training4.mat")
        X_data1 = matt["INPUT"]

    cPerm = np.zeros((nx * ny * nz, X_data1.shape[0]))  # Permeability
    for kk in range(X_data1.shape[0]):
        perm = X_data1[kk, 0, :, :]
        perm = np.reshape(X_data1[kk, 0, :, :], (nx * ny * nz,), "F")
        cPerm[:, kk] = perm
        clfye = MinMaxScaler(feature_range=(Low_K, High_K))
        (clfye.fit(cPerm))
        ini_ensemble = clfye.transform(cPerm)

    Ne = None
    while True:
        Ne = int(input("Number of realisations used for history matching (100-300) : "))
        if (Ne > int(ini_ensemble.shape[1])) or (Ne < 100):
            # raise SyntaxError('please select value between 1-2')
            value = int(ini_ensemble.shape[1])
            print("")
            print("please try again and select value between 100-" + str(value))
        else:
            break
    N_ens = Ne

    if Ne < ini_ensemble.shape[1]:
        index = np.random.choice(ini_ensemble.shape[1], Ne, replace=False)
        ini_use = ini_ensemble[:, index]
    else:
        ini_use = ini_ensemble
    ini_ensemble = ini_use

else:
    Ne = None
    while True:
        Ne = int(
            input("Number of realisations used for history matching (100-9000) : ")
        )
        if (Ne > 9000) or (Ne < 100):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 100-9000")
        else:
            break
    N_ens = Ne
    if Geostats == 1:
        if afresh == 1:
            see = intial_ensemble(nx, ny, nz, (N_ens), permx)

            ini_ensemble = see
            sio.savemat("PACKETS/Ganensemble.mat", {"Z": ini_ensemble})
            clfye = MinMaxScaler(feature_range=(Low_K, High_K))
            (clfye.fit(ini_ensemble))
            ini_ensemblef = clfye.transform(ini_ensemble)

            clfye = MinMaxScaler(feature_range=(Low_P, High_P))
            (clfye.fit(ini_ensemblef))
            ini_ensemblep = clfye.transform(ini_ensemblef)

            ensemble = ini_ensemblef
            ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K)
            ensemblep = ini_ensemblep
            ensemble, ensemblep = honour2(
                ensemblep, ensemble, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P
            )

            ini_ensemble = ensemble

        else:
            print("Use already generated ensemble from Google drive folder")
            if Ne <= 2000:
                # yes
                bb = os.path.isfile("PACKETS/Training4.mat")
                if not bb:
                    print("....Downloading Please hold.........")
                    download_file_from_google_drive(
                        "1wYyREUcpp0qLhbRItG5RMPeRMxVtntDi", "PACKETS/Training4.mat"
                    )
                    print("...Downlaod completed.......")

                    print("Load simulated labelled training data from MAT file")
                    matt = sio.loadmat("PACKETS/Training4.mat")
                    X_data1 = matt["INPUT"]
                else:
                    print("Load simulated labelled training data from MAT file")
                    matt = sio.loadmat("PACKETS/Training4.mat")
                    X_data1 = matt["INPUT"]

                cPerm = np.zeros((nx * ny * nz, X_data1.shape[0]))  # Permeability
                for kk in range(X_data1.shape[0]):
                    perm = X_data1[kk, 0, :, :]
                    perm = np.reshape(X_data1[kk, 0, :, :], (nx * ny * nz,), "F")
                    cPerm[:, kk] = perm
                    clfye = MinMaxScaler(feature_range=(Low_K, High_K))
                    (clfye.fit(cPerm))
                    ini_ensemble = clfye.transform(cPerm)

                N_ens = Ne

                if Ne < ini_ensemble.shape[1]:
                    index = np.random.choice(ini_ensemble.shape[1], Ne, replace=False)
                    ini_use = ini_ensemble[:, index]
                else:
                    ini_use = ini_ensemble
                ini_ensemble = ini_use
            else:
                bb = os.path.isfile(("PACKETS/Ganensemble.mat"))
                if not bb:
                    print(
                        "Get initial geology from saved Multiple-point-statistics run"
                    )
                    print("....Downloading Please hold.........")

                    download_file_from_google_drive(
                        "1KZvxypUSsjpkLogGm__56-bckEee3VJh", "PACKETS/Ganensemble.mat"
                    )
                    print("...Downlaod completed.......")
                else:
                    pass
                filename = "PACKETS/Ganensemble.mat"  # Ensemble generated offline

                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                index = np.random.choice(ini_ensemblef.shape[1], Ne, replace=False)

                ini_ensemblef = ini_ensemblef[:, index]

                clfye = MinMaxScaler(feature_range=(Low_K, High_K))
                (clfye.fit(ini_ensemblef))
                ini_ensemblef = clfye.transform(ini_ensemblef)

                clfye = MinMaxScaler(feature_range=(Low_P, High_P))
                (clfye.fit(ini_ensemblef))
                ini_ensemblep = clfye.transform(ini_ensemblef)

                ensemble = ini_ensemblef
                ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K)
                ensemblep = ini_ensemblep
                ensemble, ensemblep = honour2(
                    ensemblep, ensemble, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P
                )

                ini_ensemble = ensemble

    else:
        if afresh == 1:
            ini_ensemblef = initial_ensemble_gaussian(
                nx, ny, nz, (N_ens), Low_K, High_K
            )
            sio.savemat("PACKETS/Ganensemble_gauss.mat", {"Z": ini_ensemblef})
            clfye = MinMaxScaler(feature_range=(Low_K, High_K))
            (clfye.fit(ini_ensemblef))
            ini_ensemblef = clfye.transform(ini_ensemblef)

            clfye = MinMaxScaler(feature_range=(Low_P, High_P))
            (clfye.fit(ini_ensemblef))
            ini_ensemblep = clfye.transform(ini_ensemblef)

            ensemble = ini_ensemblef
            ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K)
            ensemblep = ini_ensemblep
            ensemble, ensemblep = honour2(
                ensemblep, ensemble, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P
            )

            ini_ensemble = ensemble
        else:
            filename = "PACKETS/Ganensemble_gauss.mat"  # Ensemble generated offline
            bb = os.path.isfile(filename)
            if not bb:
                print("Get initial geology from saved two-point-statistics run")
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1Kbe3F7XRzubmDU2bPkfHo2bEmjBNIZ29", filename
                )
                print("...Downlaod completed.......")
                print("")
            else:
                pass

            mat = sio.loadmat(filename)
            ini_ensemblef = mat["Z"]

            clfye = MinMaxScaler(feature_range=(Low_K, High_K))
            (clfye.fit(ini_ensemblef))
            ini_ensemblef = clfye.transform(ini_ensemblef)

            clfye = MinMaxScaler(feature_range=(Low_P, High_P))
            (clfye.fit(ini_ensemblef))
            ini_ensemblep = clfye.transform(ini_ensemblef)

            ensemble = ini_ensemblef
            ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K)
            ensemblep = ini_ensemblep
            ensemble, ensemblep = honour2(
                ensemblep, ensemble, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P
            )

            ini_ensemble = ensemble

    ini_ensemble = shuffle(ini_ensemble, axis=1)


for kk in range(N_ens):
    path_out = "Realisation" + str(kk)
    os.makedirs(path_out, exist_ok=True)

start_time = time.time()
if Technique_REKI == 1:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Denoising Autoencoder\
post-processing\n"
    )
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    if DEFAULT == 1:
        choice = 2
    else:
        choice = None
        while True:
            choice = int(input("Denoise the update:\n1=Yes\n2=No\n"))
            if (choice > 2) or (choice < 1):
                # raise SyntaxError('please select value between 1-2')
                print("")
                print("please try again and select value between 1-2")
            else:
                break

    os.chdir(oldfolder)

    if Geostats == 1:
        bb = os.path.isfile("PACKETS/denosingautoencoder.h5")
        # bb2=os.path.isfile('denosingautoencoderp.h5')
        if not bb:
            if use_pretrained == 1:
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1S71PoGY3MY3lBuuqBahstXZTslZsvXTs",
                    "PACKETS/denosingautoencoder.h5",
                )
                print("...Downlaod completed.......")

            else:
                DenosingAutoencoder(
                    nx, ny, nz, High_K1, Low_K1
                )  # Learn for permeability
            # DenosingAutoencoderp(nx,ny,nz,machine_map,N_ens,High_P) #Learn for porosity
        else:
            pass
        bb2 = os.path.isfile("PACKETS/denosingautoencoderp.h5")
        if not bb2:
            if use_pretrained == 1:
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1HZ-LeVLWO9ZsDEde4LG11aOxKOt4uBVn",
                    "PACKETS/denosingautoencoderp.h5",
                )
                print("...Downlaod completed.......")

            else:
                DenosingAutoencoderp(nx, ny, nz, N_ens, High_P, High_K, Low_K)
        else:
            pass

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)

    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )
    ax = np.zeros((Nop, 1))
    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 1e-5
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass

    snn = 0
    ii = 0
    print("Read Historical data")
    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep

        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)
        # executor.shutdown(wait=True)
        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "Initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        True_dataa = True_data

        CDd = cp.asarray(CDd)

        True_dataa = cp.asarray(True_dataa)

        Ddraw = cp.tile(True_dataa, Ne)

        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass
        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (inv(CDd)) @ (Dd - simDatafinal)), axis=1
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (inv(CDd)) @ (Dd - simDatafinal)), axis=1
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))

        sgsim = ensemble

        overall = cp.vstack([cp.asarray(sgsim), cp.asarray(ensemblep)])

        Y = overall
        Sim1 = cp.asarray(simDatafinal)

        if weighting == 1:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 2:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = Sim1[:, jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 3:
            M = cp.mean(Sim1, axis=1)

            M2 = cp.mean(overall, axis=1)

        S = cp.zeros((Sim1.shape[0], Ne))
        yprime = cp.zeros((Y.shape[0], Y.shape[1]))

        for jc in range(Ne):
            S[:, jc] = Sim1[:, jc] - M
            yprime[:, jc] = overall[:, jc] - M2
        Cydd = (yprime) / cp.sqrt(Ne - 1)
        Cdd = (S) / cp.sqrt(Ne - 1)

        GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
        inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
        Cdd = GDT.T @ GDT
        Cyd = Cydd @ GDT
        Usig, Sig, Vsig = cp.linalg.svd(
            (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
        )
        Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
        valuesig = Bsig[-1]  # last element
        valuesig = valuesig * 0.9999
        indices = (Bsig >= valuesig).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]

        (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)

        update_term = (
            (Cyd @ (X))
            @ (inv_CDd)
            @ (
                (
                    cp.tile(cp.asarray(True_dataa), Ne)
                    + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
                )
                - Sim1
            )
        )

        Ynew = Y + update_term
        sizeclem = cp.asarray(nx * ny * nz)
        if Yet == 0:
            updated_ensemble = cp.asnumpy(Ynew[:sizeclem, :])
            updated_ensemblep = cp.asnumpy(Ynew[sizeclem : 2 * sizeclem, :])

        else:
            updated_ensemble = Ynew[:sizeclem, :]
            updated_ensemblep = Ynew[sizeclem : 2 * sizeclem, :]

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior

        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest
        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)

        ensemble = updated_ensemble
        ensemblep = updated_ensemblep
        ensemble, ensemblep = honour2(
            ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
        )
        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)

    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI ENDED----------------------------")

elif Technique_REKI == 2:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Convolution Autoencoder \
Parametrisation\n"
    )
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    print("-------------------------learn Autoencoder------------------------")
    bb = os.path.isfile("PACKETS/encoder.h5")
    bb2 = os.path.isfile("PACKETS/encoderp.h5")
    if not bb:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1AEZPXAoJaC88dY9T-FFhLwJ_9U-Pl3tq", "PACKETS/encoder.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1u4Vo7XdyZQq4Z0_jNsoqG4QNWslAwyBv", "PACKETS/decoder.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "16Be2bIUWhiq8RBT48-baIHGU47XSnixS", "PACKETS/autoencoder.h5"
            )
            print("...Downlaod completed.......")

        else:
            Autoencoder2(nx, ny, nz, High_K1, Low_K1)  # Learn for permeability

    if not bb2:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1Q2oi6TmyjE-KkShhyqqi2nZRRa6I8OYB", "PACKETS/encoderp.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1Y4sQxR4Uj_OO4QjwNUocOe66Zmtu0JsK", "PACKETS/decoderp.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1fUUSBeXXZLP9PhME5oE_3yG_5ETAzIMk", "PACKETS/autoencoderp.h5"
            )
            print("...Downlaod completed.......")
        else:
            Autoencoder2p(nx, ny, nz, High_P, High_K1, Low_K1)  # Learn for porosity

    print("")
    if Jesus == 1:
        print("multiple point statistics")
        choice = None
        while True:
            choice = int(input("Denoise the update:\n1=Yes\n2=No\n"))
            if (choice > 2) or (choice < 1):
                # raise SyntaxError('please select value between 1-2')
                print("")
                print("please try again and select value between 1-2")
            else:
                break

        if choice == 1:
            bb = os.path.isfile("PACKETS/denosingautoencoder.h5")
            # bb2=os.path.isfile('denosingautoencoderp.h5')
            if not bb:
                if use_pretrained == 1:
                    print("....Downloading Please hold.........")
                    download_file_from_google_drive(
                        "1S71PoGY3MY3lBuuqBahstXZTslZsvXTs",
                        "PACKETS/denosingautoencoder.h5",
                    )
                    print("...Downlaod completed.......")
                else:
                    DenosingAutoencoder(
                        nx, ny, nz, High_K1, Low_K1
                    )  # Learn for permeability
                # DenosingAutoencoderp(nx,ny,nz,machine_map,N_ens,High_P) #Learn for porosity
            else:
                pass
            bb2 = os.path.isfile("PACKETS/denosingautoencoderp.h5")
            if not bb2:
                if use_pretrained == 1:
                    print("....Downloading Please hold.........")
                    download_file_from_google_drive(
                        "1HZ-LeVLWO9ZsDEde4LG11aOxKOt4uBVn",
                        "PACKETS/denosingautoencoderp.h5",
                    )
                    print("...Downlaod completed.......")
                else:
                    DenosingAutoencoderp(nx, ny, nz, N_ens, High_P, High_K, Low_K)
            else:
                pass
        else:
            pass
    else:
        print("Two point statistics used, passing")
    print("")
    print("--------------------Sub Section Ended--------------------------------")

    os.chdir(oldfolder)

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)

    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    # ensemble,ensemblep=honour2(ensemblep,\
    #                              ensemble,nx,ny,nz,N_ens,\
    #                                  High_K,Low_K,High_P,Low_P)

    print("Check Convolutional Autoencoder reconstruction")
    small = Get_Latent(ensemble, Ne, nx, ny, nz, High_K1)
    recc = Recover_image(small, Ne, nx, ny, nz, High_K1)

    dimms = (small.shape[0] / ensemble.shape[0]) * 100
    dimms = round(dimms, 3)

    recover = np.reshape(recc[:, 0], (nx, ny, nz), "F")
    origii = np.reshape(ensemble[:, 0], (nx, ny, nz), "F")

    plt.figure(figsize=(20, 20))
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.subplot(3, 2, 1)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 0], cmap="jet")
    plt.title("Recovered Layer 1 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 3)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 1], cmap="jet")
    plt.title("Recovered Layer 2 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 5)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 2], cmap="jet")
    plt.title("Recovered Layer 2 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 2)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 0], cmap="jet")
    plt.title("original Layer 1 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 4)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 1], cmap="jet")
    plt.title("original Layer 2 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 6)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 2], cmap="jet")
    plt.title("original Layer 3 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ttitle = "Parameter Recovery with " + str(dimms) + "% of original value"
    plt.suptitle(ttitle, fontsize=20)
    os.chdir("HM_RESULTS")
    plt.savefig("Recover_Comparison.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()

    ax = np.zeros((Nop, 1))

    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 0.0001
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass
    snn = 0
    ii = 0
    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep

        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)

        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "Initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        # Cini=CDd

        True_dataa = True_data
        CDd = cp.asarray(CDd)
        # Cini=cp.asarray(Cini)
        True_dataa = cp.asarray(True_dataa)
        Ddraw = cp.tile(True_dataa, Ne)
        # pertubations=pertubations.T
        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))
        if ii == 0:
            sgsim = ensemble
            encoded = Get_Latent(sgsim, Ne, nx, ny, nz, High_K1)
            encodedp = Get_Latentp(ensemblep, Ne, nx, ny, nz, High_P)
        else:
            encoded = updated_ensemble
            encodedp = updated_ensemblep

        shapalla = encoded.shape[0]

        # overall=cp.asarray(encoded)
        overall = cp.vstack([cp.asarray(encoded), cp.asarray(encodedp)])

        Y = overall
        Sim1 = cp.asarray(simDatafinal)

        if weighting == 1:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 2:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = Sim1[:, jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 3:
            M = cp.mean(Sim1, axis=1)

            M2 = cp.mean(overall, axis=1)

        S = cp.zeros((Sim1.shape[0], Ne))
        yprime = cp.zeros((Y.shape[0], Y.shape[1]))
        for jc in range(Ne):
            S[:, jc] = Sim1[:, jc] - M
            yprime[:, jc] = overall[:, jc] - M2
        Cydd = (yprime) / cp.sqrt(Ne - 1)
        Cdd = (S) / cp.sqrt(Ne - 1)

        GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
        inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
        Cdd = GDT.T @ GDT
        Cyd = Cydd @ GDT
        Usig, Sig, Vsig = cp.linalg.svd(
            (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
        )
        Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
        valuesig = Bsig[-1]  # last element
        valuesig = valuesig * 0.9999
        indices = (Bsig >= valuesig).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]

        (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)

        update_term = (
            (Cyd @ (X))
            @ (inv_CDd)
            @ (
                (
                    cp.tile(cp.asarray(True_dataa), Ne)
                    + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
                )
                - Sim1
            )
        )

        Ynew = Y + update_term
        sizeclem = cp.asarray(nx * ny * nz)
        if Yet == 0:
            updated_ensemble = cp.asnumpy(Ynew[:shapalla, :])
            updated_ensemblep = cp.asnumpy(Ynew[shapalla : 2 * shapalla, :])

        else:
            updated_ensemble = Ynew[:shapalla, :]
            updated_ensemblep = Ynew[shapalla : 2 * shapalla, :]

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior

        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest
        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)

        ensemble = Recover_image(updated_ensemble, Ne, nx, ny, nz, High_K1)
        ensemblep = Recover_imagep(updated_ensemblep, Ne, nx, ny, nz, High_P)
        if choice == 1:
            ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
            ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
        else:
            pass

        ensemble, ensemblep = honour2(
            ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
        )

        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_CAEE ENDED-------------------")

elif Technique_REKI == 3:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with continous Denoising Autoencoder\n"
    )
    print("Starting the History matching with ", str(Ne) + " Ensemble members")

    if Geostats == 1:
        bb = os.path.isfile("PACKETS/denosingautoencoder.h5")
        # bb2=os.path.isfile('denosingautoencoderp.h5')
        if not bb:
            if use_pretrained == 1:
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1S71PoGY3MY3lBuuqBahstXZTslZsvXTs",
                    "PACKETS/denosingautoencoder.h5",
                )
                print("...Downlaod completed.......")
            else:
                DenosingAutoencoder(
                    nx, ny, nz, High_K1, Low_K1
                )  # Learn for permeability
            # DenosingAutoencoderp(nx,ny,nz,machine_map,N_ens,High_P) #Learn for porosity
        else:
            pass
        bb2 = os.path.isfile("PACKETS/denosingautoencoderp.h5")
        if not bb2:
            if use_pretrained == 1:
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1HZ-LeVLWO9ZsDEde4LG11aOxKOt4uBVn",
                    "PACKETS/denosingautoencoderp.h5",
                )
                print("...Downlaod completed.......")
            else:
                DenosingAutoencoderp(nx, ny, nz, N_ens, High_P, High_K, Low_K)
        else:
            pass

    os.chdir(oldfolder)

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )
    ax = np.zeros((Nop, 1))

    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 0.0001
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass
    snn = 0
    ii = 0
    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep
        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)
        # executor.shutdown(wait=True)
        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "Initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        # Cini=CDd

        True_dataa = True_data
        CDd = cp.asarray(CDd)
        # Cini=cp.asarray(Cini)
        True_dataa = cp.asarray(True_dataa)
        Ddraw = cp.tile(True_dataa, Ne)
        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass
        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))
        sgsim = ensemble

        overall = cp.asarray(sgsim)

        Y = overall
        Sim1 = cp.asarray(simDatafinal)
        if weighting == 1:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 2:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = Sim1[:, jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 3:
            M = cp.mean(Sim1, axis=1)

            M2 = cp.mean(overall, axis=1)

        S = cp.zeros((Sim1.shape[0], Ne))
        yprime = cp.zeros((Y.shape[0], Y.shape[1]))
        for jc in range(Ne):
            S[:, jc] = Sim1[:, jc] - M
            yprime[:, jc] = overall[:, jc] - M2
        Cydd = (yprime) / cp.sqrt(Ne - 1)
        Cdd = (S) / cp.sqrt(Ne - 1)

        GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
        inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
        Cdd = GDT.T @ GDT
        Cyd = Cydd @ GDT
        Usig, Sig, Vsig = cp.linalg.svd(
            (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
        )
        Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
        valuesig = Bsig[-1]  # last element
        valuesig = valuesig * 0.9999
        indices = (Bsig >= valuesig).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]

        (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)

        update_term = (
            (Cyd @ (X))
            @ (inv_CDd)
            @ (
                (
                    cp.tile(cp.asarray(True_dataa), Ne)
                    + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
                )
                - Sim1
            )
        )

        Ynew = Y + update_term
        sizeclem = cp.asarray(nx * ny * nz)
        if Yet == 0:
            updated_ensemble = cp.asnumpy(Ynew[:sizeclem, :])

        else:
            updated_ensemble = Ynew[:sizeclem, :]

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior

        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest
        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)
        ensemble = updated_ensemble

        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        clfye = MinMaxScaler(feature_range=(Low_P, High_P))
        (clfye.fit(ensemble))
        ensemblep = clfye.transform(ensemble)

        ensemble, ensemblep = honour2(
            ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
        )
        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_DA ENDED--------------------")

elif Technique_REKI == 4:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Normal Score Transformation \n"
    )
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    # quantileperm = QuantileTransformer(output_distribution='normal')
    # quantileporo = QuantileTransformer(output_distribution='normal')

    os.chdir(oldfolder)

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )
    ax = np.zeros((Nop, 1))

    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 0.0001
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass
    snn = 0
    ii = 0
    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    # Plot_RSM_single(True_mat,'Historical.png')
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    # quantileperm.fit(ensemble)
    # quantileporo.fit(ensemblep)
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep
        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)
        # executor.shutdown(wait=True)
        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "Initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        Cini = CDd

        True_dataa = True_data
        CDd = cp.asarray(CDd)
        # Cini=cp.asarray(Cini)
        True_dataa = cp.asarray(True_dataa)
        Ddraw = cp.tile(True_dataa, Ne)
        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass
        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior

        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest

        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)
        ensemble, ensemblep = REKI_ASSIMILATION_NORMAL_SCORE(
            ensemble,
            ensemblep,
            simDatafinal,
            alpha,
            True_dataa,
            Ne,
            pertubations,
            Yet,
            nx,
            ny,
            nz,
            High_K1,
            Low_K1,
            High_P,
            Low_P,
            CDd,
        )
        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)
    print(
        "--------------------SECTION ADAPTIVE REKI_NORMAL_SCORE ENDED--------------------"
    )

elif Technique_REKI == 5:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Convolution Autoencoder \
Parametrisation with Generative adverserail network prior\n"
    )
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    print("-------------------------learn Autoencoder------------------------")
    bb = os.path.isfile("PACKETS/autoencoder.h5")
    # bb2=os.path.isfile('autoencoderp.h5')
    if not bb:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1AEZPXAoJaC88dY9T-FFhLwJ_9U-Pl3tq", "PACKETS/encoder.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1u4Vo7XdyZQq4Z0_jNsoqG4QNWslAwyBv", "PACKETS/decoder.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "16Be2bIUWhiq8RBT48-baIHGU47XSnixS", "PACKETS/autoencoder.h5"
            )
            print("...Downlaod completed.......")

        else:
            Autoencoder2(nx, ny, nz, High_K1, Low_K1)  # Learn for permeability
    else:
        pass
    print("--------------------Section Ended--------------------------------")
    print(" Learn the GAN module")
    aa = os.path.isfile("PACKETS/generator.h5")
    # aa2=aa=os.path.isfile('generatorp.h5')
    if not aa:
        if not os.path.exists("/images/"):
            os.makedirs("/images/")
        else:
            pass
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1DSKdVPpGKPeX-h6niGOq0Rf3TUId-AiL", "PACKETS/generator.h5"
            )
            print("...Downlaod completed.......")
        else:
            WGAN_LEARNING(High_K1, Low_K, High_P, Low_P, nx, ny, nz)
    else:
        pass
    generator_map = load_model("PACKETS/generator.h5")
    # generatorp_map = load_model('generatorp.h5')

    # clfy_gen = pickle.load(open('clfy_gen.asv' , 'rb'))
    # clfyp_gen = pickle.load(open('clfyp_gen.asv' , 'rb'))

    os.chdir(oldfolder)

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    # ensemble,ensemblep=honour2(ensemblep,\
    #                              ensemble,nx,ny,nz,N_ens,\
    #                                  High_K,Low_K,High_P,Low_P)
    ax = np.zeros((Nop, 1))

    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 0.0001
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass
    snn = 0
    ii = 0
    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    # Plot_RSM_single(True_mat,'Historical.png')
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep
        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)
        # executor.shutdown(wait=True)
        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "Initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        # Cini=CDd

        True_dataa = True_data
        CDd = cp.asarray(CDd)
        # Cini=cp.asarray(Cini)
        True_dataa = cp.asarray(True_dataa)
        Ddraw = cp.tile(True_dataa, Ne)
        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))

        if ii == 0:
            # sgsim=ensemble
            encoded = Get_Latent(ensemble, Ne, nx, ny, nz, High_K1)
            # encodedp=Get_Latentp(ensemblep,Ne,nx,ny,nz)
        else:
            encoded = updated_ensemble
            # encodedp=updated_ensemblep
        shapalla = encoded.shape[0]

        overall = cp.asarray(encoded)

        Y = overall
        Sim1 = cp.asarray(simDatafinal)

        if weighting == 1:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 2:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = Sim1[:, jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 3:
            M = cp.mean(Sim1, axis=1)

            M2 = cp.mean(overall, axis=1)

        S = cp.zeros((Sim1.shape[0], Ne))
        yprime = cp.zeros((Y.shape[0], Y.shape[1]))
        for jc in range(Ne):
            S[:, jc] = Sim1[:, jc] - M
            yprime[:, jc] = overall[:, jc] - M2
        Cydd = (yprime) / cp.sqrt(Ne - 1)
        Cdd = (S) / cp.sqrt(Ne - 1)

        GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
        inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
        Cdd = GDT.T @ GDT
        Cyd = Cydd @ GDT
        Usig, Sig, Vsig = cp.linalg.svd(
            (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
        )
        Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
        valuesig = Bsig[-1]  # last element
        valuesig = valuesig * 0.9999
        indices = (Bsig >= valuesig).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]

        (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)

        update_term = (
            (Cyd @ (X))
            @ (inv_CDd)
            @ (
                (
                    cp.tile(cp.asarray(True_dataa), Ne)
                    + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
                )
                - Sim1
            )
        )

        Ynew = Y + update_term

        sizeclem = cp.asarray(nx * ny * nz)
        if Yet == 0:
            updated_ensemble = cp.asnumpy(Ynew[:shapalla, :])
            # updated_ensemblep=cp.asnumpy(Ynew[shapalla:2*shapalla,:])

        else:
            updated_ensemble = Ynew[:shapalla, :]
            # updated_ensemblep=Ynew[shapalla:2*shapalla,:]

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior

        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest
        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)

        # ensemble=Recover_image(updated_ensemble,Ne,nx,ny,nz)

        ensemble = AE_GAN(updated_ensemble, N_ens, nx, ny, nz, generator_map, High_K1)
        # ensemblep=AE_GANP(updated_ensemblep,N_ens,nx,ny,nz,generatorp_map)
        ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
        # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        clfye = MinMaxScaler(feature_range=(Low_P, High_P))
        (clfye.fit(ensemble))
        ensemblep = clfye.transform(ensemble)
        # ensemblep=Recover_imagep(updated_ensemblep,Ne,nx,ny,nz)

        ensemble, ensemblep = honour2(
            ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
        )

        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_GAN ENDED-------------------")

elif Technique_REKI == 6:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Generative\
adverserail network prior for permeability field alone\n"
    )
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    print(" Learn the permeability field GAN module")

    aa = os.path.isfile("PACKETS/generator.h5")
    if not aa:
        if not os.path.exists("/images/"):
            os.makedirs("/images/")
        else:
            pass
        if use_pretrained == 1:
            print("....Downloadinenara Please hold.........")
            download_file_from_google_drive(
                "1DSKdVPpGKPeX-h6niGOq0Rf3TUId-AiL", "PACKETS/generator.h5"
            )
            print("...Downlaod completed.......")
        else:
            WGAN_LEARNING_PERM(High_K1, Low_K, nx, ny, nz)
    else:
        pass

    generator_map = load_model("PACKETS/generator.h5")
    print("")
    print("-------------------------------------------------------------")

    os.chdir(oldfolder)

    noise = np.random.normal(0, 1, (N_ens, 20 * 20 * 4))

    red_kini = noise.T
    ini_ensemble = AE_GAN_PERM(red_kini, Ne, nx, ny, nz, generator_map, High_K1)
    red_ensemble = red_kini
    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )
    ax = np.zeros((Nop, 1))

    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 0.0001
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass
    snn = 0
    ii = 0
    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    # Plot_RSM_single(True_mat,'Historical.png')
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep
        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)
        # executor.shutdown(wait=True)
        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "Initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        # Cini=CDd

        True_dataa = True_data
        CDd = cp.asarray(CDd)
        # Cini=cp.asarray(Cini)
        True_dataa = cp.asarray(True_dataa)
        Ddraw = cp.tile(True_dataa, Ne)
        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))
        sgsim = red_ensemble
        encoded = sgsim

        shapalla = encoded.shape[0]

        overall = cp.asarray(encoded)

        Y = overall
        Sim1 = cp.asarray(simDatafinal)

        if weighting == 1:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 2:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = Sim1[:, jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 3:
            M = cp.mean(Sim1, axis=1)

            M2 = cp.mean(overall, axis=1)

        S = cp.zeros((Sim1.shape[0], Ne))
        yprime = cp.zeros((Y.shape[0], Y.shape[1]))
        for jc in range(Ne):
            S[:, jc] = Sim1[:, jc] - M
            yprime[:, jc] = overall[:, jc] - M2
        Cydd = (yprime) / cp.sqrt(Ne - 1)
        Cdd = (S) / cp.sqrt(Ne - 1)

        GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
        inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
        Cdd = GDT.T @ GDT
        Cyd = Cydd @ GDT
        Usig, Sig, Vsig = cp.linalg.svd(
            (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
        )
        Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
        valuesig = Bsig[-1]  # last element
        valuesig = valuesig * 0.9999
        indices = (Bsig >= valuesig).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]

        (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)

        update_term = (
            (Cyd @ (X))
            @ (inv_CDd)
            @ (
                (
                    cp.tile(cp.asarray(True_dataa), Ne)
                    + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
                )
                - Sim1
            )
        )

        Ynew = Y + update_term

        sizeclem = cp.asarray(nx * ny * nz)
        if Yet == 0:
            updated_ensemble = cp.asnumpy(Ynew[:shapalla, :])

        else:
            updated_ensemble = Ynew[:shapalla, :]

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior

        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest

        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)

        # ensemble=Recover_image(updated_ensemble,Ne,nx,ny,nz)
        red_ensemble = updated_ensemble
        ensemble = AE_GAN_PERM(updated_ensemble, Ne, nx, ny, nz, generator_map, High_K1)
        ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
        # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        clfye = MinMaxScaler(feature_range=(Low_P, High_P))
        (clfye.fit(ensemble))
        ensemblep = clfye.transform(ensemble)
        ensemble, ensemblep = honour2(
            ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
        )
        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_GAN_PERM ENDED-------------------")

elif Technique_REKI == 7:
    print("")
    print("Adaptive Regularised Ensemble Kalman Inversion with KMEANS Parametrisation")
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")

    os.chdir(oldfolder)
    print("Learn KMEANS Over complete dictionary of the permeability field")
    print("")
    bb = os.path.isfile("PACKETS/Dictionary_Perm_Kmeans.mat")
    if not bb:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1rv152M5sMIhluyjGAVdpviVVzCVXHD5j",
                "PACKETS/Dictionary_Perm_Kmeans.mat",
            )
            print("...Downlaod completed.......")
            matt = sio.loadmat("PACKETS/Dictionary_Perm_Kmeans.mat")
            Dicclem = matt["Z"]
        else:
            filename = "PACKETS/Ganensemble.mat"
            mat = sio.loadmat(filename)
            ini_ensemblef = mat["Z"]

            kmeans = MiniBatchKMeans(n_clusters=Ne, random_state=0, max_iter=2000)
            kmeans = kmeans.fit(ini_ensemblef.T)
            Dicclem = kmeans.cluster_centers_.T
            Dicclem = Make_Bimodal(Dicclem, 2, High_K1, Low_K1)
            sio.savemat("PACKETS/Dictionary_Perm_Kmeans.mat", {"Z": Dicclem})
    else:
        matt = sio.loadmat("PACKETS/Dictionary_Perm_Kmeans.mat")
        Dicclem = matt["Z"]

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )
    ax = np.zeros((Nop, 1))

    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 0.0001
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass
    snn = 0
    ii = 0
    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    # Plot_RSM_single(True_mat,'Historical.png')
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep
        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)
        # executor.shutdown(wait=True)
        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "Initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        # Cini=CDd
        True_dataa = True_data

        CDd = cp.asarray(CDd)
        # Cini=cp.asarray(Cini)
        True_dataa = cp.asarray(True_dataa)
        Ddraw = cp.tile(True_dataa, Ne)
        # pertubations=pertubations.T
        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))
        sgsim = ensemble
        ensembledct = cp.linalg.lstsq(cp.asarray(Dicclem), cp.asarray(sgsim))[0]
        shapalla = ensembledct.shape[0]

        overall = ensembledct
        Y = overall
        Sim1 = cp.asarray(simDatafinal)

        if weighting == 1:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 2:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = Sim1[:, jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 3:
            M = cp.mean(Sim1, axis=1)

            M2 = cp.mean(overall, axis=1)
        S = cp.zeros((Sim1.shape[0], Ne))
        yprime = cp.zeros((Y.shape[0], Y.shape[1]))
        for jc in range(Ne):
            S[:, jc] = Sim1[:, jc] - M
            yprime[:, jc] = overall[:, jc] - M2
        Cydd = (yprime) / cp.sqrt(Ne - 1)
        Cdd = (S) / cp.sqrt(Ne - 1)

        GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
        inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
        Cdd = GDT.T @ GDT
        Cyd = Cydd @ GDT
        Usig, Sig, Vsig = cp.linalg.svd(
            (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
        )
        Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
        valuesig = Bsig[-1]  # last element
        valuesig = valuesig * 0.9999
        indices = (Bsig >= valuesig).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]

        (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)

        update_term = (
            (Cyd @ (X))
            @ (inv_CDd)
            @ (
                (
                    cp.tile(cp.asarray(True_dataa), Ne)
                    + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
                )
                - Sim1
            )
        )
        Ynew = Y + update_term
        sizeclem = cp.asarray(nx * ny * nz)
        if Yet == 0:
            updated_ensembleksvd = cp.asnumpy(Ynew[:shapalla, :])
            # updated_ensembleksvdp=cp.asnumpy(Ynew[shapalla:2*shapalla,:])
        else:
            updated_ensembleksvd = Ynew[:shapalla, :]
            # updated_ensembleksvdp=(Ynew[shapalla:2*shapalla,:])

        updated_ensemble = Dicclem @ updated_ensembleksvd
        updated_ensemble[updated_ensemble <= Low_K1] = Low_K1
        updated_ensemble[updated_ensemble >= High_K1] = High_K1
        # updated_ensemblep=Recover_Dictionary_Saarse(Dicclemp,\
        #                            updated_ensembleksvdp,Low_P,High_P)

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest

            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior
        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest

        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)

        ensemble = updated_ensemble
        ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)

        # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        clfye = MinMaxScaler(feature_range=(Low_P, High_P))
        (clfye.fit(ensemble))
        ensemblep = clfye.transform(ensemble)
        # ensemblep=np.nan_to_num(ensemblep, copy=True, nan=Low_P)
        ensemble, ensemblep = honour2(
            ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
        )
        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)
    print("--------------------SECTION ADAPTIVE REKI_KMEANS ENDED-------------------")
elif Technique_REKI == 8:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Variational Convolution Autoencoder \
Parametrisation\n"
    )
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    print(
        "-------------------------learn Variational Autoencoder------------------------"
    )
    # sizedct=int(input('Enter the percentage of required latent dimension\
    # coeficients (1%-5% (2%) is optimum): '))/100
    latent_dim = int(100)  # int(np.ceil(int(sizedct*nx*ny*nz)))
    # aa=os.path.isfile("my_model")
    # if aa==False:
    print("Learn VAE for permeability field")
    vae = Variational_Autoencoder(latent_dim, nx, ny, nz, High_K1, Low_K1, 1)
    print("Completed VAE learning for permeability field")
    print("")
    print("Learn VAE for porosity field")
    vaep = Variational_Autoencoder(latent_dim, nx, ny, nz, High_P, Low_P, 2)
    print("Completed VAE learning for porosity field")
    #     vae.save("my_model")
    # else:
    #     vae = tensorflow.keras.models.load_models("my_model")

    os.chdir(oldfolder)

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)

    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)

    # ensemble,ensemblep=honour2(ensemblep,\
    #                              ensemble,nx,ny,nz,N_ens,\
    #                                  High_K,Low_K,High_P,Low_P)

    print("Check Variational Convolutional Autoencoder reconstruction")
    small = Get_LatentV(ensemble, Ne, nx, ny, nz, latent_dim, vae, High_K1, 1)
    recc = Recover_imageV(small, Ne, nx, ny, nz, latent_dim, vae, High_K1, 1)
    recc[recc <= Low_K1] = Low_K1
    recc[recc >= High_K1] = High_K1

    smallp = Get_LatentV(ensemblep, Ne, nx, ny, nz, latent_dim, vaep, High_P, 2)
    reccp = Recover_imageV(smallp, Ne, nx, ny, nz, latent_dim, vaep, High_P, 2)
    reccp[reccp <= Low_P] = Low_P
    reccp[reccp >= High_P] = High_P

    dimms = (small.shape[0] / ensemble.shape[0]) * 100
    dimms = round(dimms, 3)

    recover = np.reshape(recc[:, 0], (nx, ny, nz), "F")
    origii = np.reshape(ensemble[:, 0], (nx, ny, nz), "F")

    recoverp = np.reshape(reccp[:, 0], (nx, ny, nz), "F")
    origiip = np.reshape(ensemblep[:, 0], (nx, ny, nz), "F")

    plt.figure(figsize=(20, 20))
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.subplot(3, 4, 1)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 0], cmap="jet")
    plt.title("Recovered Layer 1 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 4, 5)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 1], cmap="jet")
    plt.title("Recovered Layer 2", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 4, 9)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 2], cmap="jet")
    plt.title("Recovered Layer 3 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 4, 2)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 0], cmap="jet")
    plt.title("original Layer 1", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 4, 6)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 1], cmap="jet")
    plt.title("original Layer 2", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 4, 10)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 2], cmap="jet")
    plt.title("original Layer 3", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 4, 3)
    plt.pcolormesh(XX.T, YY.T, recoverp[:, :, 0], cmap="jet")
    plt.title("Recovered Layer 1 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("porosity", fontsize=13)
    plt.clim(Low_P, High_P)

    plt.subplot(3, 4, 7)
    plt.pcolormesh(XX.T, YY.T, recoverp[:, :, 1], cmap="jet")
    plt.title("Recovered Layer 2 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("porosity", fontsize=13)
    plt.clim(Low_P, High_P)

    plt.subplot(3, 4, 11)
    plt.pcolormesh(XX.T, YY.T, recoverp[:, :, 2], cmap="jet")
    plt.title("Recovered Layer 1 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("porosity", fontsize=13)
    plt.clim(Low_P, High_P)

    plt.subplot(3, 4, 4)
    plt.pcolormesh(XX.T, YY.T, origiip[:, :, 0], cmap="jet")
    plt.title("original Layer 1", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("porosity", fontsize=13)
    plt.clim(Low_P, High_P)

    plt.subplot(3, 4, 8)
    plt.pcolormesh(XX.T, YY.T, origiip[:, :, 1], cmap="jet")
    plt.title("original Layer 2", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("porosity", fontsize=13)
    plt.clim(Low_P, High_P)

    plt.subplot(3, 4, 12)
    plt.pcolormesh(XX.T, YY.T, origiip[:, :, 2], cmap="jet")
    plt.title("original Layer 3", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("porosity", fontsize=13)
    plt.clim(Low_P, High_P)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ttitle = "Parameter Recovery with " + str(dimms) + "% of original value"
    plt.suptitle(ttitle, fontsize=20)
    os.chdir("HM_RESULTS")
    plt.savefig("Recover_Comparison.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()
    ax = np.zeros((Nop, 1))

    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 0.0001
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass
    snn = 0
    ii = 0
    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    # Plot_RSM_single(True_mat,'Historical.png')
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep
        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)
        # executor.shutdown(wait=True)
        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "Initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        # Cini=CDd

        True_dataa = True_data
        CDd = cp.asarray(CDd)
        # Cini=cp.asarray(Cini)
        True_dataa = cp.asarray(True_dataa)
        Ddraw = cp.tile(True_dataa, Ne)
        # pertubations=pertubations.T
        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))
        if ii == 0:
            sgsim = ensemble
            encoded = Get_LatentV(sgsim, N_ens, nx, ny, nz, latent_dim, vae, High_K1, 1)
            encodedp = Get_LatentV(
                ensemblep, N_ens, nx, ny, nz, latent_dim, vaep, High_P, 2
            )
        else:
            # sgsim=ensemble
            encoded = updated_ensemble
            encodedp = updated_ensemblep

        # encodedp=Get_Latentp(ensemblep,Ne,nx,ny,nz)
        shapalla = encoded.shape[0]

        overall = cp.vstack([cp.asarray(encoded), cp.asarray(encodedp)])

        Y = overall
        Sim1 = cp.asarray(simDatafinal)

        if weighting == 1:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 2:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = Sim1[:, jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 3:
            M = cp.mean(Sim1, axis=1)

            M2 = cp.mean(overall, axis=1)

        S = cp.zeros((Sim1.shape[0], Ne))
        yprime = cp.zeros((Y.shape[0], Y.shape[1]))
        for jc in range(Ne):
            S[:, jc] = Sim1[:, jc] - M
            yprime[:, jc] = overall[:, jc] - M2
        Cydd = (yprime) / cp.sqrt(Ne - 1)
        Cdd = (S) / cp.sqrt(Ne - 1)

        GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
        inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
        Cdd = GDT.T @ GDT
        Cyd = Cydd @ GDT
        Usig, Sig, Vsig = cp.linalg.svd(
            (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
        )
        Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
        valuesig = Bsig[-1]  # last element
        valuesig = valuesig * 0.9999
        indices = (Bsig >= valuesig).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]

        (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)

        update_term = (
            (Cyd @ (X))
            @ (inv_CDd)
            @ (
                (
                    cp.tile(cp.asarray(True_dataa), Ne)
                    + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
                )
                - Sim1
            )
        )

        Ynew = Y + update_term
        sizeclem = cp.asarray(nx * ny * nz)
        if Yet == 0:
            updated_ensemble = cp.asnumpy(Ynew[:shapalla, :])
            updated_ensemblep = cp.asnumpy(Ynew[shapalla : 2 * shapalla, :])

        else:
            updated_ensemble = Ynew[:shapalla, :]
            updated_ensemblep = Ynew[shapalla : 2 * shapalla, :]

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior

        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest

        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)

        ensemble = Recover_imageV(
            updated_ensemble, Ne, nx, ny, nz, latent_dim, vae, High_K1, 1
        )
        ensemble[ensemble <= Low_K1] = Low_K1
        ensemble[ensemble >= High_K1] = High_K1

        ensemblep = Recover_imageV(
            updated_ensemblep, Ne, nx, ny, nz, latent_dim, vaep, High_P, 2
        )
        ensemblep[ensemblep <= Low_P] = Low_P
        ensemblep[ensemblep >= High_P] = High_P

        # clfye = MinMaxScaler(feature_range=(Low_P,High_P))
        # (clfye.fit(ensemble))
        # ensemblep = (clfye.transform(ensemble))

        ensemble, ensemblep = honour2(
            ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
        )
        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_VAE ENDED-------------------")
elif Technique_REKI == 9:
    print("")
    print("Adaptive Regularised Ensemble Kalman Inversion with DCT Parametrisation")
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")

    os.chdir(oldfolder)

    sizedct = None
    while True:
        sizedct = int(
            input(
                "Enter the percentage of required DCT \
    coeficients (15%-30%): "
            )
        )
        if (sizedct > 50) or (sizedct < 10):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 10-50")
        else:
            break
    sizedct = sizedct / 100
    size1, size2 = int(cp.ceil(int(sizedct * nx))), int(cp.ceil(int(sizedct * ny)))
    print("")

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    print("Check DCT reconstruction")
    small = dct22(ensemble, Ne, nx, ny, nz, size1, size2)
    recc = idct22(small, Ne, nx, ny, nz, size1, size2)
    dimms = (small.shape[0] / ensemble.shape[0]) * 100
    dimms = round(dimms, 3)

    recover = np.reshape(recc[:, 0], (nx, ny, nz), "F")
    origii = np.reshape(ensemble[:, 0], (nx, ny, nz), "F")

    plt.figure(figsize=(20, 20))
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.subplot(3, 2, 1)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 0], cmap="jet")
    plt.title("Recovered - Layer 1", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 3)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 1], cmap="jet")
    plt.title("Recovered - Layer 2", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 5)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 2], cmap="jet")
    plt.title("Recovered - Layer 3", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 2)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 0], cmap="jet")
    plt.title("original -Layer 1", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 4)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 1], cmap="jet")
    plt.title("original -Layer 2", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(3, 2, 6)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 2], cmap="jet")
    plt.title("original -Layer 3", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ttitle = "Parameter Recovery with " + str(dimms) + "% of original value"
    plt.suptitle(ttitle, fontsize=20)
    os.chdir("HM_RESULTS")
    plt.savefig("Recover_Comparison.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()

    ax = np.zeros((Nop, 1))

    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 0.0001
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass
    snn = 0
    ii = 0
    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep
        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)
        # executor.shutdown(wait=True)
        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        # Cini=CDd
        True_dataa = True_data

        CDd = cp.asarray(CDd)
        # Cini=cp.asarray(Cini)
        True_dataa = cp.asarray(True_dataa)
        Ddraw = cp.tile(True_dataa, Ne)
        # pertubations=pertubations.T
        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (np.linalg.inv(CDd)) @ (Dd - simDatafinal)),
            axis=1,
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))
        sgsim = ensemble

        ensembledct = dct22(ensemble, Ne, nx, ny, nz, size1, size2)
        ensembledctp = dct22(ensemblep, Ne, nx, ny, nz, size1, size2)

        shapalla = ensembledct.shape[0]

        # overall=cp.asarray(ensembledct)
        overall = cp.vstack([cp.asarray(ensembledct), cp.asarray(ensembledctp)])
        Y = overall
        Sim1 = cp.asarray(simDatafinal)

        if weighting == 1:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 2:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = Sim1[:, jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 3:
            M = cp.mean(Sim1, axis=1)

            M2 = cp.mean(overall, axis=1)
        S = cp.zeros((Sim1.shape[0], Ne))
        yprime = cp.zeros((Y.shape[0], Y.shape[1]))
        for jc in range(Ne):
            S[:, jc] = Sim1[:, jc] - M
            yprime[:, jc] = overall[:, jc] - M2
        Cydd = (yprime) / cp.sqrt(Ne - 1)
        Cdd = (S) / cp.sqrt(Ne - 1)

        GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
        inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
        Cdd = GDT.T @ GDT
        Cyd = Cydd @ GDT
        Usig, Sig, Vsig = cp.linalg.svd(
            (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
        )
        Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
        valuesig = Bsig[-1]  # last element
        valuesig = valuesig * 0.9999
        indices = (Bsig >= valuesig).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]

        (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)

        update_term = (
            (Cyd @ (X))
            @ (inv_CDd)
            @ (
                (
                    cp.tile(cp.asarray(True_dataa), Ne)
                    + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
                )
                - Sim1
            )
        )
        Ynew = Y + update_term
        sizeclem = cp.asarray(nx * ny * nz)
        if Yet == 0:
            updated_ensembleksvd = cp.asnumpy(Ynew[:shapalla, :])
            updated_ensembleksvdp = cp.asnumpy(Ynew[shapalla : 2 * shapalla, :])

        else:
            updated_ensembleksvd = Ynew[:shapalla, :]
            updated_ensembleksvdp = Ynew[shapalla : 2 * shapalla, :]

        updated_ensemble = idct22(updated_ensembleksvd, Ne, nx, ny, nz, size1, size2)

        updated_ensemblep = idct22(updated_ensembleksvdp, Ne, nx, ny, nz, size1, size2)

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest
            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior

        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest

        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)

        ensemble = updated_ensemble
        ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)

        ensemblep = updated_ensemblep
        ensemblep = np.nan_to_num(ensemblep, copy=True, nan=Low_P)

        # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
        # clfye = MinMaxScaler(feature_range=(Low_P,High_P))
        # (clfye.fit(ensemble))
        # ensemblep = (clfye.transform(ensemble))

        # ensemblep=np.nan_to_num(ensemblep, copy=True, nan=Low_P)
        ensemble, ensemblep = honour2(
            ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
        )

        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_DCT ENDED-------------------")

elif Technique_REKI == 10:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Logit \
post-processing\n"
    )
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    Low_precision = 1e-10
    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)

    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )
    ax = np.zeros((Nop, 1))
    for iq in range(Nop):
        if True_data[iq, :] == 0:
            ax[iq, :] = 1e-5
        elif (True_data[iq, :] > 0) and (True_data[iq, :] <= 10000):
            ax[iq, :] = sqrt(noise_level * True_data[iq, :])
        else:
            ax[iq, :] = 1

    R = ax**2

    CDd = np.diag(np.reshape(R, (-1,)))
    Cini = CDd
    Cini = cp.asarray(Cini)
    pertubations = cp.random.multivariate_normal(
        cp.ravel(cp.zeros((Nop, 1))), Cini, Ne, method="eigh", dtype=cp.float32
    )

    pertubations = pertubations.T
    if Yet == 0:
        pertubations = cp.asnumpy(pertubations)
    else:
        pass

    snn = 0
    ii = 0
    print("Read Historical data")
    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    alpha_big = []
    mean_cost = []
    best_cost = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")

        ini_K = ensemble
        ini_p = ensemblep
        executor = get_reusable_executor(max_workers=8)
        with executor:
            Parallel(n_jobs=8, backend="loky", verbose=50)(
                delayed(Simulator)(
                    plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
                )
                for kk in range(Ne)
            )
            executor.shutdown(wait=False)
        # executor.shutdown(wait=True)
        simDatafinal, predMatrix = RSM_extract(
            Ne, "Realisation", oldfolder, scalei, scalei2
        )

        if ii == 0:
            os.chdir("HM_RESULTS")
            Plot_RSM(predMatrix, True_mat, "Initial.png", Ne)
            os.chdir(oldfolder)
        else:
            pass

        True_dataa = True_data

        CDd = cp.asarray(CDd)

        True_dataa = cp.asarray(True_dataa)

        Ddraw = cp.tile(True_dataa, Ne)

        Dd = Ddraw  # + pertubations
        if Yet == 0:
            CDd = cp.asnumpy(CDd)
            Dd = cp.asnumpy(Dd)
            Ddraw = cp.asnumpy(Ddraw)
            True_dataa = cp.asnumpy(True_dataa)
        else:
            pass
        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (inv(CDd)) @ (Dd - simDatafinal)), axis=1
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star = np.mean(yyy, axis=0)

        yyy = np.mean(
            0.5 * ((Dd - simDatafinal).T @ (inv(CDd)) @ (Dd - simDatafinal)), axis=1
        )
        yyy = yyy.reshape(-1, 1)
        yyy = np.nan_to_num(yyy, copy=True, nan=0)
        alpha_star2 = (np.std(yyy, axis=0)) ** 2

        leftt = True_data.shape[0] / (2 * alpha_star)
        rightt = np.sqrt(True_data.shape[0] / (2 * alpha_star2))
        chok = min(max(leftt, rightt), 1 - snn)

        alpha = 1 / chok
        alpha_big.append(alpha)

        print("alpha = " + str(alpha))
        print("sn = " + str(snn))

        sgsim = Logitt(Low_K1, High_K1, ensemble, Low_precision)
        sgsimp = Logitt(Low_P, High_P, ensemblep, Low_precision)

        overall = cp.vstack([cp.asarray(sgsim), cp.asarray(sgsimp)])

        Y = overall
        Sim1 = cp.asarray(simDatafinal)

        if weighting == 1:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = (Sim1[:, jc]) * weight1[jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 2:
            weights = Get_weighting(simDatafinal, True_dataa)
            weight1 = cp.asarray(weights)

            Sp = cp.zeros((Sim1.shape[0], Ne))
            yp = cp.zeros((Y.shape[0], Y.shape[1]))

            for jc in range(Ne):
                Sp[:, jc] = Sim1[:, jc]
                yp[:, jc] = (overall[:, jc]) * weight1[jc]

            M = cp.mean(Sp, axis=1)

            M2 = cp.mean(yp, axis=1)
        if weighting == 3:
            M = cp.mean(Sim1, axis=1)

            M2 = cp.mean(overall, axis=1)

        S = cp.zeros((Sim1.shape[0], Ne))
        yprime = cp.zeros((Y.shape[0], Y.shape[1]))

        for jc in range(Ne):
            S[:, jc] = Sim1[:, jc] - M
            yprime[:, jc] = overall[:, jc] - M2
        Cydd = (yprime) / cp.sqrt(Ne - 1)
        Cdd = (S) / cp.sqrt(Ne - 1)

        GDT = Cdd.T @ ((cp.linalg.inv(cp.asarray(CDd))) ** (0.5))
        inv_CDd = (cp.linalg.inv(cp.asarray(CDd))) ** (0.5)
        Cdd = GDT.T @ GDT
        Cyd = Cydd @ GDT
        Usig, Sig, Vsig = cp.linalg.svd(
            (Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), full_matrices=False
        )
        Bsig = cp.cumsum(Sig, axis=0)  # vertically addition
        valuesig = Bsig[-1]  # last element
        valuesig = valuesig * 0.9999
        indices = (Bsig >= valuesig).ravel().nonzero()
        toluse = Sig[indices]
        tol = toluse[0]

        (V, X, U) = pinvmatt((Cdd + (cp.asarray(alpha) * cp.eye(CDd.shape[1]))), tol)

        update_term = (
            (Cyd @ (X))
            @ (inv_CDd)
            @ (
                (
                    cp.tile(cp.asarray(True_dataa), Ne)
                    + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
                )
                - Sim1
            )
        )

        Ynew = Y + update_term
        sizeclem = cp.asarray(nx * ny * nz)
        if Yet == 0:
            updated_ensemble = cp.asnumpy(Ynew[:sizeclem, :])
            updated_ensemblep = cp.asnumpy(Ynew[sizeclem : 2 * sizeclem, :])

        else:
            updated_ensemble = Ynew[:sizeclem, :]
            updated_ensemblep = Ynew[sizeclem : 2 * sizeclem, :]

        updated_ensemble = 1 / (1 + np.exp(-updated_ensemble))
        updated_ensemblep = 1 / (1 + np.exp(-updated_ensemblep))

        if ii == 0:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("Initial RMSE of the ensemble best = : " + str(tinukebest) + "... .")
            cc_ini = cc
            tinumeanprior = tinuke
            tinubestprior = tinukebest

            best_cost_mean = tinumeanprior
            best_cost_best = tinubestprior
        else:
            simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
            tinuke = (
                (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble mean = : " + str(tinuke) + "... .")
            aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
            clem = np.argmin(cc)
            simmbest = simDatafinal[:, clem].reshape(-1, 1)
            tinukebest = (
                (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
            ) / True_dataa.shape[0]
            print("RMSE of the ensemble best = : " + str(tinukebest) + "... .")

            if tinuke < tinumeanprior:
                print(
                    "ensemble mean cost decreased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke > tinumeanprior:
                print(
                    "ensemble mean cost increased by = : "
                    + str(abs(tinuke - tinumeanprior))
                    + "... ."
                )

            if tinuke == tinumeanprior:
                print("No change in ensemble mean cost")

            if tinukebest > tinubestprior:
                print(
                    "ensemble best cost increased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest < tinubestprior:
                print(
                    "ensemble best cost decreased by = : "
                    + str(abs(tinukebest - tinubestprior))
                    + "... ."
                )

            if tinukebest == tinubestprior:
                print("No change in ensemble best cost")

            tinumeanprior = tinuke
            tinubestprior = tinukebest
        if (best_cost_mean > tinuke) and (best_cost_best > tinukebest):
            print("**********************************************************")
            print("Ensemble of permeability and porosity saved             ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))
            # torch.save(model_pressure.state_dict(), oldfolder + '/pressure_model.pth')
            # torch.save(model_saturation.state_dict(), oldfolder + '/saturation_model.pth')
            best_cost_mean = tinuke
            best_cost_best = tinukebest
            use_k = ensemble
            use_p = ensemblep
        else:
            print("**********************************************************")
            print("Ensemble of permeability and porosity NOT saved        ")
            print("Current best mean cost = " + str(best_cost_mean))
            print("Current iteration mean cost = " + str(tinuke))
            print("Current best MAP cost = " + str(best_cost_best))
            print("Current iteration MAP cost = " + str(tinukebest))

        mean_cost.append(tinuke)
        best_cost.append(tinukebest)

        ensemble_bestK.append(ini_K[:, clem].reshape(-1, 1))
        ensemble_bestP.append(ini_p[:, clem].reshape(-1, 1))

        ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
        ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))

        ensembles.append(ensemble)
        ensemblesp.append(ensemblep)

        ensemble = Get_new_K(Low_K1, High_K1, updated_ensemble)
        ensemblep = Get_new_K(Low_P, High_P, updated_ensemblep)
        if snn > 1:
            print("Converged")
            break
        else:
            pass

        if ii == Termm - 1:
            print("Did not converge, Max Iteration reached")
            break
        else:
            pass

        ii = ii + 1
        snn = snn + chok
    mean_cost.append(tinuke)
    best_cost.append(tinukebest)
    meancost1 = np.vstack(mean_cost)
    chm = np.argmin(meancost1)
    print("****************************************************************")
    best_cost1 = np.vstack(best_cost)
    chb = np.argmin(best_cost1)

    ensemble_bestK = np.hstack(ensemble_bestK)
    ensemble_bestP = np.hstack(ensemble_bestP)

    yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
    yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)

    ensemble_meanK = np.hstack(ensemble_meanK)
    ensemble_meanP = np.hstack(ensemble_meanP)

    yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
    yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)

    all_ensemble = ensembles[chm]
    all_ensemblep = ensemblesp[chm]

    ensemble, ensemblep = honour2(
        use_p, use_k, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    all_ensemble, all_ensemblep = honour2(
        all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    if (choice == 1) and (Geostats == 1):
        ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
        ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
    else:
        pass

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, ensemble[:, kk], ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    simDatafinal, predMatrix = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan, all_ensemble[:, kk], all_ensemblep[:, kk], "Realisation" + str(kk)
            )
            for kk in range(Ne)
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinala, predMatrixa = RSM_extract(
        Ne, "Realisation", oldfolder, scalei, scalei2
    )

    print("Read Historical data")

    os.chdir("TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    True_data = np.hstack([Presz, Oilz, Watsz, wctz])

    True_data = np.reshape(True_data, (-1, 1), "F")
    os.chdir(oldfolder)

    os.chdir("HM_RESULTS")
    Plot_RSM(predMatrix, True_mat, "Final.png", Ne)
    Plot_RSM(predMatrixa, True_mat, "Final_cummulative_best.png", Ne)
    os.chdir(oldfolder)

    aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
    clem = np.argmin(cc)
    shpw = cc[clem]
    controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
    controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
    controlbest2 = controlj2  # controlbest
    controlbest2p = controljj2p  # controlbest

    clembad = np.argmax(cc)
    controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
    controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")

    Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
    # os.makedirs('ESMDA')
    if not os.path.exists("HM_RESULTS/ADAPT_REKI"):
        os.makedirs("HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("HM_RESULTS/ADAPT_REKI")
        os.makedirs("HM_RESULTS/ADAPT_REKI")

    print("ADAPT_REKI Model")

    Simulator(plan2, controlbest, controlbestp, "HM_RESULTS/ADAPT_REKI")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/ADAPT_REKI", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/ADAPT_REKI")
    Plot_petrophysical(
        controlbest,
        controlbestp,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/ADAPT_REKI/Compare.png"
    )
    yycheck = predMatrix

    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("HM_RESULTS")
    Plot_mean(
        controlbest,
        yes_mean_k,
        meanini,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        True_K,
        injectors,
        producers,
    )
    # Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
    os.chdir(oldfolder)

    if not os.path.exists("HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/BEST_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/BEST_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/BEST_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/BEST_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_best_k,
        yes_best_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/BEST_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best model  = : " + str(cc))

    if not os.path.exists("HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("HM_RESULTS/MEAN_RESERVOIR_MODEL")

    Simulator(plan2, yes_best_k, yes_best_p, "HM_RESULTS/MEAN_RESERVOIR_MODEL")
    simDatafinal, predMatrix = RSM_extract1(
        1, "HM_RESULTS/MEAN_RESERVOIR_MODEL", oldfolder, scalei, scalei2
    )
    os.chdir("HM_RESULTS/MEAN_RESERVOIR_MODEL")
    Plot_petrophysical(
        yes_mean_k,
        yes_mean_p,
        nx,
        ny,
        nz,
        Low_K1,
        High_K1,
        Low_P,
        High_P,
        injectors,
        producers,
    )
    os.chdir(oldfolder)
    Plot_RSM_percentile_model(
        predMatrix[0], True_mat, "HM_RESULTS/MEAN_RESERVOIR_MODEL/Compare.png"
    )

    os.chdir(oldfolder)

    yycheck = predMatrix
    yycheck = yycheck[0]
    # usesim=yycheck[:,1:]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim

    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of overall best MAP model  = : " + str(cc))

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    print(" Plot P10,P50,P90")
    for kk in range(ensembleout1.shape[1]):
        path_out = "HM_RESULTS/Percentile" + str(kk)
        os.makedirs(path_out, exist_ok=True)

    executor = get_reusable_executor(max_workers=8)
    with executor:
        Parallel(n_jobs=8, backend="loky", verbose=50)(
            delayed(Simulator)(
                plan2,
                ensembleout1[:, kk],
                ensembleoutp1[:, kk],
                "HM_RESULTS/Percentile" + str(kk),
            )
            for kk in range(ensembleout1.shape[1])
        )
        executor.shutdown(wait=False)
    # executor.shutdown(wait=True)
    simDatafinal, yzout = RSM_extract(
        ensembleout1.shape[1], "HM_RESULTS/Percentile", oldfolder, scalei, scalei2
    )

    os.chdir("HM_RESULTS")
    Plot_RSM_percentilee(yzout, True_mat, "P10_P50_P90.png")
    os.chdir(oldfolder)
    print(
        "--------------------SECTION ADAPTIVE REKI LOGIT ENDED----------------------------"
    )
elapsed_time_secs = time.time() - start_time

if Technique_REKI == 1:
    comment = "ADAPT_REKI (Vanilla Adaptive Ensemble Kalman Inversion)"
elif Technique_REKI == 2:
    comment = "ADAPT_REKI_AE  (Adaptive Ensemble Kalman Inversion + Convolution Autoencoder)\n"
elif Technique_REKI == 3:
    comment = (
        "ADAPT_REKI_DA  (Adaptive Ensemble Kalman Inversion + Denoising Autoencoder)\n"
    )
elif Technique_REKI == 4:
    comment = (
        "ADAPT_REKI_NORMAL_SCORE  (Adaptive Ensemble Kalman Inversion + Normal Score)\n"
    )
elif Technique_REKI == 5:
    comment = "ADAPT_REKI_DE_GAN  (Adaptive Ensemble Kalman Inversion + Convolution Autoencoder + GAN)\n"
elif Technique_REKI == 6:
    comment = "ADAPT_REKI_GAN  (Adaptive Ensemble Kalman Inversion + GAN on the permeability field alone\n"
elif Technique_REKI == 7:
    comment = "ADAPT_REKI_KMEANS  (Adaptive Ensemble Kalman Inversion + KMEANS on the permeability field alone\n"
elif Technique_REKI == 8:
    comment = "ADAPT_REKI_VAE  (Adaptive Ensemble Kalman Inversion + variational\
    convolutional autoencoder on the permeability field alone\n"
elif Technique_REKI == 9:
    comment = (
        "ADAPT_REKI_DCT  (Adaptive Ensemble Kalman Inversion with DCT parametrisation\n"
    )
else:
    comment = "ADAPT_REKI with Logit transformation"

Remove_folder(Ne, "Realisation")
print("Inverse problem solution used =: " + comment)
print("Ensemble size = ", str(Ne))
msg = "Execution took: %s secs (Wall clock time)" % timedelta(
    seconds=round(elapsed_time_secs)
)
print(msg)
print("-------------------PROGRAM EXECUTED-----------------------------------")
