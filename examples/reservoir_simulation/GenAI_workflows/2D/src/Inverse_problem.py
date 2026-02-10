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
Physics informed neural operator for Black oil reservoir simulation.

Train the surrogate with a mixed residual loss cost functional.

Neural operator to model the black oil model. Optimisation with ADAM and then LBFGS

Finite volume reservoir simulator with flexible solver

AMG to solve the pressure and saturation well possed inverse problem

Use the devloped Neural operator surrogate in an Inverse Problem:
    
@Parametrisation Methods:
    1)KSVD/OMP
    2)DCT
    3)DENOISING AUTOENCODER
    4)LEVEL SET
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

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import scipy.io as sio
import shutil
import time
from datetime import timedelta
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import os.path
from scipy.fftpack import dct
from scipy.fftpack.realtransforms import idct
from scipy import interpolate
import multiprocessing
import matplotlib.colors
from matplotlib import cm
from gstools import SRF, Gaussian
import mpslib as mps
from sklearn.model_selection import train_test_split
import numpy.matlib
from scipy.spatial.distance import cdist
from pyDOE import lhs
from kneed import KneeLocator
from matplotlib.colors import LinearSegmentedColormap

# rcParams['font.family'] = 'sans-serif'
# cParams['font.sans-serif'] = ['Helvetica']
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
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
from shutil import rmtree
import pickle
from physicsnemo.sym.models.fno import *
from physicsnemo.sym.models.afno.afno import *
from physicsnemo.sym.key import Key
from collections import OrderedDict
import os.path
from numpy.linalg import inv
from sklearn.decomposition import dict_learning, sparse_encode
from skimage.segmentation import find_boundaries
from math import sqrt
from PIL import Image
import numpy
import numpy.matlib

# os.environ['KERAS_BACKEND'] = 'tensorflow'
import os.path
import torch

torch.cuda.empty_cache()
import random as ra

# import torch.nn.functional as F
import scipy.linalg as sla
from torch.utils.data import DataLoader, TensorDataset

# from torch.autograd import Variable
from gstools.random import MasterRNG

# import dolfin as df
import pandas as pd
from numpy import *
import scipy.optimize.lbfgsb as lbfgsb
import numpy.linalg
from numpy.linalg import norm
import numpy.ma as ma
import logging
import os

# from joblib import Parallel, delayed
from FyeldGenerator import generate_field
from imresize import *

# from NVRS import *


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
from DIFFC import *
import warnings
import random
import sys

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


def ShowBar(Bar):
    sys.stdout.write(Bar)
    sys.stdout.flush()


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


# numpy.random.seed(1)


def Forward_model_ensemble(
    modelP,
    modelS,
    x_true,
    rwell,
    skin,
    pwf_producer,
    mazw,
    steppi,
    Ne,
    DX,
    pini_alt,
    UW,
    BW,
    UO,
    BO,
    SWI,
    SWR,
    DZ,
    dt,
    MAXZ,
):
    #### ===================================================================== ####
    #                     PINO RESERVOIR SIMULATOR
    #
    #### ===================================================================== ####

    """
    Forward model for simulating the Pinno reservoir using an ensemble of permeability fields.

    Parameters:
    -----------
    modelP : torch.nn.Module
        PyTorch model for predicting the pressure field.
    modelS : torch.nn.Module
        PyTorch model for predicting the water saturation field.
    x_true : dict
        Dictionary containing the true permeability, injection rate, water production rate, porosity, time, initial pressure, and initial water saturation.
    rwell : float
        Radius of the production well.
    skin : float
        Skin factor of the well.
    pwf_producer : float
        Bottom hole flowing pressure of the producer well.
    mazw : int
        Flag to indicate whether to apply smoothing to the pressure field.
    steppi : int
        Number of simulation timesteps.
    Ne : int
        Number of realizations in the ensemble.
    DX : float
        Grid spacing.
    pini_alt : float
        Initial reservoir pressure.
    UW : float
        Water viscosity.
    BW : float
        Water formation volume factor.
    UO : float
        Oil viscosity.
    BO : float
        Oil formation volume factor.
    SWI : float
        Initial water saturation.
    SWR : float
        Residual water saturation.
    DZ : float
        Thickness of the reservoir layer.
    dt : float
        Simulation timestep.
    MAXZ : float
        Total thickness of the reservoir.

    Returns:
    --------
    simulated : numpy.ndarray
        Array of simulated well production data.
    large_ensemble : list
        List of well production data for each realization in the ensemble.
    ouut_p : numpy.ndarray
        Array of predicted pressure fields.
    ouut_s : numpy.ndarray
        Array of predicted water saturation fields.
    ouut_oil : numpy.ndarray
        Array of predicted oil saturation fields.
    """

    ouut_p = []
    ouut_s = []
    # x_true = ensemblepy
    for clem in range(Ne):
        temp = {
            "perm": x_true["perm"][clem, :, :, :][None, :, :, :],
            "Q": x_true["Q"][clem, :, :, :][None, :, :, :],
            "Qw": x_true["Qw"][clem, :, :, :][None, :, :, :],
            "Phi": x_true["Phi"][clem, :, :, :][None, :, :, :],
            "Time": x_true["Time"][clem, :, :, :][None, :, :, :],
            "Pini": x_true["Pini"][clem, :, :, :][None, :, :, :],
            "Swini": x_true["Swini"][clem, :, :, :][None, :, :, :],
        }

        with torch.no_grad():
            ouut_p1 = modelP(temp)["pressure"]
            ouut_s1 = modelS(temp)["water_sat"]

        ouut_p.append(ouut_p1)
        ouut_s.append(ouut_s1)

        del temp
        torch.cuda.empty_cache()

    ouut_p = torch.vstack(ouut_p).detach().cpu().numpy()
    ouut_s = torch.vstack(ouut_s).detach().cpu().numpy()

    ouut_oil = np.ones_like(ouut_s) - ouut_s

    # see = Peaceman_well(x_true.detach().cpu().numpy() ,ouut_p,ouut_s,MAXZ,1,1e1)
    inn = x_true
    large_ensemble = []
    simulated = []
    for amm in range(Ne):
        Injector_location = np.where(
            inn["Qw"][amm, 0, :, :].detach().cpu().numpy().ravel() > 0
        )[0]
        producer_location = np.where(
            inn["Q"][amm, 0, :, :].detach().cpu().numpy().ravel() < 0
        )[0]

        PERM = rescale_linear_pytorch_numpy(
            np.reshape(inn["perm"][amm, 0, :, :].detach().cpu().numpy(), (-1,), "F"),
            LUB,
            HUB,
            aay,
            bby,
        )

        kuse_inj = PERM[Injector_location]
        kuse_prod = PERM[producer_location]

        RE = 0.2 * DX
        Baa = []
        Baa2 = []
        Timz = []
        for kk in range(steppi):
            Ptito = ouut_p[amm, kk, :, :]
            Stito = ouut_s[amm, kk, :, :]
            # average_pressure = np.mean(Ptito.ravel()) * pini_alt
            average_pressure = (Ptito.ravel()[producer_location]) * pini_alt
            p_inj = (Ptito.ravel()[Injector_location]) * pini_alt
            # p_inj = average_pressure#(Ptito.ravel()[Injector_location] ) * pini_alt
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
            # temp[temp == -inf] = 0
            Pwf = p_inj + temp
            Pwf = np.abs(Pwf)
            BHP = Pwf
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
            qs2 = [BHP / scalei3, qoil / scalei, qwater / scalei2, wct]
            # qs2 = [qoil/scalei,qwater/scalei2,wct]
            # print(qs.shape)
            qs = np.asarray(qs)
            qs = qs.reshape(1, -1)

            qs2 = np.asarray(qs2)
            qs2 = qs2.reshape(1, -1)

            Baa.append(qs)
            Baa2.append(qs2)
            Timz.append(timz)
        Baa = np.vstack(Baa)
        Baa2 = np.vstack(Baa2)
        Timz = np.vstack(Timz)
        overr = np.hstack([Timz, Baa])

        # sim = np.reshape(Baa,(-1,1),'F')
        sim2 = np.reshape(Baa2, (-1, 1), "F")

        simulated.append(sim2)  # For inverse problem
        large_ensemble.append(overr)  # For Plotting
    simulated = np.hstack(simulated)
    return simulated, large_ensemble, ouut_p, ouut_s, ouut_oil


def intial_ensemble(Nx, Ny, Nz, N, permx):
    # for i in range(N):
    O = mps.mpslib()
    O = mps.mpslib(method="mps_snesim_tree")
    O.par["n_real"] = N
    k = permx  # permeability field TI
    kjenn = k
    O.ti = kjenn
    O.par["simulation_grid_size"] = (Ny, Nx, Nz)
    O.run_parallel()
    ensemble = O.sim
    ens = []
    for kk in range(N):
        temp = np.reshape(ensemble[kk], (-1, 1), "F")
        ens.append(temp)
    ensemble = np.hstack(ens)
    from glob import glob

    for f3 in glob("thread*"):
        rmtree(f3)

    for f4 in glob("*mps_snesim_tree_*"):
        os.remove(f4)

    for f4 in glob("*ti_thread_*"):
        os.remove(f4)
    return ensemble


def initial_ensemble_gaussian(Nx, Ny, Nz, N, minn, maxx):
    fensemble = np.zeros((Nx * Ny * Nz, N))
    x = np.arange(Nx)
    y = np.arange(Ny)
    z = np.arange(Nz)
    model = Gaussian(dim=3, var=500, len_scale=200)  # Variance and lenght scale
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
    return fensemble


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


def Add_marker(plt, XX, YY, locc):
    for i in range(locc.shape[0]):
        a = locc[i, :]
        xloc = int(a[0])
        yloc = int(a[1])
        if a[2] == 2:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="^",
                color="white",
            )
        else:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="v",
                color="white",
            )


def honour2(sgsim2, DupdateK, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_p):
    if Yet == 0:
        sgsim2 = cp.asarray(sgsim2)
        DupdateK = cp.asarray(DupdateK)
    else:
        pass

    # uniehonour = np.reshape(rossmary,(nx,ny,nz), 'F')
    # unieporohonour = np.reshape(rossmaryporo,(nx,ny,nz), 'F')

    # Read true porosity well values

    # aa = np.zeros((nz))
    # bb = np.zeros((nz))
    # cc = np.zeros((nz))
    # dd = np.zeros((nz))
    # ee = np.zeros((nz))
    # ff = np.zeros((nz))
    # gg = np.zeros((nz))
    # hh = np.zeros((nz))

    # aa1 = np.zeros((nz))
    # bb1 = np.zeros((nz))
    # cc1 = np.zeros((nz))
    # dd1 = np.zeros((nz))
    # ee1 = np.zeros((nz))
    # ff1 = np.zeros((nz))
    # gg1 = np.zeros((nz))
    # hh1 = np.zeros((nz))

    # Read true porosity well values
    """
    for j in range(nz):
        aa[j] = uniehonour[1,24,j]
        bb[j] = uniehonour[1,1,j]
        cc[j] = uniehonour[31,0,j]
        dd[j] = uniehonour[31,31,j]
        ee[j] = uniehonour[7,9,j]
        ff[j] = uniehonour[14,12,j]
        gg[j] = uniehonour[28,19,j]
        hh[j] = uniehonour[14,27,j]

        aa1[j] = unieporohonour[1,24,j]
        bb1[j] = unieporohonour[1,1,j]
        cc1[j] = unieporohonour[31,0,j]
        dd1[j] = unieporohonour[31,31,j]
        ee1[j] = unieporohonour[7,9,j]
        ff1[j] = unieporohonour[14,12,j]
        gg1[j] = unieporohonour[28,19,j]
        hh1[j] = unieporohonour[14,27,j]
    """
    # Read permeability ensemble after EnKF update
    # A = DupdateK
    # C = sgsim2

    output = cp.zeros((nx * ny * nz, N_ens))
    outputporo = cp.zeros((nx * ny * nz, N_ens))
    """
    for j in range(N_ens):
        B = np.reshape(A[:,j],(nx,ny,nz),'F')
        D = np.reshape(C[:,j],(nx,ny,nz),'F')
    
        for jj in range(nz):
            B[1,24,jj] = aa[jj]
            B[1,1,jj] = bb[jj]
            B[31,0,jj] = cc[jj]
            B[31,31,jj] = dd[jj]
            B[7,9,jj] = ee[jj]
            B[14,12,jj] = ff[jj]
            B[28,19,jj] = gg[jj]
            B[14,27,jj] = hh[jj]

            D[1,24,jj] = aa1[jj]
            D[1,1,jj] = bb1[jj]
            D[31,0,jj] = cc1[jj]
            D[31,31,jj] = dd1[jj]
            D[7,9,jj] = ee1[jj]
            D[14,12,jj] = ff1[jj]
            D[28,19,jj] = gg1[jj]
            D[14,27,jj] = hh1[jj]
        
        output[:,j:j+1] = np.reshape(B,(nx*ny*nz,1), 'F')
        outputporo[:,j:j+1] = np.reshape(D,(nx*ny*nz,1), 'F')
    """
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


def Narrow_Band(M1, Nx, Ny, Nz):
    y = np.reshape(M1, (Nx, Ny, Nz), "F")

    accepted = []
    for kk in range(Nz):
        yuse = y[:, :, kk]
        yuse = np.reshape(yuse, (-1, 1), "F")
        kmeans = MiniBatchKMeans(n_clusters=Experts, max_iter=2000).fit(yuse)
        dd = kmeans.labels_
        dd = dd.T
        dd = np.reshape(dd, (-1, 1))
        mudd = np.reshape(dd, (Nx, Ny), "F")
        mudd1 = mudd.astype(np.int8)
        aa = find_boundaries((mudd1))
        aa = aa * 1
        aa = np.reshape(aa, (-1, 1), "F")
        aa = aa.astype(float)
        accepted.append(aa)
    return np.array(accepted)


def get_clus(y, Experts):
    """
    Parameters
    ----------
    y : Permeability field as a vector

    Experts : Number of Facies


    Returns
    -------
    sdi : Signed Distance transformation

    mexx : Cluster mean

    """
    y = np.reshape(y, (-1, 1), "F")
    kmeans = MiniBatchKMeans(n_clusters=Experts, max_iter=2000).fit(y)

    dd = kmeans.labels_
    dd = dd.T
    dd = np.reshape(dd, (-1, 1))
    kk = np.zeros((dd.shape[0], 1))
    d = dd
    mexx = kmeans.cluster_centers_
    for i in range(d.shape[0]):
        for j in range(Experts):
            if d[i, :] == j:
                kk[i, :] = np.abs(np.random.normal(mexx[j, :], 0.001, 1))

    sdi = kk

    return sdi, mexx, kmeans


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


def load_data_numpy(inn, ndata, batch_size):
    x_data = inn
    print(f"x_data: {x_data.shape}")
    data_tuple = (torch.FloatTensor(x_data),)
    data_loader = DataLoader(
        TensorDataset(*data_tuple), batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


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


def Plot_performance(PINN, PINN2, trueF, nx, ny, s1, met, namet, UIR, itt, dt, MAXZ):
    look = (PINN[itt, :, :]) * pini_alt
    look_sat = PINN2[itt, :, :]
    look_oil = 1 - look_sat

    # look = smoothn(look,3)
    if met == 1:
        look = smoothn(look, s=s1)[0]

        # kmeans = KMeans(n_clusters=2,max_iter=2000).fit(np.reshape(look_sat,(-1,1),'F'))
        # dd=kmeans.labels_
        # dd = dd.T
        # dd = np.reshape(dd,(-1,1))

        # temp = np.reshape(look_sat,(-1,1),'F')
        # temp1 = np.zeros_like(temp)
        # indi1 = np.where(dd==1)[0]
        # indi2 = np.where(dd==0)[0]

        # temp1.ravel()[indi1] = temp.ravel()[indi1]
        # temp1.ravel()[indi2] = IWSw
        # look_sat = np.reshape(temp1,(nx,ny),'F')

        # look_sat = smoothn(look_sat,s= s1)[0]

    lookf = (trueF[itt, :, :]) * pini_alt
    lookf_sat = trueF[itt + steppi, :, :]
    lookf_oil = 1 - lookf_sat

    diff1 = abs(look - lookf)
    diff1_wat = abs(look_sat - lookf_sat)
    diff1_oil = abs(look_oil - lookf_oil)

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 3, 1)
    plt.pcolormesh(XX.T, YY.T, look, cmap="jet")
    plt.title("Pressure PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(np.min(np.reshape(lookf, (-1,))), np.max(np.reshape(lookf, (-1,))))
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 2)
    plt.pcolormesh(XX.T, YY.T, lookf, cmap="jet")
    plt.title("Pressure CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 3)
    plt.pcolormesh(XX.T, YY.T, diff1, cmap="jet")
    plt.title("Pressure (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 4)
    plt.pcolormesh(XX.T, YY.T, look_sat, cmap="jet")
    plt.title("water_sat PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(np.min(np.reshape(lookf_sat, (-1,))), np.max(np.reshape(lookf_sat, (-1,))))
    cbar1.ax.set_ylabel(" water_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 5)
    plt.pcolormesh(XX.T, YY.T, lookf_sat, cmap="jet")
    plt.title("water_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 6)
    plt.pcolormesh(XX.T, YY.T, diff1_wat, cmap="jet")
    plt.title("water_sat (CFD - PINO)", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" water sat ", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 7)
    plt.pcolormesh(XX.T, YY.T, look_oil, cmap="jet")
    plt.title("oil_sat PINO", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(np.min(np.reshape(lookf_oil, (-1,))), np.max(np.reshape(lookf_oil, (-1,))))
    cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 8)
    plt.pcolormesh(XX.T, YY.T, lookf_oil, cmap="jet")
    plt.title("oil_sat CFD", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel(" oil sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(3, 3, 9)
    plt.pcolormesh(XX.T, YY.T, diff1_oil, cmap="jet")
    plt.title("oil_sat (CFD - PINO)", fontsize=13)
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

    name = namet + str(int(itt)) + ".png"
    plt.savefig(name)
    # plt.show()
    plt.clf()


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


be_verbose = False


def annealing_linear(start, end, pct):
    return start + pct * (end - start)


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycleScheduler(object):
    """
    (0, pct_start) -- linearly increase lr
    (pct_start, 1) -- cos annealing
    """

    def __init__(self, lr_max, div_factor=25.0, pct_start=0.3):
        super(OneCycleScheduler, self).__init__()
        self.lr_max = lr_max
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.lr_low = self.lr_max / self.div_factor

    def step(self, pct):
        # pct: [0, 1]
        if pct <= self.pct_start:
            return annealing_linear(self.lr_low, self.lr_max, pct / self.pct_start)

        else:
            return annealing_cos(
                self.lr_max,
                self.lr_low / 1e4,
                (pct - self.pct_start) / (1 - self.pct_start),
            )


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def ensemble_pytorch(
    param_perm,
    param_poro,
    LUB,
    HUB,
    mpor,
    hpor,
    inj_rate,
    dt,
    IWSw,
    nx,
    ny,
    nz,
    Ne,
    input_channel,
):
    """
    Input:

    1. param_perm: (numpy array) array of size (num_parameters,num_realizations)
    containing the permability values for different realizations
    2. param_poro: (numpy array) array of size (num_parameters,num_realizations) containing the porosity values for different realizations
    3. LUB: (numpy array) lower bound of the uniform distribution for scaling the permeability values
    4. HUB: (numpy array) upper bound of the uniform distribution for scaling the permeability values
    5. mpor: (float) minimum value for porosity range
    6. hpor: (float) maximum value for porosity range
    7. inj_rate: (float) value of injection rate
    8. dt: (float) time step
    9. IWSw: (float) initial water saturation
    10. nx: (int) number of cells in x direction
    11. ny: (int) number of cells in y direction
    12. nz: (int) number of cells in z direction
    13. Ne: (int) number of ensembles
    14. input_channel: (int) number of input channels in the model

    Output: Returns a dictionary containing the initial ensemble for different parameters.

    Function Description: This function takes the parameter values for permeability, porosity, uniform distribution
    for permeability, minimum and maximum values for porosity range, injection rate, time step, initial water saturation,
    number of cells in x,y and z directions, number of ensembles and number of input channels in the model.
    It returns the initial ensemble for different parameters as a dictionary.
    """
    A = np.zeros((nx, ny, nz))
    A1 = np.zeros((nx, ny, nz))
    ini_ensemble1 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble2 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble3 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble4 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble5 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble6 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble7 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)

    for kk in range(Ne):
        a = rescale_linear_numpy_pytorch(param_perm[:, kk], LUB, HUB, aay, bby)

        a = np.reshape(a, (nx, ny), "F")

        at2 = param_poro[:, kk]
        at2 = np.reshape(at2, (nx, ny), "F")

        # inj_rate = 500# kka[:,kk]# kka[kk,:]
        for jj in range(nz):
            A[1, 24, jj] = inj_rate
            A[1, 1, jj] = inj_rate
            A[31, 0, jj] = inj_rate
            A[31, 31, jj] = inj_rate
            A[7, 9, jj] = -50
            A[14, 12, jj] = -50
            A[28, 19, jj] = -50
            A[14, 27, jj] = -50

            A1[1, 24, jj] = inj_rate
            A1[1, 1, jj] = inj_rate
            A1[31, 0, jj] = inj_rate
            A1[31, 31, jj] = inj_rate

        quse1 = A
        quse_water = A1

        ini_ensemble1[kk, 0, :, :] = a  # Permeability
        ini_ensemble2[kk, 0, :, :] = quse1[:, :, 0] / UIR  # Overall_source
        ini_ensemble3[kk, 0, :, :] = quse_water[:, :, 0] / UIR  # Water injection source
        ini_ensemble4[kk, 0, :, :] = at2  # porosity
        ini_ensemble5[kk, 0, :, :] = dt * np.ones((nx, ny))  # Time
        ini_ensemble6[kk, 0, :, :] = np.ones((nx, ny))  # Initial pressure
        ini_ensemble7[kk, 0, :, :] = IWSw * np.ones(
            (nx, ny)
        )  # INitial water saturation

    # ini_ensemble = torch.from_numpy(ini_ensemble).to(device, dtype=torch.float32)
    inn = {
        "perm": torch.from_numpy(ini_ensemble1).to(device, torch.float32),
        "Q": torch.from_numpy(ini_ensemble2).to(device, dtype=torch.float32),
        "Qw": torch.from_numpy(ini_ensemble3).to(device, dtype=torch.float32),
        "Phi": torch.from_numpy(ini_ensemble4).to(device, dtype=torch.float32),
        "Time": torch.from_numpy(ini_ensemble5).to(device, dtype=torch.float32),
        "Pini": torch.from_numpy(ini_ensemble6).to(device, dtype=torch.float32),
        "Swini": torch.from_numpy(ini_ensemble7).to(device, dtype=torch.float32),
    }
    return inn


def Plot_mean(permbest, permmean, iniperm, nx, ny, Low_K, High_K, True_perm):
    Low_Ka = Low_K
    High_Ka = High_K

    permmean = np.reshape(permmean, (nx, ny, nz), "F")
    permbest = np.reshape(permbest, (nx, ny, nz), "F")
    iniperm = np.reshape(iniperm, (nx, ny, nz), "F")
    True_perm = np.reshape(True_perm, (nx, ny, nz), "F")
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 3)
    # plt.pcolormesh(XX.T,YY.T,permmean[:,:,0],cmap = 'jet')
    plt.pcolormesh(XX.T, YY.T, permmean[:, :, 0], cmap="jet")
    plt.title("mean", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, permbest[:, :, 0], cmap="jet")
    plt.title("Best", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, iniperm[:, :, 0], cmap="jet")
    plt.title("initial", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 4)
    plt.pcolormesh(XX.T, YY.T, True_perm[:, :, 0], cmap="jet")
    plt.title("TRue", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker(plt, XX, YY, wells)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Permeability comparison", fontsize=25)
    plt.savefig("Comparison.png")
    plt.close()
    plt.clf()


def Plot_petrophysical(permmean, poroo, nx, ny, nz, Low_K, High_K):
    Low_Ka = Low_K
    High_Ka = High_K

    permmean = np.reshape(permmean, (nx, ny, nz), "F")
    poroo = np.reshape(poroo, (nx, ny, nz), "F")

    from skimage.restoration import denoise_nl_means, estimate_sigma

    temp_K = np.reshape(permmean, (nx, ny), "F")
    temp_phi = np.reshape(poroo, (nx, ny), "F")

    # fast algorithm
    timk = 5
    for n in range(timk):
        sigma_est1 = np.mean(estimate_sigma(temp_K))
        sigma_est2 = np.mean(estimate_sigma(temp_phi))
        # print(f'estimated noise standard deviation for permeability = {sigma_est1}')
        print("")
        # print(f'estimated noise standard deviation for porosity = {sigma_est2}')

        patch_kw = dict(patch_size=5, patch_distance=6)  # 5x5 patches
        temp_K = denoise_nl_means(
            temp_K, h=0.8 * sigma_est1, fast_mode=True, **patch_kw
        )
        temp_phi = denoise_nl_means(
            temp_phi, h=0.8 * sigma_est2, fast_mode=True, **patch_kw
        )

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, permmean[:, :, 0], cmap="jet")
    plt.title("Permeability", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 3)
    plt.pcolormesh(XX.T, YY.T, temp_K, cmap="jet")
    plt.title("Smoothed - Permeability", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_Ka, High_Ka)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, poroo[:, :, 0], cmap="jet")
    plt.title("Porosity", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Units", fontsize=13)
    plt.clim(Low_P, High_P)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 4)
    plt.pcolormesh(XX.T, YY.T, temp_phi, cmap="jet")
    plt.title("Smoothed Porosity", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Units", fontsize=13)
    plt.clim(Low_P, High_P)
    Add_marker(plt, XX, YY, wells)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Petrophysical Reconstruction", fontsize=25)
    plt.savefig("Petro_Recon.png")
    plt.close()
    plt.clf()


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


def rectifyy(perm, ls, mexx, Experts, Nx, Ny, Nz, modelz):
    """
    rectify conlifct between pixel and level set update
    Parameters
    ----------
    perm : Pixel field

    ls : Level set field

    mexx : Clsuster means for Facies

    Experts : Number of Facies

    Nx : Dimension in X axis

    Ny : Dimension in y axis

    Nz : Dimesnion in z axis


    Returns
    -------
    valuee : Corrected pixel field

    """
    perm = np.reshape(perm, (-1, 1))
    mexx = np.reshape(mexx, (-1, 1))
    kmeans = MiniBatchKMeans(n_clusters=Experts, max_iter=2000).fit(perm)
    # ddp=modelz.predict(perm)
    ddp = np.reshape(kmeans.labels_.T, (-1, 1), "F")
    ddp = np.reshape(ddp, (-1, 1), "F")
    ls = np.reshape(ls, (-1, 1))
    kmeansp = MiniBatchKMeans(n_clusters=Experts, max_iter=2000).fit(ls)
    # dd=modelz.predict(ls)
    dd = np.reshape(kmeansp.labels_.T, (-1, 1), "F")
    dd = np.reshape(dd, (-1, 1), "F")
    valls = Nx * Ny * Nz
    valuee = np.zeros((valls, 1))
    count = Nx * Ny * Nz
    for i in range(count):
        if ddp[i, :] == dd[i, :]:
            valuee[i, :] = perm[i, :]

        if ddp[i, :] != dd[i, :]:
            jdx = np.ravel(mexx[dd[i, :], :])
            a = np.abs(np.random.normal(jdx, 0.001, 1))
            valuee[i, :] = np.ravel(
                a
            )  # np.abs(np.random.normal(mexx[dd[i,:],:],0.001,1))   #np.abs(np.random.normal(mexx[dd[i,:],:],0.01,1))

    return valuee


def Plot_RSM_percentile(pertoutt, True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    P10 = pertoutt[0]
    P50 = pertoutt[1]
    P90 = pertoutt[2]
    arekibest = pertoutt[3]
    # arekimean =  pertoutt[4]

    plt.figure(figsize=(30, 30))

    # plt.subplot(4,4,1)
    # plt.plot(timezz,True_mat[:,1], color = 'red', lw = '2',\
    #          label ='model')
    # plt.plot(timezz,P10[:,1], color = 'blue', lw = '2', \
    #          label ='P10 Model')
    # plt.plot(timezz,P50[:,1], color = 'c', lw = '2', \
    #          label ='P50 Model')
    # plt.plot(timezz,P90[:,1], color = 'green', lw = '2', \
    #      label ='P90 Model')

    # plt.plot(timezz,arekibest[:,1], color = 'k', lw = '2', \
    #      label ='aREKI cum best Model')

    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I1',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,2)
    # plt.plot(timezz,True_mat[:,2], color = 'red', lw = '2',\
    #          label ='model')
    # plt.plot(timezz,P10[:,2], color = 'blue', lw = '2', \
    #          label ='P10 Model')
    # plt.plot(timezz,P50[:,2], color = 'c', lw = '2', \
    #          label ='P50 Model')
    # plt.plot(timezz,P90[:,2], color = 'green', lw = '2', \
    #      label ='P90 Model')

    # plt.plot(timezz,arekibest[:,2], color = 'k', lw = '2', \
    #      label ='aREKI cum best Model')

    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I2',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,3)
    # plt.plot(timezz,True_mat[:,3], color = 'red', lw = '2',\
    #          label ='model')
    # plt.plot(timezz,P10[:,3], color = 'blue', lw = '2', \
    #          label ='P10 Model')
    # plt.plot(timezz,P50[:,3], color = 'c', lw = '2', \
    #          label ='P50 Model')
    # plt.plot(timezz,P90[:,3], color = 'green', lw = '2', \
    #      label ='P90 Model')
    # plt.plot(timezz,arekibest[:,3], color = 'k', lw = '2', \
    #      label ='aREKI cum best Model')

    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I3',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,4)
    # plt.plot(timezz,True_mat[:,4], color = 'red', lw = '2',\
    #          label ='model')
    # plt.plot(timezz,P10[:,4], color = 'blue', lw = '2', \
    #          label ='P10 Model')
    # plt.plot(timezz,P50[:,4], color = 'c', lw = '2', \
    #          label ='P50 Model')
    # plt.plot(timezz,P90[:,4], color = 'green', lw = '2', \
    #      label ='P90 Model')

    # plt.plot(timezz,arekibest[:,4], color = 'k', lw = '2', \
    #      label ='aREKI cum best Model')

    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I4',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    plt.subplot(3, 4, 1)
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

    plt.subplot(3, 4, 2)
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

    plt.subplot(3, 4, 3)
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

    plt.subplot(3, 4, 4)
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

    plt.subplot(3, 4, 5)
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

    plt.subplot(3, 4, 6)
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

    plt.subplot(3, 4, 7)
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

    plt.subplot(3, 4, 8)
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

    plt.subplot(3, 4, 9)
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

    plt.subplot(3, 4, 10)
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

    plt.subplot(3, 4, 11)
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

    plt.subplot(3, 4, 12)
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


def Plot_RSM_percentile_model(pertoutt, True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    P10 = pertoutt

    plt.figure(figsize=(30, 30))

    # plt.subplot(4,4,1)
    # plt.plot(timezz,True_mat[:,1], color = 'red', lw = '2',\
    #          label ='History model')
    # plt.plot(timezz,P10[:,1], color = 'blue', lw = '2', \
    #          label ='aREKI')

    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I1',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,2)
    # plt.plot(timezz,True_mat[:,2], color = 'red', lw = '2',\
    #          label ='History model')
    # plt.plot(timezz,P10[:,2], color = 'blue', lw = '2', \
    #          label ='aREKI ')

    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I2',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,3)
    # plt.plot(timezz,True_mat[:,3], color = 'red', lw = '2',\
    #          label ='History model')
    # plt.plot(timezz,P10[:,3], color = 'blue', lw = '2', \
    #          label ='aREKI ')

    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I3',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,4)
    # plt.plot(timezz,True_mat[:,4], color = 'red', lw = '2',\
    #          label ='History model')
    # plt.plot(timezz,P10[:,4], color = 'blue', lw = '2', \
    #          label ='aREKI ')

    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I4',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    plt.subplot(3, 4, 1)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 5], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,5], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 2)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 6], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,6], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 3)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 7], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,7], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 4)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 8], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,8], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 5)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 9], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,9], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 6)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 10], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,10], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 7)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 11], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,11], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 8)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 12], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,12], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 9)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 13], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,13], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 10)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 14], color="blue", lw="2", label="aREKI Surrogate")
    # plt.plot(timezz,tochi[:,14], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 11)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 15], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,15], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 12)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 16], color="blue", lw="2", label="aREKI Surrogate ")
    # plt.plot(timezz,tochi[:,16], color = 'k', lw = '2', \
    #          label ='aREKI Finite volume ')
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


def Plot_RSM_single(True_mat, Namesz):
    True_mat = True_mat[0]
    timezz = True_mat[:, 0].reshape(-1, 1)

    plt.figure(figsize=(30, 30))

    # plt.subplot(4,4,1)
    # plt.plot(timezz,True_mat[:,1], color = 'red', lw = '2',\
    #          label ='model')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I1',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,2)
    # plt.plot(timezz,True_mat[:,2], color = 'red', lw = '2',\
    #          label ='model')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I2',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,3)
    # plt.plot(timezz,True_mat[:,3], color = 'red', lw = '2',\
    #          label ='model')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I3',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,4)
    # plt.plot(timezz,True_mat[:,4], color = 'red', lw = '2',\
    #          label ='model')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I4',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    plt.subplot(3, 4, 1)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 2)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 3)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 4)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 5)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 6)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 7)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 8)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 9)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 10)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 11)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 12)
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

    plt.figure(figsize=(30, 30))

    # plt.subplot(4,4,1)
    # plt.plot(timezz,True_mat[:,1], color = 'red', lw = '2',\
    #          label ='model')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I1',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,2)
    # plt.plot(timezz,True_mat[:,2], color = 'red', lw = '2',\
    #          label ='model')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I2',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,3)
    # plt.plot(timezz,True_mat[:,3], color = 'red', lw = '2',\
    #          label ='model')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I3',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    # plt.subplot(4,4,4)
    # plt.plot(timezz,True_mat[:,4], color = 'red', lw = '2',\
    #          label ='model')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I4',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    plt.subplot(3, 4, 1)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 2)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 3)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 4)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 5)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 6)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 7)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 8)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 9)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 10)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 11)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 12)
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

    plt.figure(figsize=(30, 30))

    # plt.subplot(4,4,1)
    # plt.plot(timezz,BHPA[:,:Ne],color = 'grey', lw = '2',\
    #          label = 'Realisations')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # plt.title('I1',fontsize = 13)

    # plt.plot(timezz,True_mat[:,1], color = 'red', lw = '2',\
    #          label ='True model')
    # plt.axvline(x = 1500, color = 'black', linestyle = '--')
    # handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    # by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    # plt.legend(by_label.values(), by_label.keys())
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)

    # plt.subplot(4,4,2)
    # plt.plot(timezz,BHPB[:,:Ne],color = 'grey', lw = '2',\
    #          label = 'Realisations')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # plt.title('I2',fontsize = 13)

    # plt.plot(timezz,True_mat[:,2], color = 'red', lw = '2',\
    #          label ='True model')
    # plt.axvline(x = 1500, color = 'black', linestyle = '--')
    # handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    # by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    # plt.legend(by_label.values(), by_label.keys())
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)

    # plt.subplot(4,4,3)
    # plt.plot(timezz,BHPC[:,:Ne],color = 'grey', lw = '2',\
    #          label = 'Realisations')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # plt.title('I3',fontsize = 13)

    # plt.plot(timezz,True_mat[:,3], color = 'red', lw = '2',\
    #          label ='True model')
    # plt.axvline(x = 1500, color = 'black', linestyle = '--')
    # handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    # by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    # plt.legend(by_label.values(), by_label.keys())
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)

    # plt.subplot(4,4,4)
    # plt.plot(timezz,BHPD[:,:Ne],color = 'grey', lw = '2',\
    #          label = 'Realisations')
    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # plt.title('I4',fontsize = 13)

    # plt.plot(timezz,True_mat[:,4], color = 'red', lw = '2',\
    #          label ='True model')
    # plt.axvline(x = 1500, color = 'black', linestyle = '--')
    # handles, labels = plt.gca().get_legend_handles_labels()     # get all the labels
    # by_label = OrderedDict(zip(labels,handles))                 # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    # plt.legend(by_label.values(), by_label.keys())
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)

    plt.subplot(3, 4, 1)
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

    plt.subplot(3, 4, 2)
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

    plt.subplot(3, 4, 3)
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

    plt.subplot(3, 4, 4)
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

    plt.subplot(3, 4, 5)
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

    plt.subplot(3, 4, 6)
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

    plt.subplot(3, 4, 7)
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

    plt.subplot(3, 4, 8)
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

    plt.subplot(3, 4, 9)
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

    plt.subplot(3, 4, 10)
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

    plt.subplot(3, 4, 11)
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

    plt.subplot(3, 4, 12)
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


def Get_Latent(ini_ensemble, N_ens, nx, ny, nz, High_K):
    X_unie = np.zeros((N_ens, nx, ny, nz))
    for i in range(N_ens):
        X_unie[i, :, :, :] = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")
    ax = X_unie / High_K
    ouut = np.zeros((11 * 11 * 4, Ne))
    decoded_imgs2 = load_model("../PACKETS/encoder.h5").predict(ax)
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
    ouut = np.zeros((11 * 11 * 4, Ne))
    decoded_imgs2 = load_model("../PACKETS/encoderp.h5").predict(ax)
    for i in range(N_ens):
        jj = decoded_imgs2[i]  # .reshape(20,20,4)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def Recover_image(x, Ne, nx, ny, nz, High_K):
    X_unie = np.zeros((Ne, 11, 11, 4))
    for i in range(Ne):
        X_unie[i, :, :, :] = np.reshape(x[:, i], (11, 11, 4), "F")
    decoded_imgs2 = (load_model("../PACKETS/decoder.h5").predict(X_unie)) * High_K
    # print(decoded_imgs2.shape)
    ouut = np.zeros((nx * ny * nz, Ne))
    for i in range(Ne):
        jj = decoded_imgs2[i]  # .reshape(nx,ny,nz)
        jj = np.reshape(jj, (-1, 1), "F")
        ouut[:, i] = np.ravel(jj)
    return ouut


def Recover_imagep(x, Ne, nx, ny, nz, High_P):
    X_unie = np.zeros((Ne, 11, 11, 4))
    for i in range(Ne):
        X_unie[i, :, :, :] = np.reshape(x[:, i], (11, 11, 4), "F")
    decoded_imgs2 = load_model("../PACKETS/decoderp.h5").predict(X_unie)
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
    decoded_imgs2 = (
        load_model("../PACKETS/denosingautoencoder.h5").predict(ax)
    ) * High_K
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
    decoded_imgs2 = load_model("../PACKETS/denosingautoencoderp.h5").predict(ax)
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
    filename = "../PACKETS/Ganensemble.mat"
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
    encoded = MaxPooling2D((3, 3), padding="same")(x)  # reduces it by this value

    encoder = Model(input_image, encoded)
    encoder.summary()

    decoder_input = Input(shape=(11, 11, 32))
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(decoder_input)
    x = UpSampling2D((3, 3))(x)
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
        "../PACKETS/denosingautoencoder.h5",
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
    filename = "../PACKETS/Ganensemble.mat"
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
    encoded = MaxPooling2D((3, 3), padding="same")(x)  # reduces it by this value

    encoder = Model(input_image, encoded)
    encoder.summary()

    decoder_input = Input(shape=(11, 11, 32))
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(decoder_input)
    x = UpSampling2D((3, 3))(x)
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
        "../PACKETS/denosingautoencoderp.h5",
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
    filename = "../PACKETS/Ganensemble.mat"
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
    encoded = MaxPooling2D((3, 3), padding="same")(x)  # reduces it by this value

    encoder = Model(input_image, encoded)
    encoder.summary()

    decoder_input = Input(shape=(11, 11, 4))
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(decoder_input)
    x = UpSampling2D((3, 3))(x)
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
        "../PACKETS/autoencoder.h5",
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
    encoder.save("../PACKETS/encoder.h5")
    decoder.save("../PACKETS/decoder.h5")
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
    filename = "../PACKETS/Ganensemble.mat"
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
    encoded = MaxPooling2D((3, 3), padding="same")(x)  # reduces it by this value

    encoder = Model(input_image, encoded)
    encoder.summary()

    decoder_input = Input(shape=(11, 11, 4))
    x = Conv2D(4, (3, 3), activation="relu", padding="same")(decoder_input)
    x = UpSampling2D((3, 3))(x)
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
        "../PACKETS/autoencoderp.h5",
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
    encoder.save("../PACKETS/encoderp.h5")
    decoder.save("../PACKETS/decoderp.h5")
    os.chdir(oldfolder)


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
    k = np.reshape(k, (33, 33), "F")
    kjenn = k
    TI1 = kjenn
    see = intial_ensemble(nx, ny, nz, N_small, kjenn)

    ini_ensemble1 = see
    clfy = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfy.fit(ini_ensemble1))
    ensemblep = clfy.transform(ini_ensemble1)

    ensemblepy = ensemble_pytorch(
        ini_ensemble1,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ini_ensemble1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ini_ensemble1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

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

    ensemblepy = ensemble_pytorch(
        ini_ensemble2,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ini_ensemble2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ini_ensemble2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

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

    ensemblepy = ensemble_pytorch(
        ini_ensemble3,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ini_ensemble3.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ini_ensemble3.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
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

    ensemblepy = ensemble_pytorch(
        ini_ensemble4,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ini_ensemble4.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ini_ensemble4.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
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

    ini_ensemble5 = see

    clfy = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfy.fit(ini_ensemble5))
    ensemblep = clfy.transform(ini_ensemble5)
    # ensemblep=Getporosity_ensemble(ini_ensemble5,machine_map,N_enss)

    ensemblepy = ensemble_pytorch(
        ini_ensemble5,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ini_ensemble5.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ini_ensemble5.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
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
    # EnsemblepG=Getporosity_ensemble(ini_ensembleG,machine_map,N_enss)
    ensemblepy = ensemble_pytorch(
        ini_ensembleG,
        EnsemblepG,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ini_ensembleG.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ini_ensembleG.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
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
    namezz = "../PACKETS/" + namezz + ".h5"
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
        k = np.reshape(k, (33, 33), "F")
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
    filename = "../PACKETS/Ganensemble.mat"  # Ensemble generated offline
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


def Learn_Overcomplete_Dictionary(Ne):
    # Ne= ensemble size
    filename = "../PACKETS/Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensembley = mat["Z"]
    a = dict_learning(
        ini_ensembley.T,
        Ne,
        0,
        max_iter=100,
        method="lars",
        n_jobs=-1,
        return_n_iter=True,
        method_max_iter=10000,
        verbose=True,
    )
    second = a[1]
    Dicclem = second.T  # Over complete dictionary
    return Dicclem


def Learn_Overcomplete_Dictionaryp(N_ens):
    # Ne= ensemble size
    filename = "../PACKETS/Ganensemble.mat"
    mat = sio.loadmat(filename)
    ini_ensembley = mat["Z"]

    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ini_ensembley))
    ini_ensembleyy = clfye.transform(ini_ensembley)

    # ini_ensembleyy=Getporosity_ensemble(ini_ensembley,machine_map,N_ens)
    a = dict_learning(
        ini_ensembleyy.T,
        Ne,
        0,
        max_iter=100,
        method="lars",
        n_jobs=-1,
        return_n_iter=True,
        method_max_iter=10000,
        verbose=True,
    )
    second = a[1]
    Dicclemp = second.T  # Over complete dictionary
    return Dicclemp


def Sparse_coding(Dictionary, inpuut, N_ens):
    codde = sparse_encode(
        inpuut.T, Dictionary.T, n_jobs=-1, n_nonzero_coefs=N_ens, verbose=1
    )
    return codde.T


def Recover_Dictionary_Saarse(Dicclemk, sparsecodek, Low_Kc, High_Kc):
    recoverr = np.dot(Dicclemk, sparsecodek)
    recoverr[recoverr <= Low_Kc] = Low_Kc
    recoverr[recoverr >= High_Kc] = High_Kc
    return recoverr


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

    os.chdir("../HM_RESULTS")
    plt.savefig("Cost_Function.png")
    os.chdir(oldfolder)
    plt.close()
    plt.clf()


def REKI_ASSIMILATION_LEVELSET(
    sgsim,
    ensemblep,
    simDatafinal,
    alpha,
    True_data,
    Ne,
    pertubations,
    Experts,
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
    # sgsim=ensemble

    sizeclem = cp.asarray(nx * ny * nz)

    # Get signed distance of permeability field
    ensuse1 = sgsim
    ensuse1 = np.nan_to_num(ensuse1, copy=True, nan=Low_K)
    mexini = np.zeros((Experts, Ne))
    sdiini = np.zeros((sgsim.shape[0], sgsim.shape[1]))
    for j in range(Ne):
        sdi, mexx, _ = get_clus(ensuse1[:, j], Experts)
        sdiini[:, j] = np.ravel(sdi)
        mexini[:, j] = np.ravel(mexx)
    sdiuse = sdiini

    # Get Narow Band of permeability field
    Xspitbk = Parallel(n_jobs=8, backend="loky", verbose=0)(
        delayed(Narrow_Band)(sdiuse[:, kk], nx, ny, nz) for kk in range(Ne)
    )
    B = np.reshape(np.hstack(Xspitbk), (nx * ny * nz, Ne), "F")  #
    # B=np.hstack(Xspitbk)

    # Get signed distance of porosity field
    ensemblep = np.nan_to_num(ensemblep, copy=True, nan=Low_P)
    mexinip = np.zeros((Experts, Ne))
    sdiinip = np.zeros((ensemblep.shape[0], ensemblep.shape[1]))
    for j in range(Ne):
        sdip, mexxp, _ = get_clus(ensemblep[:, j], Experts)
        sdiinip[:, j] = np.ravel(sdip)
        mexinip[:, j] = np.ravel(mexxp)
    sdiusep = sdiinip

    # Get Narrow bands of porosity field
    Xspitb = Parallel(n_jobs=8, backend="loky", verbose=0)(
        delayed(Narrow_Band)(sdiusep[:, kk], nx, ny, nz) for kk in range(Ne)
    )
    Bp = np.reshape(np.hstack(Xspitb), (nx * ny * nz, Ne), "F")
    # Bp=np.hstack(Xspitb)
    # Make up others
    B_pixelK = cp.ones((sgsim.shape[0], sgsim.shape[1]))
    B_pixelporo = cp.ones((sgsim.shape[0], sgsim.shape[1]))

    overall = cp.vstack(
        [
            cp.asarray(sgsim),
            cp.asarray(sdiuse),
            cp.asarray(ensemblep),
            cp.asarray(sdiusep),
        ]
    )

    overallB = cp.vstack([B_pixelK, cp.asarray(B), B_pixelporo, cp.asarray(Bp)])

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
                cp.tile(cp.asarray(True_data), Ne)
                + (cp.sqrt(cp.asarray(alpha)) * cp.asarray(pertubations))
            )
            - Sim1
        )
    )

    Ynew = Y + cp.multiply(update_term, overallB)

    if Yet == 0:
        updated_ensemble = cp.asnumpy(Ynew[:sizeclem, :])
        updated_ensemble_LS = cp.asnumpy(Ynew[sizeclem : 2 * sizeclem, :])
        updated_ensemblep = cp.asnumpy(Ynew[2 * sizeclem : 3 * sizeclem, :])
        updated_ensemblep_LS = cp.asnumpy(Ynew[3 * sizeclem : 4 * sizeclem, :])

    else:
        updated_ensemble = Ynew[:sizeclem, :]
        updated_ensemble_LS = Ynew[sizeclem : 2 * sizeclem, :]
        updated_ensemblep = Ynew[2 * sizeclem : 3 * sizeclem, :]
        updated_ensemblep_LS = Ynew[3 * sizeclem : 4 * sizeclem, :]

    # Rectify the conflict for permeability ensemble
    meximkp = np.zeros((Experts, Ne))
    models_big = []
    for ja in range(Ne):
        _, mexx1, modelss = get_clus(updated_ensemble_LS[:, ja], Experts)
        meximkp[:, ja] = np.ravel(mexx1)
        models_big.append(modelss)

    Xspit2k = Parallel(n_jobs=num_cores, backend="loky", verbose=0)(
        delayed(rectifyy)(
            updated_ensemble[:, k],
            updated_ensemble_LS[:, k],
            meximkp[:, k],
            Experts,
            nx,
            ny,
            nz,
            models_big[k],
        )
        for k in range(Ne)
    )
    updated_yesK = np.reshape(np.hstack(Xspit2k), (-1, Ne), "F")  #
    # updated_yesK = np.hstack(Xspit2k)

    # Rectify the conflict for porosity ensemble
    meximkpp = np.zeros((Experts, Ne))
    models_bigp = []
    for ja in range(Ne):
        _, mexx1p, modelssp = get_clus(updated_ensemblep_LS[:, ja], Experts)
        meximkpp[:, ja] = np.ravel(mexx1p)
        models_bigp.append(modelssp)

    Xspit2kp = Parallel(n_jobs=num_cores, backend="loky", verbose=0)(
        delayed(rectifyy)(
            updated_ensemblep[:, k],
            updated_ensemblep_LS[:, k],
            meximkpp[:, k],
            Experts,
            nx,
            ny,
            nz,
            models_bigp[k],
        )
        for k in range(Ne)
    )
    updated_yesP = np.reshape(np.hstack(Xspit2kp), (-1, Ne), "F")  #
    # updated_yesP = np.hstack(Xspit2kp)
    # Honour well pixel values
    ensemblee, ensembleep = honour2(
        updated_yesP, updated_yesK, nx, ny, nz, Ne, High_K, Low_K, High_P, Low_P
    )

    return ensemblee, ensembleep


def inverse_to_pytorch(Ne, val, nx, ny, nz, device, make_up):
    X_unie = np.zeros((Ne, nz, nx + make_up, ny + make_up))
    for i in range(Ne):
        aa = np.zeros((nz, nx + make_up, ny + make_up))
        tempp = np.reshape(val[:, i], (nx, ny, nz), "F")
        for kk in range(int(nz)):
            newy = imresize(tempp[:, :, kk], output_shape=(nx + make_up, ny + make_up))
            aa[kk, :, :] = newy
        X_unie[i, :, :, :] = aa
    return torch.from_numpy(X_unie).to(device, dtype=torch.float32)


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

        model.add(Dense(128 * 11 * 11, activation="relu", input_dim=self.latent_dim))
        model.add(tensorflow.keras.layers.Reshape((11, 11, 128)))
        model.add(UpSampling2D(size=(3, 3)))
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
        plt.figure(figsize=(30, 30))
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

        plt.savefig("images/%d_perm.png" % epoch)
        plt.close()
        plt.clf()

    def save_model(self):
        def save(model, model_name):
            name = model_name + ".h5"
            model.save(name)

        save(self.generator, "generator")


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

    ax.set_box_aspect([nx, ny, 2])

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

        model.add(Dense(128 * 11 * 11, activation="relu", input_dim=self.latent_dim))
        model.add(tensorflow.keras.layers.Reshape((11, 11, 128)))
        model.add(UpSampling2D(size=(3, 3)))
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
        plt.figure(figsize=(30, 30))
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

        plt.savefig("images/%d_porosity.png" % epoch)
        plt.close()
        plt.clf()

    def save_model(self):
        def save(model, model_name):
            name = "../PACKETS/" + model_name + ".h5"
            model.save(name)

        save(self.generator, "generatorp")


def WGAN_LEARNING(High_K, Low_K, High_P, Low_P, nx, ny, nz):
    filename = "../PACKETS/Ganensemble.mat"
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
    latent_dim = 11 * 11 * 4
    wgan = WGAN(latent_dim, nx, ny, nz)
    wgan.train(epochs=4000, dataa=X_unie, batch_size=128, sample_interval=100)
    wgan.save_model()

    X_unie = np.zeros((N_enss, nx, ny, nz))
    for i in range(N_enss):
        X_unie[i, :, :, :] = np.reshape(dataasp[:, i], (nx, ny, nz), "F")
    latent_dim = 11 * 11 * 4
    wganp = WGANP(latent_dim, nx, ny, nz)
    wganp.train(epochs=4000, dataa=X_unie, batch_size=128, sample_interval=100)
    wganp.save_model()


def WGAN_LEARNING_PERM(High_K, Low_K, nx, ny, nz):
    filename = "../PACKETS/Ganensemble.mat"
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
    latent_dim = 11 * 11 * 4
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


def Regress(X_train, y_traind, namezz):
    namezz = "../PACKETS/" + namezz + ".h5"
    modelDNN = Sequential()
    modelDNN.add(Dense(5, activation="relu", input_dim=X_train.shape[1]))
    modelDNN.add(Dense(units=3, activation="relu"))
    modelDNN.add(Dense(units=1))
    # modelDNN.summary()
    modelDNN.compile(loss="mean_squared_error", optimizer="Adam", metrics=["mse"])
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)
    mc = ModelCheckpoint(
        namezz, monitor="val_loss", mode="min", verbose=0, save_best_only=True
    )
    a0 = X_train
    a0 = np.reshape(a0, (-1, X_train.shape[1]), "F")

    b0 = y_traind
    b0 = np.reshape(b0, (-1, y_traind.shape[1]), "F")
    gff = len(a0) // 100
    if gff < 1:
        gff = 1
    modelDNN.fit(
        a0,
        b0,
        validation_split=0.1,
        batch_size=gff,
        verbose=0,
        epochs=200,
        callbacks=[es, mc],
    )
    return modelDNN


def classify(X_train, y_traind, namezz):
    # from tensorflow.keras.layers import Dense
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.utils import to_categorical

    y_traind = to_categorical(y_traind.reshape(-1, 1))
    namezz = "../PACKETS/" + namezz + ".h5"
    modelDNN = Sequential()
    modelDNN.add(Dense(30, activation="relu", input_dim=X_train.shape[1]))
    modelDNN.add(Dense(units=20, activation="relu"))
    modelDNN.add(Dense(units=y_traind.shape[1], activation="softmax"))
    # modelDNN.summary()
    modelDNN.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=20)
    mc = ModelCheckpoint(
        namezz, monitor="val_accuracy", mode="max", verbose=0, save_best_only=True
    )
    a0 = X_train
    a0 = np.reshape(a0, (-1, X_train.shape[1]), "F")

    b0 = y_traind

    gff = len(a0) // 100
    if gff < 1:
        gff = 1
    modelDNN.fit(
        a0,
        b0,
        validation_split=0.1,
        batch_size=gff,
        verbose=0,
        epochs=200,
        callbacks=[es, mc],
    )
    return modelDNN


def Get_weights_Ensemble(nclusters):
    realii = []
    for k in range(nclusters):
        modeluse = load_model("../PACKETS/Regressor_Machine_Cluster_" + str(k) + ".h5")
        reali = Get_theta(modeluse)
        realii.append(reali)
    realii = np.vstack(realii)
    modelusec = load_model("../PACKETS/classifier.h5")
    realic = Get_theta(modelusec)
    ensemble = np.vstack([realii, realic])
    return ensemble


def Put_Weights_Ensemble(array, nclusters):
    array = array.reshape(-1, 1)
    regress_ensemble = array[: nclusters * 32, :].reshape(-1, 1)
    class_ensemble = array[nclusters * 32 :, :].reshape(-1, 1)
    arr = Split_Matrix(regress_ensemble, nclusters)
    for k in range(nclusters):
        modeluse = load_model("../PACKETS/Regressor_Machine_Cluster_" + str(k) + ".h5")
        modeluse = Put_Back_Regressor(arr[k].reshape(-1, 1), modeluse)
        modeluse.save("../PACKETS/Regressor_Machine_Cluster_" + str(k) + ".h5")
    modelusec = load_model("../PACKETS/classifier.h5")
    modelusec = Put_Back_Classifier(class_ensemble.reshape(-1, 1), modelusec)
    modelusec.save("../PACKETS/classifier.h5")


def Split_Matrix(matrix, sizee):
    x_split = np.split(matrix, sizee, axis=0)
    return x_split


def Get_theta(modelDNN):
    a = np.array(modelDNN.get_weights())
    a1 = np.reshape(a[0], (-1, 1), "F")
    b1 = np.reshape(a[1], (-1, 1), "F")
    a2 = np.reshape(a[2], (-1, 1), "F")
    b2 = np.reshape(a[3], (-1, 1), "F")
    a3 = np.reshape(a[4], (-1, 1), "F")
    b3 = np.reshape(a[5], (-1, 1), "F")
    reali = np.vstack([a1, b1, a2, b2, a3, b3])
    return reali


def Put_Back_Regressor(array, modelu):
    a1 = np.reshape(array[:5, :], (1, 5), "F")
    b1 = np.reshape(array[5:10, :], (5,), "F")
    a2 = np.reshape(array[10:25, :], (5, 3), "F")
    b2 = np.reshape(array[25:28, :], (3,), "F")
    a3 = np.reshape(array[28:31, :], (3, 1), "F")
    b3 = np.reshape(array[31, :], (1,), "F")

    amake = np.empty((6,), dtype=object)
    amake[0] = a1
    amake[1] = b1
    amake[2] = a2
    amake[3] = b2
    amake[4] = a3
    amake[5] = b3
    modelu.set_weights(amake)
    return modelu


def Put_Back_Classifier(array, modelu):
    a1 = np.reshape(array[:30, :], (1, 30), "F")
    b1 = np.reshape(array[30:60, :], (30,), "F")
    a2 = np.reshape(array[60:660, :], (30, 20), "F")
    b2 = np.reshape(array[660:680, :], (20,), "F")
    a3 = np.reshape(array[680:720, :], (20, 2), "F")
    b3 = np.reshape(array[720:722, :], (2,), "F")

    amake = np.empty((6,), dtype=object)
    amake[0] = a1
    amake[1] = b1
    amake[2] = a2
    amake[3] = b2
    amake[4] = a3
    amake[5] = b3
    modelu.set_weights(amake)
    return modelu


def CCR_Machine(inpuutj, outputtj):
    inpuutj = np.reshape(inpuutj, (-1, 1), "F")
    outputtj = np.reshape(outputtj, (-1, 1), "F")

    X = inpuutj
    y = outputtj
    numruth = X.shape[1]

    y_traind = y
    scaler1a = MinMaxScaler(feature_range=(0, 1))
    (scaler1a.fit(X))
    X = scaler1a.transform(X)
    scaler2a = MinMaxScaler(feature_range=(0, 1))
    (scaler2a.fit(y))
    y = scaler2a.transform(y)
    yruth = y
    filenamex = "clfx.asv"
    filenamey = "clfy.asv"
    pickle.dump(scaler1a, open(filenamex, "wb"))
    pickle.dump(scaler2a, open(filenamey, "wb"))
    y_traind = numruth * 10 * y
    matrix = np.concatenate((X, y_traind), axis=1)
    k = getoptimumk(matrix)
    nclusters = k
    nclusters = 2
    # print ('Optimal k is: ', nclusters)
    kmeans = MiniBatchKMeans(n_clusters=nclusters, max_iter=2000).fit(matrix)
    dd = kmeans.labels_
    dd = dd.T
    dd = np.reshape(dd, (-1, 1))
    # -------------------#---------------------------------#
    inputtrainclass = X
    outputtrainclass = np.reshape(dd, (-1, 1))
    classify(inputtrainclass, outputtrainclass, "Classifier")
    # Learn_Classification(inputtrainclass,outputtrainclass,nclusters)
    # print('Split for classifier problem')
    X_train = X
    y_train = dd
    # -------------------Regression----------------#
    for i in range(nclusters):
        # print('----- Learning cluster: ' + str(i+1) + ' | ' + str(nclusters))
        label0 = (np.asarray(np.where(y_train == i))).T
        a0 = X_train[label0[:, 0], :]
        a0 = np.reshape(a0, (-1, numruth), "F")
        b0 = yruth[label0[:, 0], :]
        b0 = np.reshape(b0, (-1, 1), "F")
        if a0.shape[0] != 0 and b0.shape[0] != 0:
            # model0.fit(a0, b0,verbose=False)
            Regress(a0, b0, "Regressor_Machine_Cluster_" + str(i))
    return nclusters


def Prediction_CCR__Machine(nclusters, inputtest):
    import numpy as np

    # print('Starting Prediction')
    inputtest = np.reshape(inputtest, (-1, 1), "F")
    filename1 = "../PACKETS/Classifier.h5"
    modelusec = load_model(filename1)
    loaded_model = modelusec
    filenamex = "../PACKETS/clfx.asv"
    filenamey = "../PACKETS/clfy.asv"
    clfx = pickle.load(open(filenamex, "rb"))
    clfy = pickle.load(open(filenamey, "rb"))
    inputtest = clfx.transform(inputtest)
    labelDA = loaded_model.predict(inputtest)
    labelDA = np.argmax(labelDA, axis=-1)
    numrowstest = len(inputtest)
    clementanswer = np.zeros((numrowstest, 1))
    labelDA = np.reshape(labelDA, (-1, 1), "F")
    for i in range(nclusters):
        # print('----- Predicting cluster: ' + str(i) + ' | ' + str(nclusters))
        filename2 = "../PACKETS/Regressor_Machine_Cluster_" + str(i) + ".h5"
        model0 = load_model(filename2)
        labelDA0 = (np.asarray(np.where(labelDA == i))).T
        #    ##----------------------##------------------------##
        a00 = inputtest[labelDA0[:, 0], :]
        a00 = np.reshape(a00, (-1, 1), "F")
        if a00.shape[0] != 0:
            clementanswer[labelDA0[:, 0], :] = np.reshape(model0.predict(a00), (-1, 1))

        clementanswer = clfy.inverse_transform(clementanswer)
    return clementanswer


def Learn_CCR_Ensemble(stringfaa, oldfolder, jj, inpuutj, outputtj):
    folder = stringfaa + str(jj)
    os.chdir(folder)
    CCR_Machine(inpuutj, outputtj)
    os.chdir(oldfolder)


def Theta_Get(stringfaa, oldfolder, jj, nclusters):
    folder = stringfaa + str(jj)
    os.chdir(folder)
    Ee = Get_weights_Ensemble(nclusters)
    os.chdir(oldfolder)
    return Ee


def Theta_put(stringfaa, oldfolder, jj, nclusters, arrayy):
    folder = stringfaa + str(jj)
    os.chdir(folder)
    Put_Weights_Ensemble(arrayy, nclusters)
    os.chdir(oldfolder)


def Prediction_CCR_Ensemble(stringfaa, oldfolder, jj, inpuutj, nclusters):
    folder = stringfaa + str(jj)
    os.chdir(folder)
    ouut = Prediction_CCR__Machine(nclusters, inpuutj)
    os.chdir(oldfolder)
    return ouut


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
    filename = "../PACKETS/Ganensemble.mat"
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
        32, 3, activation="gelu", strides=1, padding="same"
    )(encoder_inputs)
    x = tensorflow.keras.layers.Conv2D(
        64, 3, activation="gelu", strides=3, padding="same"
    )(x)
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(200, activation="gelu")(x)
    z_mean = tensorflow.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tensorflow.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = tensorflow.keras.layers.Dense(11 * 11 * 64, activation="gelu")(latent_inputs)
    x = tensorflow.keras.layers.Reshape((11, 11, 64))(x)
    x = tensorflow.keras.layers.Conv2DTranspose(
        64, 3, activation="gelu", strides=3, padding="same"
    )(x)
    x = tensorflow.keras.layers.Conv2DTranspose(
        32, 3, activation="gelu", strides=1, padding="same"
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


def Plot_performance_model(PINN, PINN2, nx, ny, namet, UIR, itt, dt, MAXZ, pini_alt):
    look = (PINN[itt, :, :]) * pini_alt
    look_sat = PINN2[itt, :, :]
    look_oil = 1 - look_sat

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, look, cmap="jet")
    plt.title("Pressure", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    # plt.clim(np.min(np.reshape(lookf,(-1,))),np.max(np.reshape(lookf,(-1,))))
    cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, look_sat, cmap="jet")
    plt.title("water_sat", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(0, 1)
    cbar1.ax.set_ylabel(" water_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

    plt.subplot(2, 2, 3)
    plt.pcolormesh(XX.T, YY.T, look_oil, cmap="jet")
    plt.title("oil_sat", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    plt.clim(0, 1)
    cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
    Add_marker(plt, XX, YY, wells)

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
if not os.path.exists("../PACKETS"):
    os.makedirs("../PACKETS")
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

if DEFAULT == 1:
    print(" Surrogate Model chosen = PINO")
    surrogate = 4
else:
    surrogate = None
    while True:
        surrogate = int(
            input(
                "Select surrogate method type:\n1=FNO [PhysicsNeMo Implementation]\n\
2=PINO [PhysicsNeMo Implemnation]\n3=PINO version 1 [Original paper implementation]\n\
4=PINO version 2 [Original paper implementation with Flux predicted]\n\
5=AFNO (data driven)\n6=AFNO ( data + physics driven)\n"
            )
        )
        if (surrogate > 6) or (surrogate < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-4")
        else:
            break

# load data

sizq = 1e4
nx, ny, nz = 33, 33, 1
BO = 1.1  # oil formation volume factor
BW = 1  # Water formation volume factor
UW = 1.0  # water viscosity in cP
UO = 2.5  # oil viscosity in cP
DX = 50.0  # size of pixel in x direction
DY = 50.0  # sixze of pixel in y direction
DZ = 20.0  # sizze of pixel in z direction

DX = cp.float32(DX)
DY = cp.float32(DY)
DZtempp = cp.float32(DZ)
XBLOCKS = cp.int32(nx)  # no of grid blocks in x direction
YBLOCKS = cp.int32(ny)  # no of grid blocks in x direction
ZBLOCKS = cp.int32(nz)  # no of grid blocks in z direction
UW = cp.float32(1)  # water viscosity in cP
UO = cp.float32(2.5)  # oil viscosity in cP
SWI = cp.float32(0.1)
SWR = cp.float32(0.1)
CFW = cp.float32(1e-5)  # water compressibility in 1/psi
CFO = cp.float32(1e-5)  # oil compressibility in 1/psi
CT = cp.float32(2e-5)  # Total compressibility in 1/psi
IWSo = 0.8  # initial oil saturation
IWSw = 0.2  # initial water saturation
pini = 1.0  # initial reservoir pressure
pini_alt = 1e3
sat_up = pini_alt * 10
S1 = cp.float32(IWSw)  # Initial water saturation
SO1 = cp.float32(IWSo)  # Initial oil saturation
P1 = cp.float32(pini_alt)  # Bubble point pressure psia
PB = P1
mpor, hpor = 0.05, 0.5  # minimum and maximum porosity
Low_P = mpor
High_P = hpor
BW = cp.float32(BW)  # Water formation volume factor
BO = cp.float32(BO)  # Oil formation volume factor
max_rwell = 500.0  #  maximum well radius (ft)
min_rwell = 40.0  #  minimum well radius (ft)
max_skin = 0.02  # maximum skin factor
min_skin = 0  # minimum skin factor
max_pwf_producer = 300.0  # maximum pwf in psi controller for producer
min_pwf_producer = 20.0  # maximum pwf in psi controller for producer
PATM = cp.float32(14.6959)  # Atmospheric pressure in psi
BlockTotal = XBLOCKS * YBLOCKS * ZBLOCKS  # Total Number of GridBlocks
LBLOCKS = cp.int32(XBLOCKS * YBLOCKS)  # GridBlocks per layer
# L = DX * XBLOCKS                                #Total Lengh
RE = 0.2 * DX


# training
# LUB, HUB = 1e-3,1 # Pressure rescale
LUB, HUB = 1e-1, 1  # Perm rescale limits
aay, bby = 50, 500  # Perm range mD limits
run = 1  # =run instance
scalei, scalei2, scalei3 = 1e3, 1e3, 1e2
lr = 1e-3  #'learning rate'
lr_div = 2  #'lr div factor to get the initial lr'
lr_pct = 0.3  #'percentage to reach the maximun lr, which is args.lr'
weight_decay = 0.0  # "weight decay"
weight_bound = 10  # "weight for boundary loss"

batch_size = 500  #'input batch size for training'
weight_bound2 = 1000  # "weight for data loss"
test_batch_size = batch_size  #'input batch size for testing'
timmee = 100.0  # float(input ('Enter the time step interval duration for simulation (days): '))
max_t = 3000.0  # float(input ('Enter the maximum time in days for simulation(days): '))
t_plot = int(
    timmee
)  # int(input ('Enter the plot frequency for simulation (days): ')) # plot every ...
MAXZ = 6000
steppi = int(max_t / timmee)
ntrain = (
    5000  # int((int(np.ceil((sizq - batch_size)/batch_size)) - steppi )* batch_size) \
)
# + (batch_size *2)
tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
dt = np.diff(tc2)[0]
use_LBFGS = False  # True
seed = 1  #'manual seed used in Tensor'
cuda = 0  #'cuda index'
choice = 1  #  1= Non-Gaussian prior, 2 = Gaussian prior
factorr = 0.1  # from [0 1] excluding the limits for PermZ
interior_points = ntrain
inj_rate = 500
LIR = 200  # lower injection rate
UIR = 2000  # uppwer injection rate

Nsmall = 20
rwell = 200
skin = 0
pwf_producer = 100
num_cores = 5  # cores on Local Machine
degg = 3  # Degree Polynomial
nclusters = 2  # sand and shale
N_inj = 4
N_pr = 4


# dt = timmee/max_t
print("Check the condition  CFL number")
ratioo = timmee / DX
if ratioo <= 1:
    print("Solution is unconditionally stable with time step, use IMPES method")
    CFL = 1  # IMPES
else:
    print(
        "Solution is conditionally stable with time step, Using Fully-Implicit method"
    )
    CFL = 2  # IMPLICT
    # step2 = int(input ('Enter the chop interval for time calculation (3-20): '))
    step2 = int(10)


# if seed is None:
#     seed = random.randint(1, 10000)
seed = 1  # 1 is the best
print("Random Seed: ", seed)


ra.seed(seed)
torch.manual_seed(seed)


device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")


input_channel = 7  # [K,F,Fw,phi,dt,Pini,Sini]
output_channel = 1 * steppi

if DEFAULT == 1:
    print("Default method is using the weighted adaptive ensemble Kalman inversion")
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
    print(" --------------Use constrained presure residual method--------- ")
    method = 2
    typee = 0
else:
    method = None
    while True:
        method = cp.int(
            input(
                "Select inverse problem solution method for pressure & saturation \
        solve -\n\
        1 = GMRES (Left-preconditioned)\n\
        2 = spsolve\n\
        3 = Conjugate Gradient\n\
        4 = LSQR\n\
        5 = CPR\n\
        6 = AMGX\n: "
            )
        )
        if (method > 6) or (method < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-6")
        else:
            break

    if method == 6:
        typee = cp.int(
            input(
                "Select JSON config -\n\
    1 = Standard\n\
    2 = Exotic\n\
    3 = DILU\n\
    4 = Aggregation DS\n\
    5 = Aggregation Jacobi\n\
    6 = Aggregation Thrust\n\
    7 = Aggregation Conjugate gradient\n\
    8 = PBICGSTAB_Aggregation with JACOBI smmothing\n\
    9 = CG DILU\n\
    10 = GMRES AMG\n: "
            )
        )
    else:
        typee = 0


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

print("*******************Load the trained Forward models*******************")
if surrogate == 4:
    decoder1 = ConvFullyConnectedArch(
        [Key("z", size=32)],
        [Key("pressure", size=steppi), Key("Tx", size=steppi), Key("Ty", size=steppi)],
    )
    modelP = FNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        dimension=2,
        decoder_net=decoder1,
    )

    decoder2 = ConvFullyConnectedArch(
        [Key("z", size=32)],
        [
            Key("water_sat", size=steppi),
            Key("Txw", size=steppi),
            Key("Tyw", size=steppi),
        ],
    )
    modelS = FNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        dimension=2,
        decoder_net=decoder2,
    )
if (surrogate == 1) or (surrogate == 2) or (surrogate == 3):
    decoder1 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("pressure", size=steppi)]
    )
    modelP = FNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        dimension=2,
        decoder_net=decoder1,
    )

    decoder2 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("water_sat", size=steppi)]
    )
    modelS = FNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        dimension=2,
        decoder_net=decoder2,
    )

if (surrogate == 5) or (surrogate == 6):
    modelP = AFNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        [Key("pressure", size=steppi)],
        (nx, ny),
        patch_size=3,
    )

    # Define AFNO model for forward model (saturation)
    modelS = AFNOArch(
        [
            Key("perm", size=1),
            Key("Q", size=1),
            Key("Qw", size=1),
            Key("Phi", size=1),
            Key("Time", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
        ],
        [Key("water_sat", size=steppi)],
        (nx, ny),
        patch_size=3,
    )

if surrogate == 1:
    print("-----------------Surrogate Model learned with FNO----------------")
    if not os.path.exists(("outputs/Forward_problem_FNO/ResSim/")):
        os.makedirs(("outputs/Forward_problem_FNO/ResSim/"))
    else:
        pass

    bb = os.path.isfile(
        "outputs/Forward_problem_FNO/ResSim/fno_forward_model_pressure.0.pth"
    )
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1cldZ75k-kIJQU51F1w17yRYFAAjoYOXU",
            "outputs/Forward_problem_FNO/ResSim/fno_forward_model_pressure.0.pth",
        )
        print("...Downlaod completed.......")

        os.chdir("outputs/Forward_problem_FNO/ResSim")
        print(" Surrogate model learned with FNO")

        modelP.load_state_dict(torch.load("fno_forward_model_pressure.0.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_FNO/ResSim")
        print(" Surrogate model learned with FNO")
        modelP.load_state_dict(torch.load("fno_forward_model_pressure.0.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)

    bba = os.path.isfile(
        "outputs/Forward_problem_FNO/ResSim/fno_forward_model_saturation.0.pth"
    )
    if not bba:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1IWTnWceqbCD3XdQmHOsw6Et9jS8hozML",
            "outputs/Forward_problem_FNO/ResSim/fno_forward_model_saturation.0.pth",
        )
        print("...Downlaod completed.......")
        os.chdir("outputs/Forward_problem_FNO/ResSim")
        print(" Surrogate model learned with FNO")

        modelS.load_state_dict(torch.load("fno_forward_model_saturation.0.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_FNO/ResSim")
        print(" Surrogate model learned with FNO")
        modelS.load_state_dict(torch.load("fno_forward_model_saturation.0.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
elif surrogate == 2:
    print("-----------------Surrogate Model learned with PINO----------------")
    if not os.path.exists(("outputs/Forward_problem_PINO/ResSim/")):
        os.makedirs(("outputs/Forward_problem_PINO/ResSim/"))
    else:
        pass
    bb = os.path.isfile(
        "outputs/Forward_problem_PINO/ResSim/pino_forward_model_pressure.0.pth"
    )
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1Df3NHyAMW4fdAVwdSyQEvuD8Z9gpsNwt",
            "outputs/Forward_problem_PINO/ResSim/pino_forward_model_pressure.0.pth",
        )
        print("...Downlaod completed.......")
        os.chdir("outputs/Forward_problem_PINO/ResSim")
        print(" Surrogate model learned with PINO")

        modelP.load_state_dict(torch.load("pino_forward_model_pressure.0.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_PINO/ResSim")
        print(" Surrogate model learned with PINO")
        modelP.load_state_dict(torch.load("pino_forward_model_pressure.0.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)

    bba = os.path.isfile(
        "outputs/Forward_problem_PINO/ResSim/pino_forward_model_saturation.0.pth"
    )
    if not bba:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1QAYQJy9_2FiBrxL8TYtTvTJNmsTaUSh4",
            "outputs/Forward_problem_PINO/ResSim/pino_forward_model_saturation.0.pth",
        )
        print("...Downlaod completed.......")
        os.chdir("outputs/Forward_problem_PINO/ResSim")
        print(" Surrogate model learned with PINO")

        modelS.load_state_dict(torch.load("pino_forward_model_saturation.0.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_PINO/ResSim")
        print(" Surrogate model learned with PINO")
        modelS.load_state_dict(torch.load("pino_forward_model_saturation.0.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
elif surrogate == 3:
    print("-----------------Surrogate Model learned with Original PINO----------------")
    if not os.path.exists(("outputs/Forward_problem_PINO2/")):
        os.makedirs(("outputs/Forward_problem_PINO2/"))
    else:
        pass

    bb = os.path.isfile("outputs/Forward_problem_PINO2/pressure_model.pth")
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1sv_tbB91EWcJOBWdhhnTKspoAtR564o4",
            "outputs/Forward_problem_PINO2/pressure_model.pth",
        )
        print("...Downlaod completed.......")
        os.chdir("outputs/Forward_problem_PINO2")
        print(" Surrogate model learned with original PINO")

        modelP.load_state_dict(torch.load("pressure_model.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_PINO2")
        print(" Surrogate model learned with original PINO")
        modelP.load_state_dict(torch.load("pressure_model.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)

    bba = os.path.isfile("outputs/Forward_problem_PINO2/saturation_model.pth")
    if not bba:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "13BGxwvN2IV0eo0vbUxb16BhJ2MtJ-s0W",
            "outputs/Forward_problem_PINO2/saturation_model.pth",
        )
        print("...Downlaod completed.......")
        os.chdir("outputs/Forward_problem_PINO2")
        print(" Surrogate model learned with original PINO")

        modelS.load_state_dict(torch.load("saturation_model.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_PINO2")
        print(" Surrogate model learned with PINO")
        modelS.load_state_dict(torch.load("saturation_model.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
elif surrogate == 4:
    print(
        "-----------------Surrogate Model learned with Original PINO Version 2----------------"
    )
    if not os.path.exists(("outputs/Forward_problem_PINO3/")):
        os.makedirs(("outputs/Forward_problem_PINO3/"))
    else:
        pass

    bb = os.path.isfile("outputs/Forward_problem_PINO3/pressure_model.pth")
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1jihu5qi0TWh22RuIxNEvUxvixUrnxzxB",
            "outputs/Forward_problem_PINO3/pressure_model.pth",
        )
        print("...Downlaod completed.......")
        os.chdir("outputs/Forward_problem_PINO3")
        print(" Surrogate model learned with original PINO")

        modelP.load_state_dict(torch.load("pressure_model.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_PINO3")
        print(" Surrogate model learned with original PINO")

        modelP.load_state_dict(torch.load("pressure_model.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)

    bba = os.path.isfile("outputs/Forward_problem_PINO3/saturation_model.pth")
    if not bba:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1FibuOGfBZHOrZgiPf6lggcCpKlaMaq7A",
            "outputs/Forward_problem_PINO3/saturation_model.pth",
        )
        print("...Downlaod completed.......")
        os.chdir("outputs/Forward_problem_PINO3")
        print(" Surrogate model learned with original PINO")

        modelS.load_state_dict(torch.load("saturation_model.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_PINO3")
        print(" Surrogate model learned with PINO")
        modelS.load_state_dict(torch.load("saturation_model.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
elif surrogate == 5:
    print(
        "-----------------Surrogate Model learned with AFNO (data driven)----------------"
    )
    if not os.path.exists(("outputs/Forward_problem_AFNOD/ResSim/")):
        os.makedirs(("outputs/Forward_problem_AFNOD/ResSim/"))
    else:
        pass

    bb = os.path.isfile(
        "outputs/Forward_problem_AFNOD/ResSim/afno_forward_model_pressure.0.pth"
    )
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1v1s5WtJ4HVYb1_Qm8GRLxlEbpOqzxMZU",
            "outputs/Forward_problem_AFNOD/ResSim/afno_forward_model_pressure.0.pth",
        )
        print("...Downlaod completed.......")

        os.chdir("outputs/Forward_problem_AFNOD/ResSim")
        print(" Surrogate model learned with AFNO (data driven)")

        modelP.load_state_dict(torch.load("afno_forward_model_pressure.0.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_AFNOD/ResSim")
        print(" Surrogate model learned with AFNO (data-driven)")
        modelP.load_state_dict(torch.load("afno_forward_model_pressure.0.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)

    bba = os.path.isfile(
        "outputs/Forward_problem_AFNOD/ResSim/afno_forward_model_saturation.0.pth"
    )
    if not bba:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1UdAH2WyXdY87VncoDUjiM-SzmnSSpelY",
            "outputs/Forward_problem_AFNOD/ResSim/afno_forward_model_saturation.0.pth",
        )
        print("...Downlaod completed.......")
        os.chdir("outputs/Forward_problem_AFNOD/ResSim")
        print(" Surrogate model learned with AFNO (data-driven)")

        modelS.load_state_dict(torch.load("afno_forward_model_saturation.0.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_AFNOD/ResSim")
        print(" Surrogate model learned with AFNO (data-driven)")
        modelS.load_state_dict(torch.load("afno_forward_model_saturation.0.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
elif surrogate == 6:
    print(
        "-----------------Surrogate Model learned with AFNO (physics driven)----------------"
    )
    if not os.path.exists(("outputs/Forward_problem_AFNOP/ResSim/")):
        os.makedirs(("outputs/Forward_problem_AFNOP/ResSim/"))
    else:
        pass

    bb = os.path.isfile(
        "outputs/Forward_problem_AFNOP/ResSim/afnop_forward_model_pressure.0.pth"
    )
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "14Iq6ERuHfrd6u5HMiUVNU3zzaJRusnmJ",
            "outputs/Forward_problem_AFNOP/ResSim/afnop_forward_model_pressure.0.pth",
        )
        print("...Downlaod completed.......")

        os.chdir("outputs/Forward_problem_AFNOP/ResSim")
        print(" Surrogate model learned with AFNO (physics driven)")

        modelP.load_state_dict(torch.load("afnop_forward_model_pressure.0.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_AFNOP/ResSim")
        print(" Surrogate model learned with AFNO (physics-driven)")
        modelP.load_state_dict(torch.load("afnop_forward_model_pressure.0.pth"))
        modelP = modelP.to(device)
        modelP.eval()
        os.chdir(oldfolder)

    bba = os.path.isfile(
        "outputs/Forward_problem_AFNOP/ResSim/afnop_forward_model_saturation.0.pth"
    )
    if not bba:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1qazyOOIfoXRHyJ99kfW3vICFEtPtR9sV",
            "outputs/Forward_problem_AFNOP/ResSim/afnop_forward_model_saturation.0.pth",
        )
        print("...Downlaod completed.......")
        os.chdir("outputs/Forward_problem_AFNOP/ResSim")
        print(" Surrogate model learned with AFNO (physics-driven)")

        modelS.load_state_dict(torch.load("afnop_forward_model_saturation.0.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
    else:
        os.chdir("outputs/Forward_problem_AFNOP/ResSim")
        print(" Surrogate model learned with AFNO (physics-driven)")
        modelS.load_state_dict(torch.load("afnop_forward_model_saturation.0.pth"))
        modelS = modelS.to(device)
        modelS.eval()
        os.chdir(oldfolder)
print("********************Model Loaded*************************************")


# 4 injector and 4 producer wells
wells = np.array(
    [1, 24, 1, 3, 3, 1, 31, 1, 1, 31, 31, 1, 7, 9, 2, 14, 12, 2, 28, 19, 2, 14, 27, 2]
)
wells = np.reshape(wells, (-1, 3), "C")


# True model
bba = os.path.isfile("../PACKETS/iglesias2.out")
if not bba:
    print("....Downloading Please hold.........")
    download_file_from_google_drive(
        "1_9VRt8tEOF6IV7GvUnD7CFVM40DMHkxn", "../PACKETS/iglesias2.out"
    )
    print("...Downlaod completed.......")


Truee = np.genfromtxt("../PACKETS/iglesias2.out", dtype="float")
Truee = np.reshape(Truee, (-1,))
aay1 = 50  # np.min(Truee)
bby1 = 500  # np.max(Truee)

Low_K1, High_K1 = aay1, bby1

print("Start training with Adam optimisers...................................")

perm_high = bby  # np.asscalar(np.amax(perm,axis=0)) + 300
perm_low = aay  # np.asscalar(np.amin(perm,axis=0))-50


poro_high = High_P  # np.asscalar(np.amax(poro,axis=0))+0.3
poro_low = Low_P  # np.asscalar(np.amin(poro,axis=0))-0.1

High_K, Low_K, High_P, Low_P = perm_high, perm_low, poro_high, poro_low
if not os.path.exists("../HM_RESULTS"):
    os.makedirs("../HM_RESULTS")
else:
    shutil.rmtree("../HM_RESULTS")
    os.makedirs("../HM_RESULTS")

print("")


print("")
print("---------------------------------------------------------------------")
print("-------------------------Simulation----------------------------------")
locc = 10
if DEFAULT == 1:
    BASSE = 2
else:
    BASSE = None
    while True:
        BASSE = int(
            input(
                "Generate the covariance noise matrix using\n\
percentage of data value = 1:\nConstant float value = 2:\n"
            )
        )
        if (BASSE > 2) or (BASSE < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break

print("")
print("---------------------------------------------------------------------")
print("")
print("--------------------Historical data Measurement----------------------")


print("Read Historical data")


os.chdir("../Forward_problem_results/PINO/TRUE")
True_measurement = pd.read_csv("RSM_FVM.csv")
True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
True_mat = True_measurement
Plot_RSM_singleT(True_mat, "Historical.png")
Presz = True_mat[:, 1:5] / scalei3
Oilz = True_mat[:, 5:9] / scalei
Watsz = True_mat[:, 9:13] / scalei2
wctz = True_mat[:, 13:]
True_data = np.hstack([Presz, Oilz, Watsz, wctz])
# True_data = np.hstack([Presz,Oilz,Watsz,wctz])
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


ressimmastertest = os.path.join(oldfolder, "../Training_Images")

# permx,Jesus,TIs,yett = Select_TI(oldfolder,ressimmastertest,N_small,\
#                                  nx,ny,nz,True_data,Low_K,High_K)
# """

os.chdir(oldfolder)
TII = 3
if TII == 3:
    print("TI = 3")
    os.chdir("../Training_Images")
    kq = np.genfromtxt("iglesias2.out", skip_header=0, dtype="float")
    os.chdir(oldfolder)
    kq = kq.reshape(-1, 1)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfy.fit(kq))
    kq = np.reshape(kq, (nx, ny), "F")
    kjennq = kq
    permx = kjennq

else:
    os.chdir("../Training_Images")
    kq = np.genfromtxt("TI_2.out", skip_header=3, dtype="float")
    kq = kq.reshape(-1, 1)
    os.chdir(oldfolder)
    clfy = MinMaxScaler(feature_range=(Low_K, High_K))
    (clfy.fit(kq))
    kq = clfy.transform(kq)
    kq = np.reshape(kq, (250, 250), "F")
    kjennq = kq.T
    permx = kjennq

True_K = np.reshape(permx, (-1, 1), "F")
clfyy = MinMaxScaler(feature_range=(Low_P, High_P))
(clfyy.fit(True_K))
True_P = clfyy.transform(True_K)

# bb=os.path.isfile('MAP3.h5')
# if (bb==False):
#     parad2_TI(True_K,True_P,'MAP3')

# permx=TIs['TI1']
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

"""
Select_TI_2(oldfolder,ressimmastertest,nx,ny,nz,perm_high,\
              perm_low,poro_high,poro_low,yett)
"""
namev = "../PACKETS/MAP" + str(yett) + ".h5"
# machine_map = load_model(namev)
if Jesus == 1:
    print(" Multiple point statistics chosen")
    print("Plot the TI selected")
    plt.figure(figsize=(10, 10))
    plt.imshow(permx)
    # plt.gray()
    plt.title("Training_Image Selected ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    os.chdir("../HM_RESULTS")
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
print("")
print("---------------------------------------------------------------------")
print("")
print("---------------------------------------------------------------------")

print("")
print("---------------------------------------------------------------------")
noise_level = 1e-2
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
    Technique_REKI = 3
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
        2 = ADAPT_REKI_KSVD (Adaptive Ensemble Kalman Inversion + KSVD)\n\
        3 = ADAPT_REKI_AE  (Adaptive Ensemble Kalman Inversion + Convolution Autoencoder)\n\
        4 = ADAPT_REKI_DA  (Adaptive Ensemble Kalman Inversion + Denoising Autoencoder)\n\
        5 = ADAPT_REKI_LEVELSET  (Adaptive Ensemble Kalman Inversion + Level Set)\n\
        6 = ADAPT_REKI_NORMAL_SCORE  (Adaptive Ensemble Kalman Inversion + Normal Score)\n\
        7 = ADAPT_REKI_DE_GAN  (Adaptive Ensemble Kalman Inversion + Convolution Autoencoder + GAN)\n\
        8 = ADAPT_REKI_GAN  (Adaptive Ensemble Kalman Inversion + GAN on the permeability field alone:\
This method is optimal for multiple point statistics alone (MPS))\n\
        9 = ADAPT_REKI_KMEANS  (Adaptive Ensemble Kalman Inversion + KMEANS on the permeability field alone:\
This method is optimal for multiple point statistics alone (MPS))\n\
        10 = ADAPT_REKI_VAE  (Adaptive Ensemble Kalman Inversion + variational\
convolutional autoencoder on the permeability field alone:\
This method is optimal for multiple point statistics alone (MPS))\n\
        11 = ADAPT_REKI_CCR-MoE  (Adaptive Ensemble Kalman Inversion + Mixture of\
Experts parametrisation on the permeability field alone:\
This method is optimal for multiple point statistics alone (MPS))\n\
        12 = ADAPT_REKI_DCT  (Adaptive Ensemble Kalman Inversion with DCT parametrisation\n\
        13 = ADAPT_REKI with Logit transformation\n\
        14 = ADAPT_REKI with Generative Diffusion model\n"
            )
        )
        if (Technique_REKI > 14) or (Technique_REKI < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-14")
        else:
            break
if DEFAULT == 1:
    Termm = 20
else:
    Termm = None
    while True:
        Termm = int(input("Enter number of iterations (6-20)\n"))
        if (Termm > 20) or (Termm < 6):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 6-20")
        else:
            break

if DEFAULT == 1:
    bb = os.path.isfile("../PACKETS/Training4.mat")
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1I-27_S53ORRFB_hIN_41r3Ntc6PpOE40", "../PACKETS/Training4.mat"
        )
        print("...Downlaod completed.......")

        print("Load simulated labelled training data from MAT file")
        matt = sio.loadmat("../PACKETS/Training4.mat")
        X_data1 = matt["INPUT"]
    else:
        print("Load simulated labelled training data from MAT file")
        matt = sio.loadmat("../PACKETS/Training4.mat")
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
        Ne = int(
            input("Number of realisations used for history matching (100-2000) : ")
        )
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
            see = intial_ensemble(nx, ny, nz, (N_ens + 100), permx)

            ini_ensemble = see
            sio.savemat("../PACKETS/Ganensemble.mat", {"Z": ini_ensemble})
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

            ensemblepy = ensemble_pytorch(
                ensemble,
                ensemblep,
                LUB,
                HUB,
                mpor,
                hpor,
                inj_rate,
                dt,
                IWSw,
                nx,
                ny,
                nz,
                ensemble.shape[1],
                input_channel,
            )

            mazw = 0  # Dont smooth the presure field
            simDataa, _, _, _, _ = Forward_model_ensemble(
                modelP,
                modelS,
                ensemblepy,
                rwell,
                skin,
                pwf_producer,
                mazw,
                steppi,
                ensemble.shape[1],
                DX,
                pini_alt,
                UW,
                BW,
                UO,
                BO,
                SWI,
                SWR,
                DZ,
                dt,
                MAXZ,
            )
            True_data = np.reshape(True_data, (-1, 1), "F")
            _, _, cc = funcGetDataMismatch(simDataa, True_data)
            sorted_indices = np.argsort(cc.ravel())
            first_ne = sorted_indices[:Ne]

            avs = []
            for kk in range(Ne):
                indd = first_ne[kk]
                ain = ini_ensemblef[:, indd]
                ain = ain.reshape(-1, 1)
                avs.append(ain)
            avs = np.hstack(avs)

            ini_ensemblee = avs

            clfye = MinMaxScaler(feature_range=(Low_K1, High_K1))
            (clfye.fit(ini_ensemblee))
            ini_ensemble = clfye.transform(ini_ensemblee)

        else:
            print("Use already generated ensemble from Google drive folder")
            if Ne <= 2000:
                # yes
                bb = os.path.isfile("../PACKETS/Training4.mat")
                if not bb:
                    print("....Downloading Please hold.........")
                    download_file_from_google_drive(
                        "1I-27_S53ORRFB_hIN_41r3Ntc6PpOE40", "../PACKETS/Training4.mat"
                    )
                    print("...Downlaod completed.......")

                    print("Load simulated labelled training data from MAT file")
                    matt = sio.loadmat("../PACKETS/Training4.mat")
                    X_data1 = matt["INPUT"]
                else:
                    print("Load simulated labelled training data from MAT file")
                    matt = sio.loadmat("../PACKETS/Training4.mat")
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
                bb = os.path.isfile(("../PACKETS/Ganensemble.mat"))
                if not bb:
                    print(
                        "Get initial geology from saved Multiple-point-statistics run"
                    )
                    print("....Downloading Please hold.........")
                    download_file_from_google_drive(
                        "1w81M5M2S0PD9CF2761dxmiKQ5c0OFPaH",
                        "../PACKETS/Ganensemble.mat",
                    )
                    print("...Downlaod completed.......")
                else:
                    pass
                filename = "../PACKETS/Ganensemble.mat"  # Ensemble generated offline

                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                index = np.random.choice(
                    ini_ensemblef.shape[1], Ne + 100, replace=False
                )

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

                ensemblepy = ensemble_pytorch(
                    ensemble,
                    ensemblep,
                    LUB,
                    HUB,
                    mpor,
                    hpor,
                    inj_rate,
                    dt,
                    IWSw,
                    nx,
                    ny,
                    nz,
                    ensemble.shape[1],
                    input_channel,
                )

                mazw = 0  # Dont smooth the presure field
                simDataa, _, _, _, _ = Forward_model_ensemble(
                    modelP,
                    modelS,
                    ensemblepy,
                    rwell,
                    skin,
                    pwf_producer,
                    mazw,
                    steppi,
                    ensemble.shape[1],
                    DX,
                    pini_alt,
                    UW,
                    BW,
                    UO,
                    BO,
                    SWI,
                    SWR,
                    DZ,
                    dt,
                    MAXZ,
                )
                True_data = np.reshape(True_data, (-1, 1), "F")
                _, _, cc = funcGetDataMismatch(simDataa, True_data)
                sorted_indices = np.argsort(cc.ravel())
                first_ne = sorted_indices[:Ne]

                avs = []
                for kk in range(Ne):
                    indd = first_ne[kk]
                    ain = ini_ensemblef[:, indd]
                    ain = ain.reshape(-1, 1)
                    avs.append(ain)
                avs = np.hstack(avs)

                ini_ensemblee = avs

                clfye = MinMaxScaler(feature_range=(Low_K1, High_K1))
                (clfye.fit(ini_ensemblee))
                ini_ensemble = clfye.transform(ini_ensemblee)

    else:
        if afresh == 1:
            ini_ensemblef = initial_ensemble_gaussian(
                nx, ny, nz, (N_ens + 10), Low_K, High_K
            )
            sio.savemat("../PACKETS/Ganensemble_gauss.mat", {"Z": ini_ensemblef})
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

            ensemblepy = ensemble_pytorch(
                ensemble,
                ensemblep,
                LUB,
                HUB,
                mpor,
                hpor,
                inj_rate,
                dt,
                IWSw,
                nx,
                ny,
                nz,
                ensemble.shape[1],
                input_channel,
            )

            mazw = 0  # Dont smooth the presure field
            simDataa, _, _, _, _ = Forward_model_ensemble(
                modelP,
                modelS,
                ensemblepy,
                rwell,
                skin,
                pwf_producer,
                mazw,
                steppi,
                ensemble.shape[1],
                DX,
                pini_alt,
                UW,
                BW,
                UO,
                BO,
                SWI,
                SWR,
                DZ,
                dt,
                MAXZ,
            )
            True_data = np.reshape(True_data, (-1, 1), "F")
            _, _, cc = funcGetDataMismatch(simDataa, True_data)
            sorted_indices = np.argsort(cc.ravel())
            first_ne = sorted_indices[:Ne]

            avs = []
            for kk in range(Ne):
                indd = first_ne[kk]
                ain = ini_ensemblef[:, indd]
                ain = ain.reshape(-1, 1)
                avs.append(ain)
            avs = np.hstack(avs)

            ini_ensemblee = avs

            clfye = MinMaxScaler(feature_range=(Low_K1, High_K1))
            (clfye.fit(ini_ensemblee))
            ini_ensemble = clfye.transform(ini_ensemblee)
        else:
            filename = "../PACKETS/Ganensemble_gauss.mat"  # Ensemble generated offline
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

            ensemblepy = ensemble_pytorch(
                ensemble,
                ensemblep,
                LUB,
                HUB,
                mpor,
                hpor,
                inj_rate,
                dt,
                IWSw,
                nx,
                ny,
                nz,
                ensemble.shape[1],
                input_channel,
            )

            mazw = 0  # Dont smooth the presure field
            simDataa, _, _, _, _ = Forward_model_ensemble(
                modelP,
                modelS,
                ensemblepy,
                rwell,
                skin,
                pwf_producer,
                mazw,
                steppi,
                ensemble.shape[1],
                DX,
                pini_alt,
                UW,
                BW,
                UO,
                BO,
                SWI,
                SWR,
                DZ,
                dt,
                MAXZ,
            )
            True_data = np.reshape(True_data, (-1, 1), "F")
            _, _, cc = funcGetDataMismatch(simDataa, True_data)
            sorted_indices = np.argsort(cc.ravel())
            first_ne = sorted_indices[:Ne]

            avs = []
            for kk in range(Ne):
                indd = first_ne[kk]
                ain = ini_ensemblef[:, indd]
                ain = ain.reshape(-1, 1)
                avs.append(ain)
            avs = np.hstack(avs)

            ini_ensemblee = avs

            clfye = MinMaxScaler(feature_range=(Low_K1, High_K1))
            (clfye.fit(ini_ensemblee))
            ini_ensemble = clfye.transform(ini_ensemblee)

    ini_ensemble = shuffle(ini_ensemble, axis=1)


start_time = time.time()
if Technique_REKI == 1:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Denoising Autoencoder \
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
        bb = os.path.isfile("../PACKETS/denosingautoencoder.h5")
        # bb2=os.path.isfile('denosingautoencoderp.h5')
        if not bb:
            if use_pretrained == 1:
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "17Ekju_MvRs_oOhDvpfirp8-rFy6x1D2u",
                    "../PACKETS/denosingautoencoder.h5",
                )
                print("...Downlaod completed.......")

            else:
                DenosingAutoencoder(
                    nx, ny, nz, High_K1, Low_K1
                )  # Learn for permeability
            # DenosingAutoencoderp(nx,ny,nz,machine_map,N_ens,High_P) #Learn for porosity
        else:
            pass
        bb2 = os.path.isfile("../PACKETS/denosingautoencoderp.h5")
        if not bb2:
            if use_pretrained == 1:
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1YBk_xwT175rOg3wbaTRZ6ZaCASPsvqoL",
                    "../PACKETS/denosingautoencoderp.h5",
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
    os.chdir("../Forward_problem_results/PINO/TRUE")
    True_measurement = pd.read_csv("RSM_FVM.csv")
    True_measurement = True_measurement.values.astype(np.float32)[:, 1:]
    True_mat = True_measurement
    Presz = True_mat[:, 1:5] / scalei3
    Oilz = True_mat[:, 5:9] / scalei
    Watsz = True_mat[:, 9:13] / scalei2
    wctz = True_mat[:, 13:]
    # True_data = np.hstack([Presz,Oilz,Watsz,wctz])
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

        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )

        ini_K = ensemble
        ini_p = ensemblep

        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI")
        os.makedirs("../HM_RESULTS/ADAPT_REKI")

    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI Model")

    os.chdir("../HM_RESULTS/ADAPT_REKI")
    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    print(" Plot P10,P50,P90")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    print("PINO Surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        _,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )

    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI ENDED----------------------------")
elif Technique_REKI == 2:
    print("")
    print("Adaptive Regularised Ensemble Kalman Inversion with KSVD Parametrisation")
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")

    os.chdir(oldfolder)
    print("Learn KSVD Over complete dictionary of the permeability and porosity fields")
    print("")
    bb = os.path.isfile("../PACKETS/Dictionary_Perm.mat")
    if not bb:
        Dicclem = Learn_Overcomplete_Dictionary(Ne)
        # Dicclem=Make_Bimodal(Dicclem,2)
        sio.savemat("../PACKETS/Dictionary_Perm.mat", {"Z": Dicclem})
    else:
        matt = sio.loadmat("../PACKETS/Dictionary_Perm.mat")
        Dicclem = matt["Z"]
        # Dicclem=Make_Bimodal(Dicclem,2)
    # Dicclemp=Learn_Overcomplete_Dictionaryp(Ne)

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)

    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )

        ini_K = ensemble
        ini_p = ensemblep

        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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
        ensembledct = Sparse_coding(
            Dicclem, sgsim, Ne
        )  # Get Sparse signal of permeability
        # if ii==0:
        #     ensembledct=Sparse_coding(Dicclem,sgsim,Ne)#Get Sparse signal of permeability
        #     #ensembledctp=Sparse_coding(Dicclemp,ensemblep,Ne)#Get Sparse signal of porosity
        # else:
        #     ensembledct=updated_ensembleksvd

        shapalla = ensembledct.shape[0]

        overall = cp.asarray(ensembledct)
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

        updated_ensemble = Recover_Dictionary_Saarse(
            Dicclem, updated_ensembleksvd, Low_K1, High_K1
        )
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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_KSVD"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_KSVD")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_KSVD")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_KSVD")

    # shutil.copy2('masterreal.data','ADAPT_REKI_KSVD')
    print("PINO surrogate Reservoir Simulator Forwarding - ADAPT_REKI_KSVD Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/ADAPT_REKI_KSVD")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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
    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    # shutil.copy2('masterreal.data','PERCENTILE')
    print("PINO Surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_KSVD ENDED-------------------")
elif Technique_REKI == 3:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Convolution Autoencoder \
Parametrisation\n"
    )
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    print("-------------------------learn Autoencoder------------------------")
    bb = os.path.isfile("../PACKETS/encoder.h5")
    bb2 = os.path.isfile("../PACKETS/encoderp.h5")
    if not bb:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1_99X3OrcK4Q8ym325cdJ3z0KX7YiduVw", "../PACKETS/encoder.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1appHPIKQbOdxkHkwOfLUZMpuheg6ZgKd", "../PACKETS/decoder.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1SMZ_hJ_drEVbagjPmcrjxDUB-lVniXPi", "../PACKETS/autoencoder.h5"
            )
            print("...Downlaod completed.......")

        else:
            Autoencoder2(nx, ny, nz, High_K1, Low_K1)  # Learn for permeability

    if not bb2:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1FWmvrFHwCSuMPxTBqG_9M-CSapOR6hRY", "../PACKETS/encoderp.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "19B6RmkYzDih4r3hasuaTSttIF0GfKKuw", "../PACKETS/decoderp.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1SHYUSo2DwFV7OIdfb5SWUoP27dn3nEmI", "../PACKETS/autoencoderp.h5"
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
            bb = os.path.isfile("../PACKETS/denosingautoencoder.h5")
            # bb2=os.path.isfile('denosingautoencoderp.h5')
            if not bb:
                if use_pretrained == 1:
                    print("....Downloading Please hold.........")
                    download_file_from_google_drive(
                        "17Ekju_MvRs_oOhDvpfirp8-rFy6x1D2u",
                        "../PACKETS/denosingautoencoder.h5",
                    )
                    print("...Downlaod completed.......")
                else:
                    DenosingAutoencoder(
                        nx, ny, nz, High_K1, Low_K1
                    )  # Learn for permeability
                # DenosingAutoencoderp(nx,ny,nz,machine_map,N_ens,High_P) #Learn for porosity
            else:
                pass
            bb2 = os.path.isfile("../PACKETS/denosingautoencoderp.h5")
            if not bb2:
                if use_pretrained == 1:
                    print("....Downloading Please hold.........")
                    download_file_from_google_drive(
                        "1YBk_xwT175rOg3wbaTRZ6ZaCASPsvqoL",
                        "../PACKETS/denosingautoencoderp.h5",
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

    plt.figure(figsize=(10, 10))
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 0], cmap="jet")
    plt.title("Recovered ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 0], cmap="jet")
    plt.title("original ", fontsize=15)
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
    os.chdir("../HM_RESULTS")
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep

        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_AE"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_AE")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_AE")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_AE")

    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_AE Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/ADAPT_REKI_AE")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)
    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    print("PINO Surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        5,
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )

    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_CAEE ENDED-------------------")
elif Technique_REKI == 4:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with continous Denoising Autoencoder\n"
    )
    print("Starting the History matching with ", str(Ne) + " Ensemble members")

    if Geostats == 1:
        bb = os.path.isfile("../PACKETS/denosingautoencoder.h5")
        # bb2=os.path.isfile('denosingautoencoderp.h5')
        if not bb:
            if use_pretrained == 1:
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "17Ekju_MvRs_oOhDvpfirp8-rFy6x1D2u",
                    "../PACKETS/denosingautoencoder.h5",
                )
                print("...Downlaod completed.......")
            else:
                DenosingAutoencoder(
                    nx, ny, nz, High_K1, Low_K1
                )  # Learn for permeability
            # DenosingAutoencoderp(nx,ny,nz,machine_map,N_ens,High_P) #Learn for porosity
        else:
            pass
        bb2 = os.path.isfile("../PACKETS/denosingautoencoderp.h5")
        if not bb2:
            if use_pretrained == 1:
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1YBk_xwT175rOg3wbaTRZ6ZaCASPsvqoL",
                    "../PACKETS/denosingautoencoderp.h5",
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_DA"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_DA")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_DA")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_DA")

    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_DA Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/ADAPT_REKI_DA")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")
    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    print("PINO Surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_DA ENDED--------------------")

elif Technique_REKI == 5:
    print("")
    print("Adaptive Regularised Ensemble Kalman Inversion with Level Set \n")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    Experts = int(input("Enter number of Facies: "))

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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    mean_cost = []
    best_cost = []
    alpha_big = []

    ensemble_meanK = []
    ensemble_meanP = []

    ensemble_bestK = []
    ensemble_bestP = []

    ensembles = []
    ensemblesp = []

    while snn < 1:
        print("Iteration --" + str(ii + 1) + " | " + str(Termm))
        print("****************************************************************")
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

        ensemble, ensemblep = REKI_ASSIMILATION_LEVELSET(
            ensemble,
            ensemblep,
            simDatafinal,
            alpha,
            True_dataa,
            Ne,
            pertubations,
            Experts,
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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_LEVELSET"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_LEVELSET")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_LEVELSET")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_LEVELSET")

    # shutil.copy2('masterreal.data','ADAPT_REKI_LEVELSET')
    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_LEVELSET Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/ADAPT_REKI_LEVELSET")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)
    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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
    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)
    # ensembleout1=np.hstack([controlbest,controljj2,controlbad])
    # ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    print("PINO surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)

    print(
        "--------------------SECTION ADAPTIVE REKI_REKI_LEVELSET ENDED--------------------"
    )

elif Technique_REKI == 6:
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

        # ensemble,ensemblep,CM=REKI_ASSIMILATION_NORMAL_SCORE(ensemble,\
        # ensemblep,CM,simDatafinal,inversion,alpha,True_data,Ne,\
        # pertubations,modelError,Yet,nx,ny,nz,High_K,Low_K,High_P,\
        # Low_P,CDd,quantileperm,quantileporo)

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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensmble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_NORMAL_SCORE"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_NORMAL_SCORE")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_NORMAL_SCORE")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_NORMAL_SCORE")

    # shutil.copy2('masterreal.data','ADAPT_REKI_NORMAL_SCORE')
    print(
        "PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_NORMAL_SCORE Model"
    )

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/ADAPT_REKI_NORMAL_SCORE")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)
    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")
    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )
    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    print("PINO surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)
    print(
        "--------------------SECTION ADAPTIVE REKI_NORMAL_SCORE ENDED--------------------"
    )

elif Technique_REKI == 7:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Convolution Autoencoder \
Parametrisation with Generative adverserail network prior\n"
    )
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    print("-------------------------learn Autoencoder------------------------")
    bb = os.path.isfile("../PACKETS/autoencoder.h5")
    # bb2=os.path.isfile('autoencoderp.h5')
    if not bb:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1_99X3OrcK4Q8ym325cdJ3z0KX7YiduVw", "../PACKETS/encoder.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1appHPIKQbOdxkHkwOfLUZMpuheg6ZgKd", "../PACKETS/decoder.h5"
            )
            print("...Downlaod completed.......")

            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1SMZ_hJ_drEVbagjPmcrjxDUB-lVniXPi", "../PACKETS/autoencoder.h5"
            )
            print("...Downlaod completed.......")

        else:
            Autoencoder2(nx, ny, nz, High_K1, Low_K1)  # Learn for permeability
    else:
        pass
    print("--------------------Section Ended--------------------------------")
    print(" Learn the GAN module")
    aa = os.path.isfile("../PACKETS/generator.h5")
    # aa2=aa=os.path.isfile('generatorp.h5')
    if not aa:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1DSKdVPpGKPeX-h6niGOq0Rf3TUId-AiL", "../PACKETS/generator.h5"
            )
            print("...Downlaod completed.......")
        else:
            WGAN_LEARNING(High_K1, Low_K, High_P, Low_P, nx, ny, nz)
    else:
        pass
    generator_map = load_model("../PACKETS/generator.h5")
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

        # ensemble=Recover_image(updated_ensemble,Ne,nx,ny,nz)

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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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

    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_GAN"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_GAN")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_GAN")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_GAN")

    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_GAN Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/ADAPT_REKI_GAN")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)
    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    print("PINO Surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_GAN ENDED-------------------")

elif Technique_REKI == 8:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Generative\
adverserail network prior for permeability field alone\n"
    )
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    print(" Learn the permeability field GAN module")
    aa = os.path.isfile("../PACKETS/generator.h5")
    if not aa:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1DSKdVPpGKPeX-h6niGOq0Rf3TUId-AiL", "../PACKETS/generator.h5"
            )
            print("...Downlaod completed.......")
        else:
            WGAN_LEARNING_PERM(High_K1, Low_K, nx, ny, nz)
    else:
        pass

    generator_map = load_model("../PACKETS/generator.h5")
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

        # ensemble=Recover_image(updated_ensemble,Ne,nx,ny,nz)
        red_ensemble = updated_ensemble

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

    meann = np.reshape(np.mean(red_ensemble, axis=1), (-1, 1), "F")
    gen_mean = ((generator_map.predict(meann.T)) + 1) * (High_K1 / 2)
    X_unie = gen_mean
    jud = []
    inn = X_unie[0, :, :, :]
    for k in range(nz):
        jj = inn[:, :, k]
        jj = np.reshape(jj, (-1, 1), "F")
        jud.append(jj)
    jud = np.vstack(jud)
    ouut = jud
    meann = ouut
    meann = np.nan_to_num(meann, copy=True, nan=Low_K1)
    # meann=meann.T
    meann[meann <= Low_K1] = Low_K1
    meann[meann >= High_K1] = High_K1
    # meannp=Getporosity_ensemble(meann,machine_map,1)
    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(meann))
    meannp = clfye.transform(meann)

    meanini = np.reshape(np.mean(red_kini, axis=1), (-1, 1), "F")
    gen_mean = ((generator_map.predict(meanini.T)) + 1) * (High_K1 / 2)
    X_unie = gen_mean
    jud = []
    inn = X_unie[0, :, :, :]
    for k in range(nz):
        jj = inn[:, :, k]
        jj = np.ravel(np.reshape(jj, (-1, 1), "F"))
        jud.append(jj)
    jud = np.vstack(jud)
    ouut = jud
    meanini = ouut
    meanini = np.nan_to_num(meanini, copy=True, nan=Low_K1)
    meanini = meanini.T

    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_GAN_PERM"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_GAN_PERM")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_GAN_PERM")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_GAN_PERM")

    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_GAN_PERM Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/ADAPT_REKI_GAN_PERM")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)
    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")
    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    # shutil.copy2('masterreal.data','PERCENTILE')
    print("PINO Surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_GAN_PERM ENDED-------------------")

elif Technique_REKI == 9:
    print("")
    print("Adaptive Regularised Ensemble Kalman Inversion with KMEANS Parametrisation")
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")

    os.chdir(oldfolder)
    print("Learn KMEANS Over complete dictionary of the permeability field")
    print("")
    bb = os.path.isfile("../PACKETS/Dictionary_Perm_Kmeans.mat")
    if not bb:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1rv152M5sMIhluyjGAVdpviVVzCVXHD5j",
                "../PACKETS/Dictionary_Perm_Kmeans.mat",
            )
            print("...Downlaod completed.......")
            matt = sio.loadmat("../PACKETS/Dictionary_Perm_Kmeans.mat")
            Dicclem = matt["Z"]
        else:
            filename = "../PACKETS/Ganensemble.mat"
            mat = sio.loadmat(filename)
            ini_ensemblef = mat["Z"]

            kmeans = MiniBatchKMeans(n_clusters=Ne, random_state=0, max_iter=2000)
            kmeans = kmeans.fit(ini_ensemblef.T)
            Dicclem = kmeans.cluster_centers_.T
            Dicclem = Make_Bimodal(Dicclem, 2, High_K1, Low_K1)
            sio.savemat("../PACKETS/Dictionary_Perm_Kmeans.mat", {"Z": Dicclem})
    else:
        matt = sio.loadmat("../PACKETS/Dictionary_Perm_Kmeans.mat")
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_KMEANS"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_KMEANS")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_KMEANS")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_KMEANS")

    # shutil.copy2('masterreal.data','ADAPT_REKI_KMEANS')
    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_KMEANS Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/ADAPT_REKI_KMEANS")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)
    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )
        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)
    # ensembleout1=np.hstack([controlbest,controljj2,controlbad])
    # ensembleoutp1=np.hstack([controlbestp,controljj2p,controlbadp])

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )
    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    # shutil.copy2('masterreal.data','PERCENTILE')
    print("PINO Surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)
    print("--------------------SECTION ADAPTIVE REKI_KMEANS ENDED-------------------")

elif Technique_REKI == 10:
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

    plt.figure(figsize=(10, 10))
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 0], cmap="jet")
    plt.title("Recovered ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, origii[:, :, 0], cmap="jet")
    plt.title("original", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(2, 2, 3)
    plt.pcolormesh(XX.T, YY.T, recoverp[:, :, 0], cmap="jet")
    plt.title("Recovered ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("porosity", fontsize=13)
    plt.clim(Low_P, High_P)

    plt.subplot(2, 2, 4)
    plt.pcolormesh(XX.T, YY.T, origiip[:, :, 0], cmap="jet")
    plt.title("original", fontsize=15)
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
    os.chdir("../HM_RESULTS")
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_VAE"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_VAE")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_VAE")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_VAE")

    # shutil.copy2('masterreal.data','ADAPT_REKI_VAE')
    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_VAE Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/ADAPT_REKI_VAE")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")
    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )
    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    # shutil.copy2('masterreal.data','PERCENTILE')
    print("PINO surrogate  Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_VAE ENDED-------------------")
if Technique_REKI == 11:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Mixture of Experts prior\n"
    )
    print("References for CCR-MoE include: ")
    print(
        " (1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,\
Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general\
method for learning discontinuous functions.Foundations of Data Science, \
1(2639-8001-2019-4-491):491, 2019.\n"
    )
    print(
        " (2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of\
Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.\n"
    )
    print("-------------------------------------------------------------------------")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    # choice=int(input('Denoise the update:\n1=Yes\n2=No\n'))

    for j in range(N_ens):
        folder = "Realizationn_%d" % (j)
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)
    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )
    print("Learn MoE models ... Will take a while")
    for jj in range(N_ens):
        print(str(jj + 1) + " | " + str(N_ens))
        (Learn_CCR_Ensemble)(
            "Realizationn_", oldfolder, jj, ensemble[:, jj], ensemble[:, jj]
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

        print("Extract Weights and Biases from Neural networks")
        W_E = []
        for jj in range(N_ens):
            W_Ee = Theta_Get("Realizationn_", oldfolder, jj, nclusters)
            W_E.append(W_Ee)
        W_E = np.hstack(W_E)
        unieeW = W_E

        sgsim = ensemble

        overall = cp.vstack([cp.asarray(sgsim), cp.asarray(unieeW)])

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
        # sizecm= CM.shape[0]
        if Yet == 0:
            updated_ensemble = cp.asnumpy(Ynew[:sizeclem, :])
            updated_uniee = cp.asnumpy(Ynew[sizeclem:, :])

        else:
            updated_ensemble = Ynew[:sizeclem, :]
            updated_uniee = Ynew[sizeclem:, :]

        ensembletemp = updated_ensemble

        # Put back weights and biases to neural networks
        for jj in range(N_ens):
            # print(str(jj +1) + ' | ' +  str (N_ens))
            (Theta_put)("Realizationn_", oldfolder, jj, nclusters, updated_uniee[:, jj])

        # Predict back with the MoE machine
        ensembleout = []
        for jj in range(N_ens):
            # print(str(jj +1) + ' | ' +  str (N_ens))
            spatout = Prediction_CCR_Ensemble(
                "Realizationn_", oldfolder, jj, ensembletemp[:, jj], nclusters
            )
            ensembleout.append(spatout)

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

        ensemble = np.hstack(ensembleout)
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
    Remove_folder(N_ens, "Realizationn_")

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

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_CCR"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_CCR")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_CCR")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_CCR")

    # shutil.copy2('masterreal.data','ADAPT_REKI_CCR')
    print("PINO surrogate Reservoir Simulator Forwarding - ADAPT_REKI_CCR Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/ADAPT_REKI_CCR")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)

    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )
    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    # shutil.copy2('masterreal.data','PERCENTILE')
    print("PINO Surrogate  Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)

    print(
        "--------------------SECTION ADAPTIVE REKI CCR ENDED----------------------------"
    )
elif Technique_REKI == 12:
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

    plt.figure(figsize=(10, 10))
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, recover[:, :, 0], cmap="jet")
    plt.title("Recovered", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(2, 2, 2)
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

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ttitle = "Parameter Recovery with " + str(dimms) + "% of original value"
    plt.suptitle(ttitle, fontsize=20)
    os.chdir("../HM_RESULTS")
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_DCT"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_DCT")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_DCT")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_DCT")

    # shutil.copy2('masterreal.data','ADAPT_REKI_KSVD')
    print("PINO surrogate Reservoir Simulator Forwarding - ADAPT_REKI_DCT Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/ADAPT_REKI_DCT")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)
    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")
    os.chdir(oldfolder)

    yycheck = yycheck[0]
    Presz = yycheck[:, 1:5] / scalei3
    Oilz = yycheck[:, 5:9] / scalei
    Watsz = yycheck[:, 9:13] / scalei2
    wctz = yycheck[:, 13:]
    usesim = np.hstack([Presz, Oilz, Watsz, wctz])
    usesim = np.reshape(usesim, (-1, 1), "F")
    yycheck = usesim
    cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE  = : " + str(cc))
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )
    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    # shutil.copy2('masterreal.data','PERCENTILE')
    print("PINO Surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)

    print("--------------------SECTION ADAPTIVE REKI_DCT ENDED-------------------")

elif Technique_REKI == 13:
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
    os.chdir("../Forward_problem_results/PINO/TRUE")
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

        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )

        ini_K = ensemble
        ini_p = ensemblep

        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_LOGIT"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_LOGIT")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_LOGIT")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_LOGIT")

    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_LOGIT Model")

    os.chdir("../HM_RESULTS/ADAPT_REKI_LOGIT")
    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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
    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    print(" Plot P10,P50,P90")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )

    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    print("PINO Surrogate Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        _,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )

    os.chdir(oldfolder)
    print(
        "--------------------SECTION ADAPTIVE REKI LOGIT ENDED----------------------------"
    )

elif Technique_REKI == 14:
    print("")
    print(
        "Adaptive Regularised Ensemble Kalman Inversion with Diffusion model Parametrisation\n"
    )
    print("Novel Implementation: Author: Clement Etienam SA Energy/GPU @NVIDIA")
    print("Starting the History matching with ", str(Ne) + " Ensemble members")
    print("-------------------------learn Diffusion model------------------------")
    if not os.path.exists(("../PACKETS/tochi")):
        os.makedirs(("../PACKETS/tochi"))
    else:
        pass

    aa = os.path.isfile("../PACKETS/diffusion_model.pth")
    make_up = 7
    if not aa:
        print("Learn model from scratch")
        filename = "../PACKETS/Ganensemble.mat"
        mat = sio.loadmat(filename)
        ini_ensemble = mat["Z"]
        ini_ensemble = rescale_linear(
            rescale_linear(ini_ensemble, Low_K1, High_K1), 0, 1
        )
        val = ini_ensemble.shape[1]

        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1bj-gKaRAKDPvMykF7LNTHsYfIpRB1sLD", "../PACKETS/diffusion_model.pth"
            )
            print("...Downlaod completed.......")

            print("Load pretrained model")
            cuda = 0
            device = torch.device(
                f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
            )
            model = Unet(dim=32, dim_mults=(1, 2, 4, 8), channels=nz)

            diffusion_model = GaussianDiffusion(
                model,
                image_size=nx + make_up,
                timesteps=1000,  # number of steps
                loss_type="l2",  # L1 or L2
            )

            model = model.to(device, dtype=torch.float32)
            # diffusion_model = diffusion_model.to(device,dtype=torch.float32)
            diffusion_model.load_state_dict(
                torch.load("../PACKETS/diffusion_model.pth")
            )
            diffusion_model = diffusion_model.to(device, dtype=torch.float32)
            diffusion_model.eval()

        else:
            X_unie = np.zeros((val, nz, nx + make_up, ny + make_up))

            for i in range(val):
                aa = np.zeros((nz, nx + make_up, ny + make_up))
                tempp = np.reshape(ini_ensemble[:, i], (nx, ny, nz), "F")
                for kk in range(int(nz)):
                    tempc = tempp[:, :, kk]
                    tempc = imresize(tempc, output_shape=(nx + make_up, ny + make_up))
                    tempc = rescale_linear(tempc, 0, 1)
                    aa[kk, :, :] = tempc
                X_unie[i, :, :, :] = aa
            X_unie = np.nan_to_num(X_unie, copy=True, nan=0)

            cuda = 0
            device = torch.device(
                f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
            )
            model = Unet(dim=32, dim_mults=(1, 2, 4, 8), channels=nz)

            diffusion_model = GaussianDiffusion(
                model,
                image_size=nx + make_up,
                timesteps=1000,  # number of steps
                loss_type="l2",  # L1 or L2
            )

            model = model.to(device, dtype=torch.float32)
            diffusion_model = diffusion_model.to(device, dtype=torch.float32)
            batch_size = 4
            # train_loaderr = load_data_numpy(X_unie, batch_size)

            train_lrr = 8e-5
            trainer = Trainer(
                diffusion_model,
                "../PACKETS/tochi",
                X_unie,
                train_batch_size=batch_size,
                train_lr=train_lrr,
                train_num_steps=5000,  # total training steps
                gradient_accumulate_every=2,  # gradient accumulation steps
                ema_decay=0.995,  # exponential moving average decay
                amp=True,  # turn on mixed precision
            )

            trainer.train()
            torch.save(diffusion_model.state_dict(), "../PACKETS/diffusion_model.pth")

    else:
        print("Load pretrained model")
        cuda = 0
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        model = Unet(dim=32, dim_mults=(1, 2, 4, 8), channels=nz)

        diffusion_model = GaussianDiffusion(
            model,
            image_size=nx + make_up,
            timesteps=1000,  # number of steps
            loss_type="l2",  # L1 or L2
        )

        model = model.to(device, dtype=torch.float32)
        # diffusion_model = diffusion_model.to(device,dtype=torch.float32)
        diffusion_model.load_state_dict(torch.load("../PACKETS/diffusion_model.pth"))
        diffusion_model = diffusion_model.to(device, dtype=torch.float32)
        diffusion_model.eval()

    os.chdir(oldfolder)

    print(" Generate initial ensemble")
    shape = (Ne, nz, nx + make_up, ny + make_up)
    img = torch.randn(shape).to(device, dtype=torch.float32)
    sampled_images = diffusion_model.sample_clem(img, batch_size=Ne)
    sampled_images = sampled_images.detach().cpu().numpy()

    ini_ensemble = rescale_linear(
        pytorch_to_inverse(Ne, sampled_images, nx, ny, nz), Low_K1, High_K1
    )

    encoded = pytorch_to_inverse(Ne, img.detach().cpu().numpy(), nx, ny, nz)

    ensemble = ini_ensemble
    ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
    # ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)

    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)

    # ensemble,ensemblep=honour2(ensemblep,\
    #                              ensemble,nx,ny,nz,N_ens,\
    #                                  High_K,Low_K,High_P,Low_P)

    print("Check diffusion model reconstruction")

    plt.figure(figsize=(10, 10))
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.subplot(2, 2, 1)
    plt.pcolormesh(XX.T, YY.T, np.reshape(ensemble[:, 0], (nx, ny), "F"), cmap="jet")
    plt.title("Realisation 1 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(XX.T, YY.T, np.reshape(ensemble[:, 1], (nx, ny), "F"), cmap="jet")
    plt.title("Realisation 2 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(2, 2, 3)
    plt.pcolormesh(XX.T, YY.T, np.reshape(ensemble[:, 2], (nx, ny), "F"), cmap="jet")
    plt.title("Realisation 3 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.subplot(2, 2, 4)
    plt.pcolormesh(XX.T, YY.T, np.reshape(ensemble[:, 3], (nx, ny), "F"), cmap="jet")
    plt.title("Realisation 4 ", fontsize=15)
    plt.ylabel("Y", fontsize=13)
    plt.xlabel("X", fontsize=13)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    cbar1 = plt.colorbar()
    cbar1.ax.set_ylabel("Log K (mD)", fontsize=13)
    plt.clim(Low_K1, High_K1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ttitle = "Parameter Recovery with Diffusion model "
    plt.suptitle(ttitle, fontsize=20)
    os.chdir("../HM_RESULTS")
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

    os.chdir("../Forward_problem_results/PINO/TRUE")
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
        ensemblepy = ensemble_pytorch(
            ensemble,
            ensemblep,
            LUB,
            HUB,
            mpor,
            hpor,
            inj_rate,
            dt,
            IWSw,
            nx,
            ny,
            nz,
            ensemble.shape[1],
            input_channel,
        )
        ini_K = ensemble
        ini_p = ensemblep
        mazw = 0  # Dont smooth the presure field
        simDatafinal, predMatrix, _, _, _ = Forward_model_ensemble(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            ensemble.shape[1],
            DX,
            pini_alt,
            UW,
            BW,
            UO,
            BO,
            SWI,
            SWR,
            DZ,
            dt,
            MAXZ,
        )

        if ii == 0:
            os.chdir("../HM_RESULTS")
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

        encoded = updated_ensemble  # updated noise
        # encoded = rescale_linear(encoded, 0, 1) #rescale to 0,1

        # encoded[encoded<=-1]=-1
        # encoded[encoded>=1] = 1

        updated_ensemble = inverse_to_pytorch(
            Ne, encoded, nx, ny, nz, device, make_up
        )  # take back to pytorch format
        # updated_ensemble[updated_ensemble<=-1] = -1# = rescale_linear(updated_ensemble, -1, 1)
        # updated_ensemble[updated_ensemble>=1] = 1

        sampled_images = diffusion_model.sample_clem(updated_ensemble, batch_size=Ne)
        sampled_images = sampled_images.detach().cpu().numpy()

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

        ensemble = rescale_linear(
            pytorch_to_inverse(Ne, sampled_images, nx, ny, nz), Low_K1, High_K1
        )

        ensemble[ensemble <= Low_K1] = Low_K1
        ensemble[ensemble >= High_K1] = High_K1
        # ensemblep = Getporosity_ensemble(ensemble,machine_map,N_ens)
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

    meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
    meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")

    meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
    controljj2 = np.reshape(meann, (-1, 1), "F")
    controljj2p = np.reshape(meannp, (-1, 1), "F")
    controlj2 = controljj2
    controlj2p = controljj2p

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinal,
        predMatrix,
        pressure_ensemble,
        water_ensemble,
        oil_ensemble,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    ensemblepya = ensemble_pytorch(
        all_ensemble,
        all_ensemblep,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensemble.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        simDatafinala,
        predMatrixa,
        pressure_ensemblea,
        water_ensemblea,
        oil_ensemblea,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepya,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensemble.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    print("Read Historical data")

    os.chdir("../Forward_problem_results/PINO/TRUE")
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

    os.chdir("../HM_RESULTS")
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
    if not os.path.exists("../HM_RESULTS/ADAPT_REKI_DIFFUSION"):
        os.makedirs("../HM_RESULTS/ADAPT_REKI_DIFFUSION")
    else:
        shutil.rmtree("../HM_RESULTS/ADAPT_REKI_DIFFUSION")
        os.makedirs("../HM_RESULTS/ADAPT_REKI_DIFFUSION")

    # shutil.copy2('masterreal.data','ADAPT_REKI_VAE')
    print("PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI_DIFFUSION Model")

    ensemblepy = ensemble_pytorch(
        controlbest,
        controlbestp,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    simDatafinal, yycheck, pree, wats, oilss = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    os.chdir("../HM_RESULTS/ADAPT_REKI_DIFFUSION")
    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "RESERVOIR_MODEL.mat",
        {
            "permeability": controlbest,
            "porosity": controlbestp,
            "Simulated_data_plots": yycheck,
            "Pressure": pree,
            "Water_saturation": wats,
            "Oil_saturation": oilss,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            pree[0, :, :, :],
            wats[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (pree[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = wats[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - wats[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")
    os.chdir(oldfolder)

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
    os.chdir("../HM_RESULTS")
    Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
    os.chdir(oldfolder)

    if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_best_k,
        yes_best_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "BEST_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_best_k,
            "porosity": yes_best_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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

    if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    else:
        shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
        os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")

    os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    ensemblepy = ensemble_pytorch(
        yes_mean_k,
        yes_mean_p,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        controlbest2.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    _, yycheck, preebest, watsbest, oilssbest = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        controlbest2.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )

    Plot_RSM_single(yycheck, "Performance.png")
    Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

    sio.savemat(
        "MEAN_RESERVOIR_MODEL.mat",
        {
            "permeability": yes_mean_k,
            "porosity": yes_mean_p,
            "Simulated_data_plots": yycheck,
            "Pressure": preebest,
            "Water_saturation": watsbest,
            "Oil_saturation": oilssbest,
        },
    )

    for kk in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, kk - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)
        Plot_performance_model(
            preebest[0, :, :, :],
            watsbest[0, :, :, :],
            nx,
            ny,
            "PINN_model_PyTorch.png",
            UIR,
            kk,
            dt,
            MAXZ,
            pini_alt,
        )

        Pressz = (preebest[0, kk, :, :][:, :, None]) * pini_alt
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2(
            Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii
        )

        watsz = watsbest[0, kk, :, :][:, :, None]
        maxii = max(watsz.ravel())
        minii = min(watsz.ravel())
        plot3d2(
            watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii
        )

        oilsz = 1 - watsbest[0, kk, :, :][:, :, None]
        maxii = max(oilsz.ravel())
        minii = min(oilsz.ravel())
        plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)
    print("Now predicting for a test case - Creating GIF")
    import glob

    frames = []
    imgs = sorted(glob.glob("*PINN_model_PyTorch*"), key=os.path.getmtime)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "Evolution1.gif",
        format="GIF",
        append_images=frames[1:],
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

    for f4 in glob("*unie_pressure*"):
        os.remove(f4)

    for f4 in glob("*unie_water*"):
        os.remove(f4)

    for f4 in glob("*unie_oil*"):
        os.remove(f4)

    print("")
    print("Saving prediction in CSV file")
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

    seeuset = pd.DataFrame(yycheck[0])
    seeuset.to_csv("RSM.csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

    Plot_RSM_percentile_model(yycheck[0], True_mat, "Compare.png")

    os.chdir(oldfolder)

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
    print(" Plot P10,P50,P90 and Base Measurment")
    os.chdir("../HM_RESULTS")
    sio.savemat(
        "Posterior_Ensembles.mat",
        {
            "PERM_Reali": ensemble,
            "PORO_Reali": ensemblep,
            "P10_Perm": controlbest,
            "P50_Perm": controljj2,
            "P90_Perm": controlbad,
            "P10_Poro": controlbest2p,
            "P50_Poro": controljj2p,
            "P90_Poro": controlbadp,
            "Simulated_data": simDatafinal,
            "Simulated_data_plots": predMatrix,
            "Pressures": pressure_ensemble,
            "Water_saturation": water_ensemble,
            "Oil_saturation": oil_ensemble,
            "Simulated_data_best_ensemble": simDatafinala,
            "Simulated_data_plots_best_ensemble": predMatrixa,
            "Pressures_best_ensemble": pressure_ensemblea,
            "Water_saturation_best_ensemble": water_ensemblea,
            "Oil_saturation_best_ensemble": oil_ensemblea,
        },
    )
    os.chdir(oldfolder)

    ensembleout1 = np.hstack(
        [controlbest, controljj2, controlbad, yes_best_k, yes_mean_p]
    )
    ensembleoutp1 = np.hstack(
        [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p]
    )
    # ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)

    if not os.path.exists("../HM_RESULTS/PERCENTILE"):
        os.makedirs("../HM_RESULTS/PERCENTILE")
    else:
        shutil.rmtree("../HM_RESULTS/PERCENTILE")
        os.makedirs("../HM_RESULTS/PERCENTILE")

    # shutil.copy2('masterreal.data','PERCENTILE')
    print("PINO surrogate  Reservoir Simulator Forwarding")

    ensemblepy = ensemble_pytorch(
        ensembleout1,
        ensembleoutp1,
        LUB,
        HUB,
        mpor,
        hpor,
        inj_rate,
        dt,
        IWSw,
        nx,
        ny,
        nz,
        ensembleout1.shape[1],
        input_channel,
    )

    mazw = 0  # Dont smooth the presure field
    (
        pertout,
        yzout,
        pressure_percentile,
        water_percentile,
        oil_percentile,
    ) = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        ensembleout1.shape[1],
        DX,
        pini_alt,
        UW,
        BW,
        UO,
        BO,
        SWI,
        SWR,
        DZ,
        dt,
        MAXZ,
    )
    os.chdir("../HM_RESULTS/PERCENTILE")
    Plot_RSM_percentile(yzout, True_mat, "P10_P50_P90.png")

    sio.savemat(
        "Posterior_Ensembles_percentile.mat",
        {
            "PERM_Reali": ensembleout1,
            "PORO_Reali": ensembleoutp1,
            "Simulated_data_plots": yzout,
            "Pressures": pressure_percentile,
            "Water_saturation": water_percentile,
            "Oil_saturation": oil_percentile,
        },
    )
    os.chdir(oldfolder)

    print(
        "--------------------SECTION ADAPTIVE REKI_DIFFUSION ENDED-------------------"
    )
elapsed_time_secs = time.time() - start_time

if Technique_REKI == 1:
    comment = "ADAPT_REKI (Vanilla Adaptive Ensemble Kalman Inversion)"
elif Technique_REKI == 2:
    comment = "ADAPT_REKI_KSVD\n"
elif Technique_REKI == 3:
    comment = "ADAPT_REKI_AE  (Adaptive Ensemble Kalman Inversion + Convolution Autoencoder)\n"
elif Technique_REKI == 4:
    comment = (
        "ADAPT_REKI_DA  (Adaptive Ensemble Kalman Inversion + Denoising Autoencoder)\n"
    )
elif Technique_REKI == 5:
    comment = "ADAPT_REKI_LEVELSET  (Adaptive Ensemble Kalman Inversion + Level Set)\n"
elif Technique_REKI == 6:
    comment = (
        "ADAPT_REKI_NORMAL_SCORE  (Adaptive Ensemble Kalman Inversion + Normal Score)\n"
    )
elif Technique_REKI == 7:
    comment = "ADAPT_REKI_DE_GAN  (Adaptive Ensemble Kalman Inversion + Convolution Autoencoder + GAN)\n"
elif Technique_REKI == 8:
    comment = "ADAPT_REKI_GAN  (Adaptive Ensemble Kalman Inversion + GAN on the permeability field alone\n"
elif Technique_REKI == 9:
    comment = "ADAPT_REKI_KMEANS  (Adaptive Ensemble Kalman Inversion + KMEANS on the permeability field alone\n"
elif Technique_REKI == 10:
    comment = "ADAPT_REKI_VAE  (Adaptive Ensemble Kalman Inversion + variational\
    convolutional autoencoder on the permeability field alone\n"
elif Technique_REKI == 11:
    comment = "ADAPT_REKI_CCR-MoE  (Adaptive Ensemble Kalman Inversion + Mixture of\
    Experts parametrisation on the permeability field alone)\n"
elif Technique_REKI == 12:
    comment = (
        "ADAPT_REKI_DCT  (Adaptive Ensemble Kalman Inversion with DCT parametrisation\n"
    )
elif Technique_REKI == 13:
    comment = "ADAPT_REKI with Logit transformation"
else:
    comment = "ADAPT_REKI with Generative Diffusion mode"


print("Inverse problem solution used =: " + comment)
print("Ensemble size = ", str(Ne))
msg = "Execution took: %s secs (Wall clock time)" % timedelta(
    seconds=round(elapsed_time_secs)
)
print(msg)
print("-------------------PROGRAM EXECUTED-----------------------------------")
