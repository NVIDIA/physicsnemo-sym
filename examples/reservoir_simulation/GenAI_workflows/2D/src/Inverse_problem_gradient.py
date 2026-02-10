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


@Data Assimilation Methods: LBFGS

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

import matplotlib.pyplot as plt
import scipy.io as sio
import shutil
import time
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import os.path
import torch
import torch.nn.functional as F
from scipy.fftpack import dct
from scipy.fftpack.realtransforms import idct
from scipy import interpolate
import multiprocessing
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D, MaxPooling2D, Dense
from physicsnemo.sym.models.fno import *
from physicsnemo.sym.key import Key
import os.path
from PIL import Image
import numpy
import numpy.matlib

# os.environ['KERAS_BACKEND'] = 'tensorflow'
import os.path

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
from NVRS import *

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
            # if mazw ==1:
            #     Ptito = smoothn(Ptito,s = 1e1)[0]

            # average_pressure = np.mean(Ptito.ravel()) * pini_alt
            average_pressure = (Ptito.ravel()[producer_location]) * pini_alt
            p_inj = (Ptito.ravel()[Injector_location]) * pini_alt
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

            up = UO * BO
            down = 2 * np.pi * kuse_prod * krouse * DZ
            right = np.log(RE / rwell) + skin
            J = down / (up * right)
            # drawdown = p_prod - pwf_producer
            drawdown = average_pressure - pwf_producer
            qoil = np.abs(-(drawdown * J))

            up = UW * BW
            down = 2 * np.pi * kuse_prod * krwusep * DZ
            right = np.log(RE / rwell) + skin
            J = down / (up * right)
            # drawdown = p_prod - pwf_producer
            drawdown = average_pressure - pwf_producer
            qwater = np.abs(-(drawdown * J))
            # qwater[qwater==0] = 0

            # water cut
            wct = (qwater / (qwater + qoil)) * np.float32(100)

            timz = ((kk + 1) * dt) * MAXZ
            # timz = timz.reshape(1,1)
            qs = [BHP, qoil, qwater, wct]
            qs2 = [BHP, qoil / scalei, qwater / scalei2, wct]
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


def dx(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor"

    var = inpt[:, channel : channel + 1, :, :]

    # get filter
    if order == 1:
        ddx1D = torch.Tensor(
            [
                -0.5,
                0.0,
                0.5,
            ]
        ).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor(
            [
                -1.0 / 60.0,
                3.0 / 20.0,
                -3.0 / 4.0,
                0.0,
                3.0 / 4.0,
                -3.0 / 20.0,
                1.0 / 60.0,
            ]
        ).to(inpt.device)
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])

    # apply convolution
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]

    return output


def ddx(inpt, dx, channel, dim, order=1, padding="zeros"):
    "Compute second order numerical derivatives of input tensor"

    var = inpt[:, channel : channel + 1, :, :]

    # get filter
    if order == 1:
        ddx1D = torch.Tensor(
            [
                1.0,
                -2.0,
                1.0,
            ]
        ).to(inpt.device)
    elif order == 3:
        ddx1D = torch.Tensor(
            [
                1.0 / 90.0,
                -3.0 / 20.0,
                3.0 / 2.0,
                -49.0 / 18.0,
                3.0 / 2.0,
                -3.0 / 20.0,
                1.0 / 90.0,
            ]
        ).to(inpt.device)
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (1 - dim) * [1])

    # apply convolution
    if padding == "zeros":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "replication":
        var = F.pad(var, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    output = F.conv2d(var, ddx3D, padding="valid")
    output = (1.0 / dx**2) * output
    if dim == 0:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]
    elif dim == 1:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]

    return output


def Black_oil2(
    UIR,
    pini_alt,
    LUB,
    HUB,
    aay,
    bby,
    SWI,
    SWR,
    UW,
    BW,
    UO,
    BO,
    MAXZ,
    nx,
    ny,
    input_var,
    device,
    myloss,
):
    u = input_var["pressure"]
    perm = input_var["perm"]
    fin = input_var["Q"]
    finwater = input_var["Qw"]
    dt = input_var["Time"]
    pini = input_var["Pini"]
    poro = input_var["Phi"]
    sini = input_var["Swini"]
    sat = input_var["water_sat"]
    Tx = input_var["Tx"]
    Ty = input_var["Ty"]
    Txw = input_var["Txw"]
    Tyw = input_var["Tyw"]

    siniuse = sini[0, 0, 0, 0]

    dtin = dt * MAXZ
    dxf = 1.0 / u.shape[3]

    u = u * pini_alt
    pini = pini * pini_alt
    # Pressure equation Loss
    fin = fin * UIR
    finwater = finwater * UIR

    a = perm  # absolute permeability
    v_min, v_max = LUB, HUB
    new_min, new_max = aay, bby

    m = (new_max - new_min) / (v_max - v_min)
    b = new_min - m * v_min
    a = m * a + b

    finusew = finwater
    dta = dtin

    pressure = u
    # water_sat = sat

    prior_pressure = torch.zeros(sat.shape[0], sat.shape[1], nx, ny).to(
        device, torch.float32
    )
    prior_pressure[:, 0, :, :] = pini_alt * (
        torch.ones(sat.shape[0], nx, ny).to(device, torch.float32)
    )
    prior_pressure[:, 1:, :, :] = u[:, :-1, :, :]

    prior_sat = torch.zeros(sat.shape[0], sat.shape[1], nx, ny).to(
        device, torch.float32
    )
    prior_sat[:, 0, :, :] = siniuse * (
        torch.ones(sat.shape[0], nx, ny).to(device, torch.float32)
    )
    prior_sat[:, 1:, :, :] = sat[:, :-1, :, :]

    dsw = sat - prior_sat  # ds
    dsw = torch.clip(dsw, 0.001, None)

    S = torch.div(torch.sub(prior_sat, SWI, alpha=1), (1 - SWI - SWR))

    # Pressure equation Loss
    Mw = torch.divide(torch.square(S), (UW * BW))  # Water mobility
    Mo = torch.div(
        torch.square(torch.sub(torch.ones(S.shape, device=u.device), S)), (UO * BO)
    )

    Mt = Mw + Mo
    a1 = torch.mul(Mt, a)  # overall Effective permeability
    a1water = torch.mul(Mw, a)  # water Effective permeability

    # compute first dffrential
    gulpa = []
    gulp2a = []
    for m in range(sat.shape[0]):  # Batch
        inn_now = pressure[m, :, :, :][:, None, :, :]
        dudx_fdma = dx(
            inn_now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
        )
        dudy_fdma = dx(
            inn_now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
        )
        gulpa.append(dudx_fdma)
        gulp2a.append(dudy_fdma)
    dudx_fdm = torch.stack(gulpa, 0)[:, :, 0, :, :]
    dudy_fdm = torch.stack(gulp2a, 0)[:, :, 0, :, :]

    grad_h = dudx_fdm
    grad_v = dudy_fdm
    est_sigma1 = -a1 * grad_h
    est_sigma2 = -a1 * grad_v

    yes1 = ((Tx - est_sigma1) ** 2 + (Ty - est_sigma2) ** 2).mean()

    # compute first dffrential
    gulpa = []
    for m in range(sat.shape[0]):  # Batch
        inn_now = Tx[m, :, :, :][:, None, :, :]
        dudx_fdma = dx(
            inn_now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
        )
        gulpa.append(dudx_fdma)
    sigma1_x1 = torch.stack(gulpa, 0)[:, :, 0, :, :]

    # compute first dffrential
    gulp2a = []
    for m in range(sat.shape[0]):  # Batch
        inn_now = Ty[m, :, :, :][:, None, :, :]

        dudy_fdma = dx(
            inn_now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
        )
        gulp2a.append(dudy_fdma)
    sigma2_x2 = torch.stack(gulp2a, 0)[:, :, 0, :, :]

    yes2 = (((sigma1_x1 + sigma2_x2) + fin) ** 2).mean()

    p_loss = (yes1 + yes2) * 1e-6

    # Saturation equation loss
    grad_h = dudx_fdm
    grad_v = dudy_fdm
    est_sigma1 = -a1water * grad_h
    est_sigma2 = -a1water * grad_v

    yes1w = ((Txw - est_sigma1) ** 2 + (Tyw - est_sigma2) ** 2).mean()

    # compute first dffrential
    gulpa = []
    for m in range(sat.shape[0]):  # Batch
        inn_now = Txw[m, :, :, :][:, None, :, :]
        dudx_fdma = dx(
            inn_now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
        )
        gulpa.append(dudx_fdma)
    sigma1_x1 = torch.stack(gulpa, 0)[:, :, 0, :, :]

    # compute first dffrential
    gulp2a = []
    for m in range(sat.shape[0]):  # Batch
        inn_now = Tyw[m, :, :, :][:, None, :, :]

        dudy_fdma = dx(
            inn_now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
        )
        gulp2a.append(dudy_fdma)
    sigma2_x2 = torch.stack(gulp2a, 0)[:, :, 0, :, :]

    fifth = poro * (dsw / dta)
    left = fifth - (sigma1_x1 + sigma2_x2)
    yes2w = ((left - finusew) ** 2).mean()

    s_loss = (yes1w + yes2w) * 1e-6

    return p_loss, s_loss


# [pde-loss]
# define custom class for black oil model
def Black_oil(
    UIR,
    pini_alt,
    LUB,
    HUB,
    aay,
    bby,
    SWI,
    SWR,
    UW,
    BW,
    UO,
    BO,
    MAXZ,
    nx,
    ny,
    input_var,
    device,
    myloss,
):
    "Custom Black oil PDE definition for PINO"

    # get inputs

    u = input_var["pressure"]
    perm = input_var["perm"]
    fin = input_var["Q"]
    finwater = input_var["Qw"]
    dt = input_var["Time"]
    pini = input_var["Pini"]
    poro = input_var["Phi"]
    sini = input_var["Swini"]
    sat = input_var["water_sat"]

    siniuse = sini[0, 0, 0, 0]

    dtin = dt * MAXZ
    dxf = 1.0 / u.shape[3]

    u = u * pini_alt
    pini = pini * pini_alt
    # Pressure equation Loss
    fin = fin * UIR
    finwater = finwater * UIR

    a = perm  # absolute permeability
    v_min, v_max = LUB, HUB
    new_min, new_max = aay, bby

    m = (new_max - new_min) / (v_max - v_min)
    b = new_min - m * v_min
    a = m * a + b

    finuse = fin
    finusew = finwater
    dta = dtin

    pressure = u
    # water_sat = sat

    prior_pressure = torch.zeros(sat.shape[0], sat.shape[1], nx, ny).to(
        device, torch.float32
    )
    prior_pressure[:, 0, :, :] = pini_alt * (
        torch.ones(sat.shape[0], nx, ny).to(device, torch.float32)
    )
    prior_pressure[:, 1:, :, :] = u[:, :-1, :, :]

    prior_sat = torch.zeros(sat.shape[0], sat.shape[1], nx, ny).to(
        device, torch.float32
    )
    prior_sat[:, 0, :, :] = siniuse * (
        torch.ones(sat.shape[0], nx, ny).to(device, torch.float32)
    )
    prior_sat[:, 1:, :, :] = sat[:, :-1, :, :]

    dsw = sat - prior_sat  # ds
    dsw = torch.clip(dsw, 0.001, None)

    S = torch.div(torch.sub(prior_sat, SWI, alpha=1), (1 - SWI - SWR))

    # Pressure equation Loss
    Mw = torch.divide(torch.square(S), (UW * BW))  # Water mobility
    Mo = torch.div(
        torch.square(torch.sub(torch.ones(S.shape, device=u.device), S)), (UO * BO)
    )

    kroil = torch.square(torch.sub(torch.ones(S.shape, device=u.device), S))
    Mt = Mw + Mo
    a1 = torch.mul(Mt, a)  # overall Effective permeability
    a1water = torch.mul(Mw, a)  # water Effective permeability

    # compute first dffrential
    gulpa = []
    gulp2a = []
    for m in range(sat.shape[0]):  # Batch
        inn_now = pressure[m, :, :, :][:, None, :, :]
        dudx_fdma = dx(
            inn_now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
        )
        dudy_fdma = dx(
            inn_now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
        )
        gulpa.append(dudx_fdma)
        gulp2a.append(dudy_fdma)
    dudx_fdm = torch.stack(gulpa, 0)[:, :, 0, :, :]
    dudy_fdm = torch.stack(gulp2a, 0)[:, :, 0, :, :]

    # Compute second diffrential

    gulpa = []
    gulp2a = []
    for m in range(sat.shape[0]):  # Batch
        inn_now = pressure[m, :, :, :][:, None, :, :]
        dudx_fdma = ddx(
            inn_now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
        )
        dudy_fdma = ddx(
            inn_now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
        )
        gulpa.append(dudx_fdma)
        gulp2a.append(dudy_fdma)
    dduddx_fdm = torch.stack(gulpa, 0)[:, :, 0, :, :]
    dduddy_fdm = torch.stack(gulp2a, 0)[:, :, 0, :, :]

    inn_now2 = a1
    dcdx = dx(inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication")
    dcdy = dx(inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication")

    AOIL = torch.zeros_like(kroil).to(device, torch.float32)
    AOIL[:, :, 1, 24] = finwater[0, 0, 0, 0]
    AOIL[:, :, 1, 1] = finwater[0, 0, 0, 0]
    AOIL[:, :, 31, 0] = finwater[0, 0, 0, 0]
    AOIL[:, :, 31, 31] = finwater[0, 0, 0, 0]

    AOIL[:, :, 14, 27] = fin[0, 0, 0, 0]
    AOIL[:, :, 28, 19] = fin[0, 0, 0, 0]
    AOIL[:, :, 14, 12] = fin[0, 0, 0, 0]
    AOIL[:, :, 7, 9] = fin[0, 0, 0, 0]

    finuse = AOIL
    right = (
        (dcdx * dudx_fdm) + (a1 * dduddx_fdm) + (dcdy * dudy_fdm) + (a1 * dduddy_fdm)
    )
    darcy_pressure = torch.sum(
        torch.abs((finuse.reshape(sat.shape[0], -1) - right.reshape(sat.shape[0], -1)))
        / sat.shape[0]
    )
    # Zero outer boundary
    # darcy_pressure = F.pad(darcy_pressure[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
    darcy_pressure = dxf * darcy_pressure * 1e-6

    p_loss = darcy_pressure

    # Saruration equation loss
    dudx = dudx_fdm
    dudy = dudy_fdm

    dduddx = dduddx_fdm
    dduddy = dduddy_fdm

    inn_now2 = a1water
    dadx = dx(inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication")
    dady = dx(inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication")

    flux = (dadx * dudx) + (a1water * dduddx) + (dady * dudy) + (a1water * dduddy)
    fifth = poro * (dsw / dta)
    toge = flux + finusew
    darcy_saturation = torch.sum(
        torch.abs(fifth.reshape(sat.shape[0], -1) - toge.reshape(sat.shape[0], -1))
        / sat.shape[0]
    )

    # Zero outer boundary
    # darcy_saturation = F.pad(darcy_saturation[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
    darcy_saturation = dxf * darcy_saturation * 1e-5

    s_loss = darcy_saturation
    return p_loss, s_loss


def Forward_model_pytorch(
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
    UIR,
    LUB,
    HUB,
    aay,
    bby,
    nx,
    ny,
    device,
    muloss,
):
    #### ===================================================================== ####
    #                     PINO RESERVOIR SIMULATOR
    #
    #### ===================================================================== ####
    inn = x_true

    # with torch.no_grad():
    ouut_p = modelP(x_true)["pressure"]
    ouut_s = modelS(x_true)["water_sat"]

    torch.cuda.empty_cache()

    input_var = inn
    input_var["pressure"] = ouut_p
    input_var["water_sat"] = ouut_s
    if surrogate == 4:
        input_var["Tx"] = modelP(inn)["Tx"]
        input_var["Ty"] = modelP(inn)["Ty"]
        input_var["Txw"] = modelS(inn)["Txw"]
        input_var["Tyw"] = modelS(inn)["Tyw"]

    # print(outputa_p.shape)
    if surrogate == 4:
        f_loss2, f_water2 = Black_oil2(
            UIR,
            pini_alt,
            LUB,
            HUB,
            aay,
            bby,
            SWI,
            SWR,
            UW,
            BW,
            UO,
            BO,
            MAXZ,
            nx,
            ny,
            input_var,
            device,
            myloss,
        )
    else:
        f_loss2, f_water2 = Black_oil(
            UIR,
            pini_alt,
            LUB,
            HUB,
            aay,
            bby,
            SWI,
            SWR,
            UW,
            BW,
            UO,
            BO,
            MAXZ,
            nx,
            ny,
            input_var,
            device,
            myloss,
        )

    amm = 0
    # for amm in range(Ne):
    Injector_location = torch.where(torch.ravel(inn["Qw"][amm, 0, :, :]) > 0)[0]
    producer_location = torch.where(torch.ravel(inn["Q"][amm, 0, :, :]) < 0)[0]

    permfirst = reshape_clement(inn["perm"][amm, 0, :, :], (-1,))
    v_min, v_max = (
        torch.min(inn["perm"][amm, 0, :, :]),
        torch.max(inn["perm"][amm, 0, :, :]),
    )
    new_min, new_max = aay, bby
    m = (new_max - new_min) / (v_max - v_min)
    b = new_min - m * v_min

    temp = m * torch.clip(permfirst, LUB, HUB) + b

    PERM = temp
    kuse_inj = torch.index_select(PERM, 0, Injector_location)

    kuse_prod = torch.index_select(PERM, 0, producer_location)

    RE = 0.2 * DX
    Baa = torch.zeros(steppi, 16)
    for kk in range(steppi):
        Ptito = reshape_clement(ouut_p[amm, kk, :, :], (-1,))
        Stito = reshape_clement(ouut_s[amm, kk, :, :], (-1,))

        average_pressure = (torch.index_select(Ptito, 0, producer_location)) * pini_alt

        p_inj = (torch.index_select(Ptito, 0, Injector_location)) * pini_alt
        # p_inj = average_pressure

        S = Stito
        # S = reshape_fortran(S, (-1, 1))
        # S = torch.reshape(S, (-1, 1), order='F')
        # S = reshape_clement(S,(-1,1))
        # S = Stito.ravel().reshape(-1,1)

        Sout = (S - SWI) / (1 - SWI - SWR)
        Krw = Sout**2  # Water mobility
        Kro = (1 - Sout) ** 2  # Oil mobility
        krwuse = torch.index_select(Krw, 0, Injector_location)

        krwusep = torch.index_select(Krw, 0, producer_location)

        krouse = torch.index_select(Kro, 0, producer_location)

        up = UW * BW
        down = 2 * np.pi * kuse_inj * krwuse * DZ
        right = np.log(RE / rwell) + skin
        temp = (up / down) * right
        # temp[temp == -inf] = 0
        Pwf = p_inj + temp
        Pwf = torch.abs(Pwf)
        BHP = Pwf
        # BHP = reshape_fortran(BHP, (1, -1))
        BHP = reshape_clement(BHP, (1, -1))

        up = UO * BO
        down = 2 * np.pi * kuse_prod * krouse * DZ
        right = np.log(RE / rwell) + skin
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        drawdown = average_pressure - pwf_producer
        qoil = torch.abs(-(drawdown * J))
        qoil = reshape_clement(qoil, (1, -1))

        up = UW * BW
        down = 2 * torch.pi * kuse_prod * krwusep * DZ
        right = np.log(RE / rwell) + skin
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        drawdown = average_pressure - pwf_producer
        qwater = torch.abs(-(drawdown * J))
        qwater = reshape_clement(qwater, (1, -1))
        # qwater[qwater==0] = 0

        # water cut
        wct = (qwater / (qwater + qoil)) * 100
        wct = reshape_clement(wct, (1, -1))

        # timz = ((kk + 1) *dt)* MAXZ
        # timz = timz.reshape(1,1)
        # qs = [BHP,qoil,qwater,wct]
        qs = torch.cat((BHP / scalei3, qoil / scalei, qwater / scalei, wct), 1)
        # print(qs.shape)
        # qs=torch.asarray(qs)
        qs = reshape_clement(qs, (1, -1))

        Baa[kk, :] = torch.ravel(qs)

    return reshape_clement(Baa, (-1, 1)), f_loss2, f_water2


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def reshape_clement(x, shape):
    x_reshaped = x.T.reshape(shape)
    return x_reshaped


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
        shutil.rmtree(f3)

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


import random


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
    output = DupdateK
    outputporo = sgsim2

    output[output >= High_K] = High_K  # highest value in true permeability
    output[output <= Low_K] = Low_K

    outputporo[outputporo >= High_P] = High_P
    outputporo[outputporo <= Low_P] = Low_P

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


class LpLossc(object):
    """
    loss function with rel/abs Lp loss
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLossc, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.shape[0]

        diff_norms = np.linalg.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = np.linalg.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return np.mean(diff_norms / y_norms)
            else:
                return np.sum(diff_norms / y_norms)

        return diff_norms / y_norms


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

    def new(self, x, y):
        norm = torch.abs(x - y)
        norm = torch.sum(norm) / x.shape[0]
        return norm

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
):
    Injector_location = np.where(inn["Qw"].detach().cpu().numpy().ravel() > 0)[0]
    producer_location = np.where(inn["Q"].detach().cpu().numpy().ravel() < 0)[0]

    PERM = rescale_linear_pytorch_numpy(
        np.reshape(inn["perm"].detach().cpu().numpy(), (-1,), "F"), LUB, HUB, aay, bby
    )
    kuse_inj = PERM[Injector_location]

    kuse_prod = PERM[producer_location]

    RE = 0.2 * DX
    Baa = []
    Timz = []
    for kk in range(steppi):
        Ptito = ooutp[:, kk, :, :]
        Stito = oouts[:, kk, :, :]

        average_pressure = np.mean(Ptito.ravel()) * pini_alt
        p_inj = (Ptito.ravel()[Injector_location]) * pini_alt
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
        BHP = Pwf

        up = UO * BO
        down = 2 * np.pi * kuse_prod * krouse * DZ
        right = np.log(RE / rwell) + skin
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        drawdown = average_pressure - pwf_producer
        qoil = np.abs(-(drawdown * J))

        up = UW * BW
        down = 2 * np.pi * kuse_prod * krwusep * DZ
        right = np.log(RE / rwell) + skin
        J = down / (up * right)
        # drawdown = p_prod - pwf_producer
        drawdown = average_pressure - pwf_producer
        qwater = np.abs(-(drawdown * J))
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


def Plot_mean(permbest, permmean, nx, ny, Low_K, High_K, True_perm):
    Low_Ka = Low_K
    High_Ka = High_K

    permmean = np.reshape(permmean, (nx, ny, nz), "F")
    permbest = np.reshape(permbest, (nx, ny, nz), "F")
    True_perm = np.reshape(True_perm, (nx, ny, nz), "F")
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
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

    plt.subplot(2, 2, 3)
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


def Plot_RSM_percentile(pertoutt, True_mat, Namesz):
    timezz = True_mat[:, 0].reshape(-1, 1)

    P10 = pertoutt[0]

    # arekimean =  pertoutt[4]

    plt.figure(figsize=(40, 40))

    # plt.subplot(4,4,1)
    # plt.plot(timezz,True_mat[:,1], color = 'red', lw = '2',\
    #          label ='True')
    # plt.plot(timezz,P10[:,1], color = 'blue', lw = '2', \
    #          label ='LBFGS')

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
    #          label ='LBFGS')

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
    #          label ='LBFGS')

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
    #          label ='LBFGS')

    # plt.xlabel('Time (days)',fontsize = 13)
    # plt.ylabel('BHP(Psia)',fontsize = 13)
    # #plt.ylim((0,25000))
    # plt.title('I4',fontsize = 13)
    # plt.ylim(ymin = 0)
    # plt.xlim(xmin = 0)
    # plt.legend()

    plt.subplot(3, 4, 1)
    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 5], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 2)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 6], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 3)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 7], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 4)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 8], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 5)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 9], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 6)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 10], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 7)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 11], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 8)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 12], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 9)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 13], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 10)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 14], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 11)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 15], color="blue", lw="2", label="LBFGS")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 12)
    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="model")
    plt.plot(timezz, P10[:, 16], color="blue", lw="2", label="LBFGS")

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

    plt.figure(figsize=(40, 40))

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
    plt.plot(timezz, P10[:, 5], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 2)
    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 6], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 3)
    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 7], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 4)
    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 8], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 5)
    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 9], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 6)
    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 10], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 7)
    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 11], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 8)
    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 12], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P4", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 9)
    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 13], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P1", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 10)
    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 14], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P2", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 11)
    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="History model")
    plt.plot(timezz, P10[:, 15], color="blue", lw="2", label="aREKI ")

    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$WWCT{%}$", fontsize=13)
    # plt.ylim((0,25000))
    plt.title("P3", fontsize=13)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.legend()

    plt.subplot(3, 4, 12)
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


def Plot_RSM_single(True_mat, Namesz):
    True_mat = True_mat[0]
    timezz = True_mat[:, 0].reshape(-1, 1)

    plt.figure(figsize=(40, 40))

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

    plt.figure(figsize=(40, 40))

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

    plt.scatter(reali, plttotalerror, color="k")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.xlim([1, (N - 1)])

    plt.subplot(2, 2, 2)
    plt.bar(reali, plttotalerror2, color="c")
    plt.xlabel("Realizations")
    plt.ylabel("RMSE value")
    plt.ylim(ymin=0)
    plt.title("Final Cost function")

    plt.scatter(reali, plttotalerror2, color="k")
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


locc = 10


def Remove_True(enss):
    # enss is the very large ensemble
    # Used the last one as true model
    return np.reshape(enss[:, locc - 1], (-1, 1), "F")


def Trim_ensemble(enss):
    return np.delete(enss, locc - 1, 1)  # delete last column


def shuffle(x, axis=0):
    n_axis = len(x.shape)
    t = np.arange(n_axis)
    t[0] = axis
    t[axis] = 0
    xt = np.transpose(x.copy(), t)
    np.random.shuffle(xt)
    shuffled_x = np.transpose(xt, t)
    return shuffled_x


def log_prob(x):
    ensemble = x.reshape(-1, 1)
    clfye = MinMaxScaler(feature_range=(Low_P, High_P))
    (clfye.fit(ensemble))
    ensemblep = clfye.transform(ensemble)
    # ensemblep = Getporosity_ensemble(ensemble.reshape(-1,1),machine_map,1)
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
    simDatafinal, _, _, _, _ = Forward_model_ensemble(
        modelP,
        modelS,
        ensemblepy,
        rwell,
        skin,
        pwf_producer,
        mazw,
        steppi,
        1,
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

    if loss_type == 1:
        residual = np.abs(simDatafinal - True_data)
        loss = np.sum(residual) / simDatafinal.shape[0]
    else:
        loss = myloss2.rel(simDatafinal, True_data)
    print(loss)
    return loss


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


##############################################################################
# configs
##############################################################################

oldfolder = os.getcwd()
os.chdir(oldfolder)
cur_dir = oldfolder

print("------------------Download pre-trained models------------------------")

# os.makedirs('ESMDA')
if not os.path.exists("../HM_RESULTS/"):
    os.makedirs("../HM_RESULTS/")
else:
    shutil.rmtree("../HM_RESULTS/")
    os.makedirs("../HM_RESULTS/")


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
    2=PINO [PhysicsNeMo IMplemnation]\n3=PINO version 1 \
    [Original paper implementation]\n4=PINO version 2 [Original paper implementation with Flux predicted]\n"
            )
        )
        if (surrogate > 4) or (surrogate < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-4")
        else:
            break

# load data

sizq = 1e4

filename = "../PACKETS/Ganensemble.mat"  # Ensemble generated offline

mat = sio.loadmat(filename, verify_compressed_data_integrity=False)
ini_ensemblef = mat["Z"]
sizq = ini_ensemblef.shape[1]

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


Ne = 1


N_ens = Ne
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
else:
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
else:
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

print("")
print("---------------------------------------------------------------------")
print("")
print("--------------------Historical data Measurement----------------------")
Ne = N_ens

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
print("")
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
if DEFAULT == 1:
    choicey = 2
else:
    choicey = None
    while True:
        choicey = int(
            input("Denoise the update :\n1=Denoising autoencoder\n2=Sal and pepper\n")
        )
        if (choicey > 2) or (choicey < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break
# """
# Deccor=2

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
print("|                 SOLVE INVERSE PROBLEM WITH LBFGS:               |")
print("|-----------------------------------------------------------------|")
print("")
print("History Matching using LBFGS")
print("Novel Implementation by Clement Etienam, SA-NVIDIA: SA-ML/A.I/Energy")


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
    Ne = 1
    N_ens = Ne
else:
    Ne = 1
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

            bb = os.path.isfile(("../PACKETS/Ganensemble.mat"))
            if not bb:
                print("Get initial geology from saved Multiple-point-statistics run")
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1w81M5M2S0PD9CF2761dxmiKQ5c0OFPaH", "../PACKETS/Ganensemble.mat"
                )
                print("...Downlaod completed.......")
            else:
                pass
            filename = "../PACKETS/Ganensemble.mat"  # Ensemble generated offline

            mat = sio.loadmat(filename)
            ini_ensemblef = mat["Z"]
            index = np.random.choice(ini_ensemblef.shape[1], Ne + 100, replace=False)

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

print("")
print(
    "LBFGS with Denoising Autoencoder\
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
                "17Ekju_MvRs_oOhDvpfirp8-rFy6x1D2u", "../PACKETS/denosingautoencoder.h5"
            )
            print("...Downlaod completed.......")

        else:
            DenosingAutoencoder(nx, ny, nz, High_K1, Low_K1)  # Learn for permeability
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


index = np.random.choice(ini_ensemble.shape[1], Ne, replace=False)
ensemble = ini_ensemble[:, index]
ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
# ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)

clfye = MinMaxScaler(feature_range=(Low_P, High_P))
(clfye.fit(ensemble))
ensemblep = clfye.transform(ensemble)


ensemble, ensemblep = honour2(
    ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
)


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

Trueec = torch.from_numpy(True_data).to(device, dtype=torch.float32)
os.chdir(oldfolder)


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


clement = ensemblepy["perm"].requires_grad_()
# ensemblepy['perm'].requires_grad_()
myloss = LpLoss(size_average=True)
myloss2 = LpLossc(size_average=True)
muloss = LpLoss(size_average=True)
loss_type = 1

total = int(2e3)
learning_rate = 1e-3
gamma = 0.5
step_size = 1000
loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam([clement], lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# optimizer_pressure = torch.optim.Adam(modelP.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler_pressure = torch.optim.lr_scheduler.StepLR(optimizer_pressure, step_size=step_size, gamma=gamma)


# optimizer_sat = torch.optim.Adam(modelS.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler_sat = torch.optim.lr_scheduler.StepLR(optimizer_sat, step_size=step_size, gamma=gamma)

# optimizer = torch.optim.AdamW([ensemblepy['perm']])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

start_epoch = 1
epochs = int(50000)
hista = []
aha = 0
costa = []


for na in range(2):
    for epoch in range(start_epoch, epochs + 1):
        loss_train = 0.0
        # modelP.train()
        # modelS.train()
        print("iter " + str(epoch) + " | " + str(epochs))
        print("**************************************************************")

        optimizer.zero_grad()
        # optimizer_pressure.zero_grad()
        # optimizer_sat.zero_grad()

        ensemblepy["perm"] = clement
        minimum, maximum = torch.min(clement), torch.max(clement)
        m = (hpor - mpor) / (maximum - minimum)
        b = mpor - m * minimum
        look = m * clement + b
        look = torch.clip(look, mpor, hpor)
        ensemblepy["Phi"] = look

        mazw = 0  # Dont smooth the presure field
        simDatafinal, fpp, fss = Forward_model_pytorch(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            1,
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
            UIR,
            LUB,
            HUB,
            aay,
            bby,
            nx,
            ny,
            device,
            muloss,
        )

        sim = simDatafinal.to(device, dtype=torch.float32)
        if loss_type == 1:
            residual = torch.abs(sim - Trueec)
            loss = torch.sum(residual) / sim.shape[0] + fpp + fss
            # loss = loss

        else:
            loss = myloss.rel(sim, Trueec)  # + fpp + fss
        loss.backward()

        optimizer.step()
        # optimizer_pressure.step()
        # optimizer_sat.step()

        # optimizer.zero_grad()
        # optimizer_pressure.zero_grad()
        # optimizer_sat.zero_grad()

        scheduler.step()

        ahnewa = loss.detach().cpu().numpy()
        hista.append(ahnewa)
        print("TRAINING")
        if aha < ahnewa:
            print(
                "   INVERSE PROBLEM COMMENT : Loss increased by "
                + str(abs(aha - ahnewa))
            )
        elif aha > ahnewa:
            print(
                "   INVERSE PROBLEM COMMENT : Loss decreased by "
                + str(abs(aha - ahnewa))
            )
        else:
            print("   INVERSE PROBLEM COMMENT : No change in Loss ")

        print("   loss = " + str(loss.detach().cpu().numpy()))

        aha = ahnewa
        costa.append(ahnewa)
        look1 = clement
        look1 = look1.detach().cpu().numpy()
        sio.savemat("../HM_RESULTS/Matrix1.mat", {"permeability_adam": look1})


snn = 0
steps = 100
lr = 1e-1
epochs = 2
N = epochs
total = 200


optimizer1 = torch.optim.LBFGS(
    [clement],
    lr,
    max_iter=steps,
    max_eval=None,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn="strong_wolfe",
)

# optimizer_pressure = torch.optim.LBFGS(modelP.parameters() , lr,
#                               max_iter = steps,
#                               max_eval = None,
#                               tolerance_grad = 1e-11,
#                               tolerance_change = 1e-11,
#                               history_size = 1000,
#                               line_search_fn = 'strong_wolfe')

# optimizer_sat = torch.optim.LBFGS(modelS.parameters() , lr,
#                               max_iter = steps,
#                               max_eval = None,
#                               tolerance_grad = 1e-11,
#                               tolerance_change = 1e-11,
#                               history_size = 1000,
#                               line_search_fn = 'strong_wolfe')

for kk in range(total):
    print("iter " + str(kk + 1) + " | " + str(total))
    print("**************************************************************")

    # modelP.train()
    # modelS.train()
    # clement = torch.clip(clement,LUB, HUB)
    def closure():
        if torch.is_grad_enabled():
            optimizer1.zero_grad()
            # optimizer_pressure.zero_grad()
            # optimizer_sat.zero_grad()

        mazw = 0  # Dont smooth the presure field
        ensemblepy["perm"] = clement
        minimum, maximum = torch.min(clement), torch.max(clement)
        m = (hpor - mpor) / (maximum - minimum)
        b = mpor - m * minimum
        look = m * clement + b
        ensemblepy["Phi"] = look
        look = torch.clip(look, mpor, hpor)
        simDatafinal, fpp, fss = Forward_model_pytorch(
            modelP,
            modelS,
            ensemblepy,
            rwell,
            skin,
            pwf_producer,
            mazw,
            steppi,
            1,
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
            UIR,
            LUB,
            HUB,
            aay,
            bby,
            nx,
            ny,
            device,
            muloss,
        )

        sim = simDatafinal.to(device, dtype=torch.float32)

        if loss_type == 1:
            residual = torch.abs(sim - Trueec)
            loss = torch.sum(residual) / sim.shape[0]  # + fpp + fss
        else:
            loss = myloss.rel(sim, (Trueec))  # + fpp + fss

        if loss.requires_grad:
            loss.backward()
        print("   loss = " + str(loss.detach().cpu().numpy()))
        look1 = clement
        look1 = look1.detach().cpu().numpy()
        sio.savemat("../HM_RESULTS/Matrix2.mat", {"permeability_LBFGS": look1})
        return loss

    optimizer1.step(closure)
    # optimizer_pressure.step(closure)
    # optimizer_sat.step(closure)


print("")
print("TNC")

mat = sio.loadmat("../HM_RESULTS/Matrix2.mat")
ini = mat["permeability_LBFGS"]
ini = np.reshape(ini, (-1,), "F")
# ini[ini<LUB] = LUB
# ini[ini>HUB] = HUB

# bnds = np.array([[LUB,HUB]]*nx*ny*nz, dtype=float)
# result = fmin_tnc(log_prob, ini.ravel(),bounds=bnds, approx_grad=True,args=())
# perm_mean = result[0]

perm_mean = ini

# print('')
# print('Powell algorithm routine')
# from scipy import optimize
# bounds=np.array([[LUB,HUB]]*nx*ny*nz, dtype=float)
# # perm_mean[perm_mean<LUB] = LUB
# # perm_mean[perm_mean>HUB] = HUB
# anss = optimize.minimize(log_prob, x0 = perm_mean, args=(), method='Powell', bounds=bounds,\
# tol=None, callback=print,options={'xtol': 1e-20, 'ftol': 1e-20, 'maxiter': 3,\
# 'maxfev': 1e5, 'disp': True, 'direc': None, 'return_all': False})

# perm_mean = anss.x

print("")
Best_K = rescale_linear(perm_mean, Low_K1, High_K1)
ensemble = Best_K.reshape(-1, 1)
clfye = MinMaxScaler(feature_range=(Low_P, High_P))
(clfye.fit(ensemble))
ensemblep = clfye.transform(ensemble)
Best_phi = ensemblep


print("****************************************************************")


yes_best_k = Best_K.reshape(-1, 1)
yes_best_p = Best_phi.reshape(-1, 1)


ensemblea, ensemblepa = honour2(
    Best_phi, Best_K, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
)


if choicey == 1:
    ensemble = use_denoising(ensemble, nx, ny, nz, Ne, High_K1)
    ensemblep = use_denoisingp(ensemblep, nx, ny, nz, Ne, High_P)
else:
    temp_K = np.reshape(Best_K, (nx, ny), "F")
    temp_phi = np.reshape(Best_phi, (nx, ny), "F")

    # # fast algorithm
    # timk = 5
    # for n in range(timk):
    #     sigma_est1 = np.mean(estimate_sigma(temp_K))
    #     sigma_est2 = np.mean(estimate_sigma(temp_phi))
    #     #print(f'estimated noise standard deviation for permeability = {sigma_est1}')
    #     print('')
    #     #print(f'estimated noise standard deviation for porosity = {sigma_est2}')

    #     patch_kw = dict(patch_size=5,      # 5x5 patches
    #                     patch_distance=6)
    #     temp_K = denoise_nl_means(temp_K, h=0.8 * sigma_est1, fast_mode=True,
    #                                     **patch_kw)
    #     temp_phi = denoise_nl_means(temp_phi, h=0.8 * sigma_est2, fast_mode=True,
    #                                     **patch_kw)

    ensemble = np.reshape(temp_K, (-1, 1), "F")
    ensemblep = np.reshape(temp_phi, (-1, 1), "F")

ensemble, ensemblep = honour2(
    ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
)


controlbest = ensemble
controlbestp = ensemblep

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


if not os.path.exists("../HM_RESULTS/LBFGS"):
    os.makedirs("../HM_RESULTS/LBFGS")
else:
    shutil.rmtree("../HM_RESULTS/LBFGS")
    os.makedirs("../HM_RESULTS/LBFGS")


os.chdir("../HM_RESULTS/LBFGS")
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
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk - 1, steppi - 1)
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
    plot3d2(Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii)

    watsz = wats[0, kk, :, :][:, :, None]
    maxii = max(watsz.ravel())
    minii = min(watsz.ravel())
    plot3d2(watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii)

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


residual = np.abs(yycheck - True_data)
cc = np.sum(residual) / True_data.shape[0]
print("RMSE  = : " + str(cc))
os.chdir("../HM_RESULTS")

controljj2 = ensemble


Plot_mean(controlbest, controljj2, nx, ny, Low_K1, High_K1, True_K)
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
    1,
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
    1,
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
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk - 1, steppi - 1)
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
    plot3d2(Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii)

    watsz = watsbest[0, kk, :, :][:, :, None]
    maxii = max(watsz.ravel())
    minii = min(watsz.ravel())
    plot3d2(watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii)

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


residual = np.abs(yycheck - True_data)
cc = np.sum(residual) / True_data.shape[0]

print("RMSE of overall best model  = : " + str(cc))


os.chdir(oldfolder)


print("--------------------SECTION ADAPTIVE REKI ENDED----------------------------")

elapsed_time_secs = time.time() - start_time


comment = "LBFGS Inversion"
print("Inverse problem solution used =: " + comment)
msg = "Execution took: %s secs (Wall clock time)" % timedelta(
    seconds=round(elapsed_time_secs)
)
print(msg)
print("-------------------PROGRAM EXECUTED-----------------------------------")
