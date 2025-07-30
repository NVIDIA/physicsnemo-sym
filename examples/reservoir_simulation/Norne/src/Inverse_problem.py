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

Neural operator to model the black oil model.

Use the devloped Neural operator surrogate in a Bayesian Inverse Problem Workflow:
    



@Data Assimilation Methods: 
    - Weighted Adaptive REKI - Adaptive Regularised Ensemble Kalman\
Inversion with covariance localisation

    
66 Measurements to be matched: 22 WOPR , 22 WWPR, 22 WGPR
The Field has 22 producers and 9 Water Injectors, 4 Gas Injectors

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
import scipy.io
import gzip
import xgboost as xgb
import matplotlib as mpl
import scipy.ndimage.morphology as spndmo
from datetime import timedelta
import os.path
import pickle
from scipy.fftpack import dct
from scipy.fftpack.realtransforms import idct
from scipy import interpolate
import multiprocessing

# from gstools import SRF, Gaussian
import mpslib as mps
import numpy.matlib
from pyDOE import lhs
from matplotlib.colors import LinearSegmentedColormap

# rcParams['font.family'] = 'sans-serif'
# cParams['font.sans-serif'] = ['Helvetica']
import matplotlib.lines as mlines
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from shutil import rmtree
from physicsnemo.sym.models.fno import *
from physicsnemo.sym.key import Key
from joblib import Parallel, delayed
from collections import OrderedDict
import os.path
from numpy.linalg import inv
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
# from gstools.random import MasterRNG
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
from imresize import *
import matplotlib.colors
from matplotlib import cm
import yaml
import gc
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = collect_dynamic_libs("nvidia.cuda_nvrtc", search_patterns=["lib*.so.*"])

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
text = """
|_   _|                              |  __ \         | |   | |               
   | |  _ ____   _____ _ __ ___  ___  | |__) _ __ ___ | |__ | | ___ _ __ ___  
   | | | '_ \ \ / / _ | '__/ __|/ _ \ |  ___| '__/ _ \| '_ \| |/ _ | '_ ` _ \ 
  _| |_| | | \ V |  __| |  \__ |  __/ | |   | | | (_) | |_) | |  __| | | | | |
 |_____|_| |_|\_/ \___|_|  |___/\___| |_|   |_|  \___/|_.__/|_|\___|_| |_| |_|
"""
print(text)
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


def ProgressBar2(Total, Progress):
    try:
        # Calcuting progress between 0 and 1 for percentage.
        Progress = float(Progress) / float(Total)
        # Doing this conditions at final progressing.
        if Progress >= 1.0:
            Progress = 1
            # Show the completed status
            return "100%"
        # Show the percentage of completion
        return "{:.0f}%".format(round(Progress * 100, 0))
    except:
        print("")
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


def Reinvent(matt):
    nx, ny, nz = matt.shape[1], matt.shape[2], matt.shape[0]
    dess = np.zeros((nx, ny, nz))
    for i in range(nz):
        dess[:, :, i] = matt[i, :, :]
    return dess


# numpy.random.seed(1)


def fit_clement(tensor, target_min, target_max, tensor_min, tensor_max):
    # Rescale between target min and target max
    rescaled_tensor = tensor / tensor_max
    return rescaled_tensor


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


# Define the threshold as the largest finite representable number for np.float32
threshold = np.finfo(np.float32).max


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
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    modelP1,
    modelW1,
    modelG1,
    modelO1,
    device,
    modelP2,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
    degg,
    experts,
):
    #### ===================================================================== ####
    #                     RESERVOIR SIMULATOR WITH MODULUS
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

    modelP1, modelW1, modelG1, modelP2 : torch.nn.Module
        Trained PyTorch models for predicting pressure, water saturation, gas saturation, and the secondary pressure model respectively.

    device : torch.device
        The device (CPU or GPU) where the PyTorch models will run.

    min_out_fcn, max_out_fcn : float
        Minimum and maximum values for output scaling.

    Time : array-like
        Reservoir simulation time values.

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
    Afterward, a new set of inputs for a secondary model (modelP2) is prepared
    using the processed data. Finally, the function predicts outputs using the
    modelP2 and returns the predicted values.
    """

    pressure = []
    Swater = []
    Sgas = []
    Soil = []

    for clem in range(N):
        temp = {
            "perm": x_true["perm"][clem, :, :, :, :][None, :, :, :, :],
            "Phi": x_true["Phi"][clem, :, :, :, :][None, :, :, :, :],
            "fault": x_true["fault"][clem, :, :, :, :][None, :, :, :, :],
            "Pini": x_true["Pini"][clem, :, :, :, :][None, :, :, :, :],
            "Swini": x_true["Swini"][clem, :, :, :, :][None, :, :, :, :],
        }

        with torch.no_grad():
            ouut_p1 = modelP1(temp)["pressure"]
            ouut_s1 = modelW1(temp)["water_sat"]
            ouut_sg1 = modelG1(temp)["gas_sat"]
            ouut_so1 = modelO1(temp)["oil_sat"]

        pressure.append(ouut_p1)
        Swater.append(ouut_s1)
        Sgas.append(ouut_sg1)
        Soil.append(ouut_so1)

        del temp, ouut_p1, ouut_s1, ouut_sg1, ouut_so1
        torch.cuda.empty_cache()

    pressure = torch.vstack(pressure).detach().cpu().numpy()
    pressure = Make_correct(pressure)

    Swater = torch.vstack(Swater).detach().cpu().numpy()
    Swater = Make_correct(Swater)

    Sgas = torch.vstack(Sgas).detach().cpu().numpy()
    Sgas = Make_correct(Sgas)

    Soil = torch.vstack(Soil).detach().cpu().numpy()
    Soil = Make_correct(Soil)

    # The inputs are at the normal scale

    perm = convert_back(
        x_true["perm"].detach().cpu().numpy(), target_min, target_max, minK, maxK
    )
    perm = Make_correct(perm)
    # Time = convert_back(x_true["Time"].detach().cpu().numpy() ,target_min,target_max,minT,maxT)
    # Time = Make_correct(Time)
    pressure = convert_back(pressure, target_min, target_max, minP, maxP)

    if Trainmoe == 1:
        innn = np.zeros((N, 90, steppi))
    else:
        innn = np.zeros((N, steppi, 90))

    for i in range(N):
        # progressBar = "\rEnsemble Forwarding: " + ProgressBar2(N-1, i-1)
        # ShowBar(progressBar)
        # time.sleep(1)
        ##INPUTS
        permuse = perm[i, 0, :, :, :]
        # Permeability
        a1 = np.mean(permuse[14, 30, :]) * np.ones((steppi, 1))
        a2 = np.mean(permuse[9, 31, :]) * np.ones((steppi, 1))
        a3 = np.mean(permuse[13, 33, :]) * np.ones((steppi, 1))
        a5 = np.mean(permuse[8, 36, :]) * np.ones((steppi, 1))
        a6 = np.mean(permuse[8, 45, :]) * np.ones((steppi, 1))
        a7 = np.mean(permuse[9, 28, :]) * np.ones((steppi, 1))
        a8 = np.mean(permuse[9, 23, :]) * np.ones((steppi, 1))
        a9 = np.mean(permuse[21, 21, :]) * np.ones((steppi, 1))
        a10 = np.mean(permuse[13, 27, :]) * np.ones((steppi, 1))
        a11 = np.mean(permuse[18, 37, :]) * np.ones((steppi, 1))
        a12 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a13 = np.mean(permuse[15, 65, :]) * np.ones((steppi, 1))
        a14 = np.mean(permuse[24, 36, :]) * np.ones((steppi, 1))
        a15 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a16 = np.mean(permuse[11, 71, :]) * np.ones((steppi, 1))
        a17 = np.mean(permuse[17, 67, :]) * np.ones((steppi, 1))
        a18 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a19 = np.mean(permuse[37, 97, :]) * np.ones((steppi, 1))
        a20 = np.mean(permuse[6, 63, :]) * np.ones((steppi, 1))
        a21 = np.mean(permuse[14, 75, :]) * np.ones((steppi, 1))
        a22 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a23 = np.mean(permuse[10, 27, :]) * np.ones((steppi, 1))

        permxx = np.hstack(
            (
                a1,
                a2,
                a3,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
                a15,
                a16,
                a17,
                a18,
                a19,
                a20,
                a21,
                a22,
                a23,
            )
        )

        presure_use = pressure[i, :, :, :, :]
        gas_use = Sgas[i, :, :, :, :]
        water_use = Swater[i, :, :, :, :]
        oil_use = Soil[i, :, :, :, :]
        Time_usee = Time[i, :, :, :, :]

        a1 = np.zeros((steppi, 1))
        a2 = np.zeros((steppi, 22))
        a3 = np.zeros((steppi, 22))
        a5 = np.zeros((steppi, 22))
        a4 = np.zeros((steppi, 1))

        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse * effectiveuse)

            uniegas = gas_use[k, :, :, :]
            permuse = uniegas
            a2[k, 0] = np.mean(permuse[14, 30, :])
            a2[k, 1] = np.mean(permuse[9, 31, :])
            a2[k, 2] = np.mean(permuse[13, 33, :])
            a2[k, 3] = np.mean(permuse[8, 36, :])
            a2[k, 4] = np.mean(permuse[8, 45, :])
            a2[k, 5] = np.mean(permuse[9, 28, :])
            a2[k, 6] = np.mean(permuse[9, 23, :])
            a2[k, 7] = np.mean(permuse[21, 21, :])
            a2[k, 8] = np.mean(permuse[13, 27, :])
            a2[k, 9] = np.mean(permuse[18, 37, :])
            a2[k, 10] = np.mean(permuse[18, 53, :])
            a2[k, 11] = np.mean(permuse[15, 65, :])
            a2[k, 12] = np.mean(permuse[24, 36, :])
            a2[k, 13] = np.mean(permuse[18, 54, :])
            a2[k, 14] = np.mean(permuse[11, 71, :])
            a2[k, 15] = np.mean(permuse[17, 67, :])
            a2[k, 16] = np.mean(permuse[12, 66, :])
            a2[k, 17] = np.mean(permuse[37, 97, :])
            a2[k, 18] = np.mean(permuse[6, 63, :])
            a2[k, 19] = np.mean(permuse[14, 75, :])
            a2[k, 20] = np.mean(permuse[12, 66, :])
            a2[k, 21] = np.mean(permuse[10, 27, :])

            uniewater = water_use[k, :, :, :]
            permuse = uniewater
            a3[k, 0] = np.mean(permuse[14, 30, :])
            a3[k, 1] = np.mean(permuse[9, 31, :])
            a3[k, 2] = np.mean(permuse[13, 33, :])
            a3[k, 3] = np.mean(permuse[8, 36, :])
            a3[k, 4] = np.mean(permuse[8, 45, :])
            a3[k, 5] = np.mean(permuse[9, 28, :])
            a3[k, 6] = np.mean(permuse[9, 23, :])
            a3[k, 7] = np.mean(permuse[21, 21, :])
            a3[k, 8] = np.mean(permuse[13, 27, :])
            a3[k, 9] = np.mean(permuse[18, 37, :])
            a3[k, 10] = np.mean(permuse[18, 53, :])
            a3[k, 11] = np.mean(permuse[15, 65, :])
            a3[k, 12] = np.mean(permuse[24, 36, :])
            a3[k, 13] = np.mean(permuse[18, 54, :])
            a3[k, 14] = np.mean(permuse[11, 71, :])
            a3[k, 15] = np.mean(permuse[17, 67, :])
            a3[k, 16] = np.mean(permuse[12, 66, :])
            a3[k, 17] = np.mean(permuse[37, 97, :])
            a3[k, 18] = np.mean(permuse[6, 63, :])
            a3[k, 19] = np.mean(permuse[14, 75, :])
            a3[k, 20] = np.mean(permuse[12, 66, :])
            a3[k, 21] = np.mean(permuse[10, 27, :])

            unieoil = oil_use[k, :, :, :]
            permuse = unieoil
            a5[k, 0] = np.mean(permuse[14, 30, :])
            a5[k, 1] = np.mean(permuse[9, 31, :])
            a5[k, 2] = np.mean(permuse[13, 33, :])
            a5[k, 3] = np.mean(permuse[8, 36, :])
            a5[k, 4] = np.mean(permuse[8, 45, :])
            a5[k, 5] = np.mean(permuse[9, 28, :])
            a5[k, 6] = np.mean(permuse[9, 23, :])
            a5[k, 7] = np.mean(permuse[21, 21, :])
            a5[k, 8] = np.mean(permuse[13, 27, :])
            a5[k, 9] = np.mean(permuse[18, 37, :])
            a5[k, 10] = np.mean(permuse[18, 53, :])
            a5[k, 11] = np.mean(permuse[15, 65, :])
            a5[k, 12] = np.mean(permuse[24, 36, :])
            a5[k, 13] = np.mean(permuse[18, 54, :])
            a5[k, 14] = np.mean(permuse[11, 71, :])
            a5[k, 15] = np.mean(permuse[17, 67, :])
            a5[k, 16] = np.mean(permuse[12, 66, :])
            a5[k, 17] = np.mean(permuse[37, 97, :])
            a5[k, 18] = np.mean(permuse[6, 63, :])
            a5[k, 19] = np.mean(permuse[14, 75, :])
            a5[k, 20] = np.mean(permuse[12, 66, :])
            a5[k, 21] = np.mean(permuse[10, 27, :])

            unietime = Time_usee[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]

        inn1 = np.hstack((permxx, a1, a5, a2, a3, a4))

        inn1 = fit_clement(inn1, target_min, target_max, min_inn_fcn, max_inn_fcn)

        if Trainmoe == 1:
            innn[i, :, :] = inn1.T
        else:
            innn[i, :, :] = inn1

    if Trainmoe == 1:
        innn = torch.from_numpy(innn).to(device, torch.float32)
        ouut_p = []
        # x_true = ensemblepy
        for clem in range(N):
            temp = {"X": innn[clem, :, :][None, :, :]}

            with torch.no_grad():
                ouut_p1 = modelP2(temp)["Y"]
                ouut_p1 = convert_back(
                    ouut_p1.detach().cpu().numpy(),
                    target_min,
                    target_max,
                    min_out_fcn,
                    max_out_fcn,
                )

            ouut_p.append(ouut_p1)
            del temp
            torch.cuda.empty_cache()
        ouut_p = np.vstack(ouut_p)
        ouut_p = np.transpose(ouut_p, (0, 2, 1))
        ouut_p[ouut_p <= 0] = 0
    else:
        innn = np.vstack(innn)
        cluster_all = sio.loadmat("../ML_MACHINE/clustersizescost.mat")["cluster"]
        cluster_all = np.reshape(cluster_all, (-1, 1), "F")

        clemes = Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
            delayed(PREDICTION_CCR__MACHINE)(
                ib,
                int(cluster_all[ib, :]),
                innn,
                innn.shape[1],
                "../ML_MACHINE",
                oldfolder,
                pred_type,
                degg,
                experts,
            )
            for ib in range(66)
        )

        ouut_p = np.array(Split_Matrix(np.hstack(clemes), N))
        ouut_p = convert_back(ouut_p, target_min, target_max, min_out_fcn, max_out_fcn)
        ouut_p[ouut_p <= 0] = 0

    sim = []
    for zz in range(ouut_p.shape[0]):
        Oilz = ouut_p[zz, :, :22] / scalei
        Watsz = ouut_p[zz, :, 22:44] / scalei2
        gasz = ouut_p[zz, :, 44:66] / scalei3
        spit = np.hstack([Oilz, Watsz, gasz])
        spit = np.reshape(spit, (-1, 1), "F")
        spit = remove_rows(spit, rows_to_remove).reshape(-1, 1)
        use = np.reshape(spit, (-1, 1), "F")

        sim.append(use)
    sim = np.hstack(sim)
    # progressBar = "\rEnsemble Forwarding: " + ProgressBar2(N-1, i)
    # ShowBar(progressBar)
    # time.sleep(1)
    return sim, ouut_p, pressure, Swater, Sgas, Soil


def cov(G):
    return cp.cov(G)


def KalmanGain(G, params, Gamma, N, alpha):
    CnGG = cov(G)
    mean_params = cp.mean(params, axis=1).reshape(-1, 1)
    mean_G = cp.mean(G, axis=1).reshape(-1, 1)
    Cyd = (params - mean_params) @ ((G - mean_G).T)
    # Compute the SVD of the denominator matrix
    U, s, Vh = cp.linalg.svd(CnGG + (alpha * Gamma))
    # Invert the singular values (replace small values with zero for stability)
    s_inv = 1.0 / s
    s_inv[s_inv < 1e-15] = 0
    inv_denominator = cp.dot(Vh.T, s_inv[:, cp.newaxis] * U.T)
    K = (1 / (N - 1)) * Cyd @ inv_denominator
    return K


def predict_machine3(a0, deg, model, poly):
    predicted = model.predict(poly.fit_transform(a0))
    return predicted


def PREDICTION_CCR__MACHINE(
    ii,
    nclusters,
    inputtest,
    numcols,
    training_master,
    oldfolder,
    pred_type,
    deg,
    experts,
):
    filenamex = "clfx_%d.asv" % ii
    filenamey = "clfy_%d.asv" % ii

    os.chdir(training_master)
    if experts == 1:
        filename1 = "Classifier_%d.bin" % ii
        loaded_model = xgb.Booster({"nthread": 4})  # init model
        loaded_model.load_model(filename1)  # load data
    else:
        filename1 = "Classifier_%d.pkl" % ii
        with open(filename1, "rb") as file:
            loaded_model = pickle.load(file)
    clfx = pickle.load(open(filenamex, "rb"))
    clfy = pickle.load(open(filenamey, "rb"))
    os.chdir(oldfolder)

    inputtest = clfx.transform(inputtest)
    if experts == 2:
        labelDA = loaded_model.predict(inputtest)
    else:
        labelDA = loaded_model.predict(xgb.DMatrix(inputtest))
        if nclusters == 2:
            labelDAX = 1 - labelDA
            labelDA = np.reshape(labelDA, (-1, 1))
            labelDAX = np.reshape(labelDAX, (-1, 1))
            labelDA = np.concatenate((labelDAX, labelDA), axis=1)
            labelDA = np.argmax(labelDA, axis=-1)
        else:
            labelDA = np.argmax(labelDA, axis=-1)
        labelDA = np.reshape(labelDA, (-1, 1), "F")

    numrowstest = len(inputtest)
    clementanswer = np.zeros((numrowstest, 1))
    # numcols=13
    labelDA = np.reshape(labelDA, (-1, 1), "F")
    for i in range(nclusters):
        # print('-- Predicting cluster: ' + str(i+1) + ' | ' + str(nclusters))
        if experts == 1:  # Polynomial regressor experts
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            filename2b = "polfeat_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            os.chdir(training_master)

            with open(filename2, "rb") as file:
                model0 = pickle.load(file)

            with open(filename2b, "rb") as filex:
                poly0 = pickle.load(filex)

            os.chdir(oldfolder)
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                clementanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine3(a00, deg, model0, poly0), (-1, 1)
                )
        else:  # XGBoost experts
            loaded_modelr = xgb.Booster({"nthread": 4})  # init model
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".bin"

            os.chdir(training_master)
            loaded_modelr.load_model(filename2)  # load data

            os.chdir(oldfolder)

            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                clementanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine11(a00, loaded_modelr), (-1, 1)
                )

    clementanswer = clfy.inverse_transform(clementanswer)
    return clementanswer


def predict_machine11(a0, model):
    ynew = model.predict(xgb.DMatrix(a0))
    return ynew


def Get_data_FFNN(
    oldfolder, N, pressure, Sgas, Swater, perm, Time, steppi, steppi_indices
):
    """
    Get_data_FFNN Function
    -----------------------
    This function retrieves data for Feed-Forward Neural Network (FFNN) models.

    Parameters:
    - oldfolder (str): Path to the directory/folder where the old data or relevant files are stored.
    - N (int): Represents the number of data points, samples, or another significant numerical value.
    - pressure (list/array): List or array containing pressure values or measurements.
    - Sgas (list/array): List or array representing gas saturation values or measurements.
    - Swater (list/array): List or array representing water saturation values or measurements.
    - perm (list/array): List or array representing permeability values or measurements.
    - Time (list/array): List or array containing time values or measurements, likely associated with the other data points.
    - steppi (int/float): Step size, increment value, or another significant numerical parameter for processing or iterating.
    - steppi_indices (list/array): List or array representing specific indices or locations to apply the 'steppi' value.

    Returns:
    -

    """

    ouut = np.zeros((N, steppi, 66))
    innn = np.zeros((N, steppi, 90))
    steppii = 246
    for i in range(N):
        WOIL1 = np.zeros((steppi, 22))
        WWATER1 = np.zeros((steppi, 22))
        WGAS1 = np.zeros((steppi, 22))
        WWINJ1 = np.zeros((steppi, 9))
        WGASJ1 = np.zeros((steppi, 4))

        # f = 'Realisation'
        # folder = f + str(i)
        folder = "../RUNS/Realisation" + str(i)
        os.chdir(folder)
        ## IMPORT FOR OIL
        A2oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None)

        B_1BHoilsim = A2oilsim[5].values[:steppii]
        B_1Hoilsim = A2oilsim[6].values[:steppii]
        B_2Hoilsim = A2oilsim[7].values[:steppii]
        B_3Hoilsim = A2oilsim[8].values[:steppii]
        B_4BHoilsim = A2oilsim[9].values[:steppii]

        A22oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1801, sep="\s+", header=None)
        B_4DHoilsim = A22oilsim[1].values[:steppii]
        B_4Hoilsim = A22oilsim[2].values[:steppii]
        D_1CHoilsim = A22oilsim[3].values[:steppii]
        D_1Hoilsim = A22oilsim[4].values[:steppii]
        D_2Hoilsim = A22oilsim[5].values[:steppii]
        D_3AHoilsim = A22oilsim[6].values[:steppii]
        D_3BHoilsim = A22oilsim[7].values[:steppii]
        D_4AHoilsim = A22oilsim[8].values[:steppii]
        D_4Hoilsim = A22oilsim[9].values[:steppii]

        A222oilsim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None
        )

        E_1Hoilsim = A222oilsim[1].values[:steppii]
        E_2AHoilsim = A222oilsim[2].values[:steppii]
        E_2Hoilsim = A222oilsim[3].values[:steppii]
        E_3AHoilsim = A222oilsim[4].values[:steppii]
        E_3CHoilsim = A222oilsim[5].values[:steppii]
        E_3Hoilsim = A222oilsim[6].values[:steppii]
        E_4AHoilsim = A222oilsim[7].values[:steppii]
        K_3Hoilsim = A222oilsim[8].values[:steppii]

        WOIL1[:, 0] = B_1BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 1] = B_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 2] = B_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 3] = B_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 4] = B_4BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 5] = B_4DHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 6] = B_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 7] = D_1CHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 8] = D_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 9] = D_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 10] = D_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 11] = D_3BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 12] = D_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 13] = D_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 14] = E_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 15] = E_2AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 16] = E_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 17] = E_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 18] = E_3CHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 19] = E_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 20] = E_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 21] = K_3Hoilsim.ravel()[steppi_indices - 1]

        ##IMPORT FOR WATER
        A2watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None
        )
        B_1BHwatersim = A2watersim[9].values[:steppii]

        A22watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2569, sep="\s+", header=None
        )
        B_1Hwatersim = A22watersim[1].values[:steppii]
        B_2Hwatersim = A22watersim[2].values[:steppii]
        B_3Hwatersim = A22watersim[3].values[:steppii]
        B_4BHwatersim = A22watersim[4].values[:steppii]
        B_4DHwatersim = A22watersim[5].values[:steppii]
        B_4Hwatersim = A22watersim[6].values[:steppii]
        D_1CHwatersim = A22watersim[7].values[:steppii]
        D_1Hwatersim = A22watersim[8].values[:steppii]
        D_2Hwatersim = A22watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2825, sep="\s+", header=None
        )
        D_3AHwatersim = A222watersim[1].values[:steppii]
        D_3BHwatersim = A222watersim[2].values[:steppii]
        D_4AHwatersim = A222watersim[3].values[:steppii]
        D_4Hwatersim = A222watersim[4].values[:steppii]
        E_1Hwatersim = A222watersim[5].values[:steppii]
        E_2AHwatersim = A222watersim[6].values[:steppii]
        E_2Hwatersim = A222watersim[7].values[:steppii]
        E_3AHwatersim = A222watersim[8].values[:steppii]
        E_3CHwatersim = A222watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=3081, sep="\s+", header=None
        )
        E_3Hwatersim = A222watersim[1].values[:steppii]
        E_4AHwatersim = A222watersim[2].values[:steppii]
        K_3Hwatersim = A222watersim[3].values[:steppii]

        WWATER1[:, 0] = B_1BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 1] = B_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 2] = B_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 3] = B_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 4] = B_4BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 5] = B_4DHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 6] = B_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 7] = D_1CHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 8] = D_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 9] = D_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 10] = D_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 11] = D_3BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 12] = D_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 13] = D_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 14] = E_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 15] = E_2AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 16] = E_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 17] = E_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 18] = E_3CHwatersim.ravel()[steppi_indices - 1]

        WWATER1[:, 19] = E_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 20] = E_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 21] = K_3Hwatersim.ravel()[steppi_indices - 1]

        ## GAS PRODUCTION RATE
        A2gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1033, sep="\s+", header=None)
        B_1BHgassim = A2gassim[1].values[:steppii]
        B_1Hgassim = A2gassim[2].values[:steppii]
        B_2Hgassim = A2gassim[3].values[:steppii]
        B_3Hgassim = A2gassim[4].values[:steppii]
        B_4BHgassim = A2gassim[5].values[:steppii]
        B_4DHgassim = A2gassim[6].values[:steppii]
        B_4Hgassim = A2gassim[7].values[:steppii]
        D_1CHgassim = A2gassim[8].values[:steppii]
        D_1Hgassim = A2gassim[9].values[:steppii]

        A22gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1289, sep="\s+", header=None)
        D_2Hgassim = A22gassim[1].values[:steppii]
        D_3AHgassim = A22gassim[2].values[:steppii]
        D_3BHgassim = A22gassim[3].values[:steppii]
        D_4AHgassim = A22gassim[4].values[:steppii]
        D_4Hgassim = A22gassim[5].values[:steppii]
        E_1Hgassim = A22gassim[6].values[:steppii]
        E_2AHgassim = A22gassim[7].values[:steppii]
        E_2Hgassim = A22gassim[8].values[:steppii]
        E_3AHgassim = A22gassim[9].values[:steppii]

        A222gassim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None
        )
        E_3CHgassim = A222gassim[1].values[:steppii]
        E_3Hgassim = A222gassim[2].values[:steppii]
        E_4AHgassim = A222gassim[3].values[:steppii]
        K_3Hgassim = A222gassim[4].values[:steppii]

        WGAS1[:, 0] = B_1BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 1] = B_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 2] = B_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 3] = B_3Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 4] = B_4BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 5] = B_4DHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 6] = B_4Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 7] = D_1CHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 8] = D_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 9] = D_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 10] = D_3AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 11] = D_3BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 12] = D_4AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 13] = D_4Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 14] = E_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 15] = E_2AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 16] = E_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 17] = E_3AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 18] = E_3CHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 19] = E_3Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 20] = E_4AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 21] = K_3Hgassim.ravel()[steppi_indices - 1]

        ## WATER INJECTOR RATE
        A2wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None)
        C_1Hwaterinjsim = A2wir[9].values[:steppii]

        A22wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None)
        C_2Hwaterinjsim = A22wir[1].values[:steppii]
        C_3Hwaterinjsim = A22wir[2].values[:steppii]
        C_4AHwaterinjsim = A22wir[3].values[:steppii]
        C_4Hwaterinjsim = A22wir[4].values[:steppii]
        F_1Hwaterinjsim = A22wir[5].values[:steppii]
        F_2Hwaterinjsim = A22wir[6].values[:steppii]
        F_3Hwaterinjsim = A22wir[7].values[:steppii]
        F_4Hwaterinjsim = A22wir[8].values[:steppii]

        WWINJ1[:, 0] = C_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 1] = C_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 2] = C_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 3] = C_4AHwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 4] = C_4Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 5] = F_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 6] = F_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 7] = F_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 8] = F_4Hwaterinjsim.ravel()[steppi_indices - 1]

        ## GAS INJECTOR RATE
        A2gir = pd.read_csv("FULLNORNE2.RSM", skiprows=777, sep="\s+", header=None)
        C_1Hgasinjsim = A2gir[6].values[:steppii]
        C_3Hgasinjsim = A2gir[7].values[:steppii]
        C_4AHgasinjsim = A2gir[8].values[:steppii]
        C_4Hgasinjsim = A2gir[9].values[:steppii]

        WGASJ1[:, 0] = C_1Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 1] = C_3Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 2] = C_4AHgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 3] = C_4Hgasinjsim.ravel()[steppi_indices - 1]

        # Return to the root folder
        out = np.hstack((WOIL1, WWATER1, WGAS1))
        ouut[i, :, :] = out

        ##INPUTS

        permuse = perm[i, 0, :, :, :]
        a1 = np.mean(permuse[14, 30, :]) * np.ones((steppi, 1))
        a2 = np.mean(permuse[9, 31, :]) * np.ones((steppi, 1))
        a3 = np.mean(permuse[13, 33, :]) * np.ones((steppi, 1))
        a5 = np.mean(permuse[8, 36, :]) * np.ones((steppi, 1))
        a6 = np.mean(permuse[8, 45, :]) * np.ones((steppi, 1))
        a7 = np.mean(permuse[9, 28, :]) * np.ones((steppi, 1))
        a8 = np.mean(permuse[9, 23, :]) * np.ones((steppi, 1))
        a9 = np.mean(permuse[21, 21, :]) * np.ones((steppi, 1))
        a10 = np.mean(permuse[13, 27, :]) * np.ones((steppi, 1))
        a11 = np.mean(permuse[18, 37, :]) * np.ones((steppi, 1))
        a12 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a13 = np.mean(permuse[15, 65, :]) * np.ones((steppi, 1))
        a14 = np.mean(permuse[24, 36, :]) * np.ones((steppi, 1))
        a15 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a16 = np.mean(permuse[11, 71, :]) * np.ones((steppi, 1))
        a17 = np.mean(permuse[17, 67, :]) * np.ones((steppi, 1))
        a18 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a19 = np.mean(permuse[37, 97, :]) * np.ones((steppi, 1))
        a20 = np.mean(permuse[6, 63, :]) * np.ones((steppi, 1))
        a21 = np.mean(permuse[14, 75, :]) * np.ones((steppi, 1))
        a22 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a23 = np.mean(permuse[10, 27, :]) * np.ones((steppi, 1))

        permxx = np.hstack(
            (
                a1,
                a2,
                a3,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
                a15,
                a16,
                a17,
                a18,
                a19,
                a20,
                a21,
                a22,
                a23,
            )
        )

        presure_use = pressure[i, :, :, :, :]
        gas_use = Sgas[i, :, :, :, :]
        water_use = Swater[i, :, :, :, :]
        Time_use = Time[i, :, :, :, :]

        a1 = np.zeros((steppi, 1))
        a2 = np.zeros((steppi, 22))
        a3 = np.zeros((steppi, 22))
        a4 = np.zeros((steppi, 1))

        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse)

            uniegas = gas_use[k, :, :, :]
            permuse = uniegas
            a2[k, 0] = np.mean(permuse[14, 30, :])
            a2[k, 1] = np.mean(permuse[9, 31, :])
            a2[k, 2] = np.mean(permuse[13, 33, :])
            a2[k, 3] = np.mean(permuse[8, 36, :])
            a2[k, 4] = np.mean(permuse[8, 45, :])
            a2[k, 5] = np.mean(permuse[9, 28, :])
            a2[k, 6] = np.mean(permuse[9, 23, :])
            a2[k, 7] = np.mean(permuse[21, 21, :])
            a2[k, 8] = np.mean(permuse[13, 27, :])
            a2[k, 9] = np.mean(permuse[18, 37, :])
            a2[k, 10] = np.mean(permuse[18, 53, :])
            a2[k, 11] = np.mean(permuse[15, 65, :])
            a2[k, 12] = np.mean(permuse[24, 36, :])
            a2[k, 13] = np.mean(permuse[18, 54, :])
            a2[k, 14] = np.mean(permuse[11, 71, :])
            a2[k, 15] = np.mean(permuse[17, 67, :])
            a2[k, 16] = np.mean(permuse[12, 66, :])
            a2[k, 17] = np.mean(permuse[37, 97, :])
            a2[k, 18] = np.mean(permuse[6, 63, :])
            a2[k, 19] = np.mean(permuse[14, 75, :])
            a2[k, 20] = np.mean(permuse[12, 66, :])
            a2[k, 21] = np.mean(permuse[10, 27, :])

            uniewater = water_use[k, :, :, :]
            permuse = uniewater
            a3[k, 0] = np.mean(permuse[14, 30, :])
            a3[k, 1] = np.mean(permuse[9, 31, :])
            a3[k, 2] = np.mean(permuse[13, 33, :])
            a3[k, 3] = np.mean(permuse[8, 36, :])
            a3[k, 4] = np.mean(permuse[8, 45, :])
            a3[k, 5] = np.mean(permuse[9, 28, :])
            a3[k, 6] = np.mean(permuse[9, 23, :])
            a3[k, 7] = np.mean(permuse[21, 21, :])
            a3[k, 8] = np.mean(permuse[13, 27, :])
            a3[k, 9] = np.mean(permuse[18, 37, :])
            a3[k, 10] = np.mean(permuse[18, 53, :])
            a3[k, 11] = np.mean(permuse[15, 65, :])
            a3[k, 12] = np.mean(permuse[24, 36, :])
            a3[k, 13] = np.mean(permuse[18, 54, :])
            a3[k, 14] = np.mean(permuse[11, 71, :])
            a3[k, 15] = np.mean(permuse[17, 67, :])
            a3[k, 16] = np.mean(permuse[12, 66, :])
            a3[k, 17] = np.mean(permuse[37, 97, :])
            a3[k, 18] = np.mean(permuse[6, 63, :])
            a3[k, 19] = np.mean(permuse[14, 75, :])
            a3[k, 20] = np.mean(permuse[12, 66, :])
            a3[k, 21] = np.mean(permuse[10, 27, :])

            unietime = Time_use[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]

        inn1 = np.hstack((permxx, a1, 1 - (a2 + a3), a2, a3, a4))

        innn[i, :, :] = inn1

        os.chdir(oldfolder)
    return innn, ouut


def Get_data_FFNN1(
    folder, oldfolder, N, pressure, Sgas, Swater, perm, Time, steppi, steppi_indices
):
    ouut = np.zeros((N, steppi, 66))
    innn = np.zeros((N, steppi, 90))
    steppii = 246
    for i in range(N):
        WOIL1 = np.zeros((steppi, 22))
        WWATER1 = np.zeros((steppi, 22))
        WGAS1 = np.zeros((steppi, 22))
        WWINJ1 = np.zeros((steppi, 9))
        WGASJ1 = np.zeros((steppi, 4))

        # f = 'Realisation'
        # folder = f + str(i)
        # folder = to_absolute_path('../RUNS/Realisation' + str(i))
        os.chdir(folder)
        ## IMPORT FOR OIL
        A2oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None)

        B_1BHoilsim = A2oilsim[5].values[:steppii]
        B_1Hoilsim = A2oilsim[6].values[:steppii]
        B_2Hoilsim = A2oilsim[7].values[:steppii]
        B_3Hoilsim = A2oilsim[8].values[:steppii]
        B_4BHoilsim = A2oilsim[9].values[:steppii]

        A22oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1801, sep="\s+", header=None)
        B_4DHoilsim = A22oilsim[1].values[:steppii]
        B_4Hoilsim = A22oilsim[2].values[:steppii]
        D_1CHoilsim = A22oilsim[3].values[:steppii]
        D_1Hoilsim = A22oilsim[4].values[:steppii]
        D_2Hoilsim = A22oilsim[5].values[:steppii]
        D_3AHoilsim = A22oilsim[6].values[:steppii]
        D_3BHoilsim = A22oilsim[7].values[:steppii]
        D_4AHoilsim = A22oilsim[8].values[:steppii]
        D_4Hoilsim = A22oilsim[9].values[:steppii]

        A222oilsim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None
        )

        E_1Hoilsim = A222oilsim[1].values[:steppii]
        E_2AHoilsim = A222oilsim[2].values[:steppii]
        E_2Hoilsim = A222oilsim[3].values[:steppii]
        E_3AHoilsim = A222oilsim[4].values[:steppii]
        E_3CHoilsim = A222oilsim[5].values[:steppii]
        E_3Hoilsim = A222oilsim[6].values[:steppii]
        E_4AHoilsim = A222oilsim[7].values[:steppii]
        K_3Hoilsim = A222oilsim[8].values[:steppii]

        WOIL1[:, 0] = B_1BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 1] = B_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 2] = B_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 3] = B_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 4] = B_4BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 5] = B_4DHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 6] = B_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 7] = D_1CHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 8] = D_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 9] = D_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 10] = D_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 11] = D_3BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 12] = D_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 13] = D_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 14] = E_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 15] = E_2AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 16] = E_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 17] = E_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 18] = E_3CHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 19] = E_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 20] = E_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 21] = K_3Hoilsim.ravel()[steppi_indices - 1]

        ##IMPORT FOR WATER
        A2watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None
        )
        B_1BHwatersim = A2watersim[9].values[:steppii]

        A22watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2569, sep="\s+", header=None
        )
        B_1Hwatersim = A22watersim[1].values[:steppii]
        B_2Hwatersim = A22watersim[2].values[:steppii]
        B_3Hwatersim = A22watersim[3].values[:steppii]
        B_4BHwatersim = A22watersim[4].values[:steppii]
        B_4DHwatersim = A22watersim[5].values[:steppii]
        B_4Hwatersim = A22watersim[6].values[:steppii]
        D_1CHwatersim = A22watersim[7].values[:steppii]
        D_1Hwatersim = A22watersim[8].values[:steppii]
        D_2Hwatersim = A22watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2825, sep="\s+", header=None
        )
        D_3AHwatersim = A222watersim[1].values[:steppii]
        D_3BHwatersim = A222watersim[2].values[:steppii]
        D_4AHwatersim = A222watersim[3].values[:steppii]
        D_4Hwatersim = A222watersim[4].values[:steppii]
        E_1Hwatersim = A222watersim[5].values[:steppii]
        E_2AHwatersim = A222watersim[6].values[:steppii]
        E_2Hwatersim = A222watersim[7].values[:steppii]
        E_3AHwatersim = A222watersim[8].values[:steppii]
        E_3CHwatersim = A222watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=3081, sep="\s+", header=None
        )
        E_3Hwatersim = A222watersim[1].values[:steppii]
        E_4AHwatersim = A222watersim[2].values[:steppii]
        K_3Hwatersim = A222watersim[3].values[:steppii]

        WWATER1[:, 0] = B_1BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 1] = B_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 2] = B_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 3] = B_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 4] = B_4BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 5] = B_4DHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 6] = B_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 7] = D_1CHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 8] = D_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 9] = D_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 10] = D_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 11] = D_3BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 12] = D_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 13] = D_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 14] = E_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 15] = E_2AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 16] = E_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 17] = E_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 18] = E_3CHwatersim.ravel()[steppi_indices - 1]

        WWATER1[:, 19] = E_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 20] = E_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 21] = K_3Hwatersim.ravel()[steppi_indices - 1]

        ## GAS PRODUCTION RATE
        A2gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1033, sep="\s+", header=None)
        B_1BHgassim = A2gassim[1].values[:steppii]
        B_1Hgassim = A2gassim[2].values[:steppii]
        B_2Hgassim = A2gassim[3].values[:steppii]
        B_3Hgassim = A2gassim[4].values[:steppii]
        B_4BHgassim = A2gassim[5].values[:steppii]
        B_4DHgassim = A2gassim[6].values[:steppii]
        B_4Hgassim = A2gassim[7].values[:steppii]
        D_1CHgassim = A2gassim[8].values[:steppii]
        D_1Hgassim = A2gassim[9].values[:steppii]

        A22gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1289, sep="\s+", header=None)
        D_2Hgassim = A22gassim[1].values[:steppii]
        D_3AHgassim = A22gassim[2].values[:steppii]
        D_3BHgassim = A22gassim[3].values[:steppii]
        D_4AHgassim = A22gassim[4].values[:steppii]
        D_4Hgassim = A22gassim[5].values[:steppii]
        E_1Hgassim = A22gassim[6].values[:steppii]
        E_2AHgassim = A22gassim[7].values[:steppii]
        E_2Hgassim = A22gassim[8].values[:steppii]
        E_3AHgassim = A22gassim[9].values[:steppii]

        A222gassim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None
        )
        E_3CHgassim = A222gassim[1].values[:steppii]
        E_3Hgassim = A222gassim[2].values[:steppii]
        E_4AHgassim = A222gassim[3].values[:steppii]
        K_3Hgassim = A222gassim[4].values[:steppii]

        WGAS1[:, 0] = B_1BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 1] = B_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 2] = B_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 3] = B_3Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 4] = B_4BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 5] = B_4DHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 6] = B_4Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 7] = D_1CHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 8] = D_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 9] = D_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 10] = D_3AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 11] = D_3BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 12] = D_4AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 13] = D_4Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 14] = E_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 15] = E_2AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 16] = E_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 17] = E_3AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 18] = E_3CHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 19] = E_3Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 20] = E_4AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 21] = K_3Hgassim.ravel()[steppi_indices - 1]

        ## WATER INJECTOR RATE
        A2wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None)
        C_1Hwaterinjsim = A2wir[9].values[:steppii]

        A22wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None)
        C_2Hwaterinjsim = A22wir[1].values[:steppii]
        C_3Hwaterinjsim = A22wir[2].values[:steppii]
        C_4AHwaterinjsim = A22wir[3].values[:steppii]
        C_4Hwaterinjsim = A22wir[4].values[:steppii]
        F_1Hwaterinjsim = A22wir[5].values[:steppii]
        F_2Hwaterinjsim = A22wir[6].values[:steppii]
        F_3Hwaterinjsim = A22wir[7].values[:steppii]
        F_4Hwaterinjsim = A22wir[8].values[:steppii]

        WWINJ1[:, 0] = C_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 1] = C_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 2] = C_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 3] = C_4AHwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 4] = C_4Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 5] = F_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 6] = F_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 7] = F_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 8] = F_4Hwaterinjsim.ravel()[steppi_indices - 1]

        ## GAS INJECTOR RATE
        A2gir = pd.read_csv("FULLNORNE2.RSM", skiprows=777, sep="\s+", header=None)
        C_1Hgasinjsim = A2gir[6].values[:steppii]
        C_3Hgasinjsim = A2gir[7].values[:steppii]
        C_4AHgasinjsim = A2gir[8].values[:steppii]
        C_4Hgasinjsim = A2gir[9].values[:steppii]

        WGASJ1[:, 0] = C_1Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 1] = C_3Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 2] = C_4AHgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 3] = C_4Hgasinjsim.ravel()[steppi_indices - 1]

        # Return to the root folder
        out = np.hstack((WOIL1, WWATER1, WGAS1))
        ouut[i, :, :] = out

        ##INPUTS

        permuse = perm[i, 0, :, :, :]
        a1 = np.mean(permuse[14, 30, :]) * np.ones((steppi, 1))
        a2 = np.mean(permuse[9, 31, :]) * np.ones((steppi, 1))
        a3 = np.mean(permuse[13, 33, :]) * np.ones((steppi, 1))
        a5 = np.mean(permuse[8, 36, :]) * np.ones((steppi, 1))
        a6 = np.mean(permuse[8, 45, :]) * np.ones((steppi, 1))
        a7 = np.mean(permuse[9, 28, :]) * np.ones((steppi, 1))
        a8 = np.mean(permuse[9, 23, :]) * np.ones((steppi, 1))
        a9 = np.mean(permuse[21, 21, :]) * np.ones((steppi, 1))
        a10 = np.mean(permuse[13, 27, :]) * np.ones((steppi, 1))
        a11 = np.mean(permuse[18, 37, :]) * np.ones((steppi, 1))
        a12 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a13 = np.mean(permuse[15, 65, :]) * np.ones((steppi, 1))
        a14 = np.mean(permuse[24, 36, :]) * np.ones((steppi, 1))
        a15 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a16 = np.mean(permuse[11, 71, :]) * np.ones((steppi, 1))
        a17 = np.mean(permuse[17, 67, :]) * np.ones((steppi, 1))
        a18 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a19 = np.mean(permuse[37, 97, :]) * np.ones((steppi, 1))
        a20 = np.mean(permuse[6, 63, :]) * np.ones((steppi, 1))
        a21 = np.mean(permuse[14, 75, :]) * np.ones((steppi, 1))
        a22 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a23 = np.mean(permuse[10, 27, :]) * np.ones((steppi, 1))

        permxx = np.hstack(
            (
                a1,
                a2,
                a3,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
                a15,
                a16,
                a17,
                a18,
                a19,
                a20,
                a21,
                a22,
                a23,
            )
        )

        presure_use = pressure[0, :, :, :, :]
        gas_use = Sgas[0, :, :, :, :]
        water_use = Swater[0, :, :, :, :]
        Time_use = Time[0, :, :, :, :]

        a1 = np.zeros((steppi, 1))
        a2 = np.zeros((steppi, 22))
        a3 = np.zeros((steppi, 22))
        a4 = np.zeros((steppi, 1))

        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse)

            uniegas = gas_use[k, :, :, :]
            permuse = uniegas
            a2[k, 0] = np.mean(permuse[14, 30, :])
            a2[k, 1] = np.mean(permuse[9, 31, :])
            a2[k, 2] = np.mean(permuse[13, 33, :])
            a2[k, 3] = np.mean(permuse[8, 36, :])
            a2[k, 4] = np.mean(permuse[8, 45, :])
            a2[k, 5] = np.mean(permuse[9, 28, :])
            a2[k, 6] = np.mean(permuse[9, 23, :])
            a2[k, 7] = np.mean(permuse[21, 21, :])
            a2[k, 8] = np.mean(permuse[13, 27, :])
            a2[k, 9] = np.mean(permuse[18, 37, :])
            a2[k, 10] = np.mean(permuse[18, 53, :])
            a2[k, 11] = np.mean(permuse[15, 65, :])
            a2[k, 12] = np.mean(permuse[24, 36, :])
            a2[k, 13] = np.mean(permuse[18, 54, :])
            a2[k, 14] = np.mean(permuse[11, 71, :])
            a2[k, 15] = np.mean(permuse[17, 67, :])
            a2[k, 16] = np.mean(permuse[12, 66, :])
            a2[k, 17] = np.mean(permuse[37, 97, :])
            a2[k, 18] = np.mean(permuse[6, 63, :])
            a2[k, 19] = np.mean(permuse[14, 75, :])
            a2[k, 20] = np.mean(permuse[12, 66, :])
            a2[k, 21] = np.mean(permuse[10, 27, :])

            uniewater = water_use[k, :, :, :]
            permuse = uniewater
            a3[k, 0] = np.mean(permuse[14, 30, :])
            a3[k, 1] = np.mean(permuse[9, 31, :])
            a3[k, 2] = np.mean(permuse[13, 33, :])
            a3[k, 3] = np.mean(permuse[8, 36, :])
            a3[k, 4] = np.mean(permuse[8, 45, :])
            a3[k, 5] = np.mean(permuse[9, 28, :])
            a3[k, 6] = np.mean(permuse[9, 23, :])
            a3[k, 7] = np.mean(permuse[21, 21, :])
            a3[k, 8] = np.mean(permuse[13, 27, :])
            a3[k, 9] = np.mean(permuse[18, 37, :])
            a3[k, 10] = np.mean(permuse[18, 53, :])
            a3[k, 11] = np.mean(permuse[15, 65, :])
            a3[k, 12] = np.mean(permuse[24, 36, :])
            a3[k, 13] = np.mean(permuse[18, 54, :])
            a3[k, 14] = np.mean(permuse[11, 71, :])
            a3[k, 15] = np.mean(permuse[17, 67, :])
            a3[k, 16] = np.mean(permuse[12, 66, :])
            a3[k, 17] = np.mean(permuse[37, 97, :])
            a3[k, 18] = np.mean(permuse[6, 63, :])
            a3[k, 19] = np.mean(permuse[14, 75, :])
            a3[k, 20] = np.mean(permuse[12, 66, :])
            a3[k, 21] = np.mean(permuse[10, 27, :])

            unietime = Time_use[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]

        inn1 = np.hstack((permxx, a1, 1 - (a2 + a3), a2, a3, a4))

        innn[0, :, :] = inn1

        os.chdir(oldfolder)
    return innn, ouut


def Remove_folder(N_ens, straa):
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)


def historydata(timestep, steppi, steppi_indices):
    file_path = "../NORNE/Flow.xlsx"
    seee = pd.read_excel(file_path, skiprows=1).to_numpy()[:10, 1:]

    WOIL1 = seee[:, :22]
    WWATER1 = seee[:, 22:44]
    WGAS1 = seee[:, 44:66]

    DATA = {"OIL": seee[:, :22], "WATER": seee[:, 22:44], "GAS": seee[:, 44:66]}

    oil = np.reshape(WOIL1, (-1, 1), "F")
    water = np.reshape(WWATER1, (-1, 1), "F")
    gas = np.reshape(WGAS1, (-1, 1), "F")

    # Get data for history matching
    DATA2 = np.vstack([oil, water, gas])
    new = np.hstack([WOIL1, WWATER1, WGAS1])
    return DATA, DATA2, new


def historydatano(timestep, steppi, steppi_indices):
    WOIL1 = np.zeros((steppi, 22))

    WWATER1 = np.zeros((steppi, 22))

    WGAS1 = np.zeros((steppi, 22))

    steppii = 246

    ## IMPORT FOR OIL
    A2oilsim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=1545, sep="\s+", header=None
    )

    B_1BHoilsim = A2oilsim[5].values[:steppii]
    B_1Hoilsim = A2oilsim[6].values[:steppii]
    B_2Hoilsim = A2oilsim[7].values[:steppii]
    B_3Hoilsim = A2oilsim[8].values[:steppii]
    B_4BHoilsim = A2oilsim[9].values[:steppii]

    A22oilsim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=1801, sep="\s+", header=None
    )
    B_4DHoilsim = A22oilsim[1].values[:steppii]
    B_4Hoilsim = A22oilsim[2].values[:steppii]
    D_1CHoilsim = A22oilsim[3].values[:steppii]
    D_1Hoilsim = A22oilsim[4].values[:steppii]
    D_2Hoilsim = A22oilsim[5].values[:steppii]
    D_3AHoilsim = A22oilsim[6].values[:steppii]
    D_3BHoilsim = A22oilsim[7].values[:steppii]
    D_4AHoilsim = A22oilsim[8].values[:steppii]
    D_4Hoilsim = A22oilsim[9].values[:steppii]

    A222oilsim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=2057, sep="\s+", header=None
    )

    E_1Hoilsim = A222oilsim[1].values[:steppii]
    E_2AHoilsim = A222oilsim[2].values[:steppii]
    E_2Hoilsim = A222oilsim[3].values[:steppii]
    E_3AHoilsim = A222oilsim[4].values[:steppii]
    E_3CHoilsim = A222oilsim[5].values[:steppii]
    E_3Hoilsim = A222oilsim[6].values[:steppii]
    E_4AHoilsim = A222oilsim[7].values[:steppii]
    K_3Hoilsim = A222oilsim[8].values[:steppii]

    WOIL1[:, 0] = B_1BHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 1] = B_1Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 2] = B_2Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 3] = B_3Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 4] = B_4BHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 5] = B_4DHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 6] = B_4Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 7] = D_1CHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 8] = D_1Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 9] = D_2Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 10] = D_3AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 11] = D_3BHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 12] = D_4AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 13] = D_4Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 14] = E_1Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 15] = E_2AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 16] = E_2Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 17] = E_3AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 18] = E_3CHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 19] = E_3Hoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 20] = E_4AHoilsim.ravel()[steppi_indices - 1]
    WOIL1[:, 21] = K_3Hoilsim.ravel()[steppi_indices - 1]

    ##IMPORT FOR WATER
    A2watersim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=2313, sep="\s+", header=None
    )
    B_1BHwatersim = A2watersim[9].values[:steppii]

    A22watersim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=2569, sep="\s+", header=None
    )
    B_1Hwatersim = A22watersim[1].values[:steppii]
    B_2Hwatersim = A22watersim[2].values[:steppii]
    B_3Hwatersim = A22watersim[3].values[:steppii]
    B_4BHwatersim = A22watersim[4].values[:steppii]
    B_4DHwatersim = A22watersim[5].values[:steppii]
    B_4Hwatersim = A22watersim[6].values[:steppii]
    D_1CHwatersim = A22watersim[7].values[:steppii]
    D_1Hwatersim = A22watersim[8].values[:steppii]
    D_2Hwatersim = A22watersim[9].values[:steppii]

    A222watersim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=2825, sep="\s+", header=None
    )
    D_3AHwatersim = A222watersim[1].values[:steppii]
    D_3BHwatersim = A222watersim[2].values[:steppii]
    D_4AHwatersim = A222watersim[3].values[:steppii]
    D_4Hwatersim = A222watersim[4].values[:steppii]
    E_1Hwatersim = A222watersim[5].values[:steppii]
    E_2AHwatersim = A222watersim[6].values[:steppii]
    E_2Hwatersim = A222watersim[7].values[:steppii]
    E_3AHwatersim = A222watersim[8].values[:steppii]
    E_3CHwatersim = A222watersim[9].values[:steppii]

    A222watersim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=3081, sep="\s+", header=None
    )
    E_3Hwatersim = A222watersim[1].values[:steppii]
    E_4AHwatersim = A222watersim[2].values[:steppii]
    K_3Hwatersim = A222watersim[3].values[:steppii]

    WWATER1[:, 0] = B_1BHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 1] = B_1Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 2] = B_2Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 3] = B_3Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 4] = B_4BHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 5] = B_4DHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 6] = B_4Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 7] = D_1CHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 8] = D_1Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 9] = D_2Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 10] = D_3AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 11] = D_3BHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 12] = D_4AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 13] = D_4Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 14] = E_1Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 15] = E_2AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 16] = E_2Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 17] = E_3AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 18] = E_3CHwatersim.ravel()[steppi_indices - 1]

    # if len(E_3Hwatersim<steppi):
    #     E_3Hwatersim = np.append(E_3Hwatersim.ravel(), 0)
    # if len(E_4AHwatersim<steppi):
    #     E_4AHwatersim = np.append(E_4AHwatersim.ravel(), 0)
    # if len(K_3Hwatersim<steppi):
    #     K_3Hwatersim = np.append(K_3Hwatersim.ravel(), 0)
    WWATER1[:, 19] = E_3Hwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 20] = E_4AHwatersim.ravel()[steppi_indices - 1]
    WWATER1[:, 21] = K_3Hwatersim.ravel()[steppi_indices - 1]

    ## GAS PRODUCTION RATE
    A2gassim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=1033, sep="\s+", header=None
    )
    B_1BHgassim = A2gassim[1].values[:steppii]
    B_1Hgassim = A2gassim[2].values[:steppii]
    B_2Hgassim = A2gassim[3].values[:steppii]
    B_3Hgassim = A2gassim[4].values[:steppii]
    B_4BHgassim = A2gassim[5].values[:steppii]
    B_4DHgassim = A2gassim[6].values[:steppii]
    B_4Hgassim = A2gassim[7].values[:steppii]
    D_1CHgassim = A2gassim[8].values[:steppii]
    D_1Hgassim = A2gassim[9].values[:steppii]

    A22gassim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=1289, sep="\s+", header=None
    )
    D_2Hgassim = A22gassim[1].values[:steppii]
    D_3AHgassim = A22gassim[2].values[:steppii]
    D_3BHgassim = A22gassim[3].values[:steppii]
    D_4AHgassim = A22gassim[4].values[:steppii]
    D_4Hgassim = A22gassim[5].values[:steppii]
    E_1Hgassim = A22gassim[6].values[:steppii]
    E_2AHgassim = A22gassim[7].values[:steppii]
    E_2Hgassim = A22gassim[8].values[:steppii]
    E_3AHgassim = A22gassim[9].values[:steppii]

    A222gassim = pd.read_csv(
        "../NORNE/FULLNORNE.RSM", skiprows=1545, sep="\s+", header=None
    )
    E_3CHgassim = A222gassim[1].values[:steppii]
    E_3Hgassim = A222gassim[2].values[:steppii]
    E_4AHgassim = A222gassim[3].values[:steppii]
    K_3Hgassim = A222gassim[4].values[:steppii]

    WGAS1[:, 0] = B_1BHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 1] = B_1Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 2] = B_2Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 3] = B_3Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 4] = B_4BHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 5] = B_4DHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 6] = B_4Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 7] = D_1CHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 8] = D_1Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 9] = D_2Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 10] = D_3AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 11] = D_3BHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 12] = D_4AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 13] = D_4Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 14] = E_1Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 15] = E_2AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 16] = E_2Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 17] = E_3AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 18] = E_3CHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 19] = E_3Hgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 20] = E_4AHgassim.ravel()[steppi_indices - 1]
    WGAS1[:, 21] = K_3Hgassim.ravel()[steppi_indices - 1]

    DATA = {"OIL": WOIL1, "WATER": WWATER1, "GAS": WGAS1}

    oil = np.reshape(WOIL1, (-1, 1), "F")
    water = np.reshape(WWATER1, (-1, 1), "F")
    gas = np.reshape(WGAS1, (-1, 1), "F")

    # Get data for history matching
    DATA2 = np.vstack([oil, water, gas])
    new = np.hstack([WOIL1, WWATER1, WGAS1])
    return DATA, DATA2, new


def historydata2(timestep, steppi, steppi_indices):
    WOIL1 = np.zeros((steppi, 22))
    WWATER1 = np.zeros((steppi, 22))
    WGAS1 = np.zeros((steppi, 22))
    WWINJ1 = np.zeros((steppi, 9))
    WGASJ1 = np.zeros((steppi, 4))

    indices = timestep
    # print(' Get the Well Oil Production Rate')

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 47873:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1 = df[[2, 3, 4, 5, 6, 8]].values

    B_2H = A1[:, 0][indices - 1]
    D_1H = A1[:, 1][indices - 1]
    D_2H = A1[:, 2][indices - 1]
    B_4H = A1[:, 3][indices - 1]
    D_4H = A1[:, 4][indices - 1]
    E_3H = A1[:, 5][indices - 1]

    # A2 = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 2, 5, 6, 8, 9), skip_header=48743)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 48743:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2 = df[[1, 4, 5, 7, 9]].values

    B_1H = A2[:, 0][indices - 1]
    B_3H = A2[:, 1][indices - 1]
    E_1H = A2[:, 2][indices - 1]
    E_2H = A2[:, 3][indices - 1]
    E_4AH = A2[:, 4][indices - 1]

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 49613:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3 = df[[2, 4, 7, 8, 9]].values

    D_3AH = A3[:, 0][indices - 1]
    E_3AH = A3[:, 1][indices - 1]
    B_4BH = A3[:, 2][indices - 1]
    D_4AH = A3[:, 3][indices - 1]
    D_1CH = A3[:, 4][indices - 1]

    # A4 = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 5, 6, 7, 8, 9), skip_header=50483)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 50483:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4 = df[[2, 4, 5, 6, 8, 9]].values

    B_4DH = A4[:, 0][indices - 1]
    E_3CH = A4[:, 1][indices - 1]
    E_2AH = A4[:, 2][indices - 1]
    D_3BH = A4[:, 3][indices - 1]
    B_1BH = A4[:, 4][indices - 1]
    K_3H = A4[:, 5][indices - 1]

    WOIL1[:, 0] = B_1BH.ravel()[steppi_indices - 1]
    WOIL1[:, 1] = B_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 2] = B_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 3] = B_3H.ravel()[steppi_indices - 1]
    WOIL1[:, 4] = B_4BH.ravel()[steppi_indices - 1]
    WOIL1[:, 5] = B_4DH.ravel()[steppi_indices - 1]
    WOIL1[:, 6] = B_4H.ravel()[steppi_indices - 1]
    WOIL1[:, 7] = D_1CH.ravel()[steppi_indices - 1]
    WOIL1[:, 8] = D_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 9] = D_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 10] = D_3AH.ravel()[steppi_indices - 1]
    WOIL1[:, 11] = D_3BH.ravel()[steppi_indices - 1]
    WOIL1[:, 12] = D_4AH.ravel()[steppi_indices - 1]
    WOIL1[:, 13] = D_4H.ravel()[steppi_indices - 1]
    WOIL1[:, 14] = E_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 15] = E_2AH.ravel()[steppi_indices - 1]
    WOIL1[:, 16] = E_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 17] = E_3AH.ravel()[steppi_indices - 1]
    WOIL1[:, 18] = E_3CH.ravel()[steppi_indices - 1]
    WOIL1[:, 19] = E_3H.ravel()[steppi_indices - 1]
    WOIL1[:, 20] = E_4AH.ravel()[steppi_indices - 1]
    WOIL1[:, 21] = K_3H.ravel()[steppi_indices - 1]

    # Data for Water production rate
    # print(' Get the Well water Production Rate')
    # A1w = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 3, 4, 5, 6, 8), skip_header=40913)
    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 40913:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1w = df[[2, 3, 4, 5, 6, 8]].values

    B_2Hw = A1w[:, 0][indices - 1]
    D_1Hw = A1w[:, 1][indices - 1]
    D_2Hw = A1w[:, 2][indices - 1]
    B_4Hw = A1w[:, 3][indices - 1]
    D_4Hw = A1w[:, 4][indices - 1]
    E_3Hw = A1w[:, 5][indices - 1]

    # A2w = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 2, 5, 6, 8, 9), skip_header=48743)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 41783:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2w = df[[1, 4, 5, 7, 9]].values

    B_1Hw = A2w[:, 0][indices - 1]
    B_3Hw = A2w[:, 1][indices - 1]
    E_1Hw = A2w[:, 2][indices - 1]
    E_2Hw = A2w[:, 3][indices - 1]
    E_4AHw = A2w[:, 4][indices - 1]

    # A3w = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 4, 7, 8, 9), skip_header=49613)
    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 42653:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3w = df[[2, 4, 7, 8, 9]].values

    D_3AHw = A3w[:, 0][indices - 1]
    E_3AHw = A3w[:, 1][indices - 1]
    B_4BHw = A3w[:, 2][indices - 1]
    D_4AHw = A3w[:, 3][indices - 1]
    D_1CHw = A3w[:, 4][indices - 1]

    # A4w = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 5, 6, 7, 8, 9), skip_header=50483)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 43523:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4w = df[[2, 4, 5, 6, 8, 9]].values

    B_4DHw = A4w[:, 0][indices - 1]
    E_3CHw = A4w[:, 1][indices - 1]
    E_2AHw = A4w[:, 2][indices - 1]
    D_3BHw = A4w[:, 3][indices - 1]
    B_1BHw = A4w[:, 4][indices - 1]
    K_3Hw = A4w[:, 5][indices - 1]

    WWATER1[:, 0] = B_1BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 1] = B_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 2] = B_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 3] = B_3Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 4] = B_4BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 5] = B_4DHw.ravel()[steppi_indices - 1]
    WWATER1[:, 6] = B_4Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 7] = D_1CHw.ravel()[steppi_indices - 1]
    WWATER1[:, 8] = D_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 9] = D_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 10] = D_3AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 11] = D_3BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 12] = D_4AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 13] = D_4Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 14] = E_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 15] = E_2AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 16] = E_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 17] = E_3AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 18] = E_3CHw.ravel()[steppi_indices - 1]

    WWATER1[:, 19] = E_3Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 20] = E_4AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 21] = K_3Hw.ravel()[steppi_indices - 1]

    # Data for Gas production rate
    # print(' Get the Well Gas Production Rate')
    # A1g = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 3, 4, 5, 6, 8), skip_header=54833)
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 54833:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1g = df[[2, 3, 4, 5, 6, 8]].values

    B_2Hg = A1g[:, 0][indices - 1]
    D_1Hg = A1g[:, 1][indices - 1]
    D_2Hg = A1g[:, 2][indices - 1]
    B_4Hg = A1g[:, 3][indices - 1]
    D_4Hg = A1g[:, 4][indices - 1]
    E_3Hg = A1g[:, 5][indices - 1]

    # A2g = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 2, 5, 6, 8, 9), skip_header=55703)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 55703:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2g = df[[1, 4, 5, 7, 9]].values

    B_1Hg = A2g[:, 0][indices - 1]
    B_3Hg = A2g[:, 1][indices - 1]
    E_1Hg = A2g[:, 2][indices - 1]
    E_2Hg = A2g[:, 3][indices - 1]
    E_4AHg = A2g[:, 4][indices - 1]

    # A3g = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 4, 7, 8, 9), skip_header=56573)

    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 56573:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3g = df[[2, 4, 7, 8, 9]].values

    D_3AHg = A3g[:, 0][indices - 1]
    E_3AHg = A3g[:, 1][indices - 1]
    B_4BHg = A3g[:, 2][indices - 1]
    D_4AHg = A3g[:, 3][indices - 1]
    D_1CHg = A3g[:, 4][indices - 1]

    # A4g = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 5, 6, 7, 8, 9), skip_header=57443)
    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 57443:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4g = df[[2, 4, 5, 6, 8, 9]].values
    B_4DHg = A4g[:, 0][indices - 1]
    E_3CHg = A4g[:, 1][indices - 1]
    E_2AHg = A4g[:, 2][indices - 1]
    D_3BHg = A4g[:, 3][indices - 1]
    B_1BHg = A4g[:, 4][indices - 1]
    K_3Hg = A4g[:, 5][indices - 1]

    WGAS1[:, 0] = B_1BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 1] = B_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 2] = B_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 3] = B_3Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 4] = B_4BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 5] = B_4DHg.ravel()[steppi_indices - 1]
    WGAS1[:, 6] = B_4Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 7] = D_1CHg.ravel()[steppi_indices - 1]
    WGAS1[:, 8] = D_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 9] = D_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 10] = D_3AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 11] = D_3BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 12] = D_4AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 13] = D_4Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 14] = E_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 15] = E_2AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 16] = E_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 17] = E_3AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 18] = E_3CHg.ravel()[steppi_indices - 1]
    WGAS1[:, 19] = E_3Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 20] = E_4AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 21] = K_3Hg.ravel()[steppi_indices - 1]

    # Data for Water injection rate
    # print(' Get the Well water injection Rate')
    # A1win = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 2, 3, 4, 5, 7), skip_header=72237)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 72237:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1win = df[[1, 2, 3, 4, 5, 6, 7, 8, 9]].values

    C_1Hwin = A1win[:, 0][indices - 1]
    C_2Hwin = A1win[:, 1][indices - 1]
    C_3Hwin = A1win[:, 2][indices - 1]
    C_4Hwin = A1win[:, 3][indices - 1]
    C_4AHwin = A1win[:, 4][indices - 1]
    F_1Hwin = A1win[:, 5][indices - 1]
    F_2Hwin = A1win[:, 6][indices - 1]
    F_3Hwin = A1win[:, 7][indices - 1]
    F_4Hwin = A1win[:, 8][indices - 1]

    WWINJ1[:, 0] = C_1Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 1] = C_2Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 2] = C_3Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 3] = C_4AHwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 4] = C_4Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 5] = F_1Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 6] = F_2Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 7] = F_3Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 8] = F_4Hwin.ravel()[steppi_indices - 1]

    # Data for Gas injection rate
    # print(' Get the Well Gas injection Rate')
    # A1gin = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 3, 4, 5, 6, 8), skip_header=73977)
    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 73977:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1gin = df[[1, 3, 4, 5]].values
    C_1Hgin = A1gin[:, 0][indices - 1]
    C_3Hgin = A1gin[:, 1][indices - 1]
    C_4Hgin = A1gin[:, 2][indices - 1]
    C_4AHgin = A1gin[:, 3][indices - 1]

    WGASJ1[:, 0] = C_1Hgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 1] = C_3Hgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 2] = C_4AHgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 3] = C_4Hgin.ravel()[steppi_indices - 1]

    # Get Data for plotting

    DATA = {
        "OIL": WOIL1,
        "WATER": WWATER1,
        "GAS": WGAS1,
        "WATER_INJ": WWINJ1,
        "WGAS_inj": WGASJ1,
    }

    oil = np.reshape(WOIL1, (-1, 1), "F")
    water = np.reshape(WWATER1, (-1, 1), "F")
    gas = np.reshape(WGAS1, (-1, 1), "F")

    # Get data for history matching
    DATA2 = np.vstack([oil, water, gas])
    new = np.hstack([WOIL1, WWATER1, WGAS1])
    return DATA, DATA2, new


def linear_interp(x, xp, fp):
    contiguous_xp = xp.contiguous()
    left_indices = torch.clamp(
        torch.searchsorted(contiguous_xp, x) - 1, 0, len(contiguous_xp) - 2
    )

    # Calculate denominators and handle zero case
    denominators = contiguous_xp[left_indices + 1] - contiguous_xp[left_indices]
    close_to_zero = denominators.abs() < 1e-10
    denominators[close_to_zero] = 1.0  # or any non-zero value to avoid NaN

    interpolated_value = (
        ((fp[left_indices + 1] - fp[left_indices]) / denominators)
        * (x - contiguous_xp[left_indices])
    ) + fp[left_indices]
    return interpolated_value


def replace_nan_with_zero(tensor):
    nan_mask = torch.isnan(tensor)
    return tensor * (~nan_mask).float() + nan_mask.float() * 0.0


def intial_ensemble(Nx, Ny, Nz, N, permx):
    """
    intial_ensemble Function
    ------------------------
    Initializes an ensemble of simulations using multiple-point statistics (MPS) with the snesim algorithm.

    Parameters:
    - Nx (int): Number of grid points in the x-direction.
    - Ny (int): Number of grid points in the y-direction.
    - Nz (int): Number of grid points in the z-direction.
    - N (int): Number of realizations or simulations to generate.
    - permx (array-like): The permeability field Training Image (TI).

    Returns:
    - ensemble (array): An ensemble of simulated fields, reshaped and stacked horizontally.

    Notes:
    - Make sure the 'mps.mpslib' library and its relevant methods are properly imported and set up.
    - The function cleans up temporary files and directories created during the simulation process.
    """
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
import re


def sort_key(s):
    """Extract the number from the filename for sorting."""
    return int(re.search(r"\d+", s).group())


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


def Plot_PhysicsNeMo(
    ax, nx, ny, nz, Truee, N_injw, N_pr, N_injg, varii, injectors, producers, gass
):
    """
    Plot_PhysicsNeMo Function
    ---------------------
    Plots the physicsnemo (or a relevant field) on a specified axis, marking positions of injectors and producers.

    Parameters:
    - ax (matplotlib axis object): Axis on which to plot the data.
    - nx (int): Number of grid points in the x-direction.
    - ny (int): Number of grid points in the y-direction.
    - nz (int): Number of grid points in the z-direction.
    - Truee (array-like): The true or reference field data to be plotted.
    - N_injw (int): Number of water injectors.
    - N_pr (int): Number of producers.
    - N_injg (int): Number of gas injectors.
    - varii (array-like): Variable or secondary field data, possibly representing uncertainty or variability.
    - injectors (list/array): Coordinates or indices of injector points.
    - producers (list/array): Coordinates or indices of producer points.
    - gass (list/array): Coordinates or indices of gas points or related data.

    Notes:
    - Ensure that the function is used in conjunction with the appropriate plotting libraries, e.g., matplotlib.
    """

    # matplotlib.use('Agg')
    Pressz = np.reshape(Truee, (nx, ny, nz), "F")

    avg_2d = np.mean(Pressz, axis=2)

    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs

    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii

    masked_Pressz = Pressz
    colors = plt.cm.jet(masked_Pressz)
    colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
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
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 5], "g", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="g",
            weight="bold",
            fontsize=5,
        )

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

    blue_line = mlines.Line2D([], [], color="blue", linewidth=2, label="water injector")
    green_line = mlines.Line2D(
        [], [], color="green", linewidth=2, label="oil/water/gas producer"
    )
    red_line = mlines.Line2D([], [], color="red", linewidth=2, label="gas injectors")

    # Add the legend to the plot
    ax.legend(handles=[blue_line, green_line, red_line], loc="lower left", fontsize=9)

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

    elif varii == "oil PhysicsNeMo":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation -PhysicsNeMo", fontsize=12, weight="bold")

    elif varii == "oil Numerical":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation - Numerical(Flow)", fontsize=12, weight="bold")

    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "oil saturation - (Numerical(Flow) -PhysicsNeMo))",
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

    if varii == "P10":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("P10 Reservoir Model", fontsize=12, weight="bold")

    if varii == "P50":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("P50 Reservoir Model", fontsize=12, weight="bold")

    if varii == "P90":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("P90 Reservoir Model", fontsize=12, weight="bold")

    if varii == "True model":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("True Reservoir Model", fontsize=12, weight="bold")

    if varii == "Prior":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("initial Reservoir Model", fontsize=12, weight="bold")

    if varii == "cumm-mean":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("Cummulative mean Reservoir Model", fontsize=12, weight="bold")

    if varii == "cumm-best":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title("Cummulative best Reservoir Model", fontsize=12, weight="bold")

    cbar.mappable.set_clim(minii, maxii)


def honour2(sgsim2, DupdateK, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P):
    output = DupdateK
    outputporo = sgsim2

    High_K = High_K.item()
    Low_K = Low_K.item()
    # High_P = High_P.item()
    # Low_P = Low_P.item()

    output[output >= High_K] = High_K  # highest value in true permeability
    output[output <= Low_K] = Low_K

    outputporo[outputporo >= High_P] = High_P
    outputporo[outputporo <= Low_P] = Low_P
    # if Yet==0:
    #     return cp.asnumpy(output),cp.asnumpy(outputporo)
    # else:
    output = output * effec
    outputporo = outputporo * effec
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


def Get_Time(nx, ny, nz, steppi, steppi_indices, N):
    Timee = []
    for k in range(N):
        check = np.ones((nx, ny, nz), dtype=np.float16)
        first_column = []

        with open("FULLNORNE.RSM", "r") as f:
            # Skip the first 9 lines
            for _ in range(9):
                next(f)

            # Process the remaining lines
            for line in f:
                stripped_line = line.strip()  # Remove leading and trailing spaces
                words = stripped_line.split()  # Split line into words
                if len(words) > 0 and words[0].replace(".", "", 1).isdigit():
                    # If the first word is a number (integer or float)
                    first_column.append(float(words[0]))
                else:
                    # If the first word is not a number, it might be a header line
                    break

        # Convert list to numpy array
        np_array2 = np.array(first_column)[:-1]
        np_array2 = np_array2[steppi_indices - 1]
        unie = []
        for zz in range(len(np_array2)):
            aa = np_array2[zz] * check
            unie.append(aa)
        Time = np.stack(unie, axis=0)
        Timee.append(Time)
    Timee = np.stack(Timee, axis=0)
    return Timee


def Get_source_sink(N, nx, ny, nz, waterz, gasz, steppi, steppi_indices):
    Qw = []
    Qg = []
    Qo = []

    waterz = waterz[steppi_indices - 1]
    gasz = gasz[steppi_indices - 1]

    for j in range(N):
        QW = np.zeros((steppi, nx, ny, nz), dtype=np.float16)
        QG = np.zeros((steppi, nx, ny, nz), dtype=np.float16)
        QO = np.zeros((steppi, nx, ny, nz), dtype=np.float16)
        for k in range(len(steppi_indices)):
            QW[k, 25, 43, :] = waterz[k, 0]
            QW[k, 22, 14, :] = waterz[k, 1]
            QW[k, 8, 12, :] = waterz[k, 2]
            QW[k, 10, 34, :] = waterz[k, 3]
            QW[k, 28, 50, :] = waterz[k, 4]
            QW[k, 11, 84, :] = waterz[k, 5]
            QW[k, 17, 82, :] = waterz[k, 6]
            QW[k, 5, 56, :] = waterz[k, 7]
            QW[k, 35, 67, :] = waterz[k, 8]

            QG[k, 25, 43, :] = gasz[k, 0]
            QG[k, 8, 12, :] = gasz[k, 1]
            QG[k, 10, 34, :] = gasz[k, 2]
            QG[k, 28, 50, :] = gasz[k, 3]

            QO[k, 14, 30, :] = -1
            QO[k, 9, 31, :] = -1
            QO[k, 13, 33, :] = -1
            QO[k, 8, 36, :] = -1
            QO[k, 8, 45, :] = -1
            QO[k, 9, 28, :] = -1
            QO[k, 9, 23, :] = -1
            QO[k, 21, 21, :] = -1
            QO[k, 13, 27, :] = -1
            QO[k, 18, 37, :] = -1
            QO[k, 18, 53, :] = -1
            QO[k, 15, 65, :] = -1
            QO[k, 24, 36, :] = -1
            QO[k, 18, 53, :] = -1
            QO[k, 11, 71, :] = -1
            QO[k, 17, 67, :] = -1
            QO[k, 12, 66, :] = -1
            QO[k, 37, 97, :] = -1
            QO[k, 6, 63, :] = -1
            QO[k, 14, 75, :] = -1
            QO[k, 12, 66, :] = -1
            QO[k, 10, 27, :] = -1

            # Q = QW + QO + QG

        Qw.append(QW)
        Qg.append(QG)
        Qo.append(QO)

    Qw = np.stack(Qw, axis=0)
    Qg = np.stack(Qg, axis=0)
    Qo = np.stack(Qo, axis=0)
    return Qw, Qg, Qo


def Get_falt(nx, ny, nz, floatz, N):
    Fault = np.ones((nx, ny, nz), dtype=np.float16)
    flt = []
    for k in range(N):
        floatts = floatz[:, k]
        Fault[8:14, 61:81, 0:] = floatts[0]  # E_01
        Fault[7, 56:61, 0:] = floatts[1]  # E_01_F3
        Fault[9:16, 53:69, 0:15] = floatts[2]  # DE_1
        Fault[9:16, 53:69, 15:22] = floatts[3]  # DE_1LTo
        Fault[8:10, 48:53, 0:] = floatts[4]  # DE_B3
        Fault[5:8, 46:93, 0:] = floatts[5]  # DE_0
        Fault[15, 69:100, 0:] = floatts[6]  # DE_2
        Fault[5:46, 7:10, 0:11] = floatts[7]  # BC
        Fault[10:16, 44:52, 0:11] = floatts[8]  # CD
        Fault[10:16, 44:52, 11:22] = floatts[9]  # CD_To
        Fault[6:11, 38:44, 0:] = floatts[10]  # CD_B3
        Fault[5:7, 38, 0:] = floatts[11]  # CD_0
        Fault[15:19, 52:54, 0:] = floatts[12]  # CD_1
        Fault[26, 13:49, 0:] = floatts[13]  # C_01
        Fault[26, 42:48, 0:] = floatts[14]  # C_01_Ti
        Fault[24, 45:52, 10:19] = floatts[15]  # C_08
        Fault[24, 45:52, 0:10] = floatts[16]  # C_08_Ile
        Fault[24, 40:45, 0:19] = floatts[17]  # C_08_S
        Fault[24, 45:52, 19:22] = floatts[18]  # C_08_Ti
        Fault[24, 40:45, 19:22] = floatts[19]  # C_08_S_Ti
        Fault[21:25, 48:68, 0:] = floatts[20]  # C_09
        Fault[5:7, 18:22, 0:] = floatts[21]  # C_02
        Fault[23:24, 25:35, 0:] = floatts[22]  # C_04
        Fault[12:14, 41:44, 0:] = floatts[23]  # C_05
        Fault[16:19, 47:52, 0:] = floatts[24]  # C_06
        Fault[6:18, 20:22, 0:] = floatts[25]  # C_10
        Fault[25, 51:57, 0:] = floatts[26]  # C_12
        Fault[21:24, 35:39, 0:15] = floatts[27]  # C_20
        Fault[21:24, 35:39, 15:22] = floatts[28]  # C_20_LTo
        Fault[11:13, 12:24, 0:17] = floatts[29]  # C_21
        Fault[11:13, 12:24, 17:22] = floatts[30]  # C_21_Ti
        Fault[12:14, 14:18, 0:] = floatts[31]  # C_22
        Fault[19:22, 19, 0:] = floatts[32]  # C_23
        Fault[16:18, 25:27, 0:] = floatts[33]  # C_24
        Fault[20:22, 24:30, 0:] = floatts[34]  # C_25
        Fault[7:9, 27:36, 0:] = floatts[35]  # C_26
        Fault[6:8, 36:44, 0:] = floatts[36]  # C_26N
        Fault[9, 29:38, 0:] = floatts[37]  # C_27
        Fault[14:16, 38:40, 0:] = floatts[38]  # C_28
        Fault[14:16, 43:46, 0:] = floatts[39]  # C_29
        Fault[18, 47:88, 0:] = floatts[40]  # DI
        Fault[18, 44:47, 0:] = floatts[41]  # DI_S
        Fault[16:18, 55:65, 0:] = floatts[43]  # D_05
        Fault[6:10, 78:92, 0:] = floatts[43]  # EF
        Fault[28:38, 53:103, 0:] = floatts[44]  # GH
        Fault[30:33, 71:86, 0:] = floatts[45]  # G_01
        Fault[34:36, 72:75, 0:] = floatts[46]  # G_02
        Fault[33:37, 70, 0:] = floatts[47]  # G_03
        Fault[30:34, 61:65, 0:] = floatts[48]  # G_05
        Fault[30:37, 79:92, 0:] = floatts[49]  # G_07
        Fault[34, 104:107, 0:] = floatts[50]  # G_08
        Fault[30:37, 66:70, 0:] = floatts[51]  # G_09
        Fault[35:40, 64:69, 0:] = floatts[52]  # G_13
        flt.append(Fault)

    flt = np.stack(flt, axis=0)[:, None, :, :, :]
    return np.stack(flt, axis=0)


def ensemble_pytorch(
    param_perm,
    param_poro,
    param_fault,
    nx,
    ny,
    nz,
    Ne,
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
):
    """
    ensemble_pytorch Function
    -------------------------
    Generate an ensemble of subsurface properties, formatted for PyTorch, based on provided parameters.

    Parameters:
    - param_perm (array-like): Parameters for permeability field.
    - param_poro (array-like): Parameters for porosity field.
    - param_fault (array-like): Parameters for fault field.
    - nx, ny, nz (int): Number of grid points in x, y, and z directions, respectively.
    - Ne (int): Number of ensemble members.
    - effective, oldfolder (misc): Other inputs; .
    - target_min, target_max (float): Range of target values.
    - minK, maxK, minT, maxT, ...: Min and max values for various properties such as permeability, temperature, pressure, etc.
    - steppi (int/float): Step size, increment value.
    - device (torch.device): PyTorch device, e.g., 'cpu' or 'cuda'.
    - steppi_indices (array-like): Indices or locations to apply the 'steppi' value.

    Returns:
    - inn (dict): Dictionary containing the initial ensemble for permeability, porosity, fault, initial pressure, and water saturation, all formatted as PyTorch tensors.


    """
    effective = np.reshape(effective, (nx, ny, nz), "F")
    ini_ensemble1 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)  # Permeability
    ini_ensemble2 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)  # Porosity
    ini_ensemble4 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)  # Fault
    ini_ensemble9 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    ini_ensemble10 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)

    ini_ensemble9 = 600 * np.ones((Ne, 1, nz, nx, ny))
    ini_ensemble10 = 0.2 * np.ones((Ne, 1, nz, nx, ny))
    faultz = Get_falt(nx, ny, nz, param_fault, Ne)

    for kk in range(Ne):
        a = np.reshape(param_perm[:, kk], (nx, ny, nz), "F") * effective
        a1 = np.reshape(param_poro[:, kk], (nx, ny, nz), "F") * effective

        for my in range(nz):
            ini_ensemble1[kk, 0, my, :, :] = a[:, :, my]  # Permeability
            ini_ensemble2[kk, 0, my, :, :] = a1[:, :, my]  # Porosity
            ini_ensemble4[kk, 0, my, :, :] = faultz[kk, 0, :, :, my]  # fault

    # Initial_pressure
    ini_ensemble9 = fit_clement(ini_ensemble9, target_min, target_max, minP, maxP)

    # Permeability
    ini_ensemble1 = fit_clement(ini_ensemble1, target_min, target_max, minK, maxK)

    # ini_ensemble = torch.from_numpy(ini_ensemble).to(device, dtype=torch.float32)
    inn = {
        "perm": torch.from_numpy(ini_ensemble1).to(device, torch.float32),
        "fault": torch.from_numpy(ini_ensemble4).to(device, dtype=torch.float32),
        "Phi": torch.from_numpy(ini_ensemble2).to(device, dtype=torch.float32),
        "Pini": torch.from_numpy(ini_ensemble9).to(device, dtype=torch.float32),
        "Swini": torch.from_numpy(ini_ensemble10).to(device, dtype=torch.float32),
    }
    return inn


def Get_RSM_FLOW(oldfolder, N, steppi, steppi_indices):
    WOIL1 = np.zeros((steppi, N))
    WOIL2 = np.zeros((steppi, N))
    WOIL3 = np.zeros((steppi, N))
    WOIL4 = np.zeros((steppi, N))
    WOIL5 = np.zeros((steppi, N))
    WOIL6 = np.zeros((steppi, N))
    WOIL7 = np.zeros((steppi, N))
    WOIL8 = np.zeros((steppi, N))
    WOIL9 = np.zeros((steppi, N))
    WOIL10 = np.zeros((steppi, N))
    WOIL11 = np.zeros((steppi, N))
    WOIL12 = np.zeros((steppi, N))
    WOIL13 = np.zeros((steppi, N))
    WOIL14 = np.zeros((steppi, N))
    WOIL15 = np.zeros((steppi, N))
    WOIL16 = np.zeros((steppi, N))
    WOIL17 = np.zeros((steppi, N))
    WOIL18 = np.zeros((steppi, N))
    WOIL19 = np.zeros((steppi, N))
    WOIL20 = np.zeros((steppi, N))
    WOIL21 = np.zeros((steppi, N))
    WOIL22 = np.zeros((steppi, N))

    WWATER1 = np.zeros((steppi, N))
    WWATER2 = np.zeros((steppi, N))
    WWATER3 = np.zeros((steppi, N))
    WWATER4 = np.zeros((steppi, N))
    WWATER5 = np.zeros((steppi, N))
    WWATER6 = np.zeros((steppi, N))
    WWATER7 = np.zeros((steppi, N))
    WWATER8 = np.zeros((steppi, N))
    WWATER9 = np.zeros((steppi, N))
    WWATER10 = np.zeros((steppi, N))
    WWATER11 = np.zeros((steppi, N))
    WWATER12 = np.zeros((steppi, N))
    WWATER13 = np.zeros((steppi, N))
    WWATER14 = np.zeros((steppi, N))
    WWATER15 = np.zeros((steppi, N))
    WWATER16 = np.zeros((steppi, N))
    WWATER17 = np.zeros((steppi, N))
    WWATER18 = np.zeros((steppi, N))
    WWATER19 = np.zeros((steppi, N))
    WWATER20 = np.zeros((steppi, N))
    WWATER21 = np.zeros((steppi, N))
    WWATER22 = np.zeros((steppi, N))

    WGAS1 = np.zeros((steppi, N))
    WGAS2 = np.zeros((steppi, N))
    WGAS3 = np.zeros((steppi, N))
    WGAS4 = np.zeros((steppi, N))
    WGAS5 = np.zeros((steppi, N))
    WGAS6 = np.zeros((steppi, N))
    WGAS7 = np.zeros((steppi, N))
    WGAS8 = np.zeros((steppi, N))
    WGAS9 = np.zeros((steppi, N))
    WGAS10 = np.zeros((steppi, N))
    WGAS11 = np.zeros((steppi, N))
    WGAS12 = np.zeros((steppi, N))
    WGAS13 = np.zeros((steppi, N))
    WGAS14 = np.zeros((steppi, N))
    WGAS15 = np.zeros((steppi, N))
    WGAS16 = np.zeros((steppi, N))
    WGAS17 = np.zeros((steppi, N))
    WGAS18 = np.zeros((steppi, N))
    WGAS19 = np.zeros((steppi, N))
    WGAS20 = np.zeros((steppi, N))
    WGAS21 = np.zeros((steppi, N))
    WGAS22 = np.zeros((steppi, N))

    WWINJ1 = np.zeros((steppi, N))
    WWINJ2 = np.zeros((steppi, N))
    WWINJ3 = np.zeros((steppi, N))
    WWINJ4 = np.zeros((steppi, N))
    WWINJ5 = np.zeros((steppi, N))
    WWINJ6 = np.zeros((steppi, N))
    WWINJ7 = np.zeros((steppi, N))
    WWINJ8 = np.zeros((steppi, N))
    WWINJ9 = np.zeros((steppi, N))

    WGASJ1 = np.zeros((steppi, N))
    WGASJ2 = np.zeros((steppi, N))
    WGASJ3 = np.zeros((steppi, N))
    WGASJ4 = np.zeros((steppi, N))

    steppii = 246
    for i in range(N):
        f = "Realisation_"
        folder = f + str(i)
        os.chdir(folder)
        ## IMPORT FOR OIL
        A2oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None)

        B_1BHoilsim = A2oilsim[5].values[:steppii]
        B_1Hoilsim = A2oilsim[6].values[:steppii]
        B_2Hoilsim = A2oilsim[7].values[:steppii]
        B_3Hoilsim = A2oilsim[8].values[:steppii]
        B_4BHoilsim = A2oilsim[9].values[:steppii]

        A22oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1801, sep="\s+", header=None)
        B_4DHoilsim = A22oilsim[1].values[:steppii]
        B_4Hoilsim = A22oilsim[2].values[:steppii]
        D_1CHoilsim = A22oilsim[3].values[:steppii]
        D_1Hoilsim = A22oilsim[4].values[:steppii]
        D_2Hoilsim = A22oilsim[5].values[:steppii]
        D_3AHoilsim = A22oilsim[6].values[:steppii]
        D_3BHoilsim = A22oilsim[7].values[:steppii]
        D_4AHoilsim = A22oilsim[8].values[:steppii]
        D_4Hoilsim = A22oilsim[9].values[:steppii]

        A222oilsim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None
        )

        E_1Hoilsim = A222oilsim[1].values[:steppii]
        E_2AHoilsim = A222oilsim[2].values[:steppii]
        E_2Hoilsim = A222oilsim[3].values[:steppii]
        E_3AHoilsim = A222oilsim[4].values[:steppii]
        E_3CHoilsim = A222oilsim[5].values[:steppii]
        E_3Hoilsim = A222oilsim[6].values[:steppii]
        E_4AHoilsim = A222oilsim[7].values[:steppii]
        K_3Hoilsim = A222oilsim[8].values[:steppii]

        WOIL1[:, i] = B_1BHoilsim.ravel()[steppi_indices - 1]
        WOIL2[:, i] = B_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL3[:, i] = B_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL4[:, i] = B_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL5[:, i] = B_4BHoilsim.ravel()[steppi_indices - 1]
        WOIL6[:, i] = B_4DHoilsim.ravel()[steppi_indices - 1]
        WOIL7[:, i] = B_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL8[:, i] = D_1CHoilsim.ravel()[steppi_indices - 1]
        WOIL9[:, i] = D_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL10[:, i] = D_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL11[:, i] = D_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL12[:, i] = D_3BHoilsim.ravel()[steppi_indices - 1]
        WOIL13[:, i] = D_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL14[:, i] = D_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL15[:, i] = E_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL16[:, i] = E_2AHoilsim.ravel()[steppi_indices - 1]
        WOIL17[:, i] = E_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL18[:, i] = E_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL19[:, i] = E_3CHoilsim.ravel()[steppi_indices - 1]
        WOIL20[:, i] = E_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL21[:, i] = E_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL22[:, i] = K_3Hoilsim.ravel()[steppi_indices - 1]

        ##IMPORT FOR WATER
        A2watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None
        )
        B_1BHwatersim = A2watersim[9].values[:steppi]

        A22watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2569, sep="\s+", header=None
        )
        B_1Hwatersim = A22watersim[1].values[:steppii]
        B_2Hwatersim = A22watersim[2].values[:steppii]
        B_3Hwatersim = A22watersim[3].values[:steppii]
        B_4BHwatersim = A22watersim[4].values[:steppii]
        B_4DHwatersim = A22watersim[5].values[:steppii]
        B_4Hwatersim = A22watersim[6].values[:steppii]
        D_1CHwatersim = A22watersim[7].values[:steppii]
        D_1Hwatersim = A22watersim[8].values[:steppii]
        D_2Hwatersim = A22watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2825, sep="\s+", header=None
        )
        D_3AHwatersim = A222watersim[1].values[:steppii]
        D_3BHwatersim = A222watersim[2].values[:steppii]
        D_4AHwatersim = A222watersim[3].values[:steppii]
        D_4Hwatersim = A222watersim[4].values[:steppii]
        E_1Hwatersim = A222watersim[5].values[:steppii]
        E_2AHwatersim = A222watersim[6].values[:steppii]
        E_2Hwatersim = A222watersim[7].values[:steppii]
        E_3AHwatersim = A222watersim[8].values[:steppii]
        E_3CHwatersim = A222watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=3081, sep="\s+", header=None
        )
        E_3Hwatersim = A222watersim[1].values[:steppii]
        E_4AHwatersim = A222watersim[2].values[:steppii]
        K_3Hwatersim = A222watersim[3].values[:steppii]

        WWATER1[:, i] = B_1BHwatersim.ravel()[steppi_indices - 1]
        WWATER2[:, i] = B_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER3[:, i] = B_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER4[:, i] = B_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER5[:, i] = B_4BHwatersim.ravel()[steppi_indices - 1]
        WWATER6[:, i] = B_4DHwatersim.ravel()[steppi_indices - 1]
        WWATER7[:, i] = B_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER8[:, i] = D_1CHwatersim.ravel()[steppi_indices - 1]
        WWATER9[:, i] = D_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER10[:, i] = D_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER11[:, i] = D_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER12[:, i] = D_3BHwatersim.ravel()[steppi_indices - 1]
        WWATER13[:, i] = D_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER14[:, i] = D_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER15[:, i] = E_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER16[:, i] = E_2AHwatersim.ravel()[steppi_indices - 1]
        WWATER17[:, i] = E_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER18[:, i] = E_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER19[:, i] = E_3CHwatersim.ravel()[steppi_indices - 1]

        # if len(E_3Hwatersim<steppi):
        #     E_3Hwatersim = np.append(E_3Hwatersim.ravel(), 0)
        # if len(E_4AHwatersim<steppi):
        #     E_4AHwatersim = np.append(E_4AHwatersim.ravel(), 0)
        # if len(K_3Hwatersim<steppi):
        #     K_3Hwatersim = np.append(K_3Hwatersim.ravel(), 0)
        WWATER20[:, i] = E_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER21[:, i] = E_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER22[:, i] = K_3Hwatersim.ravel()[steppi_indices - 1]

        ## GAS PRODUCTION RATE
        A2gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1033, sep="\s+", header=None)
        B_1BHgassim = A2gassim[1].values[:steppii]
        B_1Hgassim = A2gassim[2].values[:steppii]
        B_2Hgassim = A2gassim[3].values[:steppii]
        B_3Hgassim = A2gassim[4].values[:steppii]
        B_4BHgassim = A2gassim[5].values[:steppii]
        B_4DHgassim = A2gassim[6].values[:steppii]
        B_4Hgassim = A2gassim[7].values[:steppii]
        D_1CHgassim = A2gassim[8].values[:steppii]
        D_1Hgassim = A2gassim[9].values[:steppii]

        A22gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1289, sep="\s+", header=None)
        D_2Hgassim = A22gassim[1].values[:steppii]
        D_3AHgassim = A22gassim[2].values[:steppii]
        D_3BHgassim = A22gassim[3].values[:steppii]
        D_4AHgassim = A22gassim[4].values[:steppii]
        D_4Hgassim = A22gassim[5].values[:steppii]
        E_1Hgassim = A22gassim[6].values[:steppii]
        E_2AHgassim = A22gassim[7].values[:steppii]
        E_2Hgassim = A22gassim[8].values[:steppii]
        E_3AHgassim = A22gassim[9].values[:steppii]

        A222gassim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None
        )
        E_3CHgassim = A222gassim[1].values[:steppii]
        E_3Hgassim = A222gassim[2].values[:steppii]
        E_4AHgassim = A222gassim[3].values[:steppii]
        K_3Hgassim = A222gassim[4].values[:steppii]

        WGAS1[:, i] = B_1BHgassim.ravel()[steppi_indices - 1]
        WGAS2[:, i] = B_1Hgassim.ravel()[steppi_indices - 1]
        WGAS3[:, i] = B_2Hgassim.ravel()[steppi_indices - 1]
        WGAS4[:, i] = B_3Hgassim.ravel()[steppi_indices - 1]
        WGAS5[:, i] = B_4BHgassim.ravel()[steppi_indices - 1]
        WGAS6[:, i] = B_4DHgassim.ravel()[steppi_indices - 1]
        WGAS7[:, i] = B_4Hgassim.ravel()[steppi_indices - 1]
        WGAS8[:, i] = D_1CHgassim.ravel()[steppi_indices - 1]
        WGAS9[:, i] = D_1Hgassim.ravel()[steppi_indices - 1]
        WGAS10[:, i] = D_2Hgassim.ravel()[steppi_indices - 1]
        WGAS11[:, i] = D_3AHgassim.ravel()[steppi_indices - 1]
        WGAS12[:, i] = D_3BHgassim.ravel()[steppi_indices - 1]
        WGAS13[:, i] = D_4AHgassim.ravel()[steppi_indices - 1]
        WGAS14[:, i] = D_4Hgassim.ravel()[steppi_indices - 1]
        WGAS15[:, i] = E_1Hgassim.ravel()[steppi_indices - 1]
        WGAS16[:, i] = E_2AHgassim.ravel()[steppi_indices - 1]
        WGAS17[:, i] = E_2Hgassim.ravel()[steppi_indices - 1]
        WGAS18[:, i] = E_3AHgassim.ravel()[steppi_indices - 1]
        WGAS19[:, i] = E_3CHgassim.ravel()[steppi_indices - 1]
        WGAS20[:, i] = E_3Hgassim.ravel()[steppi_indices - 1]
        WGAS21[:, i] = E_4AHgassim.ravel()[steppi_indices - 1]
        WGAS22[:, i] = K_3Hgassim.ravel()[steppi_indices - 1]

        ## WATER INJECTOR RATE
        A2wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None)
        C_1Hwaterinjsim = A2wir[9].values[:steppi]

        A22wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None)
        C_2Hwaterinjsim = A22wir[1].values[:steppii]
        C_3Hwaterinjsim = A22wir[2].values[:steppii]
        C_4AHwaterinjsim = A22wir[3].values[:steppii]
        C_4Hwaterinjsim = A22wir[4].values[:steppii]
        F_1Hwaterinjsim = A22wir[5].values[:steppii]
        F_2Hwaterinjsim = A22wir[6].values[:steppii]
        F_3Hwaterinjsim = A22wir[7].values[:steppii]
        F_4Hwaterinjsim = A22wir[8].values[:steppii]

        WWINJ1[:, i] = C_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ2[:, i] = C_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ3[:, i] = C_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ4[:, i] = C_4AHwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ5[:, i] = C_4Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ6[:, i] = F_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ7[:, i] = F_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ8[:, i] = F_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ9[:, i] = F_4Hwaterinjsim.ravel()[steppi_indices - 1]

        ## GAS INJECTOR RATE
        A2gir = pd.read_csv("FULLNORNE2.RSM", skiprows=777, sep="\s+", header=None)
        C_1Hgasinjsim = A2gir[6].values[:steppii]
        C_3Hgasinjsim = A2gir[7].values[:steppii]
        C_4AHgasinjsim = A2gir[8].values[:steppii]
        C_4Hgasinjsim = A2gir[9].values[:steppii]

        WGASJ1[:, i] = C_1Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ2[:, i] = C_3Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ3[:, i] = C_4AHgasinjsim.ravel()[steppi_indices - 1]
        WGASJ4[:, i] = C_4Hgasinjsim.ravel()[steppi_indices - 1]

        # Return to the root folder
        os.chdir(oldfolder)

    OIL = {
        "B_1BH": WOIL1,
        "B_1H": WOIL2,
        "B_2H": WOIL3,
        "B_3H": WOIL4,
        "B_4BH": WOIL5,
        "B_4DH": WOIL6,
        "B_4H": WOIL7,
        "D_1CH": WOIL8,
        "D_1H": WOIL9,
        "D_2H": WOIL10,
        "D_3AH": WOIL11,
        "D_3BH": WOIL12,
        "D_4AH": WOIL13,
        "D_4H": WOIL14,
        "E_1H": WOIL15,
        "E_2AH": WOIL16,
        "E_2H": WOIL17,
        "E_3AH": WOIL18,
        "E_3CH": WOIL19,
        "E_3H": WOIL20,
        "E_4AH": WOIL21,
        "K_3H": WOIL22,
    }

    OIL = {k: abs(v) for k, v in OIL.items()}

    WATER = {
        "B_1BH": WWATER1,
        "B_1H": WWATER2,
        "B_2H": WWATER3,
        "B_3H": WWATER4,
        "B_4BH": WWATER5,
        "B_4DH": WWATER6,
        "B_4H": WWATER7,
        "D_1CH": WWATER8,
        "D_1H": WWATER9,
        "D_2H": WWATER10,
        "D_3AH": WWATER11,
        "D_3BH": WWATER12,
        "D_4AH": WWATER13,
        "D_4H": WWATER14,
        "E_1H": WWATER15,
        "E_2AH": WWATER16,
        "E_2H": WWATER17,
        "E_3AH": WWATER18,
        "E_3CH": WWATER19,
        "E_3H": WWATER20,
        "E_4AH": WWATER21,
        "K_3H": WWATER22,
    }

    WATER = {k: abs(v) for k, v in WATER.items()}

    GAS = {
        "B_1BH": WGAS1,
        "B_1H": WGAS2,
        "B_2H": WGAS3,
        "B_3H": WGAS4,
        "B_4BH": WGAS5,
        "B_4DH": WGAS6,
        "B_4H": WGAS7,
        "D_1CH": WGAS8,
        "D_1H": WGAS9,
        "D_2H": WGAS10,
        "D_3AH": WGAS11,
        "D_3BH": WGAS12,
        "D_4AH": WGAS13,
        "D_4H": WGAS14,
        "E_1H": WGAS15,
        "E_2AH": WGAS16,
        "E_2H": WGAS17,
        "E_3AH": WGAS18,
        "E_3CH": WGAS19,
        "E_3H": WGAS20,
        "E_4AH": WGAS21,
        "K_3H": WGAS22,
    }

    GAS = {k: abs(v) for k, v in GAS.items()}

    WATERINJ = {
        "C_1H": WWINJ1,
        "C_2H": WWINJ2,
        "C_3H": WWINJ3,
        "C_4AH": WWINJ4,
        "C_4H": WWINJ5,
        "F_1H": WWINJ6,
        "F_2H": WWINJ7,
        "F_3H": WWINJ8,
        "F_4H": WWINJ9,
    }

    WATERINJ = {k: abs(v) for k, v in WATERINJ.items()}

    GASINJ = {"C_1H": WGASJ1, "C_3H": WGASJ2, "C_4AH": WGASJ3, "C_4H": WGASJ4}

    GASINJ = {k: abs(v) for k, v in GASINJ.items()}
    return OIL, WATER, GAS, WATERINJ, GASINJ


def Get_predicted_data(OIL, WATER, GAS, WATERINJ, GASINJ):
    matrices = [OIL[key] for key in sorted(OIL.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_oil = np.hstack(result_columns)

    matrices = [WATER[key] for key in sorted(WATER.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_water = np.hstack(result_columns)

    matrices = [GAS[key] for key in sorted(GAS.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_gas = np.hstack(result_columns)

    matrices = [WATERINJ[key] for key in sorted(WATERINJ.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_winj = np.hstack(result_columns)

    matrices = [GASINJ[key] for key in sorted(GASINJ.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_ginj = np.hstack(result_columns)

    predicted_data = np.nan_to_num(
        np.vstack((result_oil, result_water, result_gas, result_winj, result_ginj))
    )
    return predicted_data


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
    if Truee.ndim == 3:
        # If it's a 3D array, compute the mean along axis 2
        avg_2d = np.mean(Truee, axis=2)
    else:
        avg_2d = np.reshape(Truee, (nx, ny), "F")

    # Pressz = np.reshape(Truee,(nx,ny),'F')
    maxii = max(avg_2d.ravel())
    minii = min(avg_2d.ravel())

    # avg_2d = np.mean(Pressz, axis=2)
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

    elif varii == "oil PhysicsNeMo":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation -PhysicsNeMo", fontsize=11, weight="bold")

    elif varii == "oil FLOW":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation - Flow", fontsize=11, weight="bold")

    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("oil saturation - (FLOW -PhysicsNeMo)", fontsize=11, weight="bold")

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


def Plot_2DD(
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
    if Truee.ndim == 3:
        Pressz = np.reshape(Truee, (nx, ny, nz), "F")
        avg_2d = np.mean(Pressz, axis=2)
    else:
        Pressz = np.reshape(Truee, (nx, ny), "F")
        avg_2d = Pressz

    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())

    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs
    # XX, YY = np.meshgrid(np.arange(nx),np.arange(ny))
    # plt.subplot(224)

    plt.pcolormesh(XX.T, YY.T, avg_2d, cmap="jet")
    cbar = plt.colorbar()

    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=11)
    elif varii == "water PhysicsNeMo":
        cbar.set_label("water saturation", fontsize=11)
        # plt.title('water saturation -PhysicsNeMo',fontsize=11,weight='bold')
    elif varii == "oil PhysicsNeMo":
        cbar.set_label("Oil saturation", fontsize=11)
        # plt.title('Oil saturation -PhysicsNeMo',fontsize=11,weight='bold')

    elif varii == "gas PhysicsNeMo":
        cbar.set_label("Gas saturation", fontsize=11)
        # plt.title('Gas saturation -PhysicsNeMo',fontsize=11,weight='bold')

    elif varii == "pressure PhysicsNeMo":
        cbar.set_label("pressure", fontsize=11)
        # plt.title('Pressure -PhysicsNeMo',fontsize=11,weight='bold')

    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=11)
        # plt.title('Porosity Field',fontsize=11,weight='bold')
    cbar.mappable.set_clim(minii, maxii)

    plt.ylabel("Y", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gass)


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

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gass)  # Number of gas injectors

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

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gass)  # Number of gas injectors

    for mm in range(n_inj):
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

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
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
            fontsize=9,
        )


def Plot_mean(permbest, permmean, iniperm, nx, ny, Low_K, High_K, True_perm):
    Low_Ka = Low_K
    High_Ka = High_K

    permmean = np.mean(np.reshape(permmean, (nx, ny, nz), "F") * effectiveuse, axis=2)

    permbest = np.mean(np.reshape(permbest, (nx, ny, nz), "F") * effectiveuse, axis=2)
    iniperm = np.mean(np.reshape(iniperm, (nx, ny, nz), "F") * effectiveuse, axis=2)
    True_perm = np.mean(np.reshape(True_perm, (nx, ny, nz), "F") * effectiveuse, axis=2)
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(30, 30))

    plt.subplot(2, 2, 1)
    # plt.pcolormesh(XX.T,YY.T,permmean[:,:,0],cmap = 'jet')
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        permmean,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gass,
    )
    plt.title(" MAP", fontsize=15)
    plt.clim(Low_Ka, High_Ka)

    plt.subplot(2, 2, 2)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        permbest,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gass,
    )
    plt.title(" Best", fontsize=15)
    plt.clim(Low_Ka, High_Ka)

    plt.subplot(2, 2, 3)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        iniperm,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gass,
    )
    plt.title(" initial", fontsize=15)
    plt.clim(Low_Ka, High_Ka)

    plt.subplot(2, 2, 4)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        True_perm,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gass,
    )
    plt.title(" True", fontsize=15)
    plt.clim(Low_Ka, High_Ka)

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

    permmean = np.mean(np.reshape(permmean, (nx, ny, nz), "F") * effectiveuse, axis=2)
    temp_K = np.mean(np.reshape(temp_K, (nx, ny, nz), "F") * effectiveuse, axis=2)
    poroo = np.mean(np.reshape(poroo, (nx, ny, nz), "F") * effectiveuse, axis=2)
    temp_phi = np.mean(np.reshape(temp_phi, (nx, ny, nz), "F") * effectiveuse, axis=2)

    plt.subplot(2, 2, 1)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        permmean,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gass,
    )
    plt.title(" Permeability", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers, gass)

    plt.subplot(2, 2, 2)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        temp_K,
        N_injw,
        N_pr,
        N_injg,
        "perm",
        injectors,
        producers,
        gass,
    )
    plt.title("Smoothed - Permeability", fontsize=15)
    plt.clim(Low_Ka, High_Ka)
    Add_marker2(plt, XX, YY, injectors, producers, gass)

    plt.subplot(2, 2, 3)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        poroo,
        N_injw,
        N_pr,
        N_injg,
        "poro",
        injectors,
        producers,
        gass,
    )
    plt.title("Porosity", fontsize=15)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers, gass)

    plt.subplot(2, 2, 4)
    Plot_2D(
        XX,
        YY,
        plt,
        nx,
        ny,
        nz,
        temp_phi,
        N_injw,
        N_pr,
        N_injg,
        "poro",
        injectors,
        producers,
        gass,
    )
    plt.title("Smoothed Porosity", fontsize=15)
    plt.clim(Low_P, High_P)
    Add_marker2(plt, XX, YY, injectors, producers, gass)

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


def Plot_RSM_percentile(pertoutt, True_mat, timezz):
    columns = [
        "B_1BH",
        "B_1H",
        "B_2H",
        "B_3H",
        "B_4BH",
        "B_4DH",
        "B_4H",
        "D_1CH",
        "D_1H",
        "D_2H",
        "D_3AH",
        "D_3BH",
        "D_4AH",
        "D_4H",
        "E_1H",
        "E_2AH",
        "E_2H",
        "E_3AH",
        "E_3CH",
        "E_3H",
        "E_4AH",
        "K_3H",
    ]

    # timezz = True_mat[:,0].reshape(-1,1)

    P10 = pertoutt[0]
    P50 = pertoutt[1]
    P90 = pertoutt[2]
    arekibest = pertoutt[3]
    amean = pertoutt[4]
    base = pertoutt[5]

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k], color="blue", lw="2", label="P10 Model")
        plt.plot(timezz, P50[:, k], color="c", lw="2", label="P50 Model")
        plt.plot(timezz, P90[:, k], color="green", lw="2", label="P90 Model")

        plt.plot(
            timezz, arekibest[:, k], color="k", lw="2", label="aREKI cum best Model"
        )

        plt.plot(
            timezz, amean[:, k], color="purple", lw="2", label="aREKI cum MAP Model"
        )

        plt.plot(timezz, base[:, k], color="orange", lw="2", label="Base case Model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Oil.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 22], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k + 22], color="blue", lw="2", label="P10 Model")
        plt.plot(timezz, P50[:, k + 22], color="c", lw="2", label="P50 Model")
        plt.plot(timezz, P90[:, k + 22], color="green", lw="2", label="P90 Model")

        plt.plot(
            timezz,
            arekibest[:, k + 22],
            color="k",
            lw="2",
            label="aREKI cum best Model",
        )
        plt.plot(
            timezz,
            amean[:, k + 22],
            color="purple",
            lw="2",
            label="aREKI cum MAP Model",
        )

        plt.plot(
            timezz, base[:, k + 22], color="orange", lw="2", label="Base case Model"
        )

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Water.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 44], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k + 44], color="blue", lw="2", label="P10 Model")
        plt.plot(timezz, P50[:, k + 44], color="c", lw="2", label="P50 Model")
        plt.plot(timezz, P90[:, k + 44], color="green", lw="2", label="P90 Model")

        plt.plot(
            timezz,
            arekibest[:, k + 44],
            color="k",
            lw="2",
            label="aREKI cum best Model",
        )

        plt.plot(
            timezz,
            amean[:, k + 44],
            color="purple",
            lw="2",
            label="aREKI cum MAP Model",
        )

        plt.plot(
            timezz, base[:, k + 44], color="orange", lw="2", label="Base case Model"
        )

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Gas.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    # Define your P10
    True_data = np.reshape(True_mat, (-1, 1), "F")
    P10 = np.reshape(P10, (-1, 1), "F")
    cc10 = ((np.sum((((P10) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of P10 reservoir model  =  " + str(cc10))

    P50 = np.reshape(P50, (-1, 1), "F")
    cc50 = ((np.sum((((P50) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of P50 reservoir model  =  " + str(cc50))

    P90 = np.reshape(P90, (-1, 1), "F")
    cc90 = ((np.sum((((P90) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of P90 reservoir model  = : " + str(cc90))

    P99 = np.reshape(arekibest, (-1, 1), "F")
    cc99 = ((np.sum((((P99) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of cummulative best reservoir model  =  " + str(cc99))

    Pmean = np.reshape(amean, (-1, 1), "F")
    ccmean = ((np.sum((((Pmean) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of cummulative mean reservoir model  =  " + str(ccmean))

    Pbase = np.reshape(base, (-1, 1), "F")
    ccbase = ((np.sum((((Pbase) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
    print("RMSE of base case reservoir model  =  " + str(ccbase))

    plt.figure(figsize=(10, 10))
    values = [cc10, cc50, cc90, cc99, ccmean, ccbase]
    model_names = ["P10", "P50", "P90", "cumm-best", "cumm-mean", "Base case"]
    colors = ["blue", "c", "green", "k", "purple", "orange"]

    # Find the index of the minimum RMSE value
    min_rmse_index = np.argmin(values)

    # Get the minimum RMSE value
    min_rmse = values[min_rmse_index]

    # Get the corresponding model name
    best_model = model_names[min_rmse_index]

    # Print the minimum RMSE value and its corresponding model name
    print(f"The minimum RMSE value = {min_rmse}")
    print(f"Recommended reservoir model = {best_model} reservoir model.")

    # Create a histogram
    plt.bar(model_names, values, color=colors)
    plt.xlabel("Reservoir Models")
    plt.ylabel("RMSE")
    plt.title("Histogram of RMSE Values for Different Reservoir Models")
    plt.legend(model_names)
    plt.savefig(
        "Histogram.png"
    )  # save as png                                  # preventing the figures from showing
    plt.clf()
    plt.close()


def Plot_RSM_percentile_model(pertoutt, True_mat, timezz):
    columns = [
        "B_1BH",
        "B_1H",
        "B_2H",
        "B_3H",
        "B_4BH",
        "B_4DH",
        "B_4H",
        "D_1CH",
        "D_1H",
        "D_2H",
        "D_3AH",
        "D_3BH",
        "D_4AH",
        "D_4H",
        "E_1H",
        "E_2AH",
        "E_2H",
        "E_3AH",
        "E_3CH",
        "E_3H",
        "E_4AH",
        "K_3H",
    ]

    # timezz = True_mat[:,0].reshape(-1,1)

    P10 = pertoutt

    # arekimean =  pertoutt[4]

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k], color="blue", lw="2", label="Surrogate")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Oil.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 22], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k + 22], color="blue", lw="2", label="P10 Model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Water.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 44], color="red", lw="2", label="model")
        plt.plot(timezz, P10[:, k + 44], color="blue", lw="2", label="P10 Model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Gas.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM_single(True_mat, timezz):
    True_mat = True_mat[0]

    columns = [
        "B_1BH",
        "B_1H",
        "B_2H",
        "B_3H",
        "B_4BH",
        "B_4DH",
        "B_4H",
        "D_1CH",
        "D_1H",
        "D_2H",
        "D_3AH",
        "D_3BH",
        "D_4AH",
        "D_4H",
        "E_1H",
        "E_2AH",
        "E_2H",
        "E_3AH",
        "E_3CH",
        "E_3H",
        "E_4AH",
        "K_3H",
    ]

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Oil_single.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 22], color="red", lw="2", label="model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Water_single.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 44], color="red", lw="2", label="model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Gas_single.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM_singleT(True_mat, timezz):
    columns = [
        "B_1BH",
        "B_1H",
        "B_2H",
        "B_3H",
        "B_4BH",
        "B_4DH",
        "B_4H",
        "D_1CH",
        "D_1H",
        "D_2H",
        "D_3AH",
        "D_3BH",
        "D_4AH",
        "D_4H",
        "E_1H",
        "E_2AH",
        "E_2H",
        "E_3AH",
        "E_3CH",
        "E_3H",
        "E_4AH",
        "K_3H",
    ]

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Oil_singleT.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 22], color="red", lw="2", label="model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Water_singleT.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 44], color="red", lw="2", label="model")

        plt.xlabel("Time (days)", fontsize=13)
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13)
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.savefig(
        "Gas_singleT.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def Plot_RSM(predMatrix, True_mat, Namesz, Ne, timezz):
    columns = [
        "B_1BH",
        "B_1H",
        "B_2H",
        "B_3H",
        "B_4BH",
        "B_4DH",
        "B_4H",
        "D_1CH",
        "D_1H",
        "D_2H",
        "D_3AH",
        "D_3BH",
        "D_4AH",
        "D_4H",
        "E_1H",
        "E_2AH",
        "E_2H",
        "E_3AH",
        "E_3CH",
        "E_3H",
        "E_4AH",
        "K_3H",
    ]
    # timezz = True_mat[:,0].reshape(-1,1)

    Nt = predMatrix[0].shape[0]
    # timezz=predMatrix[0][:,0].reshape(-1,1)

    WOPRA = np.zeros((Nt, Ne))
    WOPRB = np.zeros((Nt, Ne))
    WOPRC = np.zeros((Nt, Ne))
    WOPRD = np.zeros((Nt, Ne))
    WOPRE = np.zeros((Nt, Ne))
    WOPRF = np.zeros((Nt, Ne))
    WOPRG = np.zeros((Nt, Ne))
    WOPRH = np.zeros((Nt, Ne))
    WOPRI = np.zeros((Nt, Ne))
    WOPRJ = np.zeros((Nt, Ne))
    WOPRK = np.zeros((Nt, Ne))
    WOPRL = np.zeros((Nt, Ne))
    WOPRM = np.zeros((Nt, Ne))
    WOPRN = np.zeros((Nt, Ne))
    WOPRO = np.zeros((Nt, Ne))
    WOPRP = np.zeros((Nt, Ne))
    WOPRQ = np.zeros((Nt, Ne))
    WOPRR = np.zeros((Nt, Ne))
    WOPRS = np.zeros((Nt, Ne))
    WOPRT = np.zeros((Nt, Ne))
    WOPRU = np.zeros((Nt, Ne))
    WOPRV = np.zeros((Nt, Ne))

    WWPRA = np.zeros((Nt, Ne))
    WWPRB = np.zeros((Nt, Ne))
    WWPRC = np.zeros((Nt, Ne))
    WWPRD = np.zeros((Nt, Ne))
    WWPRE = np.zeros((Nt, Ne))
    WWPRF = np.zeros((Nt, Ne))
    WWPRG = np.zeros((Nt, Ne))
    WWPRH = np.zeros((Nt, Ne))
    WWPRI = np.zeros((Nt, Ne))
    WWPRJ = np.zeros((Nt, Ne))
    WWPRK = np.zeros((Nt, Ne))
    WWPRL = np.zeros((Nt, Ne))
    WWPRM = np.zeros((Nt, Ne))
    WWPRN = np.zeros((Nt, Ne))
    WWPRO = np.zeros((Nt, Ne))
    WWPRP = np.zeros((Nt, Ne))
    WWPRQ = np.zeros((Nt, Ne))
    WWPRR = np.zeros((Nt, Ne))
    WWPRS = np.zeros((Nt, Ne))
    WWPRT = np.zeros((Nt, Ne))
    WWPRU = np.zeros((Nt, Ne))
    WWPRV = np.zeros((Nt, Ne))

    WGPRA = np.zeros((Nt, Ne))
    WGPRB = np.zeros((Nt, Ne))
    WGPRC = np.zeros((Nt, Ne))
    WGPRD = np.zeros((Nt, Ne))
    WGPRE = np.zeros((Nt, Ne))
    WGPRF = np.zeros((Nt, Ne))
    WGPRG = np.zeros((Nt, Ne))
    WGPRH = np.zeros((Nt, Ne))
    WGPRI = np.zeros((Nt, Ne))
    WGPRJ = np.zeros((Nt, Ne))
    WGPRK = np.zeros((Nt, Ne))
    WGPRL = np.zeros((Nt, Ne))
    WGPRM = np.zeros((Nt, Ne))
    WGPRN = np.zeros((Nt, Ne))
    WGPRO = np.zeros((Nt, Ne))
    WGPRP = np.zeros((Nt, Ne))
    WGPRQ = np.zeros((Nt, Ne))
    WGPRR = np.zeros((Nt, Ne))
    WGPRS = np.zeros((Nt, Ne))
    WGPRT = np.zeros((Nt, Ne))
    WGPRU = np.zeros((Nt, Ne))
    WGPRV = np.zeros((Nt, Ne))

    for i in range(Ne):
        usef = predMatrix[i]

        WOPRA[:, i] = usef[:, 0]
        WOPRB[:, i] = usef[:, 1]
        WOPRC[:, i] = usef[:, 2]
        WOPRD[:, i] = usef[:, 3]
        WOPRE[:, i] = usef[:, 4]
        WOPRF[:, i] = usef[:, 5]
        WOPRG[:, i] = usef[:, 6]
        WOPRH[:, i] = usef[:, 7]
        WOPRI[:, i] = usef[:, 8]
        WOPRJ[:, i] = usef[:, 9]
        WOPRK[:, i] = usef[:, 10]
        WOPRL[:, i] = usef[:, 11]
        WOPRM[:, i] = usef[:, 12]
        WOPRN[:, i] = usef[:, 13]
        WOPRO[:, i] = usef[:, 14]
        WOPRP[:, i] = usef[:, 15]
        WOPRQ[:, i] = usef[:, 16]
        WOPRR[:, i] = usef[:, 17]
        WOPRS[:, i] = usef[:, 18]
        WOPRT[:, i] = usef[:, 19]
        WOPRU[:, i] = usef[:, 20]
        WOPRV[:, i] = usef[:, 21]

        WWPRA[:, i] = usef[:, 22]
        WWPRB[:, i] = usef[:, 23]
        WWPRC[:, i] = usef[:, 24]
        WWPRD[:, i] = usef[:, 25]
        WWPRE[:, i] = usef[:, 26]
        WWPRF[:, i] = usef[:, 27]
        WWPRG[:, i] = usef[:, 28]
        WWPRH[:, i] = usef[:, 29]
        WWPRI[:, i] = usef[:, 30]
        WWPRJ[:, i] = usef[:, 31]
        WWPRK[:, i] = usef[:, 32]
        WWPRL[:, i] = usef[:, 33]
        WWPRM[:, i] = usef[:, 34]
        WWPRN[:, i] = usef[:, 35]
        WWPRO[:, i] = usef[:, 36]
        WWPRP[:, i] = usef[:, 37]
        WWPRQ[:, i] = usef[:, 38]
        WWPRR[:, i] = usef[:, 39]
        WWPRS[:, i] = usef[:, 40]
        WWPRT[:, i] = usef[:, 41]
        WWPRU[:, i] = usef[:, 42]
        WWPRV[:, i] = usef[:, 43]

        WGPRA[:, i] = usef[:, 44]
        WGPRB[:, i] = usef[:, 45]
        WGPRC[:, i] = usef[:, 46]
        WGPRD[:, i] = usef[:, 47]
        WGPRE[:, i] = usef[:, 48]
        WGPRF[:, i] = usef[:, 49]
        WGPRG[:, i] = usef[:, 50]
        WGPRH[:, i] = usef[:, 51]
        WGPRI[:, i] = usef[:, 52]
        WGPRJ[:, i] = usef[:, 53]
        WGPRK[:, i] = usef[:, 54]
        WGPRL[:, i] = usef[:, 55]
        WGPRM[:, i] = usef[:, 56]
        WGPRN[:, i] = usef[:, 57]
        WGPRO[:, i] = usef[:, 58]
        WGPRP[:, i] = usef[:, 59]
        WGPRQ[:, i] = usef[:, 60]
        WGPRR[:, i] = usef[:, 61]
        WGPRS[:, i] = usef[:, 62]
        WGPRT[:, i] = usef[:, 63]
        WGPRU[:, i] = usef[:, 64]
        WGPRV[:, i] = usef[:, 65]

    plt.figure(figsize=(40, 40))

    plt.subplot(5, 5, 1)
    plt.plot(timezz, WOPRA[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[0], fontsize=13)

    plt.plot(timezz, True_mat[:, 0], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 2)
    plt.plot(timezz, WOPRB[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[1], fontsize=13)

    plt.plot(timezz, True_mat[:, 1], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 3)
    plt.plot(timezz, WOPRC[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[2], fontsize=13)

    plt.plot(timezz, True_mat[:, 2], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 4)
    plt.plot(timezz, WOPRD[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[3], fontsize=13)

    plt.plot(timezz, True_mat[:, 3], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 5)
    plt.plot(timezz, WOPRE[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[4], fontsize=13)

    plt.plot(timezz, True_mat[:, 4], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 6)
    plt.plot(timezz, WOPRF[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[5], fontsize=13)

    plt.plot(timezz, True_mat[:, 5], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 7)
    plt.plot(timezz, WOPRG[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[6], fontsize=13)

    plt.plot(timezz, True_mat[:, 6], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 8)
    plt.plot(timezz, WOPRH[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[7], fontsize=13)

    plt.plot(timezz, True_mat[:, 7], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 9)
    plt.plot(timezz, WOPRI[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[8], fontsize=13)

    plt.plot(timezz, True_mat[:, 8], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 10)
    plt.plot(timezz, WOPRJ[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[9], fontsize=13)

    plt.plot(timezz, True_mat[:, 9], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 11)
    plt.plot(timezz, WOPRK[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[10], fontsize=13)

    plt.plot(timezz, True_mat[:, 10], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 12)
    plt.plot(timezz, WOPRL[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[11], fontsize=13)

    plt.plot(timezz, True_mat[:, 11], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 13)
    plt.plot(timezz, WOPRM[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[12], fontsize=13)

    plt.plot(timezz, True_mat[:, 12], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 14)
    plt.plot(timezz, WOPRN[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[13], fontsize=13)

    plt.plot(timezz, True_mat[:, 13], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 15)
    plt.plot(timezz, WOPRO[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[14], fontsize=13)

    plt.plot(timezz, True_mat[:, 14], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 16)
    plt.plot(timezz, WOPRP[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[15], fontsize=13)

    plt.plot(timezz, True_mat[:, 15], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 17)
    plt.plot(timezz, WOPRQ[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[16], fontsize=13)

    plt.plot(timezz, True_mat[:, 16], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 18)
    plt.plot(timezz, WOPRR[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[17], fontsize=13)

    plt.plot(timezz, True_mat[:, 17], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 19)
    plt.plot(timezz, WOPRS[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[18], fontsize=13)

    plt.plot(timezz, True_mat[:, 18], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 20)
    plt.plot(timezz, WOPRT[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[19], fontsize=13)

    plt.plot(timezz, True_mat[:, 19], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 21)
    plt.plot(timezz, WOPRU[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[20], fontsize=13)

    plt.plot(timezz, True_mat[:, 20], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 22)
    plt.plot(timezz, WOPRV[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[21], fontsize=13)

    plt.plot(timezz, True_mat[:, 21], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.savefig(
        "Oil_" + Namesz + ".png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    plt.subplot(5, 5, 1)
    plt.plot(timezz, WWPRA[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[0], fontsize=13)

    plt.plot(timezz, True_mat[:, 22], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 2)
    plt.plot(timezz, WWPRB[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[1], fontsize=13)

    plt.plot(timezz, True_mat[:, 23], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 3)
    plt.plot(timezz, WWPRC[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[2], fontsize=13)

    plt.plot(timezz, True_mat[:, 24], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 4)
    plt.plot(timezz, WWPRD[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[3], fontsize=13)

    plt.plot(timezz, True_mat[:, 25], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 5)
    plt.plot(timezz, WWPRE[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[4], fontsize=13)

    plt.plot(timezz, True_mat[:, 26], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 6)
    plt.plot(timezz, WWPRF[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[5], fontsize=13)

    plt.plot(timezz, True_mat[:, 27], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 7)
    plt.plot(timezz, WWPRG[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[6], fontsize=13)

    plt.plot(timezz, True_mat[:, 28], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 8)
    plt.plot(timezz, WWPRH[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[7], fontsize=13)

    plt.plot(timezz, True_mat[:, 29], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 9)
    plt.plot(timezz, WWPRI[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[8], fontsize=13)

    plt.plot(timezz, True_mat[:, 30], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 10)
    plt.plot(timezz, WWPRJ[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[9], fontsize=13)

    plt.plot(timezz, True_mat[:, 31], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 11)
    plt.plot(timezz, WWPRK[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[10], fontsize=13)

    plt.plot(timezz, True_mat[:, 32], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 12)
    plt.plot(timezz, WWPRL[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[11], fontsize=13)

    plt.plot(timezz, True_mat[:, 33], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 13)
    plt.plot(timezz, WWPRM[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[12], fontsize=13)

    plt.plot(timezz, True_mat[:, 34], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 14)
    plt.plot(timezz, WWPRN[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[13], fontsize=13)

    plt.plot(timezz, True_mat[:, 35], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 15)
    plt.plot(timezz, WWPRO[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[14], fontsize=13)

    plt.plot(timezz, True_mat[:, 36], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 16)
    plt.plot(timezz, WWPRP[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[15], fontsize=13)

    plt.plot(timezz, True_mat[:, 37], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 17)
    plt.plot(timezz, WWPRQ[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[16], fontsize=13)

    plt.plot(timezz, True_mat[:, 38], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 18)
    plt.plot(timezz, WWPRR[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[17], fontsize=13)

    plt.plot(timezz, True_mat[:, 39], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 19)
    plt.plot(timezz, WWPRS[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[18], fontsize=13)

    plt.plot(timezz, True_mat[:, 40], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 20)
    plt.plot(timezz, WWPRT[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[19], fontsize=13)

    plt.plot(timezz, True_mat[:, 41], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 21)
    plt.plot(timezz, WWPRU[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13)
    plt.title(columns[20], fontsize=13)

    plt.plot(timezz, True_mat[:, 42], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 22)
    plt.plot(timezz, WWPRV[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13)
    plt.title(columns[21], fontsize=13)

    plt.plot(timezz, True_mat[:, 43], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.savefig(
        "Water_" + Namesz + ".png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    plt.subplot(5, 5, 1)
    plt.plot(timezz, WGPRA[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[0], fontsize=13)

    plt.plot(timezz, True_mat[:, 44], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 2)
    plt.plot(timezz, WGPRB[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[1], fontsize=13)

    plt.plot(timezz, True_mat[:, 45], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 3)
    plt.plot(timezz, WGPRC[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[2], fontsize=13)

    plt.plot(timezz, True_mat[:, 46], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 4)
    plt.plot(timezz, WGPRD[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[3], fontsize=13)

    plt.plot(timezz, True_mat[:, 47], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 5)
    plt.plot(timezz, WGPRE[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[4], fontsize=13)

    plt.plot(timezz, True_mat[:, 48], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 6)
    plt.plot(timezz, WGPRF[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[5], fontsize=13)

    plt.plot(timezz, True_mat[:, 49], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 7)
    plt.plot(timezz, WGPRG[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[6], fontsize=13)

    plt.plot(timezz, True_mat[:, 50], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 8)
    plt.plot(timezz, WGPRH[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[7], fontsize=13)

    plt.plot(timezz, True_mat[:, 51], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 9)
    plt.plot(timezz, WGPRI[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[8], fontsize=13)

    plt.plot(timezz, True_mat[:, 52], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 10)
    plt.plot(timezz, WGPRJ[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[9], fontsize=13)

    plt.plot(timezz, True_mat[:, 53], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 11)
    plt.plot(timezz, WGPRK[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(bbl/day)$", fontsize=13)
    plt.title(columns[10], fontsize=13)

    plt.plot(timezz, True_mat[:, 54], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 12)
    plt.plot(timezz, WGPRL[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[11], fontsize=13)

    plt.plot(timezz, True_mat[:, 55], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 13)
    plt.plot(timezz, WGPRM[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[12], fontsize=13)

    plt.plot(timezz, True_mat[:, 56], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 14)
    plt.plot(timezz, WGPRN[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[13], fontsize=13)

    plt.plot(timezz, True_mat[:, 57], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 15)
    plt.plot(timezz, WGPRO[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[14], fontsize=13)

    plt.plot(timezz, True_mat[:, 58], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 16)
    plt.plot(timezz, WGPRP[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[15], fontsize=13)

    plt.plot(timezz, True_mat[:, 59], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 17)
    plt.plot(timezz, WGPRQ[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(bbl/day)$", fontsize=13)
    plt.title(columns[16], fontsize=13)

    plt.plot(timezz, True_mat[:, 60], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 18)
    plt.plot(timezz, WGPRR[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[17], fontsize=13)

    plt.plot(timezz, True_mat[:, 61], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 19)
    plt.plot(timezz, WWPRS[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[18], fontsize=13)

    plt.plot(timezz, True_mat[:, 62], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 20)
    plt.plot(timezz, WWPRT[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[19], fontsize=13)

    plt.plot(timezz, True_mat[:, 63], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 21)
    plt.plot(timezz, WGPRU[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[20], fontsize=13)

    plt.plot(timezz, True_mat[:, 64], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.subplot(5, 5, 22)
    plt.plot(timezz, WWPRV[:, :Ne], color="grey", lw="2", label="Realisations")
    plt.xlabel("Time (days)", fontsize=13)
    plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13)
    plt.title(columns[21], fontsize=13)

    plt.plot(timezz, True_mat[:, 65], color="red", lw="2", label="True model")
    plt.axvline(x=1500, color="black", linestyle="--")
    handles, labels = plt.gca().get_legend_handles_labels()  # get all the labels
    by_label = OrderedDict(
        zip(labels, handles)
    )  # OrderedDict forms a ordered dictionary of all the labels and handles, but it only accepts one argument, and zip does this
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.savefig(
        "Gas_" + Namesz + ".png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


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


def De_correlate_ensemble(nx, ny, nz, Ne, High_K, Low_K):
    filename = "../PACKETS/Ganensemble.pkl.gz"

    with gzip.open(filename, "rb") as f2:
        mat1 = pickle.load(f2)
    mat = mat1["permeability"]
    ini_ensemblef = mat
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


def write_RSM(data, Time, Name):
    # Generate a random numpy array of shape 100x66 for demonstration
    # data = np.random.rand(100, 66)

    # Create groups and individual column labels
    groups = ["WOPR(bbl/day)", "WWPR(bbl/day)", "WGPR(scf/day)"]
    columns = [
        "B_1BH",
        "B_1H",
        "B_2H",
        "B_3H",
        "B_4BH",
        "B_4DH",
        "B_4H",
        "D_1CH",
        "D_1H",
        "D_2H",
        "D_3AH",
        "D_3BH",
        "D_4AH",
        "D_4H",
        "E_1H",
        "E_2AH",
        "E_2H",
        "E_3AH",
        "E_3CH",
        "E_3H",
        "E_4AH",
        "K_3H",
    ]

    # Generate hierarchical column headers
    headers = pd.MultiIndex.from_product([groups, columns])

    # Create a Pandas DataFrame from the numpy array with the specified column headers
    df = pd.DataFrame(data, columns=headers)
    df.insert(0, "Time(days)", Time)

    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(Name + ".xlsx", engine="xlsxwriter") as writer:
        workbook = writer.book

        # Add a new worksheet
        worksheet = workbook.add_worksheet("Sheet1")
        writer.sheets["Sheet1"] = worksheet

        # Define the format for the header: bold and centered
        header_format = workbook.add_format({"bold": True, "align": "center"})

        # Write the 'Time(days)' header with the format
        worksheet.write(0, 0, "Time(days)", header_format)

        # Write the second-level column headers
        col = 1
        for sub_col in columns * len(groups):
            worksheet.write(1, col, sub_col)
            col += 1

        # Manually merge cells for each top-level column label and apply the format
        col = 1
        for group in groups:
            end_col = col + len(columns) - 1
            worksheet.merge_range(0, col, 0, end_col, group, header_format)
            col = end_col + 1

        # Write the data rows, starting from row 2 (which is the third row in zero-based index)
        # to ensure there is no empty row between the second-level column headers and the data
        for row_num, row_data in enumerate(df.values):
            worksheet.write_row(row_num + 2, 0, row_data)

        # Write 'Time(days)' column
        time_data = np.arange(1, 101)
        for row_num, time_val in enumerate(time_data):
            worksheet.write(row_num + 2, 0, time_val)

        # Set the column width for better visibility (optional)
        worksheet.set_column(0, 0, 12)  # 'Time(days)' column
        worksheet.set_column(1, col, 10)  # Data columns


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


def Remove_True(enss):
    # enss is the very large ensemble
    # Used the last one as true model
    return np.reshape(enss[:, locc - 1], (-1, 1), "F")


def Trim_ensemble(enss):
    return np.delete(enss, locc - 1, 1)  # delete last column


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


def shuffle(x, axis=0):
    n_axis = len(x.shape)
    t = np.arange(n_axis)
    t[0] = axis
    t[axis] = 0
    xt = np.transpose(x.copy(), t)
    np.random.shuffle(xt)
    shuffled_x = np.transpose(xt, t)
    return shuffled_x


def Get_new_K(Low_K, High_K, LogS1):
    newK = (High_K * LogS1) + (1 - LogS1) * Low_K
    return newK


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


def adaptive_rho(True_dataa, simDatafinal, rho):
    # Parameters for adaptive inflation
    increase_factor = 1.05
    decrease_factor = 0.95
    threshold = 0.1  # Example threshold, adjust as necessary

    # Compute innovation
    innovation = True_dataa - simDatafinal
    MSE = cp.mean(abs(innovation))

    # Adjust rho based on innovation
    if MSE > threshold:
        rho *= increase_factor
    else:
        rho *= decrease_factor
    return rho


def extract_non_zeros(mat):
    indices = np.where(mat[:, 0] >= 0)[0]
    return mat[indices, :], indices


def place_back(extracted, indices, shape):
    result = cp.ones(shape) * -1
    for i, index in enumerate(indices):
        result[index, :] = extracted[i]
    return result.get()


def fast_gaussian(dimension, Sdev, Corr):
    """
    Generates random vector from distribution satisfying Gaussian
    variogram in 2-d.

    Input:
    - dimension: Dimension of grid
    - Sdev: Standard deviation
    - Corr: Correlation length, in units of block length.
            Corr may be replaced with a vector of length 2 with
            correlation length in x- and y- direction.

    Output:
    - x: Random vector.
    """

    # Ensure dimension is a 1D numpy array
    dimension = np.array(dimension).flatten()
    m = dimension[0]

    if len(dimension) == 1:
        n = m
    elif len(dimension) == 2:
        n = dimension[1]
    else:
        raise ValueError(
            "FastGaussian: Wrong input, dimension should have length at most 2"
        )

    # mxn = m * n

    # Compute variance
    if np.max(np.size(Sdev)) > 1:  # check input
        variance = 1
    else:
        variance = Sdev  # the variance will come out through the kronecker product.

    # Ensure Corr is a 1D numpy array

    # Corr = np.array(Corr).flatten()

    if len(Corr) == 1:
        Corr = np.array([Corr[0], Corr[0]])
    elif len(Corr) > 2:
        raise ValueError("FastGaussian: Wrong input, Corr should have length at most 2")

    # Generate the covariance matrix for one layer
    dist = np.arange(0, m) / Corr[0]
    T = scipy.linalg.toeplitz(dist)

    T = variance * np.exp(-(T**2)) + 1e-10 * np.eye(m)

    # Cholesky decomposition for one layer:
    cholT = np.linalg.cholesky(T)

    # Generate the covariance matrix for the second layer:
    # To save time - use a copy if possible:
    if Corr[0] == Corr[1] and n == m:
        cholT2 = cholT
    else:
        # Same as for the first dimension:
        dist2 = np.arange(0, n) / Corr[1]
        T2 = scipy.linalg.toeplitz(dist2)
        T2 = variance * np.exp(-(T2**2)) + 1e-10 * np.eye(n)
        cholT2 = np.linalg.cholesky(T2)

    # Draw a random variable:
    x = np.random.randn(m * n)

    # Adjust to get the correct covariance matrix,
    # Applying the matrix transformation:
    x = np.dot(cholT.T, np.dot(x.reshape(m, n), cholT2))

    # Reshape back
    x = x.flatten()

    if np.max(np.size(Sdev)) > 1:
        if np.min(np.shape(Sdev)) == 1 and len(Sdev) == len(x):
            x = Sdev * x
        else:
            raise ValueError("FastGaussian: Inconsistent dimension of Sdev")

    return x


def get_shape(t):
    shape = []
    while isinstance(t, tuple):
        shape.append(len(t))
        t = t[0]
    return tuple(shape)


def NorneInitialEnsemble(ensembleSize=100, randomNumber=1.2345e5):
    # set random generator
    np.random.seed(int(randomNumber))

    # number of ensemble members
    N = ensembleSize

    # set geostatistical parameters and initialize
    norne = NorneGeostat()
    # Assuming norne is a dictionary where properties of the Norne field are stored

    # mask for active gridcells
    A = norne["actnum"]

    # reservoir dimension
    D = norne["dim"]

    # total number of gridcells
    N_F = D[0] * D[1] * D[2]

    # initialize ensemble

    # mean values for poro, log-permx and ntg
    M = [norne["poroMean"], norne["permxLogMean"], 0.6]

    # std for poro, permx and ntg
    S = [norne["poroStd"], norne["permxStd"], norne["ntgStd"]]

    # help variable (layerwise actnum)
    A_L = [A[i : i + D[1] * D[2]] for i in range(0, len(A), D[1] * D[2])]
    A_L = np.array(A_L)

    # mean and std for multz

    # mean and std for multflt and multreg
    M_MF = 0.6
    S_MF = norne["multfltStd"]

    # mean correlation lengths (ranges)
    C = [norne["poroRange"], norne["permxRange"], norne["ntgRange"]]

    # std for correlation lengths

    C_S = 2

    # Poro / Permx correlation
    R1 = norne["poroPermxCorr"]
    ensembleperm = np.zeros((N_F, N))
    ensemblefault = np.zeros((53, N))
    ensembleporo = np.zeros((N_F, N))

    indices = np.where(A == 1)
    for i in range(N):
        # multz
        A_MZ = A_L[:, [0, 7, 10, 11, 14, 17]]  # Adjusted indexing to 0-based
        A_MZ = A_MZ.flatten()

        X = M_MF + S_MF * np.random.randn(53)
        ensemblefault[:, i] = X

        # poro
        C = np.array(C)
        X1 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[0], C_S)[0]
        X1 = X1.reshape(-1, 1)

        ensembleporo[indices, i] = (M[0] + S[0] * X1[indices]).ravel()

        # permx
        X2 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[1], C_S)[0]
        X2 = X2.reshape(-1, 1)
        X = R1 * X1 + np.sqrt(1 - R1**2) * X2
        indices = np.where(A == 1)
        ensembleperm[indices, i] = np.exp((M[1] + S[1] * X[indices]).ravel())  #
    return ensembleperm, ensembleporo, ensemblefault


# Further adjustments will be needed based on dependencies like GaussianWithVariableParameters and AdjustVariableWithInBounds.


def gaussian_with_variable_parameters(
    field_dim, mean_value, sdev, mean_corr_length, std_corr_length
):
    """
    Setup a Gaussian field with correlation length drawn from a normal distribution.
    The horizontal layers are generated independently.

    Parameters:
    - field_dim        : Tuple. Dimension of the field.
    - mean_value       : Numpy array. The mean value of the field.
    - sdev             : Numpy array or float. Standard deviation of the field.
    - mean_corr_length : Float. Mean correlation length.
    - std_corr_length  : Float. Standard deviation of the correlation length.

    Returns:
    - x                : Numpy array. The generated field.
    - corr_length      : Numpy array or float. The drawn correlation length.
    """

    corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)

    if len(field_dim) < 3:
        x = mean_value + fast_gaussian(field_dim, sdev, corr_length)
    else:
        layer_dim = np.prod(field_dim[:2])
        x = np.copy(mean_value)

        if np.isscalar(sdev):
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev, corr_length
                )
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
        else:
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev[idx_range], corr_length
                )
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)

    return x, corr_length


# Placeholder functions for the ones referred to in the MATLAB code:


def plot_and_save(
    kk,
    dt,
    pree,
    wats,
    oilss,
    gasss,
    nx,
    ny,
    nz,
    N_injw,
    N_pr,
    N_injg,
    injectors,
    producers,
    gass,
    effectiveuse,
):
    current_time = dt[kk]
    Time_vector[kk] = current_time

    f_3 = plt.figure(figsize=(20, 20), dpi=200)

    look = (pree[0, kk, :, :, :]) * effectiveuse
    ax1 = f_3.add_subplot(2, 2, 1, projection="3d")
    Plot_PhysicsNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "pressure PhysicsNeMo",
        injectors,
        producers,
        gass,
    )

    look = (wats[0, kk, :, :, :]) * effectiveuse
    ax2 = f_3.add_subplot(2, 2, 2, projection="3d")
    Plot_PhysicsNeMo(
        ax2,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "water PhysicsNeMo",
        injectors,
        producers,
        gass,
    )

    look = oilss[0, kk, :, :, :]
    look = look * effectiveuse
    ax3 = f_3.add_subplot(2, 2, 3, projection="3d")
    Plot_PhysicsNeMo(
        ax3,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "oil PhysicsNeMo",
        injectors,
        producers,
        gass,
    )

    look = (gasss[0, kk, :, :, :]) * effectiveuse
    ax4 = f_3.add_subplot(2, 2, 4, projection="3d")
    Plot_PhysicsNeMo(
        ax4,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "gas PhysicsNeMo",
        injectors,
        producers,
        gass,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    tita = "Timestep --" + str(current_time) + " days"
    plt.suptitle(tita, fontsize=16)

    # Return the kk and figure
    return kk, f_3


def add_gnoise(Ytrue, SIGMA, SQ=None):
    """
    Add noise, normally distributed, with covariance given by SIGMA.

    Parameters:
    - Ytrue    Original signal.
    - SIGMA    Specification of covariance matrix of noise. May be entered as scalar, vector or full matrix.
               If SIGMA is a vector, then it is interpreted as the covariance matrix is diag(SIGMA).
    - SQ       If provided, determine whether SIGMA or SIGMA*SIGMA' is used as the covariance matrix.
               If the square root of the covariance matrix has already been calculated previously, work may
               be saved by setting SQ to 1.

    Returns:
    - Y        Signal with noise added.
    - RTSIGMA  The square root of SIGMA; RTSIGMA @ RTSIGMA.T = SIGMA.
               (Helpful if it is cumbersome to compute).
    """

    try:
        if SQ is not None and SQ == 1:
            # Use SIGMA*SIGMA' as covariance matrix
            RTSIGMA = SIGMA
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                # SIGMA is a scalar or vector
                error = RTSIGMA * np.random.randn(1)
            else:
                error = RTSIGMA @ np.random.randn(RTSIGMA.shape[1], 1)
        else:
            # Use SIGMA as covariance matrix
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                # SIGMA is entered as a scalar or a vector
                RTSIGMA = np.sqrt(SIGMA)
                error = RTSIGMA * np.random.randn(*Ytrue.shape)
            else:
                # The matrix must be transposed.
                try:
                    RTSIGMA = np.linalg.cholesky(SIGMA).T
                except np.linalg.LinAlgError:
                    print("Problem with Cholesky factorization")
                    RTSIGMA = np.sqrtm(SIGMA).real
                    print("Finally - we got a square root!")

                error = RTSIGMA @ np.random.randn(*Ytrue.shape)

        # Add the noise:
        Y = Ytrue + error.flatten()

    except Exception as e:
        print("Error in AddGnoise")
        raise e

    return Y, RTSIGMA


def adjust_variable_within_bounds(variable, lowerbound=None, upperbound=None):
    """
    Adjust variable such that lowerbound <= variable(i) <= upperbound.

    Parameters:
    - variable: numpy array. Variables (or an ensemble of variable samples) to be checked and truncated (if necessary).
    - lowerbound: Scalar or numpy array. Lower bound(s) for the variables to be checked.
    - upperbound: Scalar or numpy array. Upper bound(s) for the variables to be checked.

    Returns:
    - variable: Variables after check/truncation.
    - n: Number of truncations.
    """

    if lowerbound is None and upperbound is None:
        raise ValueError("At least one of lowerbound or upperbound must be provided.")

    n = 0
    ne = variable.shape[1]

    if lowerbound is not None:
        if np.isscalar(lowerbound):
            n += np.sum(variable < lowerbound)
            variable[variable < lowerbound] = lowerbound
        else:
            lowerbound_repeated = np.tile(lowerbound.reshape(-1, 1), (1, ne))
            n += np.sum(variable < lowerbound_repeated)
            variable[variable < lowerbound_repeated] = lowerbound_repeated[
                variable < lowerbound_repeated
            ]

    if upperbound is not None:
        if np.isscalar(upperbound):
            n += np.sum(variable > upperbound)
            variable[variable > upperbound] = upperbound
        else:
            upperbound_repeated = np.tile(upperbound.reshape(-1, 1), (1, ne))
            n += np.sum(variable > upperbound_repeated)
            variable[variable > upperbound_repeated] = upperbound_repeated[
                variable > upperbound_repeated
            ]

    return variable, n


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


def NorneGeostat():
    norne = {}

    dim = np.array([46, 112, 22])
    ldim = dim[0] * dim[1]
    norne["dim"] = dim

    # actnum
    # act = pd.read_csv('../Norne_Initial_ensemble/ACTNUM_0704.prop', skiprows=8,nrows = 2472, sep='\s+', header=None)
    act = read_until_line("../Necessaryy/ACTNUM_0704.prop", line_num=2465, skip=1)
    act = act.T
    act = np.reshape(act, (-1,), "F")
    norne["actnum"] = act

    # porosity
    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])
    file_path = "../Necessaryy/porosity.dat"
    p = read_until_line(file_path, line_num=113345, skip=1)
    p = p[act != 0]

    for nr in range(int(dim[2])):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = p[values_range_start:values_range_end]

        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)

    norne["poroMean"] = p
    norne["poroLayerMean"] = meanv
    norne["poroLayerStd"] = stdv
    norne["poroStd"] = 0.05
    norne["poroLB"] = 0.1
    norne["poroUB"] = 0.4
    norne["poroRange"] = 26

    # permeability

    # Permeability

    k = read_until_line("../Necessaryy/permx.dat", line_num=113345, skip=1)
    k = np.log(k)
    k = k[act != 0]

    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])

    for nr in range(int(dim[2])):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = k[values_range_start:values_range_end]

        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)

    norne["permxLogMean"] = k
    norne["permxLayerLnMean"] = meanv
    norne["permxLayerStd"] = stdv
    norne["permxStd"] = 1
    norne["permxLB"] = 0.1
    norne["permxUB"] = 10
    norne["permxRange"] = 26

    # Correlation between layers

    corr_with_next_layer = np.zeros(dim[2] - 1)

    for nr in range(dim[2] - 1):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)

        index2_start = ldim * (nr + 1)
        index2_end = ldim * (nr + 2)

        act_layer1 = act[index_start:index_end]
        act_layer2 = act[index2_start:index2_end]

        active = act_layer1 * act_layer2

        values1_range_start = int(np.sum(act[:index_start]))
        values1_range_end = int(np.sum(act[:index_end]))
        values1 = np.concatenate(
            (
                k[values1_range_start:values1_range_end],
                p[values1_range_start:values1_range_end],
            )
        )

        values2_range_start = int(np.sum(act[:index2_start]))
        values2_range_end = int(np.sum(act[:index2_end]))
        values2 = np.concatenate(
            (
                k[values2_range_start:values2_range_end],
                p[values2_range_start:values2_range_end],
            )
        )

        v1 = np.concatenate((act_layer1, act_layer1))
        v1[v1 == 1] = values1.flatten()

        v2 = np.concatenate((act_layer2, act_layer2))
        v2[v2 == 1] = values2.flatten()

        active_full = np.concatenate((active, active))
        co = np.corrcoef(v1[active_full == 1], v2[active_full == 1])

        corr_with_next_layer[nr] = co[0, 1]

    norne["corr_with_next_layer"] = corr_with_next_layer.T

    # Correlation between porosity and permeability
    norne["poroPermxCorr"] = 0.7

    norne["poroNtgCorr"] = 0.6
    norne["ntgStd"] = 0.1
    norne["ntgLB"] = 0.01
    norne["ntgUB"] = 1
    norne["ntgRange"] = 26

    # rel-perm end-point scaling
    norne["krwMean"] = 1.15
    norne["krwLB"] = 0.8
    norne["krwUB"] = 1.5
    norne["krgMean"] = 0.9
    norne["krgLB"] = 0.8
    norne["krgUB"] = 1

    # oil-water contact
    norne["owcMean"] = np.array([2692.0, 2585.5, 2618.0, 2400.0, 2693.3])
    norne["owcLB"] = norne["owcMean"] - 10
    norne["owcUB"] = norne["owcMean"] + 10

    # region multipliers
    norne["multregtLogMean"] = np.log10(np.array([0.0008, 0.1, 0.05]))
    norne["multregtStd"] = 0.5
    norne["multregtLB"] = -5
    norne["multregtUB"] = 0

    # z-multipliers
    z_means = [-2, -1.3, -2, -2, -2, -2]
    z_stds = [0.5, 0.5, 0.5, 0.5, 1, 1]
    for i, (mean_, std_) in enumerate(zip(z_means, z_stds), start=1):
        norne[f"z{i}Mean"] = mean_
        norne[f"z{i}Std"] = std_
    norne["zLB"] = -4
    norne["zUB"] = 0
    norne["multzRange"] = 26

    # fault multipliers
    norne["multfltStd"] = 0.5
    norne["multfltLB"] = -5
    norne["multfltUB"] = 2

    return norne


def remove_rows(matrix, indices_to_remove):
    """
    Remove specified rows from a NumPy array.

    Parameters:
    - matrix: NumPy array
      The input array from which rows will be removed.
    - indices_to_remove: list of int
      List of row indices to be removed from the array.

    Returns:
    - modified_matrix: NumPy array
      The modified array with specified rows removed.
    """

    matrix = np.delete(matrix, indices_to_remove, axis=0)
    # indices_to_remove = sorted(indices_to_remove, reverse=True)  # Sort in reverse order
    # for index in indices_to_remove:
    #     matrix = np.delete(matrix, index, axis=0)
    return matrix


def Localisation(c, nx, ny, nz, N):
    ## Get the localization for all the wells

    A = np.zeros((nx, ny, nz))
    for jj in range(nz):
        A[14, 30, :] = 1
        A[9, 31, :] = 1
        A[13, 33, :] = 1
        A[8, 36, :] = 1
        A[8, 45, :] = 1
        A[9, 28, :] = 1
        A[9, 23, :] = 1
        A[21, 21, :] = 1
        A[13, 27, :] = 1
        A[18, 37, :] = 1
        A[18, 53, :] = 1
        A[15, 65, :] = 1
        A[24, 36, :] = 1
        A[18, 53, :] = 1
        A[11, 71, :] = 1
        A[17, 67, :] = 1
        A[12, 66, :] = 1
        A[37, 97, :] = 1
        A[6, 63, :] = 1
        A[14, 75, :] = 1
        A[12, 66, :] = 1
        A[10, 27, :] = 1
        A[10, 34, :] = 1
        A[25, 43, :] = 1
        A[22, 14, :] = 1
        A[8, 12, :] = 1
        A[11, 84, :] = 1
        A[17, 82, :] = 1
        A[5, 56, :] = 1
        A[35, 67, :] = 1
        A[28, 50, :] = 1

    print("      Calculate the Euclidean distance function to the 22 producer wells")
    lf = np.reshape(A, (nx, ny, nz), "F")
    young = np.zeros((int(nx * ny * nz / nz), nz))
    for j in range(nz):
        sdf = lf[:, :, j]
        (usdf, IDX) = spndmo.distance_transform_edt(
            np.logical_not(sdf), return_indices=True
        )
        usdf = np.reshape(usdf, (int(nx * ny * nz / nz)), "F")
        young[:, j] = usdf

    sdfbig = np.reshape(young, (nx * ny * nz, 1), "F")
    sdfbig1 = abs(sdfbig)
    z = sdfbig1
    ## the value of the range should be computed accurately.

    c0OIL1 = np.zeros((nx * ny * nz, 1))

    print("      Computing the Gaspari-Cohn coefficent")
    for i in range(nx * ny * nz):
        if 0 <= z[i, :] or z[i, :] <= c:
            c0OIL1[i, :] = (
                -0.25 * (z[i, :] / c) ** 5
                + 0.5 * (z[i, :] / c) ** 4
                + 0.625 * (z[i, :] / c) ** 3
                - (5.0 / 3.0) * (z[i, :] / c) ** 2
                + 1
            )

        elif z < 2 * c:
            c0OIL1[i, :] = (
                (1.0 / 12.0) * (z[i, :] / c) ** 5
                - 0.5 * (z[i, :] / c) ** 4
                + 0.625 * (z[i, :] / c) ** 3
                + (5.0 / 3.0) * (z[i, :] / c) ** 2
                - 5 * (z[i, :] / c)
                + 4
                - (2.0 / 3.0) * (c / z[i, :])
            )

        elif c <= z[i, :] or z[i, :] <= 2 * c:
            c0OIL1[i, :] = -5 * (z[i, :] / c) + 4 - 0.667 * (c / z[i, :])

        else:
            c0OIL1[i, :] = 0

    c0OIL1[c0OIL1 < 0] = 0
    schur = c0OIL1
    Bsch = np.tile(schur, (1, N))

    yoboschur = np.ones((2 * nx * ny * nz + 53, N))

    yoboschur[: nx * ny * nz, :] = Bsch
    yoboschur[nx * ny * nz : 2 * nx * ny * nz, 0:] = Bsch

    return yoboschur


def process_step(
    kk,
    steppi,
    dt,
    pressure,
    effectiveuse,
    Swater,
    Soil,
    Sgas,
    nx,
    ny,
    nz,
    N_injw,
    N_pr,
    N_injg,
    injectors,
    producers,
    gass,
    fol,
    fol1,
):
    os.chdir(fol)
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, kk - 1, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)

    current_time = dt[kk]
    Time_vector[kk] = current_time

    f_3 = plt.figure(figsize=(20, 20), dpi=200)

    look = ((pressure[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]
    ax1 = f_3.add_subplot(2, 2, 1, projection="3d")
    Plot_PhysicsNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "pressure PhysicsNeMo",
        injectors,
        producers,
        gass,
    )

    look = ((Swater[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]
    ax2 = f_3.add_subplot(2, 2, 2, projection="3d")
    Plot_PhysicsNeMo(
        ax2,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "water PhysicsNeMo",
        injectors,
        producers,
        gass,
    )

    look = Soil[0, kk, :, :, :]
    look = (look * effectiveuse)[:, :, ::-1]

    ax3 = f_3.add_subplot(2, 2, 3, projection="3d")
    Plot_PhysicsNeMo(
        ax3,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "oil PhysicsNeMo",
        injectors,
        producers,
        gass,
    )

    look = ((Sgas[0, kk, :, :, :]) * effectiveuse)[:, :, ::-1]
    ax4 = f_3.add_subplot(2, 2, 4, projection="3d")
    Plot_PhysicsNeMo(
        ax4,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        N_injg,
        "gas PhysicsNeMo",
        injectors,
        producers,
        gass,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(current_time) + " days"
    plt.suptitle(tita, fontsize=16)
    # plt.savefig('Dynamic' + str(int(kk)))
    plt.savefig("Dynamic" + str(int(kk)))
    plt.clf()
    plt.close()
    os.chdir(fol1)


##############################################################################
# configs
##############################################################################
text = """
 _____                              ______          _     _                
|_   _|                             | ___ \        | |   | |               
  | | _ ____   _____ _ __ ___  ___  | |_/ _ __ ___ | |__ | | ___ _ __ ___  
  | || '_ \ \ / / _ | '__/ __|/ _ \ |  __| '__/ _ \| '_ \| |/ _ | '_ ` _ \ 
 _| || | | \ V |  __| |  \__ |  __/ | |  | | | (_) | |_) | |  __| | | | | |
 \___|_| |_|\_/ \___|_|  |___/\___| \_|  |_|  \___/|_.__/|_|\___|_| |_| |_|
"""
print(text)
oldfolder = os.getcwd()
os.chdir(oldfolder)
cur_dir = oldfolder


# njobs = int((multiprocessing.cpu_count() // 4) - 1)
njobs = 3
num_cores = njobs

print("------------------Download pre-trained models------------------------")
if not os.path.exists("../PACKETS"):
    os.makedirs("../PACKETS")
else:
    pass


# default = int(input('Select 1 = use default | 2 = Use user defined \n'))


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
    print(" Default configuration selected, sit back and relax.....")
else:
    pass
TEMPLATEFILE = {}

print("")

surrogate = 1
print("PINO surrogate for forwarding")


TEMPLATEFILE["Surrogate model"] = "PINO (PhysicsNeMo implementation)"

# load data


fname = "conf/config_PINO.yaml"


sizq = 1e4

exper = sio.loadmat("../PACKETS/exper.mat")
experts = exper["expert"]
mat = sio.loadmat("../PACKETS/conversions.mat")
minK = mat["minK"]
maxK = mat["maxK"]
minT = mat["minT"]
maxT = mat["maxT"]
minP = mat["minP"]
maxP = mat["maxP"]
minQw = mat["minQW"]
maxQw = mat["maxQW"]
minQg = mat["minQg"]
maxQg = mat["maxQg"]
minQ = mat["minQ"]
maxQ = mat["maxQ"]
min_inn_fcn = mat["min_inn_fcn"]
max_inn_fcn = mat["max_inn_fcn"]
min_out_fcn = mat["min_out_fcn"]
max_out_fcn = mat["max_out_fcn"]
steppi = int(mat["steppi"])
# print(steppi)
steppi_indices = mat["steppi_indices"].flatten()

target_min = 0.01
target_max = 1


plan = read_yaml(fname)
nx = plan["custom"]["PROPS"]["nx"]
ny = plan["custom"]["PROPS"]["ny"]
nz = plan["custom"]["PROPS"]["nz"]
effective = np.genfromtxt("../NORNE/actnum.out", dtype="float")
effec = np.reshape(effective, (-1, 1), "F")

# True_K = np.genfromtxt('../NORNE/rossmary.GRDECL')
# True_P = np.genfromtxt('../NORNE/rossmaryporo.GRDECL')


True_K = np.genfromtxt("../NORNE/sgsim.out").astype(np.float32)[:, 19]
True_P = np.genfromtxt("../NORNE/sgsimporo.out").astype(np.float32)[:, 19]
True_fault = np.genfromtxt("../NORNE/faultensemble.dat").astype(np.float32)[:, 19]


effectiveuse = np.reshape(effective, (nx, ny, nz), "F")

injectors = plan["custom"]["WELLSPECS"]["water_injector_wells"]
producers = plan["custom"]["WELLSPECS"]["producer_wells"]
gass = plan["custom"]["WELLSPECS"]["gas_injector_wells"]

N_injw = len(
    plan["custom"]["WELLSPECS"]["water_injector_wells"]
)  # Number of water injectors
N_pr = len(plan["custom"]["WELLSPECS"]["producer_wells"])  # Number of producers
N_injg = len(
    plan["custom"]["WELLSPECS"]["gas_injector_wells"]
)  # Number of gas injectors


oldfolder2 = os.getcwd()


# training
# LUB, HUB = 1e-3,1 # Pressure rescale
LUB, HUB = target_min, target_max  # Perm rescale limits
aay, bby = minK, maxK  # Perm range mD limits


# if seed is None:
#     seed = random.randint(1, 10000)
seed = 1  # 1 is the best
print("Random Seed: ", seed)


ra.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:  # Choose GPU 1 (index 1)
        device = torch.device("cuda:0")
    else:  # If there's only one GPU or no GPUs, choose the first one (index 0)
        device = torch.device("cuda:0")
else:  # If CUDA is not available, use the CPU
    raise RuntimeError("No GPU found. Please run on a system with a GPU.")
torch.cuda.set_device(device)


input_channel = 5  # [K,phi,FTM,Pini,Sini]
output_channel = 1 * steppi


print("")
Nnnena = 1


TEMPLATEFILE["Kalman update"] = "Exotic"


print("")


TEMPLATEFILE["weighting"] = "Non Weighted innovation"


print("")

if DEFAULT == 1:
    do_localisation = 1
    print("Doing covariance localisation\n")
else:
    do_localisation = None
    while True:
        do_localisation = int(
            input(
                "Do covariance localisation duirng Kalman update:\n\
    1 = Yes\n\
    2 = No\n"
            )
        )
        if (do_localisation > 2) or (do_localisation < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break
if do_localisation == 1:
    TEMPLATEFILE["Covariance localisation"] = "Covariance localisaion = Yes"
else:
    TEMPLATEFILE["Covariance localisation"] = "Covariance localisaion = No"


print("")
if DEFAULT == 1:
    use_pretrained = 2
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

TEMPLATEFILE["Use pretrained model"] = use_pretrained
print("")
print("*******************Load the trained Forward models*******************")

decoder1 = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
modelP = FNOArch(
    [
        Key("perm", size=1),
        Key("Phi", size=1),
        Key("Pini", size=1),
        Key("Swini", size=1),
        Key("fault", size=1),
    ],
    dimension=3,
    decoder_net=decoder1,
)

decoder2 = ConvFullyConnectedArch([Key("z", size=32)], [Key("water_sat", size=steppi)])
modelW = FNOArch(
    [
        Key("perm", size=1),
        Key("Phi", size=1),
        Key("Pini", size=1),
        Key("Swini", size=1),
        Key("fault", size=1),
    ],
    dimension=3,
    decoder_net=decoder2,
)

decoder3 = ConvFullyConnectedArch([Key("z", size=32)], [Key("gas_sat", size=steppi)])
modelG = FNOArch(
    [
        Key("perm", size=1),
        Key("Phi", size=1),
        Key("Pini", size=1),
        Key("Swini", size=1),
        Key("fault", size=1),
    ],
    dimension=3,
    decoder_net=decoder3,
)


decoder5 = ConvFullyConnectedArch([Key("z", size=32)], [Key("oil_sat", size=steppi)])
modelO = FNOArch(
    [
        Key("perm", size=1),
        Key("Phi", size=1),
        Key("Pini", size=1),
        Key("Swini", size=1),
        Key("fault", size=1),
    ],
    dimension=3,
    decoder_net=decoder5,
)


decoder4 = ConvFullyConnectedArch([Key("z", size=32)], [Key("Y", size=66)])
model_peacemann = FNOArch(
    [Key("X", size=90)],
    fno_modes=13,
    dimension=1,
    padding=20,
    nr_fno_layers=5,
    decoder_net=decoder4,
)


print("-----------------Surrogate Model learned with PINO----------------")
if not os.path.exists(("outputs/Forward_problem_PINO/ResSim/")):
    os.makedirs(("outputs/Forward_problem_PINO/ResSim/"))
else:
    pass


os.chdir("outputs/Forward_problem_PINO/ResSim")
print(" Surrogate model learned with PINO for dynamic properties pressure model")
modelP.load_state_dict(torch.load("fno_forward_model_pressure.0.pth"))
modelP = modelP.to(device)
modelP.eval()


print(" Surrogate model learned with PINO for dynamic properties- water model")
modelW.load_state_dict(torch.load("fno_forward_model_water.0.pth"))
modelW = modelW.to(device)
modelW.eval()


print(" Surrogate model learned with PINO for dynamic properties - Gas model")
modelG.load_state_dict(torch.load("fno_forward_model_gas.0.pth"))
modelG = modelG.to(device)
modelG.eval()


print(" Surrogate model learned with PINO for dynamic properties - Oil model")
modelO.load_state_dict(torch.load("fno_forward_model_oil.0.pth"))
modelO = modelO.to(device)
modelO.eval()


print(" Surrogate model learned with PINO for peacemann well model")
model_peacemann.load_state_dict(torch.load("fno_forward_model_peacemann.0.pth"))
model_peacemann = model_peacemann.to(device)
model_peacemann.eval()
os.chdir(oldfolder)

print("********************Model Loaded*************************************")

print("")
if DEFAULT == 1:
    Trainmoe = 1
    print("Inference peacemann with Mixture of Experts\n")
else:
    Trainmoe = None
    while True:
        Trainmoe = int(
            input(
                "Select 1 = Inference peacemann with FNO | 2 = Inference peacemann with Mixture of Experts \n"
            )
        )
        if (Trainmoe > 2) or (Trainmoe < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break
if Trainmoe == 2:
    TEMPLATEFILE["Peaceman modelling inference"] = (
        "Inference peacemann = Mixture of Experts"
    )
else:
    TEMPLATEFILE["Peaceman modelling inference"] = "Inference peacemann = FNO"
print("")
if Trainmoe == 2:
    print("------------------------------------------------------L-----------")
    print("Using Cluster Classify Regress (CCR) for peacemann model          ")
    print("")
    print("References for CCR include: ")
    print(
        "(1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,\n\
    Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general\n\
    method for learning discontinuous functions.Foundations of Data Science,\n\
    1(2639-8001-2019-4-491):491, 2019.\n"
    )
    print("")
    print(
        "(2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of\n\
    Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.\n"
    )
    print("-----------------------------------------------------------------------")
    pred_type = 1
else:
    pred_type = 1
degg = 3


rho = 1.05

# True model
Truee = True_K
Truee = np.reshape(Truee, (-1,))
aay1 = minK  # np.min(Truee)
bby1 = maxK  # np.max(Truee)

Low_K1, High_K1 = aay1, bby1

perm_high = maxK  # np.asscalar(np.amax(perm,axis=0)) + 300
perm_low = minK  # np.asscalar(np.amin(perm,axis=0))-50

High_P, Low_P = 0.5, 0.05
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
    BASSE = 1
    print("Covarance data noise matrix using percentage of measured value\n")
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
if BASSE == 1:
    TEMPLATEFILE["Covariance matrix generation"] = (
        "Covariance noise matrix generation = data percentage\n"
    )
else:
    TEMPLATEFILE["Covariance matrix generation"] = (
        "Covariance noise matrix generation = constant value\n"
    )

print("")
print("---------------------------------------------------------------------")
print("")
print("--------------------Historical data Measurement----------------------")

# rows_to_remove = list(range(59, 69)) + list(range(189, 199)) + list(range(279, 289))+ list(range(409, 419))\
# + list(range(429, 439)) + list(range(499, 509)) + list(range(619, 639))+ list(range(649, 659))


os.chdir("../NORNE")
timestep = np.genfromtxt(("../NORNE/timestep.out"))
timestep = timestep.astype(int)
os.chdir(oldfolder)

os.chdir("../NORNE")
Time = Get_Time(nx, ny, nz, steppi, steppi_indices, 1)
Time_unie1 = np.zeros((steppi))
for i in range(steppi):
    Time_unie1[i] = Time[0, i, 0, 0, 0]
os.chdir(oldfolder)

print("Read Historical data")
_, True_data1, True_mat = historydata(timestep, steppi, steppi_indices)
True_mat[True_mat <= 0] = 0
# True_mat = True_data1
os.chdir("../HM_RESULTS")
Plot_RSM_singleT(True_mat, Time_unie1)
os.chdir(oldfolder)
scalei, scalei2, scalei3 = 1e2, 1e2, 1e4
Oilz = True_mat[:, :22] / scalei
Watsz = True_mat[:, 22:44] / scalei2
gasz = True_mat[:, 44:66] / scalei3
True_data = np.hstack([Oilz, Watsz, gasz])
# True_data = np.hstack([Oilz,Watsz])
True_data = np.reshape(True_data, (-1, 1), "F")

rows_to_remove = np.where(True_data <= 1e-4)[0]
# matrix = np.delete(True_data, rows_to_remove, axis=0)

True_data = remove_rows(True_data, rows_to_remove).reshape(-1, 1)
True_yet = True_data


sdjesus = np.std(True_data, axis=0)
sdjesus = np.reshape(sdjesus, (1, -1), "F")

menjesus = np.mean(True_data, axis=0)
menjesus = np.reshape(menjesus, (1, -1), "F")


True_dataTI = True_data
True_dataTI = np.reshape(True_dataTI, (-1, 1), "F")


print("")
print("-----------------------Select Good prior-----------------------------")
# """

path = os.getcwd()

os.chdir(oldfolder)
print("-------------------------------- Pior Selected------------------------")
print("---------------------------------------------------------------------")
print("")
print("---------------------------------------------------------------------")


noise_level = None
while True:
    noise_level = float(input("Enter the masurement data noise level in % (5%-25%): "))

    if (noise_level > 25) or (noise_level < 5):
        # raise SyntaxError('please select value between 1-2')
        print("")
        print("please try again and select value between 5%-25%")
    else:
        break


print("")
noise_level = noise_level / 100
print("---------------------------------------------------------------------")

print("")

choice = 2

sizeclem = nx * ny * nz

# Specify the rows to remove (rows 2 to 5 and 7 to 10)


# Use the remove_rows function to remove the specified rows

print("")
print("-------------------Decorrelate the ensemble---------------------------")
# """
if DEFAULT == 1:
    Deccor = 2
    print("No initial ensemble decorrrlation\n")
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

if Deccor == 1:
    TEMPLATEFILE["Ensemble decorrelation"] = "ensemble decorrelation = Yes"
else:
    TEMPLATEFILE["Ensemble decorrelation"] = "ensemble decorrelation = No"

# """
# Deccor=2

print("")
print("-----------------------Alpha Parameter-------------------------------")

if DEFAULT == 1:
    DE_alpha = 1
    print("Using reccomended alpha value\n ")
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
    print("Random generated ensemble\n")
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
print(
    "History Matching using the Adaptive Regularised Ensemble Kalman Inversion with covariance localisation"
)
print("Novel Implementation by Clement Etienam, SA-Nvidia: SA-ML/A.I/Energy")

batch_clem = 1

Technique_REKI = 1
TEMPLATEFILE["Data assimilation method"] = (
    "ADAPT_REKI (Vanilla Adaptive Ensemble Kalman Inversion)\n"
)


print("")
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

TEMPLATEFILE["Iterations"] = Termm
print("")

if DEFAULT == 1:
    Ne = 150
else:
    Ne = None
    while True:
        Ne = int(input("Number of realisations used for history matching (100-300) : "))
        N_ens = Ne
        if (Ne > 500) or (Ne < 20):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 20-500")
        else:
            break

N_ens = Ne

if Ne == 100:
    perm_ensemble = np.genfromtxt("../NORNE/sgsim.out").astype(np.float32)
    poro_ensemble = np.genfromtxt("../NORNE/sgsimporo.out").astype(np.float32)
    fault_ensemblep = np.genfromtxt("../NORNE/faultensemble.dat").astype(np.float32)

    perm_ensemble = np.delete(perm_ensemble, 20, axis=1)
    poro_ensemble = np.delete(poro_ensemble, 20, axis=1)
    fault_ensemblep = np.delete(fault_ensemblep, 20, axis=1)

    Neuse = int(Ne - 99)
    perm, poro, fault = NorneInitialEnsemble(ensembleSize=Neuse, randomNumber=1.2345e5)
    ini_ensemble = perm
    ini_ensemblep = poro
    ini_ensemblefault = fault

    ini_ensemble = np.hstack((ini_ensemble, perm_ensemble))

    ini_ensemblep = np.hstack((ini_ensemblep, poro_ensemble))

    ini_ensemblefault = np.hstack((ini_ensemblefault, fault_ensemblep))

    ini_ensemble, ini_ensemblep = honour2(
        ini_ensemblep, ini_ensemble, nx, ny, nz, N_ens, High_K, Low_K, High_P, Low_P
    )

    os.chdir("../NORNE")
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
    Time_unie = np.zeros((steppi))
    for i in range(steppi):
        Time_unie[i] = Time[0, i, 0, 0, 0]
    os.chdir(oldfolder)
    dt = Time_unie

else:
    pass

if (Ne > 100) and (Ne < 5000):
    os.chdir("../NORNE")
    Time = Get_Time(nx, ny, nz, steppi, steppi_indices, Ne)
    Time_unie = np.zeros((steppi))
    for i in range(steppi):
        Time_unie[i] = Time[0, i, 0, 0, 0]
    os.chdir(oldfolder)
    dt = Time_unie

    perm_ensemble = np.genfromtxt("../NORNE/sgsim.out").astype(np.float32)
    poro_ensemble = np.genfromtxt("../NORNE/sgsimporo.out").astype(np.float32)
    fault_ensemblep = np.genfromtxt("../NORNE/faultensemble.dat").astype(np.float32)

    perm_ensemble = np.delete(perm_ensemble, 20, axis=1)
    poro_ensemble = np.delete(poro_ensemble, 20, axis=1)
    fault_ensemblep = np.delete(fault_ensemblep, 20, axis=1)

    print("Read Historical data")
    _, True_data1, True_mat = historydata(timestep, steppi, steppi_indices)
    True_mat[True_mat <= 0] = 0
    # True_mat = True_data1
    os.chdir("../HM_RESULTS")
    Plot_RSM_singleT(True_mat, Time_unie1)
    os.chdir(oldfolder)
    # scalei,scalei2,scalei3 = 1e3,1e3,1e3
    Oilz = True_mat[:, :22] / scalei
    Watsz = True_mat[:, 22:44] / scalei2
    gasz = True_mat[:, 44:66] / scalei3
    True_data = np.hstack([Oilz, Watsz, gasz])
    # True_data = np.hstack([Oilz,Watsz])
    True_data = np.reshape(True_data, (-1, 1), "F")
    True_data = remove_rows(True_data, rows_to_remove).reshape(-1, 1)
    True_yet = True_data
    Nop = True_data.shape[0]

    Neuse = int(Ne - 99)
    perm, poro, fault = NorneInitialEnsemble(ensembleSize=Neuse, randomNumber=1.2345e5)
    ini_ensemble = perm
    ini_ensemblep = poro
    ini_ensemblefault = fault

    ini_ensemble = np.hstack((ini_ensemble, perm_ensemble))

    ini_ensemblep = np.hstack((ini_ensemblep, poro_ensemble))

    ini_ensemblefault = np.hstack((ini_ensemblefault, fault_ensemblep))

else:
    pass

if (Ne > 5000) or (Ne < 100):
    # raise SyntaxError('please select value between 1-2')
    print("")
    print("Generating ensemble")
    perm, poro, fault = NorneInitialEnsemble(ensembleSize=Ne, randomNumber=1.2345e5)
    ini_ensemble = perm
    ini_ensemblep = poro
    ini_ensemblefault = fault
    del perm, poro, fault
else:
    pass

TEMPLATEFILE["Ensemble size"] = Ne


# Printing the keys and values to the screen
print("")
print("--------------History Matching Operational conditions:----------------")
print("------------------------------------------------------------------")
for key, value in TEMPLATEFILE.items():
    print(f"{key}: {value}")

# Saving the dictionary to a YAML file
yaml_filename = "../HM_RESULTS/History_Matching_Template_file.yaml"
with open(yaml_filename, "w") as yaml_file:
    yaml.dump(TEMPLATEFILE, yaml_file)


print("")
# effec[effec==0]=-1
ini_ensemble = shuffle(ini_ensemble, axis=1) * effec
ini_ensemblep = shuffle(ini_ensemblep, axis=1) * effec
ini_ensemblefault = shuffle(ini_ensemblefault, axis=1)

start_time = time.time()
print("")

print("")
print("------Adaptive Regularised Ensemble Kalman Inversion with----------- \n")
print("----Starting the History matching with ", str(Ne) + " Ensemble members--")

os.chdir(oldfolder)

ensemble = ini_ensemble
ensemble = np.nan_to_num(ensemble, copy=True, nan=Low_K1)
# ensemblep=Getporosity_ensemble(ensemble,machine_map,N_ens)


ensemblep = ini_ensemblep
ensemblef = ini_ensemblefault

ensemble, ensemblep = honour2(
    ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
)

ensemble = ensemble * effec
ensemblep = ensemblep * effec
Nop = True_data.shape[0]
ax = np.zeros((Nop, 1))
# print(Nop)
for iq in range(Nop):
    if (True_data[iq, :] > 0) and (True_data[iq, :] <= 1000000):
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


alpha_big = []
mean_cost = []
best_cost = []


ensemble_meanK = []
ensemble_meanP = []
ensemble_meanf = []

ensemble_bestK = []
ensemble_bestP = []
ensemble_bestf = []

ensembles = []
ensemblesp = []
ensemblesf = []


# rho = 1.08

updated_ensemble = ensemble
updated_ensemblep = ensemblep

base_k = np.mean(ensemble, axis=1).reshape(-1, 1)
base_p = np.mean(ensemblep, axis=1).reshape(-1, 1)
base_f = np.mean(ensemblef, axis=1).reshape(-1, 1)

text = """
                                                                                                                                                                                   
                  RRRRRRRRRRRRRRRRR   EEEEEEEEEEEEEEEEEEEEEEKKKKKKKKK    KKKKKKKIIIIIIIIII
                  R::::::::::::::::R  E::::::::::::::::::::EK:::::::K    K:::::KI::::::::I
                  R::::::RRRRRR:::::R E::::::::::::::::::::EK:::::::K    K:::::KI::::::::I
                  RR:::::R     R:::::REE::::::EEEEEEEEE::::EK:::::::K   K::::::KII::::::II
  aaaaaaaaaaaaa     R::::R     R:::::R  E:::::E       EEEEEEKK::::::K  K:::::KKK  I::::I  
  a::::::::::::a    R::::R     R:::::R  E:::::E               K:::::K K:::::K     I::::I  
  aaaaaaaaa:::::a   R::::RRRRRR:::::R   E::::::EEEEEEEEEE     K::::::K:::::K      I::::I  
           a::::a   R:::::::::::::RR    E:::::::::::::::E     K:::::::::::K       I::::I  
    aaaaaaa:::::a   R::::RRRRRR:::::R   E:::::::::::::::E     K:::::::::::K       I::::I  
  aa::::::::::::a   R::::R     R:::::R  E::::::EEEEEEEEEE     K::::::K:::::K      I::::I  
 a::::aaaa::::::a   R::::R     R:::::R  E:::::E               K:::::K K:::::K     I::::I  
a::::a    a:::::a   R::::R     R:::::R  E:::::E       EEEEEEKK::::::K  K:::::KKK  I::::I  
a::::a    a:::::a RR:::::R     R:::::REE::::::EEEEEEEE:::::EK:::::::K   K::::::KII::::::II
a:::::aaaa::::::a R::::::R     R:::::RE::::::::::::::::::::EK:::::::K    K:::::KI::::::::I
 a::::::::::aa:::aR::::::R     R:::::RE::::::::::::::::::::EK:::::::K    K:::::KI::::::::I
  aaaaaaaaaa  aaaaRRRRRRRR     RRRRRRREEEEEEEEEEEEEEEEEEEEEEKKKKKKKKK    KKKKKKKIIIIIIIIII
"""
print(text)

if Trainmoe == 2:
    texta = """
    PPPPPPPPPPPPPPPPP  IIIIIIIIINNNNNNNN        NNNNNNNN    OOOOOOOOO                              CCCCCCCCCCCCC      CCCCCCCCCCCCRRRRRRRRRRRRRRRRR   
    P::::::::::::::::P I::::::::N:::::::N       N::::::N  OO:::::::::OO                         CCC::::::::::::C   CCC::::::::::::R::::::::::::::::R  
    P::::::PPPPPP:::::PI::::::::N::::::::N      N::::::NOO:::::::::::::OO                     CC:::::::::::::::C CC:::::::::::::::R::::::RRRRRR:::::R 
    PP:::::P     P:::::II::::::IN:::::::::N     N::::::O:::::::OOO:::::::O                   C:::::CCCCCCCC::::CC:::::CCCCCCCC::::RR:::::R     R:::::R
      P::::P     P:::::P I::::I N::::::::::N    N::::::O::::::O   O::::::O                  C:::::C       CCCCCC:::::C       CCCCCC R::::R     R:::::R
      P::::P     P:::::P I::::I N:::::::::::N   N::::::O:::::O     O:::::O                 C:::::C            C:::::C               R::::R     R:::::R
      P::::PPPPPP:::::P  I::::I N:::::::N::::N  N::::::O:::::O     O:::::O                 C:::::C            C:::::C               R::::RRRRRR:::::R 
      P:::::::::::::PP   I::::I N::::::N N::::N N::::::O:::::O     O:::::O --------------- C:::::C            C:::::C               R:::::::::::::RR  
      P::::PPPPPPPPP     I::::I N::::::N  N::::N:::::::O:::::O     O:::::O -:::::::::::::- C:::::C            C:::::C               R::::RRRRRR:::::R 
      P::::P             I::::I N::::::N   N:::::::::::O:::::O     O:::::O --------------- C:::::C            C:::::C               R::::R     R:::::R
      P::::P             I::::I N::::::N    N::::::::::O:::::O     O:::::O                 C:::::C            C:::::C               R::::R     R:::::R
      P::::P             I::::I N::::::N     N:::::::::O::::::O   O::::::O                  C:::::C       CCCCCC:::::C       CCCCCC R::::R     R:::::R
    PP::::::PP         II::::::IN::::::N      N::::::::O:::::::OOO:::::::O                   C:::::CCCCCCCC::::CC:::::CCCCCCCC::::RR:::::R     R:::::R
    P::::::::P         I::::::::N::::::N       N:::::::NOO:::::::::::::OO                     CC:::::::::::::::C CC:::::::::::::::R::::::R     R:::::R
    P::::::::P         I::::::::N::::::N        N::::::N  OO:::::::::OO                         CCC::::::::::::C   CCC::::::::::::R::::::R     R:::::R
    PPPPPPPPPP         IIIIIIIIINNNNNNNN         NNNNNNN    OOOOOOOOO                              CCCCCCCCCCCCC      CCCCCCCCCCCCRRRRRRRR     RRRRRRR
                                                                                                                                                      
    """
    # print(texta)
else:
    texta = """
                                                                                                                                                          
    PPPPPPPPPPPPPPPPP  IIIIIIIIINNNNNNNN        NNNNNNNN    OOOOOOOOO                      FFFFFFFFFFFFFFFFFFFFFNNNNNNNN        NNNNNNNN    OOOOOOOOO     
    P::::::::::::::::P I::::::::N:::::::N       N::::::N  OO:::::::::OO                    F::::::::::::::::::::N:::::::N       N::::::N  OO:::::::::OO   
    P::::::PPPPPP:::::PI::::::::N::::::::N      N::::::NOO:::::::::::::OO                  F::::::::::::::::::::N::::::::N      N::::::NOO:::::::::::::OO 
    PP:::::P     P:::::II::::::IN:::::::::N     N::::::O:::::::OOO:::::::O                 FF::::::FFFFFFFFF::::N:::::::::N     N::::::O:::::::OOO:::::::O
      P::::P     P:::::P I::::I N::::::::::N    N::::::O::::::O   O::::::O                   F:::::F       FFFFFN::::::::::N    N::::::O::::::O   O::::::O
      P::::P     P:::::P I::::I N:::::::::::N   N::::::O:::::O     O:::::O                   F:::::F            N:::::::::::N   N::::::O:::::O     O:::::O
      P::::PPPPPP:::::P  I::::I N:::::::N::::N  N::::::O:::::O     O:::::O                   F::::::FFFFFFFFFF  N:::::::N::::N  N::::::O:::::O     O:::::O
      P:::::::::::::PP   I::::I N::::::N N::::N N::::::O:::::O     O:::::O ---------------   F:::::::::::::::F  N::::::N N::::N N::::::O:::::O     O:::::O
      P::::PPPPPPPPP     I::::I N::::::N  N::::N:::::::O:::::O     O:::::O -:::::::::::::-   F:::::::::::::::F  N::::::N  N::::N:::::::O:::::O     O:::::O
      P::::P             I::::I N::::::N   N:::::::::::O:::::O     O:::::O ---------------   F::::::FFFFFFFFFF  N::::::N   N:::::::::::O:::::O     O:::::O
      P::::P             I::::I N::::::N    N::::::::::O:::::O     O:::::O                   F:::::F            N::::::N    N::::::::::O:::::O     O:::::O
      P::::P             I::::I N::::::N     N:::::::::O::::::O   O::::::O                   F:::::F            N::::::N     N:::::::::O::::::O   O::::::O
    PP::::::PP         II::::::IN::::::N      N::::::::O:::::::OOO:::::::O                 FF:::::::FF          N::::::N      N::::::::O:::::::OOO:::::::O
    P::::::::P         I::::::::N::::::N       N:::::::NOO:::::::::::::OO                  F::::::::FF          N::::::N       N:::::::NOO:::::::::::::OO 
    P::::::::P         I::::::::N::::::N        N::::::N  OO:::::::::OO                    F::::::::FF          N::::::N        N::::::N  OO:::::::::OO   
    PPPPPPPPPP         IIIIIIIIINNNNNNNN         NNNNNNN    OOOOOOOOO                      FFFFFFFFFFF          NNNNNNNN         NNNNNNN    OOOOOOOOO   
    """
print(texta)

while snn < 1:
    print("Iteration --" + str(ii + 1) + " | " + str(Termm))
    print("****************************************************************")

    ensemblepy = ensemble_pytorch(
        ensemble,
        ensemblep,
        ensemblef,
        nx,
        ny,
        nz,
        Ne,
        effective,
        oldfolder,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        minQ,
        maxQ,
        minQw,
        maxQw,
        minQg,
        maxQg,
        steppi,
        device,
        steppi_indices,
    )

    ini_K = ensemble
    ini_p = ensemblep
    ini_f = ensemblef

    mazw = 0  # Dont smooth the presure field

    simDatafinal, predMatrix, _, _, _, _ = Forward_model_ensemble(
        ensemble.shape[1],
        ensemblepy,
        steppi,
        min_inn_fcn,
        max_inn_fcn,
        target_min,
        target_max,
        minK,
        maxK,
        minT,
        maxT,
        minP,
        maxP,
        modelP,
        modelW,
        modelG,
        modelO,
        device,
        model_peacemann,
        min_out_fcn,
        max_out_fcn,
        Time,
        effectiveuse,
        Trainmoe,
        num_cores,
        pred_type,
        oldfolder,
        degg,
        experts,
    )

    if ii == 0:
        os.chdir("../HM_RESULTS")
        Plot_RSM(predMatrix, True_mat, "Initial.png", Ne, Time_unie1)
        os.chdir(oldfolder)
    else:
        pass

    # print('-----------------------------Read Historical data----------------')
    _, True_data1, True_mat = historydata(timestep, steppi, steppi_indices)
    True_mat[True_mat <= 0] = 0
    # True_mat = True_data1
    os.chdir("../HM_RESULTS")
    Plot_RSM_singleT(True_mat, Time_unie1)
    os.chdir(oldfolder)
    # scalei,scalei2,scalei3 = 1e3,1e3,1e3
    Oilz = True_mat[:, :22] / scalei
    Watsz = True_mat[:, 22:44] / scalei2
    gasz = True_mat[:, 44:66] / scalei3
    True_data = np.hstack([Oilz, Watsz, gasz])
    # True_data = np.hstack([Oilz,Watsz])
    True_data = np.reshape(True_data, (-1, 1), "F")
    True_data = remove_rows(True_data, rows_to_remove).reshape(-1, 1)
    True_yet = True_data
    True_dataa = True_data

    rho = adaptive_rho(True_dataa, simDatafinal, rho)

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
    # Extract non-zero rows from each column
    # sgsim1,indices = extract_non_zeros(sgsim)
    sgsim1 = sgsim

    # ensemblep1,_ = extract_non_zeros(ensemblep)
    ensemblep1 = ensemblep

    overall = cp.vstack(
        [cp.asarray(sgsim1), cp.asarray(ensemblep1), cp.asarray(ensemblef)]
    )

    Y = overall
    Sim1 = cp.asarray(simDatafinal)

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

    pertubations_cu = cp.asarray(pertubations)
    true_data_cu = cp.asarray(True_data)
    alpha_cu = cp.asarray(alpha)
    tile_true_ne = cp.tile(true_data_cu, Ne)
    pertu_alpha = cp.sqrt(alpha_cu) * pertubations_cu
    factor_sum = (tile_true_ne + pertu_alpha) - Sim1
    del pertubations_cu, true_data_cu, alpha_cu, Usig, Vsig
    gc.collect()

    # print(f"Cyd {Cyd.shape}, X {X.shape}, inv_CDd {inv_CDd.shape}")
    update_term = Cyd @ X
    del Cyd, X
    gc.collect()
    update_term @= inv_CDd
    del inv_CDd
    gc.collect()
    update_term @= factor_sum
    del factor_sum
    gc.collect()

    if do_localisation == 1:
        if ii == 0:
            locmat = Localisation(10, nx, ny, nz, Ne)
            see1 = locmat[: nx * ny * nz, :] * effec
            XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
            look = np.reshape(see1[:, 1], (nx, ny, nz), "F")
            look[look == 0] = np.nan
            plt.figure(figsize=(40, 40))
            for kkt in range(nz):
                plt.subplot(5, 5, int(kkt + 1))
                plt.pcolormesh(XX.T, YY.T, look[:, :, kkt], cmap="jet")
                string = "Layer " + str(kkt + 1)
                plt.title(string, fontsize=13)
                plt.ylabel("Y", fontsize=13)
                plt.xlabel("X", fontsize=13)
                plt.axis([0, (nx - 1), 0, (ny - 1)])
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                cbar1 = plt.colorbar()
                cbar1.ax.set_ylabel(" Localisation Matrix", fontsize=13)
                Add_marker2(plt, XX, YY, injectors, producers, gass)

            plt.savefig("../HM_RESULTS/Localisation_matrix.png")
            plt.clf()
            plt.close()
            locmat = cp.asarray(locmat)
        else:
            pass

        update_term = update_term * locmat
    else:
        pass
    Ynew = Y + update_term

    sizeclem = cp.asarray(sgsim1.shape[0])
    if Yet == 0:
        updated_ensemble1 = cp.asnumpy(Ynew[:sizeclem, :])
        updated_ensemblep1 = cp.asnumpy(Ynew[sizeclem : 2 * sizeclem, :])
        updated_ensemblef = cp.asnumpy(Ynew[2 * sizeclem :, :])

    else:
        updated_ensemble1 = Ynew[:sizeclem, :]
        updated_ensemblep1 = Ynew[sizeclem : 2 * sizeclem, :]
        updated_ensemblef = Ynew[2 * sizeclem :, :]

    # updated_ensemble = place_back(updated_ensemble1,indices, (nx*ny*nz, Ne))
    # updated_ensemblep = place_back(updated_ensemblep1,indices, (nx*ny*nz, Ne))

    updated_ensemble = updated_ensemble1
    updated_ensemblep = updated_ensemblep1

    if ii == 0:
        simmean = np.reshape(np.mean(simDatafinal, axis=1), (-1, 1), "F")
        tinuke = (
            (np.sum((((simmean) - True_dataa) ** 2))) ** (0.5)
        ) / True_dataa.shape[0]
        print("Initial RMSE of the ensemble mean =  " + str(tinuke) + "... .")
        aa, bb, cc = funcGetDataMismatch(simDatafinal, True_dataa)
        clem = np.argmin(cc)
        simmbest = simDatafinal[:, clem].reshape(-1, 1)
        tinukebest = (
            (np.sum((((simmbest) - True_dataa) ** 2))) ** (0.5)
        ) / True_dataa.shape[0]
        print("Initial RMSE of the ensemble best =  " + str(tinukebest) + "... .")
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
        print("RMSE of the ensemble best =  " + str(tinukebest) + "... .")

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
                "ensemble best cost increased by =  "
                + str(abs(tinukebest - tinubestprior))
                + "... ."
            )

        if tinukebest < tinubestprior:
            print(
                "ensemble best cost decreased by =  "
                + str(abs(tinukebest - tinubestprior))
                + "... ."
            )

        if tinukebest == tinubestprior:
            print("No change in ensemble best cost")

        tinumeanprior = tinuke
        tinubestprior = tinukebest
    if best_cost_mean > tinuke:
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
        use_f = ensemblef
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
    ensemble_bestf.append(ini_f[:, clem].reshape(-1, 1))

    ensemble_meanK.append(np.reshape(np.mean(ini_K, axis=1), (-1, 1), "F"))
    ensemble_meanP.append(np.reshape(np.mean(ini_p, axis=1), (-1, 1), "F"))
    ensemble_meanf.append(np.reshape(np.mean(ini_f, axis=1), (-1, 1), "F"))

    # ensembles.append(ensemble)
    # ensemblesp.append(ensemblep)
    # ensemblesf.append(ensemblef)

    ensemble = updated_ensemble
    ensemblep = updated_ensemblep
    ensemblef = updated_ensemblef

    ensemble, ensemblep = honour2(
        ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
    )
    ensemblef = np.clip(ensemblef, 0, 1)
    ensemble = ensemble * effec
    ensemblep = ensemblep * effec

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
print("**********************************************************************")
best_cost1 = np.vstack(best_cost)
chb = np.argmin(best_cost1)

ensemble_bestK = np.hstack(ensemble_bestK)
ensemble_bestP = np.hstack(ensemble_bestP)
ensemble_bestf = np.hstack(ensemble_bestf)

yes_best_k = ensemble_bestK[:, chb].reshape(-1, 1)
yes_best_p = ensemble_bestP[:, chb].reshape(-1, 1)
yes_best_f = ensemble_bestf[:, chb].reshape(-1, 1)


ensemble_meanK = np.hstack(ensemble_meanK)
ensemble_meanP = np.hstack(ensemble_meanP)
ensemble_meanf = np.hstack(ensemble_meanf)

yes_mean_k = ensemble_meanK[:, chm].reshape(-1, 1)
yes_mean_p = ensemble_meanP[:, chm].reshape(-1, 1)
yes_mean_f = ensemble_meanf[:, chm].reshape(-1, 1)

all_ensemble = use_k  # ensembles[chm]
all_ensemblep = use_p  # ensemblesp[chm]
all_ensemblef = use_f  # ensemblesf[chm]

ensemble, ensemblep = honour2(
    ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
)
use_f = np.clip(use_f, 0, 1)

all_ensemble, all_ensemblep = honour2(
    all_ensemblep, all_ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
)

all_ensemblef = np.clip(all_ensemblef, 0, 1)


ensemble, ensemblep = honour2(
    ensemblep, ensemble, nx, ny, nz, N_ens, High_K1, Low_K1, High_P, Low_P
)

meann = np.reshape(np.mean(ensemble, axis=1), (-1, 1), "F")
meannp = np.reshape(np.mean(ensemblep, axis=1), (-1, 1), "F")
meannf = np.reshape(np.mean(ensemblef, axis=1), (-1, 1), "F")


meanini = np.reshape(np.mean(ini_ensemble, axis=1), (-1, 1), "F")
controljj2 = np.reshape(meann, (-1, 1), "F")
controljj2p = np.reshape(meannp, (-1, 1), "F")
controljj2f = np.reshape(meannf, (-1, 1), "F")

controlj2 = controljj2
controlj2p = controljj2p
controlj2f = controljj2f


ensemblepy = ensemble_pytorch(
    ensemble,
    ensemblep,
    ensemblef,
    nx,
    ny,
    nz,
    ensemble.shape[1],
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
)

mazw = 0  # Dont smooth the presure field


(
    simDatafinal,
    predMatrix,
    pressure_ensemble,
    water_ensemble,
    oil_ensemble,
    gas_ensemble,
) = Forward_model_ensemble(
    ensemble.shape[1],
    ensemblepy,
    steppi,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    modelP,
    modelW,
    modelG,
    modelO,
    device,
    model_peacemann,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
    degg,
    experts,
)


ensemblepya = ensemble_pytorch(
    all_ensemble,
    all_ensemblep,
    all_ensemblef,
    nx,
    ny,
    nz,
    all_ensemble.shape[1],
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
)


mazw = 0  # Dont smooth the presure field
(
    simDatafinala,
    predMatrixa,
    pressure_ensemblea,
    water_ensemblea,
    oil_ensemblea,
    gas_ensemblea,
) = Forward_model_ensemble(
    all_ensemble.shape[1],
    ensemblepya,
    steppi,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    modelP,
    modelW,
    modelG,
    modelO,
    device,
    model_peacemann,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
    degg,
    experts,
)

os.chdir("../HM_RESULTS")
# Plot_RSM(predMatrix,True_mat,"Final.png",Ne,Time_unie1)
Plot_RSM(predMatrixa, True_mat, "Final.png", Ne, Time_unie1)
os.chdir(oldfolder)


aa, bb, cc = funcGetDataMismatch(simDatafinal, True_data)
clem = np.argmin(cc)
shpw = cc[clem]
controlbest = np.reshape(ensemble[:, clem], (-1, 1), "F")
controlbestp = np.reshape(ensemblep[:, clem], (-1, 1), "F")
controlbestf = np.reshape(ensemblef[:, clem], (-1, 1), "F")
controlbest2 = controlj2  # controlbest
controlbest2p = controljj2p  # controlbest
controlbest2f = controljj2f  # controlbest

clembad = np.argmax(cc)
controlbad = np.reshape(ensemble[:, clembad], (-1, 1), "F")
controlbadp = np.reshape(ensemblep[:, clembad], (-1, 1), "F")
controlbadf = np.reshape(ensemblef[:, clembad], (-1, 1), "F")

Plot_Histogram_now(Ne, cc_ini, cc, mean_cost, best_cost)
# os.makedirs('ESMDA')
if not os.path.exists("../HM_RESULTS/ADAPT_REKI"):
    os.makedirs("../HM_RESULTS/ADAPT_REKI")
else:
    shutil.rmtree("../HM_RESULTS/ADAPT_REKI")
    os.makedirs("../HM_RESULTS/ADAPT_REKI")

print("-----PINO Surrogate Reservoir Simulator Forwarding - ADAPT_REKI Model--")


ensemblepy = ensemble_pytorch(
    controlbest,
    controlbestp,
    controlbestf,
    nx,
    ny,
    nz,
    controlbest.shape[1],
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
)

mazw = 0  # Dont smooth the presure field
_, yycheck, pree, wats, oilss, gasss = Forward_model_ensemble(
    controlbest.shape[1],
    ensemblepy,
    steppi,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    modelP,
    modelW,
    modelG,
    modelO,
    device,
    model_peacemann,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
    degg,
    experts,
)

os.chdir("../HM_RESULTS/ADAPT_REKI")
Plot_RSM_single(yycheck, Time_unie1)
Plot_petrophysical(controlbest, controlbestp, nx, ny, nz, Low_K1, High_K1)


X_data1 = {
    "permeability": controlbest,
    "porosity": controlbestp,
    "Simulated_data_plots": yycheck,
    "Pressure": pree,
    "Water_saturation": wats,
    "Oil_saturation": oilss,
    "gas_saturation": gasss,
}

with gzip.open("RESERVOIR_MODEL.pkl.gz", "wb") as f1:
    pickle.dump(X_data1, f1)
os.chdir(oldfolder)

Time_vector = np.zeros((steppi))

for kk in range(steppi):
    current_time = dt[kk]
    Time_vector[kk] = current_time


folderrin = os.path.join(oldfolder, "..", "HM_RESULTS", "ADAPT_REKI")


Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
    delayed(process_step)(
        kk,
        steppi,
        dt,
        pree,
        effectiveuse,
        wats,
        oilss,
        gasss,
        nx,
        ny,
        nz,
        N_injw,
        N_pr,
        N_injg,
        injectors,
        producers,
        gass,
        folderrin,
        oldfolder,
    )
    for kk in range(steppi)
)

progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, steppi - 1, steppi - 1)
ShowBar(progressBar)
time.sleep(1)


os.chdir("../HM_RESULTS/ADAPT_REKI")
print("-------------------------Creating GIF---------------------------------")
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


print("")
print("--------------------------Saving prediction in CSV file---------------")
write_RSM(yycheck[0, :, :66], Time_vector, "PhysicsNeMo")


Plot_RSM_percentile_model(yycheck[0, :, :66], True_mat, Time_unie1)

os.chdir(oldfolder)


yycheck = yycheck[0, :, :66]
# usesim=yycheck[:,1:]
Oilz = yycheck[:, :22] / scalei
Watsz = yycheck[:, 22:44] / scalei2
wctz = yycheck[:, 44:] / scalei3
usesim = np.hstack([Oilz, Watsz, wctz])
usesim = np.reshape(usesim, (-1, 1), "F")
usesim = remove_rows(usesim, rows_to_remove).reshape(-1, 1)
usesim = np.reshape(usesim, (-1, 1), "F")
yycheck = usesim

cc = ((np.sum((((usesim) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
print("RMSE  =  " + str(cc))
os.chdir("../HM_RESULTS")
Plot_mean(controlbest, yes_mean_k, meanini, nx, ny, Low_K1, High_K1, True_K)
# Plot_mean(controlbest,controljj2,meanini,nx,ny,Low_K1,High_K1,True_K)
os.chdir(oldfolder)


if not os.path.exists("../HM_RESULTS/BEST_RESERVOIR_MODEL"):
    os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")
else:
    shutil.rmtree("../HM_RESULTS/BEST_RESERVOIR_MODEL")
    os.makedirs("../HM_RESULTS/BEST_RESERVOIR_MODEL")


ensemblepy = ensemble_pytorch(
    yes_best_k,
    yes_best_p,
    yes_best_f,
    nx,
    ny,
    nz,
    controlbest2.shape[1],
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
)

os.chdir(oldfolder)
mazw = 0  # Dont smooth the presure field
_, yycheck, preebest, watsbest, oilssbest, gasbest = Forward_model_ensemble(
    controlbest2.shape[1],
    ensemblepy,
    steppi,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    modelP,
    modelW,
    modelG,
    modelO,
    device,
    model_peacemann,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
    degg,
    experts,
)

os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
Plot_RSM_single(yycheck, Time_unie1)
Plot_petrophysical(yes_best_k, yes_best_p, nx, ny, nz, Low_K1, High_K1)

X_data1 = {
    "permeability": yes_best_k,
    "porosity": yes_best_p,
    "Simulated_data_plots": yycheck,
    "Pressure": preebest,
    "Water_saturation": watsbest,
    "Oil_saturation": oilssbest,
    "gas_saturation": gasbest,
}

with gzip.open("BEST_RESERVOIR_MODEL.pkl.gz", "wb") as f1:
    pickle.dump(X_data1, f1)

os.chdir(oldfolder)


folderrin = os.path.join(oldfolder, "..", "HM_RESULTS", "BEST_RESERVOIR_MODEL")
Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
    delayed(process_step)(
        kk,
        steppi,
        dt,
        preebest,
        effectiveuse,
        watsbest,
        oilssbest,
        gasbest,
        nx,
        ny,
        nz,
        N_injw,
        N_pr,
        N_injg,
        injectors,
        producers,
        gass,
        folderrin,
        oldfolder,
    )
    for kk in range(steppi)
)

progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, steppi - 1, steppi - 1)
ShowBar(progressBar)
time.sleep(1)


os.chdir("../HM_RESULTS/BEST_RESERVOIR_MODEL")
print("Creating GIF")
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


print("")
write_RSM(yycheck[0, :, :66], Time_vector, "PhysicsNeMo")


Plot_RSM_percentile_model(yycheck[0, :, :66], True_mat, Time_unie1)

os.chdir(oldfolder)


yycheck = yycheck[0, :, :66]
# usesim=yycheck[:,1:]
Oilz = yycheck[:, :22] / scalei
Watsz = yycheck[:, 22:44] / scalei2
wctz = yycheck[:, 44:] / scalei3
usesim = np.hstack([Oilz, Watsz, wctz])
usesim = np.reshape(usesim, (-1, 1), "F")
usesim = remove_rows(usesim, rows_to_remove).reshape(-1, 1)
usesim = np.reshape(usesim, (-1, 1), "F")
yycheck = usesim

cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
print("RMSE of overall best model  =  " + str(cc))


if not os.path.exists("../HM_RESULTS/MEAN_RESERVOIR_MODEL"):
    os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
else:
    shutil.rmtree("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
    os.makedirs("../HM_RESULTS/MEAN_RESERVOIR_MODEL")


ensemblepy = ensemble_pytorch(
    yes_mean_k,
    yes_mean_p,
    yes_mean_f,
    nx,
    ny,
    nz,
    controlbest2.shape[1],
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
)


os.chdir(oldfolder)
mazw = 0  # Dont smooth the presure field
_, yycheck, preebest, watsbest, oilssbest, gasbest = Forward_model_ensemble(
    controlbest2.shape[1],
    ensemblepy,
    steppi,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    modelP,
    modelW,
    modelG,
    modelO,
    device,
    model_peacemann,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
    degg,
    experts,
)

os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
Plot_RSM_single(yycheck, Time_unie1)
Plot_petrophysical(yes_mean_k, yes_mean_p, nx, ny, nz, Low_K1, High_K1)

X_data1 = {
    "permeability": yes_mean_k,
    "porosity": yes_mean_p,
    "Simulated_data_plots": yycheck,
    "Pressure": preebest,
    "Water_saturation": watsbest,
    "Oil_saturation": oilssbest,
    "Gas_saturation": gasbest,
}

with gzip.open("MEAN_RESERVOIR_MODEL.pkl.gz", "wb") as f1:
    pickle.dump(X_data1, f1)

os.chdir(oldfolder)


folderrin = os.path.join(oldfolder, "..", "HM_RESULTS", "MEAN_RESERVOIR_MODEL")
Parallel(n_jobs=num_cores, backend="loky", verbose=10)(
    delayed(process_step)(
        kk,
        steppi,
        dt,
        preebest,
        effectiveuse,
        watsbest,
        oilssbest,
        gasbest,
        nx,
        ny,
        nz,
        N_injw,
        N_pr,
        N_injg,
        injectors,
        producers,
        gass,
        folderrin,
        oldfolder,
    )
    for kk in range(steppi)
)

progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, steppi - 1, steppi - 1)
ShowBar(progressBar)
time.sleep(1)

os.chdir("../HM_RESULTS/MEAN_RESERVOIR_MODEL")
print("Creating GIF")
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


print("")
write_RSM(yycheck[0, :, :66], Time_vector, "PhysicsNeMo")


Plot_RSM_percentile_model(yycheck[0, :, :66], True_mat, Time_unie1)


os.chdir(oldfolder)

yycheck = yycheck[0, :, :66]
# usesim=yycheck[:,1:]
Oilz = yycheck[:, :22] / scalei
Watsz = yycheck[:, 22:44] / scalei2
wctz = yycheck[:, 44:] / scalei3
usesim = np.hstack([Oilz, Watsz, wctz])
usesim = np.reshape(usesim, (-1, 1), "F")
usesim = remove_rows(usesim, rows_to_remove).reshape(-1, 1)
usesim = np.reshape(usesim, (-1, 1), "F")
yycheck = usesim

cc = ((np.sum((((yycheck) - True_data) ** 2))) ** (0.5)) / True_data.shape[0]
print("RMSE of overall best MAP model  =  " + str(cc))


os.chdir("../HM_RESULTS")
X_data1 = {
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
    "Gas_saturation": gas_ensemble,
    "Simulated_data_best_ensemble": simDatafinala,
    "Simulated_data_plots_best_ensemble": predMatrixa,
    "Pressures_best_ensemble": pressure_ensemblea,
    "Water_saturation_best_ensemble": water_ensemblea,
    "Oil_saturation_best_ensemble": oil_ensemblea,
    "Gas_saturation_best_ensemble": gas_ensemblea,
}


os.chdir(oldfolder)

ensembleout1 = np.hstack(
    [controlbest, controljj2, controlbad, yes_best_k, yes_mean_k, base_k]
)
ensembleoutp1 = np.hstack(
    [controlbestp, controljj2p, controlbadp, yes_best_p, yes_mean_p, base_p]
)
ensembleoutf1 = np.hstack(
    [controlbestf, controljj2f, controlbadf, yes_best_f, yes_mean_f, base_f]
)

# ensembleoutp=Getporosity_ensemble(ensembleout,machine_map,3)
print("-------------------Plot P10,P50,P90 Reservoir Models------------------")
if not os.path.exists("../HM_RESULTS/PERCENTILE"):
    os.makedirs("../HM_RESULTS/PERCENTILE")
else:
    shutil.rmtree("../HM_RESULTS/PERCENTILE")
    os.makedirs("../HM_RESULTS/PERCENTILE")

print("PINO Surrogate Reservoir Simulator Forwarding")


ensemblepy = ensemble_pytorch(
    ensembleout1,
    ensembleoutp1,
    ensembleoutf1,
    nx,
    ny,
    nz,
    ensembleoutf1.shape[1],
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
)

os.chdir(oldfolder)
mazw = 0  # Dont smooth the presure field
(
    _,
    yzout,
    pressure_percentile,
    water_percentile,
    oil_percentile,
    gas_percentile,
) = Forward_model_ensemble(
    ensembleoutf1.shape[1],
    ensemblepy,
    steppi,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    modelP,
    modelW,
    modelG,
    modelO,
    device,
    model_peacemann,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
    degg,
    experts,
)

os.chdir("../HM_RESULTS/PERCENTILE")
Plot_RSM_percentile(yzout, True_mat, Time_unie1)


X_data1 = {
    "PERM_Reali": ensembleout1,
    "PORO_Reali": ensembleoutp1,
    "Simulated_data_plots": yzout,
    "Pressures": pressure_percentile,
    "Water_saturation": water_percentile,
    "Oil_saturation": oil_percentile,
    "Gas_saturation": gas_percentile,
}

with gzip.open("Posterior_Ensembles_percentile.pkl.gz", "wb") as f1:
    pickle.dump(X_data1, f1)


f_3 = plt.figure(figsize=(20, 20), dpi=200)

look = ((np.reshape(True_K, (nx, ny, nz), "F")) * effectiveuse)[:, :, ::-1]
ax1 = f_3.add_subplot(3, 3, 1, projection="3d")
Plot_PhysicsNeMo(
    ax1,
    nx,
    ny,
    nz,
    look,
    N_injw,
    N_pr,
    N_injg,
    "True model",
    injectors,
    producers,
    gass,
)

look = ((np.reshape(base_k, (nx, ny, nz), "F")) * effectiveuse)[:, :, ::-1]
ax1 = f_3.add_subplot(3, 3, 2, projection="3d")
Plot_PhysicsNeMo(
    ax1, nx, ny, nz, look, N_injw, N_pr, N_injg, "Prior", injectors, producers, gass
)


look = ((np.reshape(controlbest, (nx, ny, nz), "F")) * effectiveuse)[:, :, ::-1]
ax1 = f_3.add_subplot(3, 3, 3, projection="3d")
Plot_PhysicsNeMo(
    ax1, nx, ny, nz, look, N_injw, N_pr, N_injg, "P10", injectors, producers, gass
)


look = ((np.reshape(controljj2, (nx, ny, nz), "F")) * effectiveuse)[:, :, ::-1]
ax1 = f_3.add_subplot(3, 3, 4, projection="3d")
Plot_PhysicsNeMo(
    ax1, nx, ny, nz, look, N_injw, N_pr, N_injg, "P50", injectors, producers, gass
)

look = ((np.reshape(controlbad, (nx, ny, nz), "F")) * effectiveuse)[:, :, ::-1]
ax1 = f_3.add_subplot(3, 3, 5, projection="3d")
Plot_PhysicsNeMo(
    ax1, nx, ny, nz, look, N_injw, N_pr, N_injg, "P90", injectors, producers, gass
)


look = ((np.reshape(yes_best_k, (nx, ny, nz), "F")) * effectiveuse)[:, :, ::-1]
ax1 = f_3.add_subplot(3, 3, 6, projection="3d")
Plot_PhysicsNeMo(
    ax1, nx, ny, nz, look, N_injw, N_pr, N_injg, "cumm-best", injectors, producers, gass
)

look = ((np.reshape(yes_mean_k, (nx, ny, nz), "F")) * effectiveuse)[:, :, ::-1]
ax1 = f_3.add_subplot(3, 3, 7, projection="3d")
Plot_PhysicsNeMo(
    ax1, nx, ny, nz, look, N_injw, N_pr, N_injg, "cumm-mean", injectors, producers, gass
)


plt.tight_layout(rect=[0, 0, 1, 0.95])

tita = "Reservoir Models permeability Fields"
plt.suptitle(tita, fontsize=16)
plt.savefig("Reservoir_models.png")
plt.clf()
plt.close()

os.chdir(oldfolder)
print("--------------------SECTION ADAPTIVE REKI ENDED----------------------------")

elapsed_time_secs = time.time() - start_time


comment = "Adaptive Regularised Ensemble Kalman Inversion"

if Trainmoe == 2:
    comment2 = "PINO-CCR"
else:
    comment2 = "PINO-FNO"

print("Inverse problem solution used =: " + comment)
print("Forward model surrogate =: " + comment2)
print("Ensemble size = ", str(Ne))
msg = "Execution took: %s secs (Wall clock time)" % timedelta(
    seconds=round(elapsed_time_secs)
)
print(msg)
textaa = """
______                                                      _____                    _           _   _ _ 
| ___ \                                                    |  ___|                  | |         | | | | |
| |_/ _ __ ___   __ _ _ __ __ _ _ __ ___  _ __ ___   ___   | |____  _____  ___ _   _| |_ ___  __| | | | |
|  __| '__/ _ \ / _` | '__/ _` | '_ ` _ \| '_ ` _ \ / _ \  |  __\ \/ / _ \/ __| | | | __/ _ \/ _` | | | |
| |  | | | (_) | (_| | | | (_| | | | | | | | | | | |  __/  | |___>  |  __| (__| |_| | ||  __| (_| | |_|_|
\_|  |_|  \___/ \__, |_|  \__,_|_| |_| |_|_| |_| |_|\___|  \____/_/\_\___|\___|\__,_|\__\___|\__,_| (_(_)
                 __/ |                                                                                   
                |___/                                                                                                                                                               
"""
print(textaa)
print("-------------------PROGRAM EXECUTED-----------------------------------")
