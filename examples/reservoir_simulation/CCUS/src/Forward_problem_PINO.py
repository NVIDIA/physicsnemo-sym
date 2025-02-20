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
import torch.nn.functional as F
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
from physicsnemo.sym.node import Node
import requests
from typing import Union, Tuple
from pathlib import Path
from functools import reduce
import torch
from torch.optim import Optimizer
import warnings
import numbers
from scipy.optimize import Bounds, NonlinearConstraint
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import OptimizeResult

# from scipy.optimize.optimize import _status_message
from abc import ABC, abstractmethod
from torch import Tensor
from typing import List, Optional
from collections import namedtuple
import torch.autograd as autograd
from torch._vmap_internals import _vmap
from scipy.sparse.linalg import eigsh
from torch.optim.lbfgs import _strong_wolfe, _cubic_interpolate
from scipy.optimize import minimize_scalar
from torch.linalg import norm
from scipy.linalg import eigh_tridiagonal, get_lapack_funcs

print(" ")
print(" This computer has %d cores, which will all be utilised in parallel " % cores)
print(" ")
print("......................DEFINE SOME FUNCTIONS.....................")


def _minimize_trust_ncg(fun, x0, **trust_region_options):
    """Minimization of scalar function of one or more variables using
    the Newton conjugate gradient trust-region algorithm.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    initial_trust_radius : float
        Initial trust-region radius.
    max_trust_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than ``gtol`` before successful
        termination.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    Notes
    -----
    This is algorithm (7.2) of Nocedal and Wright 2nd edition.
    Only the function that computes the Hessian-vector product is required.
    The Hessian itself is not required, and the Hessian does
    not need to be positive semidefinite.

    """
    return _minimize_trust_region(
        fun, x0, subproblem=CGSteihaugSubproblem, **trust_region_options
    )


class BaseQuadraticSubproblem(ABC):
    """
    Base/abstract class defining the quadratic model for trust-region
    minimization. Child classes must implement the ``solve`` method and
    ``hess_prod`` property.
    """

    def __init__(self, x, closure):
        # evaluate closure
        f, g, hessp, hess = closure(x)

        self._x = x
        self._f = f
        self._g = g
        self._h = hessp if self.hess_prod else hess
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None

        # buffer for boundaries computation
        self._tab = x.new_empty(2)

    def __call__(self, p):
        return self.fun + self.jac.dot(p) + 0.5 * p.dot(self.hessp(p))

    @property
    def fun(self):
        """Value of objective function at current iteration."""
        return self._f

    @property
    def jac(self):
        """Value of Jacobian of objective function at current iteration."""
        return self._g

    @property
    def hess(self):
        """Value of Hessian of objective function at current iteration."""
        if self.hess_prod:
            raise Exception(
                "class {} does not have " "method `hess`".format(type(self))
            )
        return self._h

    def hessp(self, p):
        """Value of Hessian-vector product at current iteration for a
        particular vector ``p``.

        Note: ``self._h`` is either a Tensor or a LinearOperator. In either
        case, it has a method ``mv()``.
        """
        return self._h.mv(p)

    @property
    def jac_mag(self):
        """Magnitude of jacobian of objective function at current iteration."""
        if self._g_mag is None:
            self._g_mag = norm(self.jac)
        return self._g_mag

    def get_boundaries_intersections(self, z, d, trust_radius):
        """
        Solve the scalar quadratic equation ||z + t d|| == trust_radius.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
        a = d.dot(d)
        b = 2 * z.dot(d)
        c = z.dot(z) - trust_radius**2
        sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)

        # The following calculation is mathematically equivalent to:
        #   ta = (-b - sqrt_discriminant) / (2*a)
        #   tb = (-b + sqrt_discriminant) / (2*a)
        # but produces smaller round off errors.
        aux = b + torch.copysign(sqrt_discriminant, b)
        self._tab[0] = -aux / (2 * a)
        self._tab[1] = -2 * c / aux
        return self._tab.sort()[0]

    @abstractmethod
    def solve(self, trust_radius):
        pass

    @property
    @abstractmethod
    def hess_prod(self):
        """A property that must be set by every sub-class indicating whether
        to use full hessian matrix or hessian-vector products."""
        pass


# class PointwiseLossNormC(Loss):
#     """
#     L-p loss function for pointwise data
#     Computes the p-th order loss of each output tensor

#     Parameters
#     ----------
#     ord : int
#         Order of the loss. For example, `ord=2` would be the L2 loss.
#     """

#     def __init__(self, ord: int = 2):
#         super().__init__()
#         self.ord: int = ord

#     @staticmethod
#     def _loss(
#         invar: Dict[str, Tensor],
#         pred_outvar: Dict[str, Tensor],
#         true_outvar: Dict[str, Tensor],
#         lambda_weighting: Dict[str, Tensor],
#         step: int,
#         ord: float,
#     ) -> Dict[str, Tensor]:
#         losses = {}
#         for key, value in pred_outvar.items():
#             l = lambda_weighting[key] * torch.abs((pred_outvar[key] - true_outvar[key])).pow(ord)
#             if "area" in invar.keys():
#                 l *= invar["area"]
#             losses[key] = l.sum()
#         return losses

#     def forward(
#         self,
#         invar: Dict[str, Tensor],
#         pred_outvar: Dict[str, Tensor],
#         true_outvar: Dict[str, Tensor],
#         lambda_weighting: Dict[str, Tensor],
#         step: int,
#     ) -> Dict[str, Tensor]:
#         return PointwiseLossNormC._loss(
#             invar, pred_outvar, true_outvar, lambda_weighting, step, self.ord
#         )


# def replace_nans_and_infs_with_mean(tensor):
#     """
#     Replace NaNs and infinities in a tensor with the mean of its finite values.
#     """
#     finite_vals = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]
#     if len(finite_vals) > 0:  # Ensure there are finite values to compute the mean
#         mean_val = finite_vals.mean()
#         tensor[torch.isnan(tensor) | torch.isinf(tensor)] = mean_val
#     else:
#         # Optional: Handle case where all values are NaN or inf by setting to a default value, e.g., 0
#         tensor[:] = 0
#     return tensor


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
            0.0, device=tensor.device
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

        l = lambda_weighting["Pco2_g"] * torch.abs(pred_outvar["Pco2_g"])
        # l = l/(torch.prod(torch.tensor(pred_outvar["Pco2_g"].size())).item())

        if "area" in invar.keys():
            l *= invar["area"]
        losses["Pco2_g"] = l.sum()  # It is fine

        l = lambda_weighting["Pco2_g"] * torch.abs(pred_outvar["Pco2_l"])
        # l = l/(torch.prod(torch.tensor(pred_outvar["Pco2_g"].size())).item())
        if "area" in invar.keys():
            l *= invar["area"]
        losses["Pco2_l"] = l.sum()  # Bad

        l = lambda_weighting["Pco2_g"] * torch.abs(pred_outvar["Ph2o_l"])
        # l = l/(torch.prod(torch.tensor(pred_outvar["Pco2_g"].size())).item())
        if "area" in invar.keys():
            l *= invar["area"]
        losses["Ph2o_l"] = l.sum()  # bad

        l = lambda_weighting["Pco2_g"] * torch.abs(pred_outvar["Sco2_g"])
        # l = l/(torch.prod(torch.tensor(pred_outvar["Pco2_g"].size())).item())
        if "area" in invar.keys():
            l *= invar["area"]
        losses["Sco2_g"] = l.sum()

        l = lambda_weighting["Pco2_g"] * torch.abs(pred_outvar["Sco2_l"])
        # l = l/(torch.prod(torch.tensor(pred_outvar["Pco2_g"].size())).item())
        if "area" in invar.keys():
            l *= invar["area"]
        losses["Sco2_l"] = l.sum()

        l = lambda_weighting["Pco2_g"] * torch.abs(pred_outvar["Sh2o_l"])
        # l = l/(torch.prod(torch.tensor(pred_outvar["Pco2_g"].size())).item())
        if "area" in invar.keys():
            l *= invar["area"]
        losses["Sh2o_l"] = l.sum()

        l = lambda_weighting["Pco2_g"] * torch.abs(pred_outvar["satwp"])
        # l = l/(torch.prod(torch.tensor(pred_outvar["Pco2_g"].size())).item())
        if "area" in invar.keys():
            l *= invar["area"]
        losses["satwp"] = l.sum()

        l = lambda_weighting["Pco2_g"] * torch.abs(pred_outvar["satgp"])
        # l = l/(torch.prod(torch.tensor(pred_outvar["Pco2_g"].size())).item())
        if "area" in invar.keys():
            l *= invar["area"]
        losses["satgp"] = l.sum()
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


def corey_relative_permeability_torch(S, Swi, Sor, Krend, n):
    """
    Compute the relative permeability using the Corey model for PyTorch tensors.

    Parameters:
    - S: Tensor of saturations for the phase of interest.
    - Swi: Irreducible water saturation (scalar or tensor with the same shape as S).
    - Sor: Residual oil saturation (scalar or tensor with the same shape as S).
    - Krend: End-point relative permeability for the phase (scalar).
    - n: Corey exponent for the phase (scalar).

    Returns:
    - Kr: Tensor of relative permeabilities to the phase.
    """
    Se = (S - Swi) / (1 - Sor - Swi)
    Kr = Krend * Se**n
    return Kr


class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method"""

    hess_prod = True

    def solve(self, trust_radius):
        """Solve the subproblem using a conjugate gradient method.

        Parameters
        ----------
        trust_radius : float
            We are allowed to wander only this far away from the origin.

        Returns
        -------
        p : Tensor
            The proposed step.
        hits_boundary : bool
            True if the proposed step is on the boundary of the trust region.

        """

        # get the norm of jacobian and define the origin
        p_origin = torch.zeros_like(self.jac)

        # define a default tolerance
        tolerance = self.jac_mag * self.jac_mag.sqrt().clamp(max=0.5)

        # Stop the method if the search direction
        # is a direction of nonpositive curvature.
        if self.jac_mag < tolerance:
            hits_boundary = False
            return p_origin, hits_boundary

        # init the state for the first iteration
        z = p_origin
        r = self.jac
        d = -r

        # Search for the min of the approximation of the objective function.
        while True:

            # do an iteration
            Bd = self.hessp(d)
            dBd = d.dot(Bd)
            if dBd <= 0:
                # Look at the two boundary points.
                # Find both values of t to get the boundary points such that
                # ||z + t d|| == trust_radius
                # and then choose the one with the predicted min value.
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                p_boundary = torch.where(self(pa).lt(self(pb)), pa, pb)
                hits_boundary = True
                return p_boundary, hits_boundary
            r_squared = r.dot(r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if norm(z_next) >= trust_radius:
                # Find t >= 0 to get the boundary point such that
                # ||z + t d|| == trust_radius
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return p_boundary, hits_boundary
            r_next = r + alpha * Bd
            r_next_squared = r_next.dot(r_next)
            if r_next_squared.sqrt() < tolerance:
                hits_boundary = False
                return z_next, hits_boundary
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d

            # update the state for the next iteration
            z = z_next
            r = r_next
            d = d_next


def _minimize_trust_krylov(fun, x0, **trust_region_options):
    """Minimization of scalar function of one or more variables using
    the GLTR Krylov subspace trust-region algorithm.

    .. warning::
        This minimizer is in early stages and has not been rigorously
        tested. It may change in the near future.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    initial_tr_radius : float
        Initial trust-region radius.
    max_tr_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than ``gtol`` before successful
        termination.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    Notes
    -----
    This trust-region solver is based on the GLTR algorithm as
    described in [1]_ and [2]_.

    References
    ----------
    .. [1] F. Lenders, C. Kirches, and A. Potschka, "trlib: A vector-free
           implementation of the GLTR method for...",
           arXiv:1611.04718.
    .. [2] N. Gould, S. Lucidi, M. Roma, P. Toint: “Solving the Trust-Region
           Subproblem using the Lanczos Method”,
           SIAM J. Optim., 9(2), 504–525, 1999.
    .. [3] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.

    """
    return _minimize_trust_region(
        fun, x0, subproblem=KrylovSubproblem, **trust_region_options
    )


class KrylovSubproblem(BaseQuadraticSubproblem):
    """The GLTR trust region sub-problem defined on an expanding
    Krylov subspace.

    Based on the implementation of GLTR described in [1]_.

    References
    ----------
    .. [1] F. Lenders, C. Kirches, and A. Potschka, "trlib: A vector-free
           implementation of the GLTR method for...",
           arXiv:1611.04718.
    .. [2] N. Gould, S. Lucidi, M. Roma, P. Toint: “Solving the Trust-Region
           Subproblem using the Lanczos Method”,
           SIAM J. Optim., 9(2), 504–525, 1999.
    .. [3] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    """

    hess_prod = True
    max_lanczos = None
    max_ms_iters = 500  # max iterations of the Moré-Sorensen loop

    def __init__(
        self, x, fun, k_easy=0.1, k_hard=0.2, tol=1e-5, ortho=True, debug=False
    ):
        super().__init__(x, fun)
        self.eps = torch.finfo(x.dtype).eps
        self.k_easy = k_easy
        self.k_hard = k_hard
        self.tol = tol
        self.ortho = ortho
        self._debug = debug

    def tridiag_subproblem(self, Ta, Tb, tr_radius):
        """Solve the GLTR tridiagonal subproblem.

        Based on Algorithm 5.2 of [2]_. We factorize as follows:

        .. math::
            T + lambd * I = LDL^T

        Where `D` is diagonal and `L` unit (lower) bi-diagonal.
        """
        device, dtype = Ta.device, Ta.dtype

        # convert to numpy
        Ta = Ta.cpu().numpy()
        Tb = Tb.cpu().numpy()
        tr_radius = float(tr_radius)

        # right hand side
        rhs = np.zeros_like(Ta)
        rhs[0] = -float(self.jac_mag)

        # get LAPACK routines for factorizing and solving sym-PD tridiagonal
        ptsv, pttrs = get_lapack_funcs(("ptsv", "pttrs"), (Ta, Tb, rhs))

        eig0 = None
        lambd_lb = 0.0
        lambd = 0.0
        for _ in range(self.max_ms_iters):
            lambd = max(lambd, lambd_lb)

            # factor T + lambd * I = LDL^T and solve LDL^T p = rhs
            d, e, p, info = ptsv(Ta + lambd, Tb, rhs)
            assert info >= 0  # sanity check
            if info > 0:
                assert eig0 is None  # sanity check; should only happen once
                # estimate smallest eigenvalue and continue
                eig0 = eigh_tridiagonal(
                    Ta,
                    Tb,
                    eigvals_only=True,
                    select="i",
                    select_range=(0, 0),
                    lapack_driver="stebz",
                ).item()
                lambd_lb = max(1e-3 - eig0, 0)
                continue

            p_norm = np.linalg.norm(p)
            if p_norm < tr_radius:
                # TODO: add extra checks
                status = 0
                break
            elif abs(p_norm - tr_radius) / tr_radius <= self.k_easy:
                status = 1
                break

            # solve LDL^T q = p and compute <q, p>
            v, info = pttrs(d, e, p)
            q_norm2 = v.dot(p)

            # update lambd
            lambd += (p_norm**2 / q_norm2) * (p_norm - tr_radius) / tr_radius
        else:
            status = -1

        p = torch.tensor(p, device=device, dtype=dtype)

        return p, status, lambd

    def solve(self, tr_radius):
        g = self.jac
        gamma_0 = self.jac_mag
        (n,) = g.shape
        m = n if self.max_lanczos is None else min(n, self.max_lanczos)
        dtype = g.dtype
        device = g.device
        h_best = None
        error_best = float("inf")

        # Lanczos Q matrix buffer
        Q = torch.zeros(m, n, dtype=dtype, device=device)
        Q[0] = g / gamma_0

        # Lanczos T matrix buffers
        # a and b are the diagonal and off-diagonal entries of T, respectively
        a = torch.zeros(m, dtype=dtype, device=device)
        b = torch.zeros(m, dtype=dtype, device=device)

        # first lanczos iteration
        r = self.hessp(Q[0])
        torch.dot(Q[0], r, out=a[0])
        r.sub_(Q[0] * a[0])
        torch.linalg.norm(r, out=b[0])
        if b[0] < self.eps:
            raise RuntimeError("initial beta is zero.")

        # remaining iterations
        for i in range(1, m):
            torch.div(r, b[i - 1], out=Q[i])
            r = self.hessp(Q[i])
            r.sub_(Q[i - 1] * b[i - 1])
            torch.dot(Q[i], r, out=a[i])
            r.sub_(Q[i] * a[i])
            if self.ortho:
                # Re-orthogonalize with Gram-Schmidt
                r.addmv_(Q[: i + 1].T, Q[: i + 1].mv(r), alpha=-1)
            torch.linalg.norm(r, out=b[i])
            if b[i] < self.eps:
                # This should never occur when self.ortho=True
                raise RuntimeError("reducible T matrix encountered.")

            # GLTR sub-problem
            h, status, lambd = self.tridiag_subproblem(a[: i + 1], b[:i], tr_radius)

            if status >= 0:
                # convergence check; see Algorithm 1 of [1]_ and
                # Algorithm 5.1 of [2]_. Equivalent to the following:
                #     p = Q[:i+1].T.mv(h)
                #     error = torch.linalg.norm(self.hessp(p) + lambd * p + g)
                error = b[i] * h[-1].abs()
                if self._debug:
                    print(
                        "iter %3d - status: %d - lambd: %0.4e - error: %0.4e"
                        % (i + 1, status, lambd, error)
                    )
                if error < error_best:
                    # we've found a new best
                    hits_boundary = status != 0
                    h_best = h
                    error_best = error
                    if error_best <= self.tol:
                        break

            elif self._debug:
                print("iter %3d - status: %d - lambd: %0.4e" % (i + 1, status, lambd))

        if h_best is None:
            # TODO: what should we do here?
            raise RuntimeError("gltr solution not found")

        # project h back to R^n
        p_best = Q[: i + 1].T.mv(h_best)

        return p_best, hits_boundary


def _minimize_trust_exact(fun, x0, **trust_region_options):
    """Minimization of scalar function of one or more variables using
    a nearly exact trust-region algorithm.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    initial_tr_radius : float
        Initial trust-region radius.
    max_tr_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than ``gtol`` before successful
        termination.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    Notes
    -----
    This trust-region solver was based on [1]_, [2]_ and [3]_,
    which implement similar algorithms. The algorithm is basically
    that of [1]_ but ideas from [2]_ and [3]_ were also used.

    References
    ----------
    .. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",
           Siam, pp. 169-200, 2000.
    .. [2] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    .. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",
           SIAM Journal on Scientific and Statistical Computing, vol. 4(3),
           pp. 553-572, 1983.

    """
    return _minimize_trust_region(
        fun, x0, subproblem=IterativeSubproblem, **trust_region_options
    )


def solve_triangular(A, b, **kwargs):
    return torch.triangular_solve(b.unsqueeze(1), A, **kwargs)[0].squeeze(1)


def solve_cholesky(A, b, **kwargs):
    return torch.cholesky_solve(b.unsqueeze(1), A, **kwargs).squeeze(1)


@torch.jit.script
def estimate_smallest_singular_value(U) -> Tuple[Tensor, Tensor]:
    """Given upper triangular matrix ``U`` estimate the smallest singular
    value and the correspondent right singular vector in O(n**2) operations.

    A vector `e` with components selected from {+1, -1}
    is selected so that the solution `w` to the system
    `U.T w = e` is as large as possible. Implementation
    based on algorithm 3.5.1, p. 142, from reference [1]_
    adapted for lower triangular matrix.

    References
    ----------
    .. [1] G.H. Golub, C.F. Van Loan. "Matrix computations".
           Forth Edition. JHU press. pp. 140-142.
    """

    U = torch.atleast_2d(U)
    UT = U.T
    m, n = U.shape
    if m != n:
        raise ValueError("A square triangular matrix should be provided.")

    p = torch.zeros(n, dtype=U.dtype, device=U.device)
    w = torch.empty(n, dtype=U.dtype, device=U.device)

    for k in range(n):
        wp = (1 - p[k]) / UT[k, k]
        wm = (-1 - p[k]) / UT[k, k]
        pp = p[k + 1 :] + UT[k + 1 :, k] * wp
        pm = p[k + 1 :] + UT[k + 1 :, k] * wm

        if wp.abs() + norm(pp, 1) >= wm.abs() + norm(pm, 1):
            w[k] = wp
            p[k + 1 :] = pp
        else:
            w[k] = wm
            p[k + 1 :] = pm

    # The system `U v = w` is solved using backward substitution.
    v = torch.triangular_solve(w.view(-1, 1), U)[0].view(-1)
    v_norm = norm(v)

    s_min = norm(w) / v_norm  # Smallest singular value
    z_min = v / v_norm  # Associated vector

    return s_min, z_min


def gershgorin_bounds(H):
    """
    Given a square matrix ``H`` compute upper
    and lower bounds for its eigenvalues (Gregoshgorin Bounds).
    """
    H_diag = torch.diag(H)
    H_diag_abs = H_diag.abs()
    H_row_sums = H.abs().sum(dim=1)
    lb = torch.min(H_diag + H_diag_abs - H_row_sums)
    ub = torch.max(H_diag - H_diag_abs + H_row_sums)

    return lb, ub


def singular_leading_submatrix(A, U, k):
    """
    Compute term that makes the leading ``k`` by ``k``
    submatrix from ``A`` singular.
    """
    # Compute delta
    delta = torch.sum(U[: k - 1, k - 1] ** 2) - A[k - 1, k - 1]

    # Initialize v
    v = A.new_zeros(A.shape[0])
    v[k - 1] = 1

    # Compute the remaining values of v by solving a triangular system.
    if k != 1:
        v[: k - 1] = solve_triangular(U[: k - 1, : k - 1], -U[: k - 1, k - 1])

    return delta, v


class IterativeSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by nearly exact iterative method."""

    # UPDATE_COEFF appears in reference [1]_
    # in formula 7.3.14 (p. 190) named as "theta".
    # As recommended there it value is fixed in 0.01.
    UPDATE_COEFF = 0.01
    hess_prod = False

    def __init__(self, x, fun, k_easy=0.1, k_hard=0.2):

        super().__init__(x, fun)

        # When the trust-region shrinks in two consecutive
        # calculations (``tr_radius < previous_tr_radius``)
        # the lower bound ``lambda_lb`` may be reused,
        # facilitating  the convergence. To indicate no
        # previous value is known at first ``previous_tr_radius``
        # is set to -1  and ``lambda_lb`` to None.
        self.previous_tr_radius = -1
        self.lambda_lb = None

        self.niter = 0
        self.EPS = torch.finfo(x.dtype).eps

        # ``k_easy`` and ``k_hard`` are parameters used
        # to determine the stop criteria to the iterative
        # subproblem solver. Take a look at pp. 194-197
        # from reference _[1] for a more detailed description.
        self.k_easy = k_easy
        self.k_hard = k_hard

        # Get Lapack function for cholesky decomposition.
        # NOTE: cholesky_ex requires pytorch >= 1.9.0
        if "cholesky_ex" in dir(torch.linalg):
            self.torch_cholesky = True
        else:
            # if we don't have torch cholesky, use potrf from scipy
            (self.cholesky,) = get_lapack_funcs(("potrf",), (self.hess.cpu().numpy(),))
            self.torch_cholesky = False

        # Get info about Hessian
        self.dimension = len(self.hess)
        self.hess_gershgorin_lb, self.hess_gershgorin_ub = gershgorin_bounds(self.hess)
        self.hess_inf = norm(self.hess, float("inf"))
        self.hess_fro = norm(self.hess, "fro")

        # A constant such that for vectors smaler than that
        # backward substituition is not reliable. It was stabilished
        # based on Golub, G. H., Van Loan, C. F. (2013).
        # "Matrix computations". Forth Edition. JHU press., p.165.
        self.CLOSE_TO_ZERO = self.dimension * self.EPS * self.hess_inf

    def _initial_values(self, tr_radius):
        """Given a trust radius, return a good initial guess for
        the damping factor, the lower bound and the upper bound.
        The values were chosen accordingly to the guidelines on
        section 7.3.8 (p. 192) from [1]_.
        """
        hess_norm = torch.min(self.hess_fro, self.hess_inf)

        # Upper bound for the damping factor
        lambda_ub = self.jac_mag / tr_radius + torch.min(
            -self.hess_gershgorin_lb, hess_norm
        )
        lambda_ub = torch.clamp(lambda_ub, min=0)

        # Lower bound for the damping factor
        lambda_lb = self.jac_mag / tr_radius - torch.min(
            self.hess_gershgorin_ub, hess_norm
        )
        lambda_lb = torch.max(lambda_lb, -self.hess.diagonal().min())
        lambda_lb = torch.clamp(lambda_lb, min=0)

        # Improve bounds with previous info
        if tr_radius < self.previous_tr_radius:
            lambda_lb = torch.max(self.lambda_lb, lambda_lb)

        # Initial guess for the damping factor
        if lambda_lb == 0:
            lambda_initial = lambda_lb.clone()
        else:
            lambda_initial = torch.max(
                torch.sqrt(lambda_lb * lambda_ub),
                lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb),
            )

        return lambda_initial, lambda_lb, lambda_ub

    def solve(self, tr_radius):
        """Solve quadratic subproblem"""

        lambda_current, lambda_lb, lambda_ub = self._initial_values(tr_radius)
        n = self.dimension
        hits_boundary = True
        already_factorized = False
        self.niter = 0

        while True:
            # Compute Cholesky factorization
            if already_factorized:
                already_factorized = False
            else:
                H = self.hess.clone()
                H.diagonal().add_(lambda_current)
                if self.torch_cholesky:
                    U, info = torch.linalg.cholesky_ex(H)
                    U = U.t().contiguous()
                else:
                    U, info = self.cholesky(
                        H.cpu().numpy(), lower=False, overwrite_a=False, clean=True
                    )
                    U = H.new_tensor(U)

            self.niter += 1

            # Check if factorization succeeded
            if info == 0 and self.jac_mag > self.CLOSE_TO_ZERO:
                # Successful factorization

                # Solve `U.T U p = s`
                p = solve_cholesky(U, -self.jac, upper=True)
                p_norm = norm(p)

                # Check for interior convergence
                if p_norm <= tr_radius and lambda_current == 0:
                    hits_boundary = False
                    break

                # Solve `U.T w = p`
                w = solve_triangular(U, p, transpose=True)
                w_norm = norm(w)

                # Compute Newton step accordingly to
                # formula (4.44) p.87 from ref [2]_.
                delta_lambda = (p_norm / w_norm) ** 2 * (p_norm - tr_radius) / tr_radius
                lambda_new = lambda_current + delta_lambda

                if p_norm < tr_radius:  # Inside boundary
                    s_min, z_min = estimate_smallest_singular_value(U)

                    ta, tb = self.get_boundaries_intersections(p, z_min, tr_radius)

                    # Choose `step_len` with the smallest magnitude.
                    # The reason for this choice is explained at
                    # ref [3]_, p. 6 (Immediately before the formula
                    # for `tau`).
                    step_len = torch.min(ta.abs(), tb.abs())

                    # Compute the quadratic term  (p.T*H*p)
                    quadratic_term = p.dot(H.mv(p))

                    # Check stop criteria
                    relative_error = (step_len**2 * s_min**2) / (
                        quadratic_term + lambda_current * tr_radius**2
                    )
                    if relative_error <= self.k_hard:
                        p.add_(step_len * z_min)
                        break

                    # Update uncertanty bounds
                    lambda_ub = lambda_current
                    lambda_lb = torch.max(lambda_lb, lambda_current - s_min**2)

                    # Compute Cholesky factorization
                    H = self.hess.clone()
                    H.diagonal().add_(lambda_new)
                    if self.torch_cholesky:
                        _, info = torch.linalg.cholesky_ex(H)
                    else:
                        _, info = self.cholesky(
                            H.cpu().numpy(), lower=False, overwrite_a=False, clean=True
                        )

                    if info == 0:
                        lambda_current = lambda_new
                        already_factorized = True
                    else:
                        lambda_lb = torch.max(lambda_lb, lambda_new)
                        lambda_current = torch.max(
                            torch.sqrt(lambda_lb * lambda_ub),
                            lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb),
                        )

                else:  # Outside boundary
                    # Check stop criteria
                    relative_error = torch.abs(p_norm - tr_radius) / tr_radius
                    if relative_error <= self.k_easy:
                        break

                    # Update uncertanty bounds
                    lambda_lb = lambda_current

                    # Update damping factor
                    lambda_current = lambda_new

            elif info == 0 and self.jac_mag <= self.CLOSE_TO_ZERO:
                # jac_mag very close to zero

                # Check for interior convergence
                if lambda_current == 0:
                    p = self.jac.new_zeros(n)
                    hits_boundary = False
                    break

                s_min, z_min = estimate_smallest_singular_value(U)
                step_len = tr_radius

                # Check stop criteria
                if (
                    step_len**2 * s_min**2
                    <= self.k_hard * lambda_current * tr_radius**2
                ):
                    p = step_len * z_min
                    break

                # Update uncertainty bounds and dampening factor
                lambda_ub = lambda_current
                lambda_lb = torch.max(lambda_lb, lambda_current - s_min**2)
                lambda_current = torch.max(
                    torch.sqrt(lambda_lb * lambda_ub),
                    lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb),
                )

            else:
                # Unsuccessful factorization

                delta, v = singular_leading_submatrix(H, U, info)
                v_norm = norm(v)

                lambda_lb = torch.max(lambda_lb, lambda_current + delta / v_norm**2)

                # Update damping factor
                lambda_current = torch.max(
                    torch.sqrt(lambda_lb * lambda_ub),
                    lambda_lb + self.UPDATE_COEFF * (lambda_ub - lambda_lb),
                )

        self.lambda_lb = lambda_lb
        self.lambda_current = lambda_current
        self.previous_tr_radius = tr_radius

        return p, hits_boundary


def _minimize_dogleg(fun, x0, **trust_region_options):
    """Minimization of scalar function of one or more variables using
    the dog-leg trust-region algorithm.

    .. warning::
        The Hessian is required to be positive definite at all times;
        otherwise this algorithm will fail.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    initial_trust_radius : float
        Initial trust-region radius.
    max_trust_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    References
    ----------
    .. [1] Jorge Nocedal and Stephen Wright,
           Numerical Optimization, second edition,
           Springer-Verlag, 2006, page 73.

    """
    return _minimize_trust_region(
        fun, x0, subproblem=DoglegSubproblem, **trust_region_options
    )


class DoglegSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by the dogleg method"""

    hess_prod = False

    def cauchy_point(self):
        """
        The Cauchy point is minimal along the direction of steepest descent.
        """
        if self._cauchy_point is None:
            g = self.jac
            Bg = self.hessp(g)
            self._cauchy_point = -(g.dot(g) / g.dot(Bg)) * g
        return self._cauchy_point

    def newton_point(self):
        """
        The Newton point is a global minimum of the approximate function.
        """
        if self._newton_point is None:
            p = -torch.cholesky_solve(
                self.jac.view(-1, 1), torch.linalg.cholesky(self.hess)
            )
            self._newton_point = p.view(-1)
        return self._newton_point

    def solve(self, trust_radius):
        """Solve quadratic subproblem"""

        # Compute the Newton point.
        # This is the optimum for the quadratic model function.
        # If it is inside the trust radius then return this point.
        p_best = self.newton_point()
        if norm(p_best) < trust_radius:
            hits_boundary = False
            return p_best, hits_boundary

        # Compute the Cauchy point.
        # This is the predicted optimum along the direction of steepest descent.
        p_u = self.cauchy_point()

        # If the Cauchy point is outside the trust region,
        # then return the point where the path intersects the boundary.
        p_u_norm = norm(p_u)
        if p_u_norm >= trust_radius:
            p_boundary = p_u * (trust_radius / p_u_norm)
            hits_boundary = True
            return p_boundary, hits_boundary

        # Compute the intersection of the trust region boundary
        # and the line segment connecting the Cauchy and Newton points.
        # This requires solving a quadratic equation.
        # ||p_u + t*(p_best - p_u)||**2 == trust_radius**2
        # Solve this for positive time t using the quadratic formula.
        _, tb = self.get_boundaries_intersections(p_u, p_best - p_u, trust_radius)
        p_boundary = p_u + tb * (p_best - p_u)
        hits_boundary = True
        return p_boundary, hits_boundary


def _minimize_trust_region(
    fun,
    x0,
    subproblem=None,
    initial_trust_radius=1.0,
    max_trust_radius=1000.0,
    eta=0.15,
    gtol=1e-4,
    max_iter=None,
    disp=False,
    return_all=False,
    callback=None,
):
    """
    Minimization of scalar function of one or more variables using a
    trust-region algorithm.

    Options for the trust-region algorithm are:
        initial_trust_radius : float
            Initial trust radius.
        max_trust_radius : float
            Never propose steps that are longer than this value.
        eta : float
            Trust region related acceptance stringency for proposed steps.
        gtol : float
            Gradient norm must be less than `gtol`
            before successful termination.
        max_iter : int
            Maximum number of iterations to perform.
        disp : bool
            If True, print convergence message.

    This function is called by :func:`torchmin.minimize`.
    It is not supposed to be called directly.
    """
    if subproblem is None:
        raise ValueError(
            "A subproblem solving strategy is required for " "trust-region methods"
        )
    if not (0 <= eta < 0.25):
        raise Exception("invalid acceptance stringency")
    if max_trust_radius <= 0:
        raise Exception("the max trust radius must be positive")
    if initial_trust_radius <= 0:
        raise ValueError("the initial trust radius must be positive")
    if initial_trust_radius >= max_trust_radius:
        raise ValueError(
            "the initial trust radius must be less than the " "max trust radius"
        )

    # Input check/pre-process
    disp = int(disp)
    if max_iter is None:
        max_iter = x0.numel() * 200

    # Construct scalar objective function
    hessp = subproblem.hess_prod
    sf = ScalarFunction(fun, x0.shape, hessp=hessp, hess=not hessp)
    closure = sf.closure

    # init the search status
    warnflag = 1  # maximum iterations flag
    k = 0

    # initialize the search
    trust_radius = torch.as_tensor(
        initial_trust_radius, dtype=x0.dtype, device=x0.device
    )
    x = x0.detach().flatten()
    if return_all:
        allvecs = [x]

    # initial subproblem
    m = subproblem(x, closure)

    # search for the function min
    # do not even start if the gradient is small enough
    while k < max_iter:

        # Solve the sub-problem.
        # This gives us the proposed step relative to the current position
        # and it tells us whether the proposed step
        # has reached the trust region boundary or not.
        try:
            p, hits_boundary = m.solve(trust_radius)
        except RuntimeError as exc:
            # TODO: catch general linalg error like np.linalg.linalg.LinAlgError
            if "singular" in exc.args[0]:
                warnflag = 3
                break
            else:
                raise

        # calculate the predicted value at the proposed point
        predicted_value = m(p)

        # define the local approximation at the proposed point
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, closure)

        # evaluate the ratio defined in equation (4.4)
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction

        # update the trust radius according to the actual/predicted ratio
        if rho < 0.25:
            trust_radius = trust_radius.mul(0.25)
        elif rho > 0.75 and hits_boundary:
            trust_radius = torch.clamp(2 * trust_radius, max=max_trust_radius)

        # if the ratio is high enough then accept the proposed step
        if rho > eta:
            x = x_proposed
            m = m_proposed

        # append the best guess, call back, increment the iteration count
        if return_all:
            allvecs.append(x.clone())
        if callback is not None:
            callback(x.clone())
        k += 1

        # verbosity check
        if disp > 1:
            print("iter %d - fval: %0.4f" % (k, m.fun))

        # check if the gradient is small enough to stop
        if m.jac_mag < gtol:
            warnflag = 0
            break

    # print some stuff if requested
    if disp:
        msg = status_messages[warnflag]
        if warnflag != 0:
            msg = "Warning: " + msg
        print(msg)
        print("         Current function value: %f" % m.fun)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        # print("         Gradient evaluations: %d" % sf.ngev)
        # print("         Hessian evaluations: %d" % (sf.nhev + nhessp[0]))

    result = OptimizeResult(
        x=x.view_as(x0),
        fun=m.fun,
        grad=m.jac.view_as(x0),
        success=(warnflag == 0),
        status=warnflag,
        nfev=sf.nfev,
        nit=k,
        message=status_messages[warnflag],
    )

    if not subproblem.hess_prod:
        result["hess"] = m.hess.view(*x0.shape, *x0.shape)

    if return_all:
        result["allvecs"] = allvecs

    return result


class LinearOperatorr:
    """A generic linear operator to use with Minimizer"""

    def __init__(self, matvec, shape, dtype=torch.float, device=None):
        self.rmv = matvec
        self.mv = matvec
        self.shape = shape
        self.dtype = dtype
        self.device = device


class Minimizer(Optimizer):
    """A general-purpose PyTorch optimizer for unconstrained function
    minimization.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    Parameters
    ----------
    params : iterable
        An iterable of :class:`torch.Tensor` s. Specifies what Tensors
        should be optimized.
    method : str
        One of the various optimization methods offered in scipy minimize.
        Defaults to 'bfgs'.
    **minimize_kwargs : dict
        Additional keyword arguments that will be passed to
        :func:`torchmin.minimize()`.

    """

    def __init__(self, params, method="bfgs", **minimize_kwargs):
        assert isinstance(method, str)
        method_ = method.lower()

        self._hessp = self._hess = False
        if method_ in ["bfgs", "l-bfgs", "cg"]:
            pass
        elif method_ in ["newton-cg", "trust-ncg", "trust-krylov"]:
            self._hessp = True
        elif method_ in ["newton-exact", "dogleg", "trust-exact"]:
            self._hess = True
        else:
            raise ValueError("Unknown method {}".format(method))

        defaults = dict(method=method_, **minimize_kwargs)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Minimizer doesn't support per-parameter options")

        self._nfev = [0]
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self._closure = None

    @property
    def nfev(self):
        return self._nfev[0]

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    def _gather_flat_param(self):
        params = []
        for p in self._params:
            if p.data.is_sparse:
                p = p.data.to_dense().view(-1)
            else:
                p = p.data.view(-1)
            params.append(p)
        return torch.cat(params)

    def _gather_flat_grad(self):
        grads = []
        for p in self._params:
            if p.grad is None:
                g = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                g = p.grad.to_dense().view(-1)
            else:
                g = p.grad.view(-1)
            grads.append(g)
        return torch.cat(grads)

    def _set_flat_param(self, value):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.copy_(value[offset : offset + numel].view_as(p))
            offset += numel
        assert offset == self._numel()

    def fun(self, x):
        raise NotImplementedError

    def closure(self, x):
        from torchmin.function import sf_value

        assert self._closure is not None
        self._set_flat_param(x)
        with torch.enable_grad():
            f = self._closure()
            f.backward(create_graph=self._hessp or self._hess)
            grad = self._gather_flat_grad()

        grad_out = grad.detach().clone()
        hessp = None
        hess = None
        if self._hessp or self._hess:
            grad_accum = grad.detach().clone()

            def hvp(v):
                assert v.shape == grad.shape
                grad.backward(gradient=v, retain_graph=True)
                output = self._gather_flat_grad().detach() - grad_accum
                grad_accum.add_(output)
                return output

            numel = self._numel()
            if self._hessp:
                hessp = LinearOperator(
                    hvp, shape=(numel, numel), dtype=grad.dtype, device=grad.device
                )
            if self._hess:
                eye = torch.eye(numel, dtype=grad.dtype, device=grad.device)
                hess = torch.zeros(numel, numel, dtype=grad.dtype, device=grad.device)
                for i in range(numel):
                    hess[i] = hvp(eye[i])

        return sf_value(f=f.detach(), grad=grad_out.detach(), hessp=hessp, hess=hess)

    def dir_evaluate(self, x, t, d):
        from torchmin.function import de_value

        self._set_flat_param(x + d.mul(t))
        with torch.enable_grad():
            f = self._closure()
        f.backward()
        grad = self._gather_flat_grad()
        self._set_flat_param(x)

        return de_value(f=float(f), grad=grad)

    @torch.no_grad()
    def step(self, closure):
        """Perform an optimization step.

        The function "closure" should have a slightly different
        form vs. the PyTorch standard: namely, it should not include any
        `backward()` calls. Backward steps will be performed internally
        by the optimizer.

        >>> def closure():
        >>>    optimizer.zero_grad()
        >>>    output = model(input)
        >>>    loss = loss_fn(output, target)
        >>>    # loss.backward() <-- skip this step!
        >>>    return loss

        Parameters
        ----------
        closure : callable
            A function that re-evaluates the model and returns the loss.

        """
        from torchmin.minimize import minimize

        # sanity check
        assert len(self.param_groups) == 1

        # overwrite closure
        closure_ = closure

        def closure():
            self._nfev[0] += 1
            return closure_()

        self._closure = closure

        # get initial value
        x0 = self._gather_flat_param()

        # perform parameter update
        kwargs = {k: v for k, v in self.param_groups[0].items() if k != "params"}
        result = minimize(self, x0, **kwargs)

        # set final value
        self._set_flat_param(result.x)

        return result


@torch.no_grad()
def _minimize_cg(
    fun,
    x0,
    max_iter=None,
    gtol=1e-5,
    normp=float("inf"),
    callback=None,
    disp=0,
    return_all=False,
):
    """Minimize a scalar function of one or more variables using
    nonlinear conjugate gradient.

    The algorithm is described in Nocedal & Wright (2006) chapter 5.2.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    max_iter : int
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    normp : float
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    """
    disp = int(disp)
    if max_iter is None:
        max_iter = x0.numel() * 200

    # Construct scalar objective function
    sf = ScalarFunction(fun, x_shape=x0.shape)
    closure = sf.closure
    dir_evaluate = sf.dir_evaluate

    # initialize
    x = x0.detach().flatten()
    f, g, _, _ = closure(x)
    if disp > 1:
        print("initial fval: %0.4f" % f)
    if return_all:
        allvecs = [x]
    d = g.neg()
    grad_norm = g.norm(p=normp)
    old_f = f + g.norm() / 2  # Sets the initial step guess to dx ~ 1

    for niter in range(1, max_iter + 1):
        # delta/gtd
        delta = dot(g, g)
        gtd = dot(g, d)

        # compute initial step guess based on (f - old_f) / gtd
        t0 = torch.clamp(2.02 * (f - old_f) / gtd, max=1.0)
        if t0 <= 0:
            warnflag = 4
            msg = "Initial step guess is negative."
            break
        old_f = f

        # buffer to store next direction vector
        cached_step = [None]

        def polak_ribiere_powell_step(t, g_next):
            y = g_next - g
            beta = torch.clamp(dot(y, g_next) / delta, min=0)
            d_next = -g_next + d.mul(beta)
            torch.norm(g_next, p=normp, out=grad_norm)
            return t, d_next

        def descent_condition(t, f_next, g_next):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            cached_step[:] = polak_ribiere_powell_step(t, g_next)
            t, d_next = cached_step

            # Accept step if it leads to convergence.
            cond1 = grad_norm <= gtol

            # Accept step if sufficient descent condition applies.
            cond2 = dot(d_next, g_next) <= -0.01 * dot(g_next, g_next)

            return cond1 | cond2

        # Perform CG step
        f, g, t, ls_evals = strong_wolfe(
            dir_evaluate, x, t0, d, f, g, gtd, c2=0.4, extra_condition=descent_condition
        )

        # Update x and then update d (in that order)
        x = x + d.mul(t)
        if t == cached_step[0]:
            # Reuse already computed results if possible
            d = cached_step[1]
        else:
            d = polak_ribiere_powell_step(t, g)[1]

        if disp > 1:
            print("iter %3d - fval: %0.4f" % (niter, f))
        if return_all:
            allvecs.append(x)
        if callback is not None:
            callback(x)

        # check optimality
        if grad_norm <= gtol:
            warnflag = 0
            break

    else:
        # if we get to the end, the maximum iterations was reached
        warnflag = 1

    if disp:
        # print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % niter)
        print("         Function evaluations: %d" % sf.nfev)

    result = OptimizeResult(
        x=x,
        fun=f,
        grad=g,
        nit=niter,
        nfev=sf.nfev,
        status=warnflag,
        success=(warnflag == 0),
    )
    if return_all:
        result["allvecs"] = allvecs
    return result


def _build_obj(f, x0):
    numel = x0.numel()

    def to_tensor(x):
        return torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0)

    def f_with_jac(x):
        x = to_tensor(x).requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
        (grad,) = torch.autograd.grad(fval, x)
        return fval.detach().cpu().numpy(), grad.view(-1).cpu().numpy()

    def f_hess(x):
        x = to_tensor(x).requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
            (grad,) = torch.autograd.grad(fval, x, create_graph=True)

        def matvec(p):
            p = to_tensor(p)
            (hvp,) = torch.autograd.grad(grad, x, p, retain_graph=True)
            return hvp.view(-1).cpu().numpy()

        return LinearOperator((numel, numel), matvec=matvec)

    return f_with_jac, f_hess


def _build_constr(constr, x0):
    assert isinstance(constr, dict)
    assert set(constr.keys()).issubset(_constr_keys)
    assert "fun" in constr
    assert "lb" in constr or "ub" in constr
    if "lb" not in constr:
        constr["lb"] = -np.inf
    if "ub" not in constr:
        constr["ub"] = np.inf
    f_ = constr["fun"]
    numel = x0.numel()

    def to_tensor(x):
        return torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0)

    def f(x):
        x = to_tensor(x)
        return f_(x).cpu().numpy()

    def f_jac(x):
        x = to_tensor(x)
        if "jac" in constr:
            grad = constr["jac"](x)
        else:
            x.requires_grad_(True)
            with torch.enable_grad():
                (grad,) = torch.autograd.grad(f_(x), x)
        return grad.view(-1).cpu().numpy()

    def f_hess(x, v):
        x = to_tensor(x)
        if "hess" in constr:
            hess = constr["hess"](x)
            return v[0] * hess.view(numel, numel).cpu().numpy()
        elif "hessp" in constr:

            def matvec(p):
                p = to_tensor(p)
                hvp = constr["hessp"](x, p)
                return v[0] * hvp.view(-1).cpu().numpy()

            return LinearOperator((numel, numel), matvec=matvec)
        else:
            x.requires_grad_(True)
            with torch.enable_grad():
                if "jac" in constr:
                    grad = constr["jac"](x)
                else:
                    (grad,) = torch.autograd.grad(f_(x), x, create_graph=True)

            def matvec(p):
                p = to_tensor(p)
                (hvp,) = torch.autograd.grad(grad, x, p, retain_graph=True)
                return v[0] * hvp.view(-1).cpu().numpy()

            return LinearOperator((numel, numel), matvec=matvec)

    return NonlinearConstraint(
        fun=f,
        lb=constr["lb"],
        ub=constr["ub"],
        jac=f_jac,
        hess=f_hess,
        keep_feasible=constr.get("keep_feasible", False),
    )


def _check_bound(val, x0):
    if isinstance(val, numbers.Number):
        return np.full(x0.numel(), val)
    elif isinstance(val, torch.Tensor):
        assert val.numel() == x0.numel()
        return val.detach().cpu().numpy().flatten()
    elif isinstance(val, np.ndarray):
        assert val.size == x0.numel()
        return val.flatten()
    else:
        raise ValueError("Bound value has unrecognized format.")


def _build_bounds(bounds, x0):
    assert isinstance(bounds, dict)
    assert set(bounds.keys()).issubset(_bounds_keys)
    assert "lb" in bounds or "ub" in bounds
    lb = _check_bound(bounds.get("lb", -np.inf), x0)
    ub = _check_bound(bounds.get("ub", np.inf), x0)
    keep_feasible = bounds.get("keep_feasible", False)

    return Bounds(lb, ub, keep_feasible)


@torch.no_grad()
def minimize_constr(
    f,
    x0,
    constr=None,
    bounds=None,
    max_iter=None,
    tol=None,
    callback=None,
    disp=0,
    **kwargs,
):
    """Minimize a scalar function of one or more variables subject to
    bounds and/or constraints.

    .. note::
        This is a wrapper for SciPy's
        `'trust-constr' <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_
        method. It uses autograd behind the scenes to build jacobian & hessian
        callables before invoking scipy. Inputs and objectivs should use
        PyTorch tensors like other routines. CUDA is supported; however,
        data will be transferred back-and-forth between GPU/CPU.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    constr : dict, optional
        Constraint specifications. Should be a dictionary with the
        following fields:

            * fun (callable) - Constraint function
            * lb (Tensor or float, optional) - Constraint lower bounds
            * ub : (Tensor or float, optional) - Constraint upper bounds

        One of either `lb` or `ub` must be provided. When `lb` == `ub` it is
        interpreted as an equality constraint.
    bounds : dict, optional
        Bounds on variables. Should a dictionary with at least one
        of the following fields:

            * lb (Tensor or float) - Lower bounds
            * ub (Tensor or float) - Upper bounds

        Bounds of `-inf`/`inf` are interpreted as no bound. When `lb` == `ub`
        it is interpreted as an equality constraint.
    max_iter : int, optional
        Maximum number of iterations to perform. If unspecified, this will
        be set to the default of the selected method.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int
        Level of algorithm's verbosity:

            * 0 : work silently (default).
            * 1 : display a termination report.
            * 2 : display progress during iterations.
            * 3 : display progress during iterations (more complete report).
    **kwargs
        Additional keyword arguments passed to SciPy's trust-constr solver.
        See options `here <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html>`_.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    """
    if max_iter is None:
        max_iter = 1000
    x0 = x0.detach()
    if x0.is_cuda:
        warnings.warn(
            "GPU is not recommended for trust-constr. "
            "Data will be moved back-and-forth from CPU."
        )

    # handle callbacks
    if callback is not None:
        callback_ = callback
        callback = lambda x: callback_(
            torch.tensor(x, dtype=x0.dtype, device=x0.device).view_as(x0)
        )

    # handle bounds
    if bounds is not None:
        bounds = _build_bounds(bounds, x0)

    # build objective function (and hessian)
    f_with_jac, f_hess = _build_obj(f, x0)

    # build constraints
    if constr is not None:
        constraints = [_build_constr(constr, x0)]
    else:
        constraints = []

    # optimize
    x0_np = x0.cpu().numpy().flatten().copy()
    result = minimize(
        f_with_jac,
        x0_np,
        method="trust-constr",
        jac=True,
        hess=f_hess,
        callback=callback,
        tol=tol,
        bounds=bounds,
        constraints=constraints,
        options=dict(verbose=int(disp), maxiter=max_iter, **kwargs),
    )

    # convert the important things to torch tensors
    for key in ["fun", "grad", "x"]:
        result[key] = torch.tensor(result[key], dtype=x0.dtype, device=x0.device)
    result["x"] = result["x"].view_as(x0)

    return result


def _cg_iters(grad, hess, max_iter, normp=1):
    """A CG solver specialized for the NewtonCG sub-problem.

    Derived from Algorithm 7.1 of "Numerical Optimization (2nd Ed.)"
    (Nocedal & Wright, 2006; pp. 169)
    """
    # generalized dot product that supports batch inputs
    # TODO: let the user specify dot fn?
    dot = lambda u, v: u.mul(v).sum(-1, keepdim=True)

    g_norm = grad.norm(p=normp)
    tol = g_norm * g_norm.sqrt().clamp(0, 0.5)
    eps = torch.finfo(grad.dtype).eps
    n_iter = 0  # TODO: remove?
    maxiter_reached = False

    # initialize state and iterate
    x = torch.zeros_like(grad)
    r = grad.clone()
    p = grad.neg()
    rs = dot(r, r)
    for n_iter in range(max_iter):
        if r.norm(p=normp) < tol:
            break
        Bp = hess.mv(p)
        curv = dot(p, Bp)
        curv_sum = curv.sum()
        if curv_sum < 0:
            # hessian is not positive-definite
            if n_iter == 0:
                # if first step, fall back to steepest descent direction
                # (scaled by Rayleigh quotient)
                x = grad.mul(rs / curv)
                # x = grad.neg()
            break
        elif curv_sum <= 3 * eps:
            break
        alpha = rs / curv
        x.addcmul_(alpha, p)
        r.addcmul_(alpha, Bp)
        rs_new = dot(r, r)
        p.mul_(rs_new / rs).sub_(r)
        rs = rs_new
    else:
        # curvature keeps increasing; bail
        maxiter_reached = True

    return x, n_iter, maxiter_reached


@torch.no_grad()
def _minimize_newton_cg(
    fun,
    x0,
    lr=1.0,
    max_iter=None,
    cg_max_iter=None,
    twice_diffable=True,
    line_search="strong-wolfe",
    xtol=1e-5,
    normp=1,
    callback=None,
    disp=0,
    return_all=False,
):
    """Minimize a scalar function of one or more variables using the
    Newton-Raphson method, with Conjugate Gradient for the linear inverse
    sub-problem.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    cg_max_iter : int, optional
        Maximum number of iterations for CG subproblem. Recommended to
        leave this at the default of ``20 * x0.numel()``.
    twice_diffable : bool
        Whether to assume the function is twice continuously differentiable.
        If True, hessian-vector products will be much faster.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    lr = float(lr)
    disp = int(disp)
    xtol = x0.numel() * xtol
    if max_iter is None:
        max_iter = x0.numel() * 200
    if cg_max_iter is None:
        cg_max_iter = x0.numel() * 20

    # construct scalar objective function
    sf = ScalarFunction(fun, x0.shape, hessp=True, twice_diffable=twice_diffable)
    closure = sf.closure
    if line_search == "strong-wolfe":
        dir_evaluate = sf.dir_evaluate

    # initial settings
    x = x0.detach().clone(memory_format=torch.contiguous_format)
    f, g, hessp, _ = closure(x)
    if disp > 1:
        print("initial fval: %0.4f" % f)
    if return_all:
        allvecs = [x]
    ncg = 0  # number of cg iterations
    n_iter = 0

    # begin optimization loop
    for n_iter in range(1, max_iter + 1):

        # ============================================================
        #  Compute a search direction pk by applying the CG method to
        #       H_f(xk) p = - J_f(xk) starting from 0.
        # ============================================================

        # Compute search direction with conjugate gradient (GG)
        d, cg_iters, cg_fail = _cg_iters(g, hessp, cg_max_iter, normp)
        ncg += cg_iters

        if cg_fail:
            warnflag = 3
            break

        # =====================================================
        #  Perform variable update (with optional line search)
        # =====================================================

        if line_search == "none":
            update = d.mul(lr)
            x = x + update
        elif line_search == "strong-wolfe":
            # strong-wolfe line search
            _, _, t, ls_nevals = strong_wolfe(dir_evaluate, x, lr, d, f, g)
            update = d.mul(t)
            x = x + update
        else:
            raise ValueError("invalid line_search option {}.".format(line_search))

        # re-evaluate function
        f, g, hessp, _ = closure(x)

        if disp > 1:
            print("iter %3d - fval: %0.4f" % (n_iter, f))
        if callback is not None:
            callback(x)
        if return_all:
            allvecs.append(x)

        # ==========================
        #  check for convergence
        # ==========================

        if update.norm(p=normp) <= xtol:
            warnflag = 0
            break

        if not f.isfinite():
            warnflag = 3
            break

    else:
        # if we get to the end, the maximum num. iterations was reached
        warnflag = 1

    if disp:
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)
        print("         Function evaluations: %d" % sf.nfev)
        print("         CG iterations: %d" % ncg)
    result = OptimizeResult(
        fun=f,
        grad=g,
        nfev=sf.nfev,
        ncg=ncg,
        status=warnflag,
        success=(warnflag == 0),
        x=x,
        nit=n_iter,
    )
    if return_all:
        result["allvecs"] = allvecs
    return result


@torch.no_grad()
def _minimize_newton_exact(
    fun,
    x0,
    lr=1.0,
    max_iter=None,
    line_search="strong-wolfe",
    xtol=1e-5,
    normp=1,
    tikhonov=0.0,
    handle_npd="grad",
    callback=None,
    disp=0,
    return_all=False,
):
    """Minimize a scalar function of one or more variables using the
    Newton-Raphson method.

    This variant uses an "exact" Newton routine based on Cholesky factorization
    of the explicit Hessian matrix.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    tikhonov : float
        Optional diagonal regularization (Tikhonov) parameter for the Hessian.
    handle_npd : str
        Mode for handling non-positive definite hessian matrices. Can be one
        of the following:

            * 'grad' : use steepest descent direction (gradient)
            * 'lu' : solve the inverse hessian with LU factorization
            * 'eig' : use symmetric eigendecomposition to determine a
              diagonal regularization parameter
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    lr = float(lr)
    disp = int(disp)
    xtol = x0.numel() * xtol
    if max_iter is None:
        max_iter = x0.numel() * 200

    # Construct scalar objective function
    sf = ScalarFunction(fun, x0.shape, hess=True)
    closure = sf.closure
    if line_search == "strong-wolfe":
        dir_evaluate = sf.dir_evaluate

    # initial settings
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    f, g, _, hess = closure(x)
    if tikhonov > 0:
        hess.diagonal().add_(tikhonov)
    if disp > 1:
        print("initial fval: %0.4f" % f)
    if return_all:
        allvecs = [x]
    nfail = 0
    n_iter = 0

    # begin optimization loop
    for n_iter in range(1, max_iter + 1):

        # ==================================================
        #  Compute a search direction d by solving
        #          H_f(x) d = - J_f(x)
        #  with the true Hessian and Cholesky factorization
        # ===================================================

        # Compute search direction with Cholesky solve
        try:
            d = torch.cholesky_solve(
                g.neg().unsqueeze(1), torch.linalg.cholesky(hess)
            ).squeeze(1)
            chol_fail = False
        except:
            chol_fail = True
            nfail += 1
            if handle_npd == "lu":
                d = torch.linalg.solve(hess, g.neg())
            elif handle_npd == "grad":
                d = g.neg()
            elif handle_npd == "cauchy":
                gnorm = g.norm(p=2)
                scale = 1 / gnorm
                gHg = g.dot(hess.mv(g))
                if gHg > 0:
                    scale *= torch.clamp_max_(gnorm.pow(3) / gHg, max=1)
                d = scale * g.neg()
            elif handle_npd == "eig":
                # this setting is experimental! use with caution
                # TODO: why chose the factor 1.5 here? Seems to work best
                eig0 = eigsh(
                    hess.cpu().numpy(),
                    k=1,
                    which="SA",
                    tol=1e-4,
                    return_eigenvectors=False,
                ).item()
                tau = max(1e-3 - 1.5 * eig0, 0)
                hess.diagonal().add_(tau)
                d = torch.cholesky_solve(
                    g.neg().unsqueeze(1), torch.linalg.cholesky(hess)
                ).squeeze(1)
            else:
                raise RuntimeError("invalid handle_npd encountered.")

        # =====================================================
        #  Perform variable update (with optional line search)
        # =====================================================

        if line_search == "none":
            update = d.mul(lr)
            x = x + update
        elif line_search == "strong-wolfe":
            # strong-wolfe line search
            _, _, t, ls_nevals = strong_wolfe(dir_evaluate, x, lr, d, f, g)
            update = d.mul(t)
            x = x + update
        else:
            raise ValueError("invalid line_search option {}.".format(line_search))

        # ===================================
        #  Re-evaluate func/Jacobian/Hessian
        # ===================================

        f, g, _, hess = closure(x)
        if tikhonov > 0:
            hess.diagonal().add_(tikhonov)

        if disp > 1:
            print("iter %3d - fval: %0.4f - chol_fail: %r" % (n_iter, f, chol_fail))
        if callback is not None:
            callback(x)
        if return_all:
            allvecs.append(x)

        # ==========================
        #  check for convergence
        # ==========================

        if update.norm(p=normp) <= xtol:
            warnflag = 0
            break

        if not f.isfinite():
            warnflag = 3
            break

    else:
        # if we get to the end, the maximum num. iterations was reached
        warnflag = 1

    if disp:
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)
        print("         Function evaluations: %d" % sf.nfev)
    result = OptimizeResult(
        fun=f,
        grad=g,
        nfev=sf.nfev,
        nfail=nfail,
        status=warnflag,
        success=(warnflag == 0),
        x=x.view_as(x0),
        nit=n_iter,
    )
    if return_all:
        result["allvecs"] = allvecs
    return result


class LinearOperator:
    """A generic linear operator to use with Minimizer"""

    def __init__(self, matvec, shape, dtype=torch.float, device=None):
        self.rmv = matvec
        self.mv = matvec
        self.shape = shape
        self.dtype = dtype
        self.device = device


def _strong_wolfe_extra(
    obj_func,
    x,
    t,
    d,
    f,
    g,
    gtd,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-9,
    max_ls=25,
    extra_condition=None,
):
    """A modified variant of pytorch's strong-wolfe line search that supports
    an "extra_condition" argument (callable).

    This is required for methods such as Conjugate Gradient (polak-ribiere)
    where the strong wolfe conditions do not guarantee that we have a
    descent direction.

    Code borrowed from pytorch::
        Copyright (c) 2016 Facebook, Inc.
        All rights reserved.
    """
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    if extra_condition is None:
        extra_condition = lambda *args: True
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd and extra_condition(t, f_new, g_new):
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd and extra_condition(t, f_new, g_new):
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


def strong_wolfe(fun, x, t, d, f, g, gtd=None, **kwargs):
    """
    Expects `fun` to take arguments {x, t, d} and return {f(x1), f'(x1)},
    where x1 is the new location after taking a step from x in direction d
    with step size t.
    """
    if gtd is None:
        gtd = g.mul(d).sum()

    # use python floats for scalars as per torch.optim.lbfgs
    f, t = float(f), float(t)

    if "extra_condition" in kwargs:
        f, g, t, ls_nevals = _strong_wolfe_extra(
            fun, x.view(-1), t, d.view(-1), f, g.view(-1), gtd, **kwargs
        )
    else:
        # in theory we shouldn't need to use pytorch's native _strong_wolfe
        # since the custom implementation above is equivalent with
        # extra_codition=None. But we will keep this in case they make any
        # changes.
        f, g, t, ls_nevals = _strong_wolfe(
            fun, x.view(-1), t, d.view(-1), f, g.view(-1), gtd, **kwargs
        )

    # convert back to torch scalar
    f = torch.as_tensor(f, dtype=x.dtype, device=x.device)

    return f, g.view_as(x), t, ls_nevals


def brent(fun, x, d, bounds=(0, 10)):
    """
    Expects `fun` to take arguments {x} and return {f(x)}
    """

    def line_obj(t):
        return float(fun(x + t * d))

    res = minimize_scalar(line_obj, bounds=bounds, method="bounded")
    return res.x


def backtracking(fun, x, t, d, f, g, mu=0.1, decay=0.98, max_ls=500, tmin=1e-5):
    """
    Expects `fun` to take arguments {x, t, d} and return {f(x1), x1},
    where x1 is the new location after taking a step from x in direction d
    with step size t.

    We use a generalized variant of the armijo condition that supports
    arbitrary step functions x' = step(x,t,d). When step(x,t,d) = x + t * d
    it is equivalent to the standard condition.
    """
    x_new = x
    f_new = f
    success = False
    for i in range(max_ls):
        f_new, x_new = fun(x, t, d)
        if f_new <= f + mu * g.mul(x_new - x).sum():
            success = True
            break
        if t <= tmin:
            warnings.warn("step size has reached the minimum threshold.")
            break
        t = t.mul(decay)
    else:
        warnings.warn("backtracking did not converge.")

    return x_new, f_new, t, success


@torch.jit.script
class JacobianLinearOperator(object):
    def __init__(
        self,
        x: Tensor,
        f: Tensor,
        gf: Optional[Tensor] = None,
        gx: Optional[Tensor] = None,
        symmetric: bool = False,
    ) -> None:
        self.x = x
        self.f = f
        self.gf = gf
        self.gx = gx
        self.symmetric = symmetric
        # tensor-like properties
        self.shape = (x.numel(), x.numel())
        self.dtype = x.dtype
        self.device = x.device

    def mv(self, v: Tensor) -> Tensor:
        if self.symmetric:
            return self.rmv(v)
        assert v.shape == self.x.shape
        gx, gf = self.gx, self.gf
        assert (gx is not None) and (gf is not None)
        outputs: List[Tensor] = [gx]
        inputs: List[Tensor] = [gf]
        grad_outputs: List[Optional[Tensor]] = [v]
        jvp = autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)[0]
        if jvp is None:
            raise Exception
        return jvp

    def rmv(self, v: Tensor) -> Tensor:
        assert v.shape == self.f.shape
        outputs: List[Tensor] = [self.f]
        inputs: List[Tensor] = [self.x]
        grad_outputs: List[Optional[Tensor]] = [v]
        vjp = autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)[0]
        if vjp is None:
            raise Exception
        return vjp


class ScalarFunction(object):
    """Scalar-valued objective function with autograd backend.

    This class provides a general-purpose objective wrapper which will
    compute first- and second-order derivatives via autograd as specified
    by the parameters of __init__.
    """

    def __new__(cls, fun, x_shape, hessp=False, hess=False, twice_diffable=True):
        if isinstance(fun, Minimizer):
            assert fun._hessp == hessp
            assert fun._hess == hess
            return fun
        return super(ScalarFunction, cls).__new__(cls)

    def __init__(self, fun, x_shape, hessp=False, hess=False, twice_diffable=True):
        self._fun = fun
        self._x_shape = x_shape
        self._hessp = hessp
        self._hess = hess
        self._I = None
        self._twice_diffable = twice_diffable
        self.nfev = 0

    def fun(self, x):
        if x.shape != self._x_shape:
            x = x.view(self._x_shape)
        f = self._fun(x)
        if f.numel() != 1:
            raise RuntimeError(
                "ScalarFunction was supplied a function "
                "that does not return scalar outputs."
            )
        self.nfev += 1

        return f

    def closure(self, x):
        """Evaluate the function, gradient, and hessian/hessian-product

        This method represents the core function call. It is used for
        computing newton/quasi newton directions, etc.
        """
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.fun(x)
            grad = autograd.grad(f, x, create_graph=self._hessp or self._hess)[0]
        hessp = None
        hess = None
        if self._hessp:
            hessp = JacobianLinearOperator(x, grad, symmetric=self._twice_diffable)
        if self._hess:
            if self._I is None:
                self._I = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            hvp = lambda v: autograd.grad(grad, x, v, retain_graph=True)[0]
            hess = _vmap(hvp)(self._I)

        return sf_value(f=f.detach(), grad=grad.detach(), hessp=hessp, hess=hess)

    def dir_evaluate(self, x, t, d):
        """Evaluate a direction and step size.

        We define a separate "directional evaluate" function to be used
        for strong-wolfe line search. Only the function value and gradient
        are needed for this use case, so we avoid computational overhead.
        """
        x = x + d.mul(t)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.fun(x)
        grad = autograd.grad(f, x)[0]

        return de_value(f=float(f), grad=grad)


class VectorFunction(object):
    """Vector-valued objective function with autograd backend."""

    def __init__(self, fun, x_shape, jacp=False, jac=False):
        self._fun = fun
        self._x_shape = x_shape
        self._jacp = jacp
        self._jac = jac
        self._I = None
        self.nfev = 0

    def fun(self, x):
        if x.shape != self._x_shape:
            x = x.view(self._x_shape)
        f = self._fun(x)
        if f.dim() == 0:
            raise RuntimeError(
                "VectorFunction expected vector outputs but " "received a scalar."
            )
        elif f.dim() > 1:
            f = f.view(-1)
        self.nfev += 1

        return f

    def closure(self, x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = self.fun(x)
        jacp = None
        jac = None
        if self._jacp:
            jacp = JacobianLinearOperator(x, f)
        if self._jac:
            if self._I is None:
                self._I = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
            jvp = lambda v: autograd.grad(f, x, v, retain_graph=True)[0]
            jac = _vmap(jvp)(self._I)

        return vf_value(f=f.detach(), jacp=jacp, jac=jac)


class HessianUpdateStrategy(ABC):
    def __init__(self):
        self.n_updates = 0

    @abstractmethod
    def solve(self, grad):
        pass

    @abstractmethod
    def _update(self, s, y, rho_inv):
        pass

    def update(self, s, y):
        rho_inv = y.dot(s)
        if rho_inv <= 1e-6:
            # curvature is negative; do not update
            return
        self._update(s, y, rho_inv)
        self.n_updates += 1


class L_BFGS(HessianUpdateStrategy):
    def __init__(self, x, history_size=100):
        super().__init__()
        self.y = []
        self.s = []
        self.rho = []
        self.H_diag = 1.0
        self.alpha = x.new_empty(history_size)
        self.history_size = history_size

    def solve(self, grad):
        mem_size = len(self.y)
        d = grad.neg()
        for i in reversed(range(mem_size)):
            self.alpha[i] = self.s[i].dot(d) * self.rho[i]
            d.add_(self.y[i], alpha=-self.alpha[i])
        d.mul_(self.H_diag)
        for i in range(mem_size):
            beta_i = self.y[i].dot(d) * self.rho[i]
            d.add_(self.s[i], alpha=self.alpha[i] - beta_i)

        return d

    def _update(self, s, y, rho_inv):
        if len(self.y) == self.history_size:
            self.y.pop(0)
            self.s.pop(0)
            self.rho.pop(0)
        self.y.append(y)
        self.s.append(s)
        self.rho.append(rho_inv.reciprocal())
        self.H_diag = rho_inv / y.dot(y)


class BFGS(HessianUpdateStrategy):
    def __init__(self, x, inverse=True):
        super().__init__()
        self.inverse = inverse
        if inverse:
            self.I = torch.eye(x.numel(), device=x.device, dtype=x.dtype)
            self.H = self.I.clone()
        else:
            self.B = torch.eye(x.numel(), device=x.device, dtype=x.dtype)

    def solve(self, grad):
        if self.inverse:
            return torch.matmul(self.H, grad.neg())
        else:
            return torch.cholesky_solve(
                grad.neg().unsqueeze(1), torch.linalg.cholesky(self.B)
            ).squeeze(1)

    def _update(self, s, y, rho_inv):
        rho = rho_inv.reciprocal()
        if self.inverse:
            if self.n_updates == 0:
                self.H.mul_(rho_inv / y.dot(y))
            torch.addr(
                torch.chain_matmul(
                    torch.addr(self.I, s, y, alpha=-rho),
                    self.H,
                    torch.addr(self.I, y, s, alpha=-rho),
                ),
                s,
                s,
                alpha=rho,
                out=self.H,
            )
        else:
            if self.n_updates == 0:
                self.B.mul_(rho * y.dot(y))
            Bs = torch.mv(self.B, s)
            torch.addr(
                torch.addr(self.B, y, y, alpha=rho),
                Bs,
                Bs,
                alpha=s.dot(Bs).reciprocal().neg(),
                out=self.B,
            )


@torch.no_grad()
def _minimize_bfgs_core(
    fun,
    x0,
    lr=1.0,
    low_mem=False,
    history_size=100,
    inv_hess=True,
    max_iter=None,
    line_search="strong-wolfe",
    gtol=1e-5,
    xtol=1e-9,
    normp=float("inf"),
    callback=None,
    disp=0,
    return_all=False,
):
    """Minimize a multivariate function with BFGS or L-BFGS.

    We choose from BFGS/L-BFGS with the `low_mem` argument.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    low_mem : bool
        Whether to use L-BFGS, the "low memory" variant of the BFGS algorithm.
    history_size : int
        History size for L-BFGS hessian estimates. Ignored if `low_mem=False`.
    inv_hess : bool
        Whether to parameterize the inverse hessian vs. the hessian with BFGS.
        Ignored if `low_mem=True` (L-BFGS always parameterizes the inverse).
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to 200 * x0.numel()
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    xtol : float
        Termination tolerance on function/parameter changes.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by `torch.norm` p argument.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    lr = float(lr)
    disp = int(disp)
    if max_iter is None:
        max_iter = x0.numel() * 200
    if low_mem and not inv_hess:
        raise ValueError("inv_hess=False is not available for L-BFGS.")

    # construct scalar objective function
    sf = ScalarFunction(fun, x0.shape)
    closure = sf.closure
    if line_search == "strong-wolfe":
        dir_evaluate = sf.dir_evaluate

    # compute initial f(x) and f'(x)
    x = x0.detach().view(-1).clone(memory_format=torch.contiguous_format)
    f, g, _, _ = closure(x)
    if disp > 1:
        print("initial fval: %0.4f" % f)
    if return_all:
        allvecs = [x]

    # initial settings
    if low_mem:
        hess = L_BFGS(x, history_size)
    else:
        hess = BFGS(x, inv_hess)
    d = g.neg()
    t = min(1.0, g.norm(p=1).reciprocal()) * lr
    n_iter = 0

    # BFGS iterations
    for n_iter in range(1, max_iter + 1):

        # ==================================
        #   compute Quasi-Newton direction
        # ==================================

        if n_iter > 1:
            d = hess.solve(g)

        # directional derivative
        gtd = g.dot(d)

        # check if directional derivative is below tolerance
        if gtd > -xtol:
            warnflag = 4
            # msg = 'A non-descent direction was encountered.'
            break

        # ======================
        #   update parameter
        # ======================

        if line_search == "none":
            # no line search, move with fixed-step
            x_new = x + d.mul(t)
            f_new, g_new, _, _ = closure(x_new)
        elif line_search == "strong-wolfe":
            #  Determine step size via strong-wolfe line search
            f_new, g_new, t, ls_evals = strong_wolfe(dir_evaluate, x, t, d, f, g, gtd)
            x_new = x + d.mul(t)
        else:
            raise ValueError("invalid line_search option {}.".format(line_search))

        if disp > 1:
            print("iter %3d - fval: %0.4f" % (n_iter, f_new))
        if return_all:
            allvecs.append(x_new)
        if callback is not None:
            callback(x_new)

        # ================================
        #   update hessian approximation
        # ================================

        s = x_new.sub(x)
        y = g_new.sub(g)

        hess.update(s, y)

        # =========================================
        #   check conditions and update buffers
        # =========================================

        # convergence by insufficient progress
        if (s.norm(p=normp) <= xtol) | ((f_new - f).abs() <= xtol):
            warnflag = 0
            break

        # update state
        f[...] = f_new
        x.copy_(x_new)
        g.copy_(g_new)
        t = lr

        # convergence by 1st-order optimality
        if g.norm(p=normp) <= gtol:
            warnflag = 0
            break

        # precision loss; exit
        if ~f.isfinite():
            warnflag = 2
            break

    else:
        # if we get to the end, the maximum num. iterations was reached
        warnflag = 1

    if disp:

        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)
        print("         Function evaluations: %d" % sf.nfev)
    result = OptimizeResult(
        fun=f,
        grad=g,
        nfev=sf.nfev,
        status=warnflag,
        success=(warnflag == 0),
        x=x.view_as(x0),
        nit=n_iter,
    )
    if return_all:
        result["allvecs"] = allvecs

    return result


def _minimize_bfgs(
    fun,
    x0,
    lr=1.0,
    inv_hess=True,
    max_iter=None,
    line_search="strong-wolfe",
    gtol=1e-5,
    xtol=1e-9,
    normp=float("inf"),
    callback=None,
    disp=0,
    return_all=False,
):
    """Minimize a multivariate function with BFGS

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    inv_hess : bool
        Whether to parameterize the inverse hessian vs. the hessian with BFGS.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    xtol : float
        Termination tolerance on function/parameter changes.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    return _minimize_bfgs_core(
        fun,
        x0,
        lr,
        low_mem=False,
        inv_hess=inv_hess,
        max_iter=max_iter,
        line_search=line_search,
        gtol=gtol,
        xtol=xtol,
        normp=normp,
        callback=callback,
        disp=disp,
        return_all=return_all,
    )


def _minimize_lbfgs(
    fun,
    x0,
    lr=1.0,
    history_size=100,
    max_iter=None,
    line_search="strong-wolfe",
    gtol=1e-5,
    xtol=1e-9,
    normp=float("inf"),
    callback=None,
    disp=0,
    return_all=False,
):
    """Minimize a multivariate function with L-BFGS

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    lr : float
        Step size for parameter updates. If using line search, this will be
        used as the initial step size for the search.
    history_size : int
        History size for L-BFGS hessian estimates.
    max_iter : int, optional
        Maximum number of iterations to perform. Defaults to
        ``200 * x0.numel()``.
    line_search : str
        Line search specifier. Currently the available options are
        {'none', 'strong_wolfe'}.
    gtol : float
        Termination tolerance on 1st-order optimality (gradient norm).
    xtol : float
        Termination tolerance on function/parameter changes.
    normp : Number or str
        The norm type to use for termination conditions. Can be any value
        supported by :func:`torch.norm`.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.
    """
    return _minimize_bfgs_core(
        fun,
        x0,
        lr,
        low_mem=True,
        history_size=history_size,
        max_iter=max_iter,
        line_search=line_search,
        gtol=gtol,
        xtol=xtol,
        normp=normp,
        callback=callback,
        disp=disp,
        return_all=return_all,
    )


def minimize(
    fun,
    x0,
    method,
    max_iter=None,
    tol=None,
    options=None,
    callback=None,
    disp=0,
    return_all=False,
):
    """Minimize a scalar function of one or more variables.

    .. note::
        This is a general-purpose minimizer that calls one of the available
        routines based on a supplied `method` argument.

    Parameters
    ----------
    fun : callable
        Scalar objective function to minimize.
    x0 : Tensor
        Initialization point.
    method : str
        The minimization routine to use. Should be one of

            - 'bfgs'
            - 'l-bfgs'
            - 'cg'
            - 'newton-cg'
            - 'newton-exact'
            - 'dogleg'
            - 'trust-ncg'
            - 'trust-exact'
            - 'trust-krylov'

        At the moment, method must be specified; there is no default.
    max_iter : int, optional
        Maximum number of iterations to perform. If unspecified, this will
        be set to the default of the selected method.
    tol : float
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of keyword arguments to pass to the selected minimization
        routine.
    callback : callable, optional
        Function to call after each iteration with the current parameter
        state, e.g. ``callback(x)``.
    disp : int or bool
        Display (verbosity) level. Set to >0 to print status messages.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization routine.

    """
    x0 = torch.as_tensor(x0)
    method = method.lower()
    assert method in [
        "bfgs",
        "l-bfgs",
        "cg",
        "newton-cg",
        "newton-exact",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ]
    if options is None:
        options = {}
    if tol is not None:
        options.setdefault(_tolerance_keys[method], tol)
    options.setdefault("max_iter", max_iter)
    options.setdefault("callback", callback)
    options.setdefault("disp", disp)
    options.setdefault("return_all", return_all)

    if method == "bfgs":
        return _minimize_bfgs(fun, x0, **options)
    elif method == "l-bfgs":
        return _minimize_lbfgs(fun, x0, **options)
    elif method == "cg":
        return _minimize_cg(fun, x0, **options)
    elif method == "newton-cg":
        return _minimize_newton_cg(fun, x0, **options)
    elif method == "newton-exact":
        return _minimize_newton_exact(fun, x0, **options)
    elif method == "dogleg":
        return _minimize_dogleg(fun, x0, **options)
    elif method == "trust-ncg":
        return _minimize_trust_ncg(fun, x0, **options)
    elif method == "trust-exact":
        return _minimize_trust_exact(fun, x0, **options)
    elif method == "trust-krylov":
        return _minimize_trust_krylov(fun, x0, **options)
    else:
        raise RuntimeError('invalid method "{}" encountered.'.format(method))


_tolerance_keys = {
    "l-bfgs": "gtol",
    "bfgs": "gtol",
    "cg": "gtol",
    "newton-cg": "xtol",
    "newton-exact": "xtol",
    "dogleg": "gtol",
    "trust-ncg": "gtol",
    "trust-exact": "gtol",
    "trust-krylov": "gtol",
}
__all__ = ["ScalarFunction", "VectorFunction"]

__all__ = ["strong_wolfe", "brent", "backtracking"]
status_messages = (
    "A bad approximation caused failure to predict improvement.",
    "A linalg error occurred, such as a non-psd Hessian.",
)

_constr_keys = {"fun", "lb", "ub", "jac", "hess", "hessp", "keep_feasible"}
_bounds_keys = {"lb", "ub", "keep_feasible"}
dot = lambda u, v: torch.dot(u.view(-1), v.view(-1))
sf_value = namedtuple("sf_value", ["f", "grad", "hessp", "hess"])
de_value = namedtuple("de_value", ["f", "grad"])
vf_value = namedtuple("vf_value", ["f", "jacp", "jac"])


def print_header(title, num_breaks=1):
    print("\n" * num_breaks + "=" * 50)
    print(" " * 20 + title)
    print("=" * 50 + "\n")


def residual_energy_equation(delta, tau):

    delta = (
        torch.tensor(delta, dtype=torch.float32)
        if not isinstance(delta, torch.Tensor)
        else delta
    )
    tau = (
        torch.tensor(tau, dtype=torch.float32)
        if not isinstance(tau, torch.Tensor)
        else tau
    )

    # Clamp values of delta and tau to avoid division by zero or negative square roots
    delta = torch.clamp(delta, min=1e-6)
    tau = torch.clamp(tau, min=1e-6)

    a1 = 0.3886 * (delta**1) * (tau**0)
    a2 = 0.2938e1 * (delta**1) * (tau**0.75)
    a3 = -0.5587e1 * (delta**1) * (tau**1)
    a4 = -0.7675e0 * (delta**1) * (tau**2)
    a5 = 0.3173e0 * (delta**2) * (tau**0.75)
    a6 = 0.548e0 * (delta**2) * (tau**2)
    a7 = 0.1228e0 * (delta**3) * (tau**0.75)

    a8 = 0.2166e1 * (delta**1) * (tau**1.5) * torch.exp(-(delta**1))
    a9 = 0.158e1 * (delta**2) * (tau**1.5) * torch.exp(-(delta**1))
    a10 = -0.231e0 * (delta**4) * (tau**2.5) * torch.exp(-(delta**1))
    a11 = 0.581e-1 * (delta**4) * (tau**0) * torch.exp(-(delta**1))
    a12 = -0.5537e0 * (delta**5) * (tau**1.5) * torch.exp(-(delta**1))
    a13 = 0.4895e0 * (delta**5) * (tau**2) * torch.exp(-(delta**1))
    a14 = -0.243e-1 * (delta**6) * (tau**0) * torch.exp(-(delta**1))
    a15 = 0.625e-1 * (delta**6) * (tau**1) * torch.exp(-(delta**1))
    a16 = -0.122e0 * (delta**6) * (tau**2) * torch.exp(-(delta**1))
    a17 = -0.371e0 * (delta**1) * (tau**3) * torch.exp(-(delta**2))
    a18 = -0.168e-1 * (delta**1) * (tau**6) * torch.exp(-(delta**2))
    a19 = -0.119e0 * (delta**4) * (tau**3) * torch.exp(-(delta**2))
    a20 = -0.456e-1 * (delta**4) * (tau**6) * torch.exp(-(delta**2))
    a21 = 0.356e-1 * (delta**4) * (tau**8) * torch.exp(-(delta**2))
    a22 = -0.744e-2 * (delta**7) * (tau**6) * torch.exp(-(delta**2))
    a23 = -0.174e-2 * (delta**8) * (tau**0) * torch.exp(-(delta**2))
    a24 = -0.128e-1 * (delta**2) * (tau**7) * torch.exp(-(delta**3))
    a25 = 0.243e-1 * (delta**3) * (tau**12) * torch.exp(-(delta**3))
    a26 = -0.374e-1 * (delta**3) * (tau**16) * torch.exp(-(delta**3))
    a27 = 0.143e0 * (delta**5) * (tau**22) * torch.exp(-(delta**4))
    a28 = -0.135e0 * (delta**5) * (tau**24) * torch.exp(-(delta**4))
    a29 = -0.231e-1 * (delta**6) * (tau**16) * torch.exp(-(delta**4))
    a30 = 0.124e-1 * (delta**7) * (tau**24) * torch.exp(-(delta**4))
    a31 = 0.211e-2 * (delta**8) * (tau**8) * torch.exp(-(delta**4))
    a32 = -0.339e-3 * (delta**10) * (tau**2) * torch.exp(-(delta**4))
    a33 = 0.559e-2 * (delta**4) * (tau**8) * torch.exp(-(delta**5))
    a34 = -0.303e-3 * (delta**8) * (tau**14) * torch.exp(-(delta**6))

    expp = torch.exp(-(25 * (delta - 1) ** 2) - 325 * (tau - 1.16) ** 2)
    a35 = -0.2e3 * (delta**2) * (tau**1) * expp

    expp = torch.exp(-(25 * (delta - 1) ** 2) - 300 * (tau - 1.19) ** 2)
    a36 = -0.3e5 * (delta**2) * (tau**1) * expp

    expp = torch.exp(-(25 * (delta - 1) ** 2) - 300 * (tau - 1.19) ** 2)
    a37 = -0.2e5 * (delta**2) * (tau**1) * expp

    expp = torch.exp(-(15 * (delta - 1) ** 2) - 275 * (tau - 1.25) ** 2)
    a38 = -0.3e3 * (delta**3) * (tau**3) * expp

    expp = torch.exp(-(20 * (delta - 1) ** 2) - 275 * (tau - 1.22) ** 2)
    a39 = 0.2e3 * (delta**3) * (tau**3) * expp

    expp = torch.exp(-(10 * (delta - 1) ** 2) - 275 * (tau - 1) ** 2)
    deltaa = ((1 - tau) + 0.7 * ((delta - 1) ** 2)) ** (1 / (2 * 0.3)) ** 2 + 0.3 * (
        (delta - 1) ** 2
    ) ** 3.5
    a40 = -0.666e0 * (deltaa**0.875) * (delta) * expp

    expp = torch.exp(-(10 * (delta - 1) ** 2) - 275 * (tau - 1) ** 2)
    deltaa = ((1 - tau) + 0.7 * ((delta - 1) ** 2)) ** (1 / (2 * 0.3)) ** 2 + 0.3 * (
        (delta - 1) ** 2
    ) ** 3.5
    a41 = 0.726e0 * (deltaa**0.925) * (delta) * expp

    expp = torch.exp(-(12.5 * (delta - 1) ** 2) - 275 * (tau - 1) ** 2)
    deltaa = ((1 - tau) + 0.7 * ((delta - 1) ** 2)) ** (1 / (2 * 0.3)) ** 2 + 1 * (
        (delta - 1) ** 2
    ) ** 3
    a42 = 0.551e-1 * (deltaa**0.875) * (delta) * expp

    total = (
        a1
        + a2
        + a3
        + a4
        + a5
        + a6
        + a7
        + a8
        + a9
        + a10
        + a11
        + a12
        + a13
        + a14
        + a15
        + a16
        + a17
        + a18
        + a19
        + a20
        + a21
        + a22
        + a23
        + a24
        + a25
        + a26
        + a27
        + a28
        + a29
        + a30
        + a31
        + a32
        + a33
        + a34
        + a35
        + a36
        + a37
        + a38
        + a39
        + a40
        + a41
        + a42
    )

    # print(total.shape)
    # Assuming 'total' is your tensor
    if torch.isfinite(total).any():  # Check if there are any finite values
        valid_elements = total[torch.isfinite(total)]  # Extract finite elements
        mean_value = valid_elements.mean()  # Calculate mean of valid elements
    else:
        mean_value = torch.tensor(
            0.0, device=total.device
        )  # Default to 0 if no valid elements

    # Replace NaN and Inf values with the mean of valid elements
    total = torch.where(torch.isnan(total) | torch.isinf(total), mean_value, total)
    return total


def residual_energy_equationn(delta, tau):
    delta = (
        np.array(delta, dtype=np.float32)
        if not isinstance(delta, np.ndarray)
        else delta
    )
    tau = np.array(tau, dtype=np.float32) if not isinstance(tau, np.ndarray) else tau

    # Clamp values to avoid division by zero or negative square roots
    delta = np.clip(delta, 1e-6, None)
    tau = np.clip(tau, 1e-6, None)

    # Coefficients and calculations using numpy
    a1 = 0.3886 * (delta**1) * (tau**0)
    a2 = 0.2938e1 * (delta**1) * (tau**0.75)
    a3 = -0.5587e1 * (delta**1) * (tau**1)
    a4 = -0.7675e0 * (delta**1) * (tau**2)
    a5 = 0.3173e0 * (delta**2) * (tau**0.75)
    a6 = 0.548e0 * (delta**2) * (tau**2)
    a7 = 0.1228e0 * (delta**3) * (tau**0.75)
    a8 = 0.2166e1 * (delta**1) * (tau**1.5) * np.exp(-(delta**1))
    a9 = 0.158e1 * (delta**2) * (tau**1.5) * np.exp(-(delta**1))
    a10 = -0.231e0 * (delta**4) * (tau**2.5) * np.exp(-(delta**1))
    a11 = 0.581e-1 * (delta**4) * (tau**0) * np.exp(-(delta**1))
    a12 = -0.5537e0 * (delta**5) * (tau**1.5) * np.exp(-(delta**1))
    a13 = 0.4895e0 * (delta**5) * (tau**2) * np.exp(-(delta**1))
    a14 = -0.243e-1 * (delta**6) * (tau**0) * np.exp(-(delta**1))
    a15 = 0.625e-1 * (delta**6) * (tau**1) * np.exp(-(delta**1))
    a16 = -0.122e0 * (delta**6) * (tau**2) * np.exp(-(delta**1))
    a17 = -0.371e0 * (delta**1) * (tau**3) * np.exp(-(delta**2))
    a18 = -0.168e-1 * (delta**1) * (tau**6) * np.exp(-(delta**2))
    a19 = -0.119e0 * (delta**4) * (tau**3) * np.exp(-(delta**2))
    a20 = -0.456e-1 * (delta**4) * (tau**6) * np.exp(-(delta**2))
    a21 = 0.356e-1 * (delta**4) * (tau**8) * np.exp(-(delta**2))
    a22 = -0.744e-2 * (delta**7) * (tau**6) * np.exp(-(delta**2))
    a23 = -0.174e-2 * (delta**8) * (tau**0) * np.exp(-(delta**2))
    a24 = -0.128e-1 * (delta**2) * (tau**7) * np.exp(-(delta**3))
    a25 = 0.243e-1 * (delta**3) * (tau**12) * np.exp(-(delta**3))
    a26 = -0.374e-1 * (delta**3) * (tau**16) * np.exp(-(delta**3))
    a27 = 0.143e0 * (delta**5) * (tau**22) * np.exp(-(delta**4))
    a28 = -0.135e0 * (delta**5) * (tau**24) * np.exp(-(delta**4))
    a29 = -0.231e-1 * (delta**6) * (tau**16) * np.exp(-(delta**4))
    a30 = 0.124e-1 * (delta**7) * (tau**24) * np.exp(-(delta**4))
    a31 = 0.211e-2 * (delta**8) * (tau**8) * np.exp(-(delta**4))
    a32 = -0.34e-3 * (delta**10) * (tau**2) * np.exp(-(delta**4))
    a33 = 0.559e-2 * (delta**4) * (tau**8) * np.exp(-(delta**5))
    a34 = -0.30e-3 * (delta**8) * (tau**14) * np.exp(-(delta**6))
    # Additional expressions involving complex exponential terms
    expp = np.exp(-(25 * (delta - 1) ** 2) - 325 * (tau - 1.16) ** 2)
    a35 = -0.21e3 * (delta**2) * (tau**1) * expp
    expp = np.exp(-(25 * (delta - 1) ** 2) - 300 * (tau - 1.19) ** 2)
    a36 = -0.3e5 * (delta**2) * (tau**1) * expp

    expp = np.exp(-(25 * (delta - 1) ** 2) - 300 * (tau - 1.19) ** 2)
    a37 = -0.2e5 * (delta**2) * (tau**1) * expp

    expp = np.exp(-(15 * (delta - 1) ** 2) - 275 * (tau - 1.25) ** 2)
    a38 = -0.28e3 * (delta**3) * (tau**3) * expp

    expp = np.exp(-(20 * (delta - 1) ** 2) - 275 * (tau - 1.22) ** 2)
    a39 = 0.212e3 * (delta**3) * (tau**3) * expp

    expp = np.exp(-(10 * (delta - 1) ** 2) - 275 * (tau - 1) ** 2)
    deltaa = ((1 - tau) + 0.7 * ((delta - 1) ** 2)) ** (1 / (2 * 0.3)) ** 2 + 0.3 * (
        (delta - 1) ** 2
    ) ** 3.5
    a40 = -0.666e0 * (deltaa**0.875) * (delta) * expp

    expp = np.exp(-(10 * (delta - 1) ** 2) - 275 * (tau - 1) ** 2)
    deltaa = ((1 - tau) + 0.7 * ((delta - 1) ** 2)) ** (1 / (2 * 0.3)) ** 2 + 0.3 * (
        (delta - 1) ** 2
    ) ** 3.5
    a41 = 0.726e0 * (deltaa**0.925) * (delta) * expp

    expp = np.exp(-(12.5 * (delta - 1) ** 2) - 275 * (tau - 1) ** 2)
    deltaa = ((1 - tau) + 0.7 * ((delta - 1) ** 2)) ** (1 / (2 * 0.3)) ** 2 + 1 * (
        (delta - 1) ** 2
    ) ** 3
    a42 = 0.551e-1 * (deltaa**0.875) * (delta) * expp

    # Sum up all the terms for the total
    total = (
        a1
        + a2
        + a3
        + a4
        + a5
        + a6
        + a7
        + a8
        + a9
        + a10
        + a11
        + a12
        + a13
        + a14
        + a15
        + a16
        + a17
        + a18
        + a19
        + a20
        + a21
        + a22
        + a23
        + a24
        + a25
        + a26
        + a27
        + a28
        + a29
        + a30
        + a31
        + a32
        + a33
        + a34
        + a35
        + a36
        + a37
        + a38
        + a39
        + a40
        + a41
        + a42
    )

    # Handling NaNs or infinities if any
    # Assuming 'total' is your NumPy array
    valid_elements = total[np.isfinite(total)]  # Extract finite elements
    if (
        valid_elements.size > 0
    ):  # Ensure there are valid elements to calculate mean from
        mean_value = valid_elements.mean()  # Calculate mean of valid elements
    else:
        mean_value = 0.0  # Default to 0 if no valid elements

    # Replace NaN and Inf values with the calculated mean
    total = np.where(np.isnan(total) | np.isinf(total), mean_value, total)

    return total


# Example usage


def Helmhotz(x, P, T, Pc, Tc, R, reduce=True):

    T = abs(T)
    rhoc = 467  # kg/m3 is the critical density of co2
    delta = x / rhoc
    Tr = T / Tc
    tau = 1 / Tr

    left = (P) / (R * T * x)

    ouut = residual_energy_equation(delta, tau)
    right = 1 + (delta * ouut)

    val = (left - right) ** 2
    if reduce:
        return val.sum()
    else:
        # don't reduce batch dimensions
        return val.sum(-1)


def Helmhotzn(x, P, T, Pc, Tc, R):

    T = abs(T)
    rhoc = 467  # kg/m3 is the critical density of co2
    delta = x / rhoc
    Tr = T / Tc
    tau = 1 / Tr

    left = (P) / (R * T * x)

    ouut = residual_energy_equationn(delta, tau)
    right = 1 + (delta * ouut)

    val = np.sum((left - right) ** 2)
    return val


def EOS(x, P, T, Pc, Tc, AS, reduce=True):

    T = abs(T)

    Pr = P / Pc
    Tr = T / Tc

    left = (Pr * x) / Tr
    right = (
        1
        + (AS["a1"] + (AS["a2"] / Tr**2) + (AS["a3"] / Tr**3)) / x
        + (AS["a4"] + (AS["a5"] / Tr**2) + (AS["a6"] / Tr**3)) / (x**2)
        + (AS["a7"] + (AS["a8"] / Tr**2) + (AS["a9"] / Tr**3)) / (x**4)
        + (AS["a10"] + (AS["a11"] / Tr**2) + (AS["a12"] / Tr**3)) / (x**5)
        + (
            (AS["a13"] / ((Tr**3) * (x**2)))
            * (AS["a14"] + (AS["a15"] / (x**2)))
            * (torch.exp(-(AS["a15"] / (x**2))))
        )
    )

    val = (left - right) ** 2
    if reduce:
        return val.sum()
    else:
        # don't reduce batch dimensions
        return val.sum(-1)


def EOSn(x, P, T, Pc, Tc, AS):

    T = abs(T)
    Pr = P / Pc
    Tr = T / Tc

    left = (Pr * x) / Tr
    right = (
        1
        + (AS["a1"] + (AS["a2"] / Tr**2) + (AS["a3"] / Tr**3)) / x
        + (AS["a4"] + (AS["a5"] / Tr**2) + (AS["a6"] / Tr**3)) / (x**2)
        + (AS["a7"] + (AS["a8"] / Tr**2) + (AS["a9"] / Tr**3)) / (x**4)
        + (AS["a10"] + (AS["a11"] / Tr**2) + (AS["a12"] / Tr**3)) / (x**5)
        + (
            (AS["a13"] / ((Tr**3) * (x**2)))
            * (AS["a14"] + (AS["a15"] / (x**2)))
            * (np.exp(-(AS["a15"] / (x**2))))
        )
    )
    val = np.sum((left - right) ** 2) / 2
    return val


def sol_co2_brine(R, T, potential, fugg, salinityy, pressure):

    T = abs(T)
    eps = 1e-6

    right = (potential / (R * T) + eps) - fugg + salinityy
    right = torch.exp(right)
    x_calculated = pressure / (right - eps)
    # #x_calculated = pressure / torch.exp((potential / (R * T)) - fugg + salinityy**2) - eps
    # #x_calculated = pressure / torch.min(torch.exp((potential / (R * T)),1e4) - fugg + salinityy**2) - eps

    # # Calculate the denominator part
    # denominator = torch.exp(potential / (R * T)+eps) - fugg + salinityy**2

    # # Assuming you want to compare this denominator with a constant threshold and take the minimum
    # threshold = torch.tensor(1e4)  # Example threshold
    # capped_denominator = torch.min(denominator, threshold)

    # #capped_denominator = denominator

    # # Now use the capped_denominator in your calculation
    # x_calculated = pressure / capped_denominator - eps

    return x_calculated


def fugacity(Vr, AS, P, T, Pc, Tc):

    T = abs(T)
    Pr = P / Pc
    Tr = T / Tc
    Z = (Pr * Vr) / Tr

    fugacity = (
        Z
        - 1
        - torch.log(Z)
        + (AS["a1"] + (AS["a2"] / Tr**2) + (AS["a3"] / Tr**3)) / Vr
        + (AS["a4"] + (AS["a5"] / Tr**2) + (AS["a6"] / Tr**3)) / (2 * Vr**2)
        + (AS["a7"] + (AS["a8"] / Tr**2) + (AS["a9"] / Tr**3)) / (4 * Vr**4)
        + (AS["a10"] + (AS["a11"] / Tr**2) + (AS["a12"] / Tr**3)) / (5 * Vr**5)
        + (AS["a13"] / (2 * Tr**3 * AS["a15"]))
        * (AS["a14"] + 1 - (AS["a14"] + 1 + (AS["a15"] / Vr**2)))
        * (torch.exp(-(AS["a15"] / (Vr**2))))
    )
    return fugacity


def calculate_mu_co2(ρg, T):
    # Constants
    ω = 1 / 1251.196  # K
    x = torch.tensor([0.2352, -0.493, 5.21e-2, 5.35e-2, -1.54e-2], dtype=torch.float32)
    d = torch.tensor([0.407e-2, 0.72e-4, 1e-6, 1e-6, -1e-6], dtype=torch.float32)

    def B_star(T_star):
        ln_T_star = torch.log(T_star)
        powers_ln_T_star = torch.pow(
            ln_T_star.unsqueeze(-1), torch.arange(5, dtype=torch.float32)
        )
        yes = torch.sum(x * powers_ln_T_star, dim=-1)
        yes = torch.exp(yes)
        return yes

    def mu_o(T):
        T = abs(T)
        T_star = ω * T
        B_star_value = B_star(torch.tensor(T_star))
        see = (1.00697 * torch.sqrt(torch.tensor(T + 1e-6))) / (B_star_value + 1e-6)
        return see

    def mu_excess(ρg, T):
        return (
            d[0] * ρg
            + d[1] * ρg**2
            + d[2] * ρg**6 * T**3
            + d[3] * ρg**8
            + d[4] * ρg**8 * T
        )

    # Calculate the overall chemical potential μg
    μg = mu_o(T) + mu_excess(ρg, T)
    return μg


def calculate_h2o_density_viscosity(m, T, P, y_co2l):

    T = abs(T)
    A = -3.033405
    B = 10.128163
    C = -8.750567
    D = 2.663107

    m = torch.tensor(m)
    T = torch.tensor(T)

    x = (
        (-9.9594 * torch.exp(-0.004539 * m))
        + (7.0845 * torch.exp(-0.000164 * T))
        + (3.9093 * torch.exp(0.000025 * P))
    )

    x = torch.min(x, torch.tensor(1e4))

    rho_table = A + B * (x) + C * (x**2) + D * (x**3)

    Cco2 = (y_co2l * rho_table) / (44 * (1 - (y_co2l + 1e-6)))
    Vphi = 37.51 - (T * 9.585e-2) + ((T**2) * 8.74e-4) - ((T**3) * 5.044e-7)
    rho_h2o = rho_table + (44 * Cco2) - (Cco2 * rho_table * Vphi)

    # Constants for the correlation
    A = 2.414e-5  # Pa·s
    B = 247.8  # K
    C = 140  # K

    # Convert temperature from Celsius to Kelvin
    T_K = T

    # Calculate the viscosity using the correlation
    mu_h2o_pure = A * 10 ** (B / ((T_K - C) + 1e-6))

    a = mu_h2o_pure * 0.000629 * (1 - torch.exp(-0.7 * m))
    b = mu_h2o_pure * (1 + (0.0816 * m) + (0.0122 * (m**2)) + (0.000128 * (m**3)))

    mew_h2o = (a * T) + b
    return rho_h2o, mew_h2o


def convert_pressure_temperature(pressure, temperature):
    """
    Convert pressure from ATM to pascals (Pa) and temperature from degrees Celsius (°C) to kelvin (K).

    Args:
        pressure (float): Pressure value in atmospheres (ATM).
        temperature (float): Temperature value in degrees Celsius (°C).

    Returns:
        tuple: A tuple containing two values: converted pressure in pascals (Pa) and converted temperature in kelvin (K).
    """
    pascal_pressure = pressure * 101325.0  # 1 ATM is equal to 101325 pascals
    kelvin_temperature = temperature + 273.15  # Conversion from °C to K
    return pascal_pressure, kelvin_temperature


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


def calc_mu_g(p):
    # Average reservoir pressure
    mu_g = 3e-10 * p**2 + 1e-6 * p + 0.0133
    return mu_g


def calc_rs(p_bub, p):
    # p=average reservoir pressure
    cuda = 0
    device1 = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    rs_factor = torch.where(
        p < p_bub,
        torch.tensor(1.0).to(device1, torch.float32),
        torch.tensor(0.0).to(device1, torch.float32),
    )
    rs = (
        (178.11**2)
        / 5.615
        * (torch.pow(p / p_bub, 1.3) * rs_factor + (1 - rs_factor))
    )
    return rs


def calc_dp(p_bub, p_atm, p):
    dp = torch.where(p < p_bub, p_atm - p, p_atm - p_bub)
    return dp


def calc_bg(p_bub, p_atm, p):
    # P is average reservoir pressure
    b_g = torch.divide(1, torch.exp(1.7e-3 * calc_dp(p_bub, p_atm, p)))
    return b_g


def calc_bo(p_bub, p_atm, CFO, p):
    # p is average reservoir pressure
    exp_term1 = torch.where(p < p_bub, -8e-5 * (p_atm - p), -8e-5 * (p_atm - p_bub))
    exp_term2 = -CFO * torch.where(p < p_bub, torch.zeros_like(p), p - p_bub)
    b_o = torch.divide(1, torch.exp(exp_term1) * torch.exp(exp_term2))
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


def compute_differential(u, dxf):
    # Assuming u has shape: [batch_size, channels, nz, height, width]
    # Aim is to compute derivatives along height, width, and nz for each slice in nz

    batch_size, channels, nz, height, width = u.shape
    derivatives_x = []
    derivatives_y = []
    derivatives_z = []  # List to store derivatives in z direction

    for i in range(nz):
        slice_u = u[:, :, i, :, :]  # shape: [batch_size, channels, height, width]

        # Compute derivatives for this slice
        dudx_fdm = dx(slice_u, dx=dxf, channel=0, dim=0, order=1, padding="replication")
        dudy_fdm = dx(slice_u, dx=dxf, channel=0, dim=1, order=1, padding="replication")

        derivatives_x.append(dudx_fdm)
        derivatives_y.append(dudy_fdm)

        # Compute the derivative in z direction
        # Avoid the boundaries of the volume in z direction
        if i > 0 and i < nz - 1:
            dudz_fdm = (u[:, :, i + 1, :, :] - u[:, :, i - 1, :, :]) / (2 * dxf)
            derivatives_z.append(dudz_fdm)
        else:
            # This handles the boundaries where the derivative might not be well-defined
            # Depending on your application, you can either use forward/backward differences or pad with zeros or replicate values
            # Here, as an example, I'm padding with zeros
            dudz_fdm = torch.zeros_like(slice_u)
            derivatives_z.append(dudz_fdm)

    # Stack results to get tensors of shape [batch_size, channels, nz, height, width]
    dudx_fdm = torch.stack(derivatives_x, dim=2)
    dudy_fdm = torch.stack(derivatives_y, dim=2)
    dudz_fdm = torch.stack(derivatives_z, dim=2)  # Stack the z derivatives

    return dudx_fdm, dudy_fdm, dudz_fdm  # Return the z derivatives as well


def compute_second_differential(u, dxf):
    """Computes the x, y, and z second derivatives for each slice in the nz dimension of tensor u."""

    batch_size, channels, nz, height, width = u.shape
    second_derivatives_x = []
    second_derivatives_y = []
    second_derivatives_z = []  # List to store second derivatives in z direction

    for i in range(nz):
        slice_u = u[:, :, i, :, :]  # Extract the ith slice in the nz dimension

        # Compute second derivatives for this slice in x and y
        dduddx_fdm = ddx(
            slice_u, dx=dxf, channel=0, dim=0, order=1, padding="replication"
        )
        dduddy_fdm = ddx(
            slice_u, dx=dxf, channel=0, dim=1, order=1, padding="replication"
        )

        second_derivatives_x.append(dduddx_fdm)
        second_derivatives_y.append(dduddy_fdm)

        # Compute the second derivative in z direction
        # Avoid the boundaries of the volume in z direction
        if i > 1 and i < nz - 2:
            dduddz_fdm = (u[:, :, i + 2, :, :] - 2 * slice_u + u[:, :, i - 2, :, :]) / (
                dxf**2
            )
            second_derivatives_z.append(dduddz_fdm)
        else:
            # This handles the boundaries where the derivative might not be well-defined
            # Padding with zeros for simplicity. You may need to handle this differently based on your application
            dduddz_fdm = torch.zeros_like(slice_u)
            second_derivatives_z.append(dduddz_fdm)

    # Stack results along the nz dimension to get tensors of shape [batch_size, channels, nz, height, width]
    dduddx_fdm = torch.stack(second_derivatives_x, dim=2)
    dduddy_fdm = torch.stack(second_derivatives_y, dim=2)
    dduddz_fdm = torch.stack(
        second_derivatives_z, dim=2
    )  # Stack the z second derivatives

    return dduddx_fdm, dduddy_fdm, dduddz_fdm  # Return the z second derivatives as well


def compute_gradient_3d(inpt, dx, dim, order=1, padding="zeros"):
    "Compute first order numerical derivatives of input tensor for 3D data"

    # Define filter
    if order == 1:
        ddx1D = torch.Tensor([-0.5, 0.0, 0.5]).to(inpt.device)
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

    # Reshape filter for 3D convolution
    padding_sizes = [(0, 0), (0, 0), (0, 0)]
    if dim == 0:
        ddx3D = ddx1D.view(1, 1, -1, 1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    elif dim == 1:
        ddx3D = ddx1D.view(1, 1, 1, -1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    else:  # dim == 2
        ddx3D = ddx1D.view(1, 1, 1, 1, -1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)

    # Iterate over channels and compute the gradient for each channel
    outputs = []
    for ch in range(inpt.shape[1]):
        channel_data = inpt[:, ch : ch + 1]
        if padding == "zeros":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "constant",
                0,
            )
        elif padding == "replication":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "replicate",
            )
        out_ch = F.conv3d(channel_data, ddx3D, padding=0) * (1.0 / dx)
        outputs.append(out_ch)

    # Stack results along the channel dimension
    output = torch.cat(outputs, dim=1)

    return output


def compute_second_order_gradient_3d(inpt, dx, dim, padding="zeros"):
    "Compute second order numerical derivatives (Laplacian) of input tensor for 3D data"

    # Define filter for second order derivative
    ddx1D = torch.Tensor([-1.0, 2.0, -1.0]).to(inpt.device)

    # Reshape filter for 3D convolution
    padding_sizes = [(0, 0), (0, 0), (0, 0)]
    if dim == 0:
        ddx3D = ddx1D.view(1, 1, -1, 1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    elif dim == 1:
        ddx3D = ddx1D.view(1, 1, 1, -1, 1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)
    else:  # dim == 2
        ddx3D = ddx1D.view(1, 1, 1, 1, -1)
        padding_sizes[dim] = ((ddx1D.shape[0] - 1) // 2, (ddx1D.shape[0] - 1) // 2)

    # Iterate over channels and compute the gradient for each channel
    outputs = []
    for ch in range(inpt.shape[1]):
        channel_data = inpt[:, ch : ch + 1]
        if padding == "zeros":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "constant",
                0,
            )
        elif padding == "replication":
            channel_data = F.pad(
                channel_data,
                (
                    padding_sizes[2][0],
                    padding_sizes[2][1],
                    padding_sizes[1][0],
                    padding_sizes[1][1],
                    padding_sizes[0][0],
                    padding_sizes[0][1],
                ),
                "replicate",
            )
        out_ch = F.conv3d(channel_data, ddx3D, padding=0) * (1.0 / (dx**2))
        outputs.append(out_ch)

    # Stack results along the channel dimension
    output = torch.cat(outputs, dim=1)

    return output


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


def RelPerm(Sa, Sg, SWI, SWR, SWOW, SWOG):
    one_minus_swi_swr = 1 - (SWI + SWR)

    so = (((1 - (Sa + Sg))) - SWR) / one_minus_swi_swr
    sw = (Sa - SWI) / one_minus_swi_swr
    sg = Sg / one_minus_swi_swr

    KROW = linear_interp(Sa, SWOW[:, 0], SWOW[:, 1])
    KRW = linear_interp(Sa, SWOW[:, 0], SWOW[:, 2])
    KROG = linear_interp(Sg, SWOG[:, 0], SWOG[:, 1])
    KRG = linear_interp(Sg, SWOG[:, 0], SWOG[:, 2])

    KRO = ((KROW / (1 - sw)) * (KROG / (1 - sg))) * so
    return KRW, KRO, KRG


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


def dx1(inpt, dx, channel, dim, order=1, padding="zeros"):
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


def ddx1(inpt, dx, channel, dim, order=1, padding="zeros"):
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


def StoneIIModel(params, device, Sg, Sw):
    # device = params["device"]
    k_rwmax = params["k_rwmax"].to(device)
    k_romax = params["k_romax"].to(device)
    k_rgmax = params["k_rgmax"].to(device)
    n = params["n"].to(device)
    p = params["p"].to(device)
    q = params["q"].to(device)
    m = params["m"].to(device)
    Swi = params["Swi"].to(device)
    Sor = params["Sor"].to(device)

    denominator = 1 - Swi - Sor

    krw = k_rwmax * ((Sw - Swi) / denominator).pow(n)
    kro = (
        k_romax * (1 - (Sw - Swi) / denominator).pow(p) * (1 - Sg / denominator).pow(q)
    )
    krg = k_rgmax * (Sg / denominator).pow(m)

    return krw, kro, krg


def improve(tensor, maxK):
    # Replace NaN values with maxK
    tensor = torch.where(torch.isnan(tensor), torch.full_like(tensor, maxK), tensor)

    # Replace positive and negative infinity with maxK
    tensor = torch.where(torch.isinf(tensor), torch.full_like(tensor, maxK), tensor)

    # Ensure no values exceed maxK
    tensor = torch.where(tensor > maxK, torch.full_like(tensor, maxK), tensor)
    return tensor


class compositional_oil(torch.nn.Module):
    "Custom CO2-Brine PDE definition for PINO"

    def __init__(
        self,
        neededM,
        SWI,
        SWR,
        UW,
        BW,
        UO,
        BO,
        nx,
        ny,
        nz,
        SWOW,
        SWOG,
        target_min,
        target_max,
        minK,
        maxK,
        minP,
        maxP,
        p_bub,
        p_atm,
        CFO,
        Relperm,
        params,
        pde_method,
        RE,
        DZ,
        AS,
        Pc,
        Tc,
        T,
        device,
        pytt,
    ):
        super().__init__()
        self.neededM = neededM
        self.SWI = SWI
        self.SWR = SWR
        self.UW = UW
        self.BW = BW
        self.UO = UO
        self.BO = BO
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.SWOW = SWOW
        self.SWOG = SWOG
        self.target_min = target_min
        self.target_max = target_max
        self.minK = minK
        self.maxK = maxK
        self.minP = minP
        self.maxP = maxP
        self.p_bub = p_bub
        self.p_atm = p_atm
        self.CFO = CFO
        self.Relperm = Relperm
        self.params = params
        self.pde_method = pde_method
        self.RE = RE
        self.Pc = Pc
        self.Tc = Tc
        self.T = T
        self.DZ = DZ
        self.AS = AS
        self.device = device
        self.pytt = pytt

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        u = input_var["pressure"]
        perm = input_var["perm"]
        fin = self.neededM["Q"]
        fin = fin.repeat(u.shape[0], 1, 1, 1, 1)
        fin = fin.clamp(min=0)
        finwater = self.neededM["Qw"]
        finwater = finwater.repeat(u.shape[0], 1, 1, 1, 1)
        finwater = finwater.clamp(min=0)
        dt = self.neededM["Time"]
        pini = input_var["Pini"]
        poro = input_var["Phi"]
        sini = input_var["Swini"]
        sat = input_var["water_sat"]
        satg = input_var["gas_sat"]
        fingas = self.neededM["Qg"]
        fingas = fingas.repeat(u.shape[0], 1, 1, 1, 1)
        fingas = fingas.clamp(min=0)
        siniuse = sini[0, 0, 0, 0, 0]
        dxf = 1e-2
        Rgas = 8.314
        brine_salinity = 3  # ppt
        Zco2 = 0.02  # co2 component fraction

        # maxin = self.maxP.detach().cpu().numpy()

        # u = improve(u,1)
        # sat = improve(sat,1)
        # satg = improve(satg,1)

        # Rescale back

        # pressure
        u = u * self.maxP

        # Initial_pressure
        pini = pini * self.maxP
        # Permeability
        a = perm * self.maxK
        # permyes = a

        # Pressure equation Loss
        device = self.device

        u = abs(u)
        u = u * 6894.76  # psi to pascal
        # a = a * 9.869233e-16 # mD to m2
        pressure_mean = u.mean(dim=(2, 3, 4))

        ft3_to_m3 = 0.0283168  # Conversion factor from cubic feet to cubic meters
        seconds_per_day = 86400  # Number of seconds in a day

        fingas = fingas * ft3_to_m3 / seconds_per_day

        days_per_year = 365.25  # Average, including leap years
        hours_per_day = 24
        minutes_per_hour = 60
        seconds_per_minute = 60

        dt = dt * days_per_year * hours_per_day * minutes_per_hour * seconds_per_minute

        fugacitybig = torch.zeros(sat.shape[0], sat.shape[1]).to(device, torch.float32)
        VL = torch.zeros(sat.shape[0], sat.shape[1]).to(device, torch.float32)
        VG = torch.zeros(sat.shape[0], sat.shape[1]).to(device, torch.float32)
        y_co2_L = torch.zeros(sat.shape[0], sat.shape[1]).to(device, torch.float32)
        y_h2o_L = torch.zeros(sat.shape[0], sat.shape[1]).to(device, torch.float32)
        Rho_co2L = torch.zeros(sat.shape[0], sat.shape[1]).to(device, torch.float32)
        mew_co2L = torch.zeros(sat.shape[0], sat.shape[1]).to(device, torch.float32)
        Rho_h2oL = torch.zeros(sat.shape[0], sat.shape[1]).to(device, torch.float32)
        mew_h2oL = torch.zeros(sat.shape[0], sat.shape[1]).to(device, torch.float32)

        for mm in range(sat.shape[0]):
            # print('starting thermodynamic')

            if self.pytt == 1:
                x0 = torch.randn(sat.shape[1], 1).to(device, torch.float64)

                # compute the reduced volume Vr
                fn_Vr = lambda x: EOS(
                    x,
                    pressure_mean[mm, :].reshape(-1, 1),
                    self.T,
                    self.Pc,
                    self.Tc,
                    self.AS,
                )

                res = minimize(
                    fn_Vr, x0, method="l-bfgs", tol=1e-3, max_iter=5, disp=False
                )
                Vr = res.x

                # compute the fugacity
                fugac = fugacity(
                    Vr,
                    self.AS,
                    pressure_mean[mm, :].reshape(-1, 1),
                    self.T,
                    self.Pc,
                    self.Tc,
                )
                fugacitybig[mm, :] = fugac.ravel()

                # Calcultae the chemical potential of CO2
                uco2 = 1 + (
                    (Rgas * self.T) * torch.log(pressure_mean[mm, :].reshape(-1, 1))
                )

                # calculate solubility of co2 in brine
                # solco2brine = sol_co2_brine(Rgas,self.T,uco2,fugac,brine_salinity, pressure_mean[mm,:].reshape(-1,1))
                solco2brine = 10 * torch.ones(sat.shape[1], 1).to(device, torch.float32)

                # compute phase fractions
                vl = (1 + (solco2brine + 1e-6)) / (1 + (Zco2 / (1 - Zco2)))
                vg = 1 - vl

                VL[mm, :] = vl.ravel()
                VG[mm, :] = vg.ravel()

                # compute phase component fractions
                yco2_l = solco2brine / (1 - (solco2brine + 1e-6))
                yh2o_l = 1 - yco2_l
                yco2_g = 1

                y_co2_L[mm, :] = yco2_l.ravel()
                y_h2o_L[mm, :] = yh2o_l.ravel()

                # Compute co2 density
                x0 = torch.randn(sat.shape[1], 1).to(device, torch.float64)
                fhelmotz = lambda x: Helmhotz(
                    x,
                    pressure_mean[mm, :].reshape(-1, 1),
                    self.T,
                    self.Pc,
                    self.Tc,
                    Rgas,
                )

                Rho_co2 = minimize(
                    fhelmotz, x0, method="l-bfgs", max_iter=5, tol=1e-3, disp=False
                ).x

                # Rho_co2 = 20 * torch.ones(sat.shape[1],1).to(device,torch.float32)

                Rho_co2L[mm, :] = Rho_co2.ravel()

                # compute co2 viscosity
                mew_co2 = calculate_mu_co2(Rho_co2, self.T)
                mew_co2L[mm, :] = mew_co2.ravel()

                # compute h2o density and viscosity
                Rho_h2o, mew_h2o = calculate_h2o_density_viscosity(
                    brine_salinity, self.T, pressure_mean[mm, :].reshape(-1, 1), yco2_l
                )

                Rho_h2oL[mm, :] = Rho_h2o.ravel()
                mew_h2oL[mm, :] = mew_h2o.ravel()

            else:
                x0 = abs(np.random.rand(sat.shape[1], 1)) * 2

                # compute the reduced volume Vr
                fn_Vrn = lambda x: EOSn(
                    x,
                    pressure_mean[mm, :].reshape(-1, 1).detach().cpu().numpy(),
                    self.T,
                    self.Pc,
                    self.Tc,
                    self.AS,
                )

                Vr = scipy.optimize.fmin_powell(
                    fn_Vrn, x0, xtol=1e-6, ftol=1e-6, disp=False
                )

                # print(Vr.shape)

                Vr = torch.from_numpy(Vr).to(device, torch.float32)
                Vr = Vr.reshape(-1, 1)

                # compute the fugacity
                fugac = fugacity(
                    Vr,
                    self.AS,
                    pressure_mean[mm, :].reshape(-1, 1),
                    self.T,
                    self.Pc,
                    self.Tc,
                )
                fugacitybig[mm, :] = fugac.ravel()

                # Calcultae the chemical potential of CO2
                uco2 = 1 + (
                    (Rgas * self.T) * torch.log(pressure_mean[mm, :].reshape(-1, 1))
                )

                # calculate solubility of co2 in brine
                # solco2brine = sol_co2_brine(Rgas,self.T,uco2,fugac,brine_salinity, pressure_mean[mm,:].reshape(-1,1))

                solco2brine = 10 * torch.ones(sat.shape[1], 1).to(device, torch.float32)

                # compute phase fractions
                vl = (1 + (solco2brine + 1e-6)) / (1 + (Zco2 / (1 - Zco2)))
                vg = 1 - vl

                VL[mm, :] = vl.ravel()
                VG[mm, :] = vg.ravel()

                # compute phase component fractions
                yco2_l = solco2brine / (1 - (solco2brine + 1e-6))
                yh2o_l = 1 - yco2_l
                yco2_g = 1

                y_co2_L[mm, :] = yco2_l.ravel()
                y_h2o_L[mm, :] = yh2o_l.ravel()

                # Compute co2 density
                x0 = abs(np.random.rand(sat.shape[1], 1)) * 10
                fhelmotzn = lambda x: Helmhotzn(
                    x,
                    pressure_mean[mm, :].reshape(-1, 1).detach().cpu().numpy(),
                    self.T,
                    self.Pc,
                    self.Tc,
                    Rgas,
                )

                Rho_co2 = scipy.optimize.fmin_powell(
                    fhelmotzn, x0, xtol=1e-6, ftol=1e-6, disp=False
                )
                # Rho_co2 = 20 * np.ones((sat.shape[1],1))

                Rho_co2 = Rho_co2.reshape(-1, 1)
                Rho_co2 = torch.from_numpy(Rho_co2).to(device, torch.float32)

                Rho_co2L[mm, :] = Rho_co2.ravel()

                # compute co2 viscosity
                mew_co2 = calculate_mu_co2(Rho_co2, self.T)
                mew_co2L[mm, :] = mew_co2.ravel()

                # compute h2o density and viscosity
                Rho_h2o, mew_h2o = calculate_h2o_density_viscosity(
                    brine_salinity, self.T, pressure_mean[mm, :].reshape(-1, 1), yco2_l
                )

                Rho_h2oL[mm, :] = Rho_h2o.ravel()
                mew_h2oL[mm, :] = mew_h2o.ravel()
            # print('Finished thermodynamic')
            # print('')
        # print(pressurey.shape)
        VL = VL.view(sat.shape[0], sat.shape[1], 1, 1, 1)
        VG = VG.view(sat.shape[0], sat.shape[1], 1, 1, 1)
        y_co2_L = y_co2_L.view(sat.shape[0], sat.shape[1], 1, 1, 1)
        y_h2o_L = y_h2o_L.view(sat.shape[0], sat.shape[1], 1, 1, 1)

        Rho_co2L = Rho_co2L.view(sat.shape[0], sat.shape[1], 1, 1, 1)
        mew_co2L = mew_co2L.view(sat.shape[0], sat.shape[1], 1, 1, 1)
        Rho_h2oL = Rho_h2oL.view(sat.shape[0], sat.shape[1], 1, 1, 1)
        mew_h2oL = mew_h2oL.view(sat.shape[0], sat.shape[1], 1, 1, 1)

        prior_pressure = torch.zeros(
            sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny
        ).to(device, torch.float32)
        prior_pressure[:, 0, :, :, :] = pini[:, 0, :, :, :]
        prior_pressure[:, 1:, :, :, :] = u[:, :-1, :, :, :]

        avg_p = (
            prior_pressure.mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
            .mean(dim=4, keepdim=True)
        )
        # UG = calc_mu_g(avg_p)

        avg_p = replace_with_mean(avg_p)

        UG = mew_co2L
        UW = mew_h2oL

        prior_sat = torch.zeros(
            sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny
        ).to(device, torch.float32)
        prior_sat[:, 0, :, :, :] = siniuse * (
            torch.ones(sat.shape[0], self.nz, self.nx, self.ny).to(
                device, torch.float32
            )
        )
        prior_sat[:, 1:, :, :, :] = sat[:, :-1, :, :, :]

        prior_gas = torch.zeros(
            sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny
        ).to(device, torch.float32)
        prior_gas[:, 0, :, :, :] = torch.zeros(
            sat.shape[0], self.nz, self.nx, self.ny
        ).to(device, torch.float32)
        prior_gas[:, 1:, :, :, :] = satg[:, :-1, :, :, :]

        prior_time = torch.zeros(
            sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny
        ).to(device, torch.float32)
        prior_time[:, 0, :, :, :] = torch.zeros(
            sat.shape[0], self.nz, self.nx, self.ny
        ).to(device, torch.float32)
        prior_time[:, 1:, :, :, :] = dt[:, :-1, :, :, :]

        dsg = satg - prior_gas  # ds
        dsg = torch.clip(dsg, 0.001, None)

        dtime = dt - prior_time  # ds
        dtime = replace_with_mean(dtime)

        # Pressure equation Loss

        if self.Relperm == 2:

            KRW, _, KRG = StoneIIModel(
                self.params, device, prior_gas, prior_sat
            )  # Highly tuned to use case

        else:
            # parameters, adjust as necessary
            Swi_CO2 = 0.2  # Irreducible water saturation
            Sor_CO2 = 0.2  # Residual oil/non-wetting phase saturation
            Krend_CO2 = 0.8  # End-point relative permeability for CO2
            n_CO2 = 3  # Corey exponent for CO2

            Swi_brine = 0.2  # Irreducible water saturation
            Sor_brine = 0.2  # Residual non-wetting phase saturation
            Krend_brine = 1.0  # End-point relative permeability for brine
            n_brine = 2  # Corey exponent for brine

            KRW = corey_relative_permeability_torch(
                prior_sat, Swi_brine, Sor_brine, Krend_brine, n_brine
            )
            KRG = corey_relative_permeability_torch(
                prior_gas, Swi_CO2, Sor_CO2, Krend_CO2, n_CO2
            )

        Mw = torch.divide(KRW, (UW))
        Mg = torch.divide(KRG, (UG))

        Mg = replace_with_mean(Mg)
        Mw = replace_with_mean(Mw)

        Mt = torch.add(Mw, Mg)

        a1 = Mt * a  # overall Effective permeability
        a1water = Mw * a  # water Effective permeability
        a1gas = Mg * a  # gas Effective permeability

        if self.pde_method == 1:

            # Pressure equation for CO2 in gas
            # compute first dffrential for pressure
            dudx_fdm = compute_gradient_3d(
                u, dx=dxf, dim=0, order=1, padding="replication"
            )
            dudy_fdm = compute_gradient_3d(
                u, dx=dxf, dim=1, order=1, padding="replication"
            )
            dudz_fdm = compute_gradient_3d(
                u, dx=dxf, dim=2, order=1, padding="replication"
            )

            # Compute second diffrential for pressure
            dduddx_fdm = compute_second_order_gradient_3d(
                u, dx=dxf, dim=0, padding="replication"
            )
            dduddy_fdm = compute_second_order_gradient_3d(
                u, dx=dxf, dim=1, padding="replication"
            )
            dduddz_fdm = compute_second_order_gradient_3d(
                u, dx=dxf, dim=2, padding="replication"
            )

            # compute first dffrential for effective gas permeability
            dcdx = compute_gradient_3d(
                a1gas.float(), dx=dxf, dim=0, order=1, padding="replication"
            )
            dcdy = compute_gradient_3d(
                a1gas.float(), dx=dxf, dim=1, order=1, padding="replication"
            )
            dcdz = compute_gradient_3d(
                a1gas.float(), dx=dxf, dim=2, order=1, padding="replication"
            )

            # Apply the function to all tensors that might contain NaNs

            fin = replace_nan_with_zero(fin)
            dcdx = replace_nan_with_zero(dcdx)
            dudx_fdm = replace_nan_with_zero(dudx_fdm)
            a1 = replace_nan_with_zero(a1)
            dduddx_fdm = replace_nan_with_zero(dduddx_fdm)
            dcdy = replace_nan_with_zero(dcdy)
            dudy_fdm = replace_nan_with_zero(dudy_fdm)
            dduddy_fdm = replace_nan_with_zero(dduddy_fdm)

            # fin = fingas + finwater

            a1gas_prime = a1gas * Rho_co2L * yco2_g
            fingas_prime = fingas * Rho_co2L * yco2_g

            darcy_pressure1 = torch.mul(
                1,
                (
                    fingas_prime
                    + dcdx * dudx_fdm
                    + a1gas_prime * dduddx_fdm
                    + dcdy * dudy_fdm
                    + a1gas_prime * dduddy_fdm
                    + dcdz * dudz_fdm
                    + a1gas_prime * dduddz_fdm
                ),
            )

            darcy_pressure1 = dxf * darcy_pressure1

            darcy_pressure1 = dxf * darcy_pressure1
            p_loss1 = darcy_pressure1

            # Pressure equation for CO2 in liquid
            # compute first dffrential for effective gas permeability
            dcdx = compute_gradient_3d(
                a1water.float(), dx=dxf, dim=0, order=1, padding="replication"
            )
            dcdy = compute_gradient_3d(
                a1water.float(), dx=dxf, dim=1, order=1, padding="replication"
            )
            dcdz = compute_gradient_3d(
                a1water.float(), dx=dxf, dim=2, order=1, padding="replication"
            )

            a1water_prime = a1water * Rho_h2oL * y_co2_L
            darcy_pressure2 = torch.mul(
                1,
                (
                    dcdx * dudx_fdm
                    + a1water_prime * dduddx_fdm
                    + dcdy * dudy_fdm
                    + a1water_prime * dduddy_fdm
                    + dcdz * dudz_fdm
                    + a1water_prime * dduddz_fdm
                ),
            )

            darcy_pressure2 = dxf * darcy_pressure2
            p_loss2 = darcy_pressure2

            # Pressure equation for H2o in liquid
            a1water_prime = a1water * Rho_h2oL * y_h2o_L
            darcy_pressure3 = torch.mul(
                1,
                (
                    dcdx * dudx_fdm
                    + a1water_prime * dduddx_fdm
                    + dcdy * dudy_fdm
                    + a1water_prime * dduddy_fdm
                    + dcdz * dudz_fdm
                    + a1water_prime * dduddz_fdm
                ),
            )

            darcy_pressure3 = dxf * darcy_pressure3
            p_loss3 = darcy_pressure3

            # Gas Saturation equation loss in gas,Sco2_g:
            dudx = dudx_fdm
            dudy = dudy_fdm
            dudz = dudz_fdm

            dduddx = dduddx_fdm
            dduddy = dduddy_fdm
            dduddz = dduddz_fdm

            # compute first diffrential for effective gas permeability
            dadx = compute_gradient_3d(
                a1gas.float(), dx=dxf, dim=0, order=1, padding="replication"
            )
            dady = compute_gradient_3d(
                a1gas.float(), dx=dxf, dim=1, order=1, padding="replication"
            )
            dadz = compute_gradient_3d(
                a1gas.float(), dx=dxf, dim=2, order=1, padding="replication"
            )
            # Apply the function to all tensors that could possibly have NaNs
            poro = replace_nan_with_zero(poro)
            dtime = replace_nan_with_zero(dtime)
            dadx = replace_nan_with_zero(dadx)
            dudx = replace_nan_with_zero(dudx)
            a1gas = replace_nan_with_zero(a1gas)
            dduddx = replace_nan_with_zero(dduddx)
            dady = replace_nan_with_zero(dady)
            dudy = replace_nan_with_zero(dudy)
            dduddy = replace_nan_with_zero(dduddy)
            fingas_prime = replace_nan_with_zero(fingas_prime)

            a1gas_prime = a1gas * Rho_co2L * yco2_g
            inner_diff = (
                dadx * dudx
                + a1gas_prime * dduddx
                + dady * dudy
                + a1gas_prime * dduddy
                + dadz * dudz
                + a1gas_prime * dduddz
                + fingas_prime
            )

            satg_prime = satg * Rho_co2L * yco2_g
            prior_gas_prime = prior_gas * Rho_co2L * yco2_g
            dsg_prime = satg_prime - prior_gas_prime  # ds
            dsg_prime = torch.clip(dsg_prime, 0.001, None)

            darcy_saturation1 = poro * torch.divide(dsg_prime, dtime) - inner_diff
            darcy_saturation1 = dxf * darcy_saturation1

            s_loss1 = torch.zeros_like(u).to(device, torch.float32)
            s_loss1 = darcy_saturation1

            # Gas Saturation equation loss in liquid,Sco2_l:
            dudx = dudx_fdm
            dudy = dudy_fdm
            dudz = dudz_fdm

            dduddx = dduddx_fdm
            dduddy = dduddy_fdm
            dduddz = dduddz_fdm

            # compute first diffrential for effective gas permeability
            dadx = compute_gradient_3d(
                a1water.float(), dx=dxf, dim=0, order=1, padding="replication"
            )
            dady = compute_gradient_3d(
                a1water.float(), dx=dxf, dim=1, order=1, padding="replication"
            )
            dadz = compute_gradient_3d(
                a1water.float(), dx=dxf, dim=2, order=1, padding="replication"
            )

            a1water_prime = a1water * Rho_h2oL * y_co2_L
            inner_diff = (
                dadx * dudx
                + a1water_prime * dduddx
                + dady * dudy
                + a1water_prime * dduddy
                + dadz * dudz
                + a1water_prime * dduddz
            )
            sat_prime = sat * Rho_h2oL * y_co2_L
            prior_water_prime = prior_sat * Rho_co2L * y_co2_L
            ds_prime = sat_prime - prior_water_prime  # ds
            ds_prime = torch.clip(ds_prime, 0.001, None)
            darcy_saturation2 = poro * torch.divide(ds_prime, dtime) - inner_diff
            darcy_saturation2 = dxf * darcy_saturation2
            s_loss2 = darcy_saturation2

            # brine Saturation equation loss in liquid,Sh2o_l:
            dudx = dudx_fdm
            dudy = dudy_fdm
            dudz = dudz_fdm

            dduddx = dduddx_fdm
            dduddy = dduddy_fdm
            dduddz = dduddz_fdm

            # compute first diffrential for effective gas permeability
            dadx = compute_gradient_3d(
                a1water.float(), dx=dxf, dim=0, order=1, padding="replication"
            )
            dady = compute_gradient_3d(
                a1water.float(), dx=dxf, dim=1, order=1, padding="replication"
            )
            dadz = compute_gradient_3d(
                a1water.float(), dx=dxf, dim=2, order=1, padding="replication"
            )

            a1water_prime = a1water * Rho_h2oL * y_h2o_L
            inner_diff = (
                dadx * dudx
                + a1water_prime * dduddx
                + dady * dudy
                + a1water_prime * dduddy
                + dadz * dudz
                + a1water_prime * dduddz
            )
            sat_prime = sat * Rho_h2oL * y_h2o_L
            prior_water_prime = prior_sat * Rho_co2L * y_h2o_L
            ds_prime = sat_prime - prior_water_prime  # ds
            ds_prime = torch.clip(ds_prime, 0.001, None)
            darcy_saturation3 = poro * torch.divide(ds_prime, dtime) - inner_diff
            darcy_saturation3 = dxf * darcy_saturation3
            s_loss3 = darcy_saturation3

            s_loss4 = sat - torch.abs(1 - satg)
            s_loss4 = dxf * s_loss4

            s_loss5 = satg - torch.abs(1 - sat)
            s_loss5 = dxf * s_loss5

        else:
            # dxf = 1/1000

            # compute first dffrential
            gulpa = []
            gulp2a = []
            for m in range(sat.shape[0]):  # Batch
                inn_now = u[m, :, :, :, :]
                gulp = []
                gulp2 = []
                for i in range(self.nz):
                    now = inn_now[:, i, :, :][:, None, :, :]
                    dudx_fdma = dx1(
                        now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                    )
                    dudy_fdma = dx1(
                        now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                    )
                    gulp.append(dudx_fdma)
                    gulp2.append(dudy_fdma)
                check = torch.stack(gulp, 2)[:, 0, :, :, :]
                check2 = torch.stack(gulp2, 2)[:, 0, :, :]
                gulpa.append(check)
                gulp2a.append(check2)
            dudx_fdm = torch.stack(gulpa, 0)
            dudx_fdm = dudx_fdm.clamp(min=1e-6)  # ensures that all values are at least

            dudy_fdm = torch.stack(gulp2a, 0)
            dudy_fdm = dudy_fdm.clamp(min=1e-6)

            # Compute second diffrential
            gulpa = []
            gulp2a = []
            for m in range(sat.shape[0]):  # Batch
                inn_now = u[m, :, :, :, :]
                gulp = []
                gulp2 = []
                for i in range(self.nz):
                    now = inn_now[:, i, :, :][:, None, :, :]
                    dudx_fdma = ddx1(
                        now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                    )
                    dudy_fdma = ddx1(
                        now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                    )
                    gulp.append(dudx_fdma)
                    gulp2.append(dudy_fdma)
                check = torch.stack(gulp, 2)[:, 0, :, :, :]
                check2 = torch.stack(gulp2, 2)[:, 0, :, :]
                gulpa.append(check)
                gulp2a.append(check2)
            dduddx_fdm = torch.stack(gulpa, 0)
            dduddx_fdm = dduddx_fdm.clamp(min=1e-6)
            dduddy_fdm = torch.stack(gulp2a, 0)
            dduddy_fdm = dduddy_fdm.clamp(min=1e-6)

            gulp = []
            gulp2 = []
            for i in range(self.nz):
                inn_now2 = a1gas.float()[:, :, i, :, :]
                dudx_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulp.append(dudx_fdma)
                gulp2.append(dudy_fdma)
            dcdx = torch.stack(gulp, 2)
            dcdx = dcdx.clamp(min=1e-6)
            dcdy = torch.stack(gulp2, 2)
            dcdy = dcdy.clamp(min=1e-6)

            fin = replace_nan_with_zero(fin)
            dcdx = replace_nan_with_zero(dcdx)
            dudx_fdm = replace_nan_with_zero(dudx_fdm)
            a1 = replace_nan_with_zero(a1)
            dduddx_fdm = replace_nan_with_zero(dduddx_fdm)
            dcdy = replace_nan_with_zero(dcdy)
            dudy_fdm = replace_nan_with_zero(dudy_fdm)
            dduddy_fdm = replace_nan_with_zero(dduddy_fdm)

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dcdx.shape[0]):
                see = dcdx[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dcdx = dsout

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dcdx.shape[0]):
                see = dcdy[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dcdy = dsout

            fin = fingas + finwater

            a1gas_prime = a1gas * Rho_co2L * yco2_g
            fingas_prime = fingas * Rho_co2L * yco2_g

            darcy_pressure1 = (
                fingas_prime
                + (dcdx * dudx_fdm)
                + (a1gas_prime * dduddx_fdm)
                + (dcdy * dudy_fdm)
                + (a1gas_prime * dduddy_fdm)
            )

            p_loss1 = darcy_pressure1
            p_loss1 = (torch.abs(p_loss1)) / sat.shape[0]
            p_loss1 = dxf * p_loss1

            gulp = []
            gulp2 = []
            for i in range(self.nz):
                inn_now2 = a1water.float()[:, :, i, :, :]
                dudx_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulp.append(dudx_fdma)
                gulp2.append(dudy_fdma)
            dcdx = torch.stack(gulp, 2)
            dcdx = dcdx.clamp(min=1e-6)
            dcdy = torch.stack(gulp2, 2)
            dcdy = dcdy.clamp(min=1e-6)

            fin = replace_nan_with_zero(fin)
            dcdx = replace_nan_with_zero(dcdx)
            dudx_fdm = replace_nan_with_zero(dudx_fdm)
            a1 = replace_nan_with_zero(a1)
            dduddx_fdm = replace_nan_with_zero(dduddx_fdm)
            dcdy = replace_nan_with_zero(dcdy)
            dudy_fdm = replace_nan_with_zero(dudy_fdm)
            dduddy_fdm = replace_nan_with_zero(dduddy_fdm)

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dcdx.shape[0]):
                see = dcdx[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dcdx = dsout

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dcdx.shape[0]):
                see = dcdy[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dcdy = dsout

            a1water_prime = a1water * Rho_h2oL * y_co2_L
            darcy_pressure2 = (
                +(dcdx * dudx_fdm)
                + (a1water_prime * dduddx_fdm)
                + (dcdy * dudy_fdm)
                + (a1water_prime * dduddy_fdm)
            )

            p_loss2 = darcy_pressure2
            p_loss2 = (torch.abs(p_loss2)) / sat.shape[0]
            p_loss2 = dxf * p_loss2

            gulp = []
            gulp2 = []
            for i in range(self.nz):
                inn_now2 = a1water.float()[:, :, i, :, :]
                dudx_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulp.append(dudx_fdma)
                gulp2.append(dudy_fdma)
            dcdx = torch.stack(gulp, 2)
            dcdx = dcdx.clamp(min=1e-6)
            dcdy = torch.stack(gulp2, 2)
            dcdy = dcdy.clamp(min=1e-6)

            fin = replace_nan_with_zero(fin)
            dcdx = replace_nan_with_zero(dcdx)
            dudx_fdm = replace_nan_with_zero(dudx_fdm)
            a1 = replace_nan_with_zero(a1)
            dduddx_fdm = replace_nan_with_zero(dduddx_fdm)
            dcdy = replace_nan_with_zero(dcdy)
            dudy_fdm = replace_nan_with_zero(dudy_fdm)
            dduddy_fdm = replace_nan_with_zero(dduddy_fdm)

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dcdx.shape[0]):
                see = dcdx[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dcdx = dsout

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dcdx.shape[0]):
                see = dcdy[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dcdy = dsout
            a1water_prime = a1water * Rho_h2oL * y_h2o_L
            darcy_pressure3 = (
                +(dcdx * dudx_fdm)
                + (a1water_prime * dduddx_fdm)
                + (dcdy * dudy_fdm)
                + (a1water_prime * dduddy_fdm)
            )

            p_loss3 = darcy_pressure3
            p_loss3 = (torch.abs(p_loss3)) / sat.shape[0]
            p_loss3 = dxf * p_loss3

            # Saruration equation loss 1
            dudx = dudx_fdm
            dudy = dudy_fdm

            dduddx = dduddx_fdm
            dduddy = dduddy_fdm

            gulp = []
            gulp2 = []
            for i in range(self.nz):
                inn_now2 = a1gas.float()[:, :, i, :, :]
                dudx_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulp.append(dudx_fdma)
                gulp2.append(dudy_fdma)
            dadx = torch.stack(gulp, 2)
            dady = torch.stack(gulp2, 2)
            dadx = dadx.clamp(min=1e-6)
            dady = dady.clamp(min=1e-6)

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dadx.shape[0]):
                see = dadx[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dadx = dsout

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dady.shape[0]):
                see = dady[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dady = dsout

            a1gas_prime = a1gas * Rho_co2L * yco2_g
            satg_prime = satg * Rho_co2L * yco2_g
            prior_gas_prime = prior_gas * Rho_co2L * yco2_g
            dsg_prime = satg_prime - prior_gas_prime  # ds
            dsg_prime = torch.clip(dsg_prime, 0.001, None)

            flux = (
                (dadx * dudx)
                + (a1gas_prime * dduddx)
                + (dady * dudy)
                + (a1gas_prime * dduddy)
            )
            fifth = poro * (dsg_prime / dtime)
            fingas_prime = fingas * Rho_co2L * yco2_g
            toge = flux + fingas_prime
            darcy_saturation1 = fifth - toge

            s_loss1 = darcy_saturation1
            s_loss1 = (torch.abs(s_loss1)) / sat.shape[0]
            # s_loss = s_loss.reshape(1, 1)
            s_loss1 = dxf * s_loss1

            # Saruration equation loss 2
            dudx = dudx_fdm
            dudy = dudy_fdm

            dduddx = dduddx_fdm
            dduddy = dduddy_fdm

            gulp = []
            gulp2 = []
            for i in range(self.nz):
                inn_now2 = a1water.float()[:, :, i, :, :]
                dudx_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulp.append(dudx_fdma)
                gulp2.append(dudy_fdma)
            dadx = torch.stack(gulp, 2)
            dady = torch.stack(gulp2, 2)
            dadx = dadx.clamp(min=1e-6)
            dady = dady.clamp(min=1e-6)

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dadx.shape[0]):
                see = dadx[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dadx = dsout

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dady.shape[0]):
                see = dady[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dady = dsout

            a1water_prime = a1water * Rho_h2oL * y_co2_L
            sat_prime = sat * Rho_h2oL * y_co2_L
            prior_water_prime = prior_sat * Rho_co2L * y_co2_L
            ds_prime = sat_prime - prior_water_prime  # ds
            ds_prime = torch.clip(ds_prime, 0.001, None)

            flux = (
                (dadx * dudx)
                + (a1water_prime * dduddx)
                + (dady * dudy)
                + (a1water_prime * dduddy)
            )
            fifth = poro * (ds_prime / dtime)
            toge = flux
            darcy_saturation2 = fifth - toge

            s_loss2 = darcy_saturation2
            s_loss2 = (torch.abs(s_loss2)) / sat.shape[0]
            s_loss2 = dxf * s_loss2

            # Saruration equation loss 3
            dudx = dudx_fdm
            dudy = dudy_fdm

            dduddx = dduddx_fdm
            dduddy = dduddy_fdm

            gulp = []
            gulp2 = []
            for i in range(self.nz):
                inn_now2 = a1water.float()[:, :, i, :, :]
                dudx_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = dx1(
                    inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulp.append(dudx_fdma)
                gulp2.append(dudy_fdma)
            dadx = torch.stack(gulp, 2)
            dady = torch.stack(gulp2, 2)
            dadx = dadx.clamp(min=1e-6)
            dady = dady.clamp(min=1e-6)

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dadx.shape[0]):
                see = dadx[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dadx = dsout

            dsout = torch.zeros(
                (sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)
            ).to(device, torch.float32)
            for k in range(dady.shape[0]):
                see = dady[k, :, :, :, :]
                gulp = []
                for i in range(sat.shape[1]):
                    gulp.append(see)

                checkken = torch.vstack(gulp)
                dsout[k, :, :, :, :] = checkken

            dady = dsout

            a1water_prime = a1water * Rho_h2oL * y_h2o_L
            sat_prime = sat * Rho_h2oL * y_h2o_L
            prior_water_prime = prior_sat * Rho_co2L * y_h2o_L
            ds_prime = sat_prime - prior_water_prime  # ds
            ds_prime = torch.clip(ds_prime, 0.001, None)

            flux = (
                (dadx * dudx)
                + (a1water_prime * dduddx)
                + (dady * dudy)
                + (a1water_prime * dduddy)
            )
            fifth = poro * (ds_prime / dtime)
            toge = flux
            darcy_saturation3 = fifth - toge

            s_loss3 = darcy_saturation3
            s_loss3 = (torch.abs(s_loss3)) / sat.shape[0]
            s_loss3 = dxf * s_loss3

            s_loss4 = (sat - torch.abs(1 - satg)) / sat.shape[0]
            s_loss4 = dxf * s_loss4

            s_loss5 = (satg - torch.abs(1 - sat)) / sat.shape[0]
            s_loss5 = dxf * s_loss5

        # Apply the function to each tensor
        p_loss1 = replace_with_mean(p_loss1) / 25000
        p_loss2 = replace_with_mean(p_loss2) / 25000
        p_loss3 = replace_with_mean(p_loss3) / 25000

        s_loss1 = replace_with_mean(s_loss1) / 25000
        s_loss2 = replace_with_mean(s_loss2) / 25000
        s_loss3 = replace_with_mean(s_loss3) / 25000
        s_loss4 = replace_with_mean(s_loss4) / 25000
        s_loss5 = replace_with_mean(s_loss5) / 25000

        output_var = {
            "Pco2_g": p_loss1,
            "Pco2_l": p_loss2,
            "Ph2o_l": p_loss3,
            "Sco2_g": s_loss1,
            "Sco2_l": s_loss2,
            "Sh2o_l": s_loss3,
            "satwp": s_loss4,
            "satgp": s_loss5,
        }

        return normalize_tensors_adjusted(output_var)


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


@physicsnemo.sym.main(config_path="conf", config_name="config_PINO")
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
    print("|                 TRAIN THE MODEL USING A PINO APPROACH:        |")
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

    learn_pde = "PINO"

    pytt = None
    while True:
        pytt = int(
            input("Thermodynamics non-linear optimsiation:\n1 = bfgs\n2 = Powell\n")
        )
        if (pytt > 2) or (pytt < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:

            break
    print("")

    if DEFAULT == 1:
        print("Default configuration selected, sit back and relax.....")
        learn_pde = "PINO"
    else:
        pass

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

    print("")
    if DEFAULT == 1:
        Relperm = 1
    else:
        Relperm = None
        while True:
            Relperm = int(input("Select 1 = Correy | 2 = Stone II \n"))
            if (Relperm > 2) or (Relperm < 1):
                # raise SyntaxError('please select value between 1-2')
                print("")
                print("please try again and select value between 1-2")
            else:

                break

    paramss = {
        "k_rwmax": torch.tensor(0.3),
        "k_romax": torch.tensor(0.9),
        "k_rgmax": torch.tensor(0.8),
        "n": torch.tensor(2.0),
        "p": torch.tensor(2.0),
        "q": torch.tensor(2.0),
        "m": torch.tensor(2.0),
        "Swi": torch.tensor(0.1),
        "Sor": torch.tensor(0.2),
    }

    print("")
    if DEFAULT == 1:
        pde_method = 2
    else:
        pde_method = None
        while True:
            pde_method = int(input("Select 1 = approximate | 2 = Extensive \n"))
            if (pde_method > 2) or (pde_method < 1):
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

    Pc, Tc = convert_pressure_temperature(cfg.custom.PROPS.Pc, cfg.custom.PROPS.Tc)
    T = 18.4  # C
    T = T + 273.15
    AS = dict()
    AS["a1"] = float(cfg.custom.PROPS.a1)
    AS["a2"] = float(cfg.custom.PROPS.a2)
    AS["a3"] = float(cfg.custom.PROPS.a3)
    AS["a4"] = float(cfg.custom.PROPS.a4)
    AS["a5"] = float(cfg.custom.PROPS.a5)
    AS["a6"] = float(cfg.custom.PROPS.a6)
    AS["a7"] = float(cfg.custom.PROPS.a7)
    AS["a8"] = float(cfg.custom.PROPS.a8)
    AS["a9"] = float(cfg.custom.PROPS.a9)
    AS["a10"] = float(cfg.custom.PROPS.a10)
    AS["a11"] = float(cfg.custom.PROPS.a11)
    AS["a12"] = float(cfg.custom.PROPS.a12)
    AS["a13"] = float(cfg.custom.PROPS.a13)
    AS["a14"] = float(cfg.custom.PROPS.a14)
    AS["a15"] = float(cfg.custom.PROPS.a15)

    SWOW = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOW), dtype=float)).to(
        device
    )

    SWOG = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOG), dtype=float)).to(
        device
    )

    BO = float(cfg.custom.PROPS.BO)
    BW = float(cfg.custom.PROPS.BW)
    UW = float(cfg.custom.PROPS.UW)
    UO = float(cfg.custom.PROPS.UO)
    SWI = cp.float32(cfg.custom.PROPS.SWI)
    SWR = cp.float32(cfg.custom.PROPS.SWR)
    CFO = cp.float32(cfg.custom.PROPS.CFO)
    p_atm = cp.float32(float(cfg.custom.PROPS.PATM))
    p_bub = cp.float32(float(cfg.custom.PROPS.PB))

    params1 = {
        "BO": torch.tensor(BO),
        "UO": torch.tensor(UO),
        "BW": torch.tensor(BW),
        "UW": torch.tensor(UW),
    }

    DZ = torch.tensor(100).to(device)
    RE = torch.tensor(0.2 * 100).to(device)

    # cgrid = np.genfromtxt("NORNE/clementgrid.out", dtype='float')

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

    SWI = torch.from_numpy(np.array(SWI)).to(device)
    SWR = torch.from_numpy(np.array(SWR)).to(device)
    UW = torch.from_numpy(np.array(UW)).to(device)
    BW = torch.from_numpy(np.array(BW)).to(device)
    UO = torch.from_numpy(np.array(UO)).to(device)
    BO = torch.from_numpy(np.array(BO)).to(device)

    # SWOW = torch.from_numpy(SWOW).to(device)
    # SWOG = torch.from_numpy(SWOG).to(device)
    p_bub = torch.from_numpy(np.array(p_bub)).to(device)
    p_atm = torch.from_numpy(np.array(p_atm)).to(device)
    CFO = torch.from_numpy(np.array(CFO)).to(device)

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

    outvar_train["Pco2_g"] = np.zeros_like(outvar_train["pressure"])

    outvar_train["Pco2_l"] = np.zeros_like(outvar_train["pressure"])

    outvar_train["Ph2o_l"] = np.zeros_like(outvar_train["pressure"])

    outvar_train["Sco2_g"] = np.zeros_like(outvar_train["water_sat"])

    outvar_train["Sco2_l"] = np.zeros_like(outvar_train["water_sat"])

    outvar_train["Sh2o_l"] = np.zeros_like(outvar_train["gas_sat"])

    outvar_train["satwp"] = np.zeros_like(outvar_train["gas_sat"])

    outvar_train["satgp"] = np.zeros_like(outvar_train["gas_sat"])

    train_dataset = DictGridDataset(invar_train, outvar_train)

    test_dataset = DictGridDataset(invar_test, outvar_test)

    # [init-node]
    # Make custom compositional residual node for PINO

    # Define FNO model
    # Pressure
    decoder = ConvFullyConnectedArch(
        [Key("z", size=32)],
        [
            Key("pressure", size=steppi),
            Key("water_sat", size=steppi),
            Key("gas_sat", size=steppi),
        ],
        activation_fn=Activation.RELU,
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
        activation_fn=Activation.RELU,
    )

    inputs = [
        "perm",
        "Phi",
        "Pini",
        "Swini",
        "pressure",
        "water_sat",
        "gas_sat",
    ]

    compositionall = Node(
        inputs=inputs,
        outputs=[
            "Pco2_g",
            "Pco2_l",
            "Ph2o_l",
            "Sco2_g",
            "Sco2_l",
            "Sh2o_l",
            "satwp",
            "satgp",
        ],
        evaluate=compositional_oil(
            neededM,
            SWI,
            SWR,
            UW,
            BW,
            UO,
            BO,
            nx,
            ny,
            nz,
            SWOW,
            SWOG,
            target_min,
            target_max,
            minKx,
            maxKx,
            minPx,
            maxPx,
            p_bub,
            p_atm,
            CFO,
            Relperm,
            paramss,
            pde_method,
            RE,
            DZ,
            AS,
            Pc,
            Tc,
            T,
            device,
            pytt,
        ),
        name="compositional node",
    )

    nodes = [fno_supervised.make_node("fno_forward_model")] + [compositionall]

    # make domain
    domain = Domain()

    # add constraints to domain
    supervised_dynamic = SupervisedGridConstraint(
        nodes=nodes,
        loss=PointwiseLossNormC(ord=1),
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
