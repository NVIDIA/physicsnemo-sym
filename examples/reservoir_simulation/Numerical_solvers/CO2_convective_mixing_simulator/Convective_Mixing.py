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
    from numba import cuda

    print(cuda.detect())  # Print the GPU information
    import tensorflow as tf

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)
else:
    print("No GPU Available")
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "tensorflow"
import os.path
import multiprocessing
import numba as nb
from numba import prange
import numpy.matlib
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica"]
from matplotlib import rcParams

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
from matplotlib import rcParams
import decimal
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
# numpy.random.seed(99)
print(" ")
print(" This computer has #d cores, which will all be utilised in parallel ")
print(" ")
print("......................DEFINE SOME FUNCTIONS.....................")


@nb.njit(parallel=True)
def tridiagonal(km, a, b, c, d, u):
    # function for Thomas algorithm:
    h = np.zeros((km))
    g = np.zeros((km))
    h[0] = 0
    g[0] = u[0]
    for i in prange(1, km - 1):
        h[i] = c[i] / (b[i] - a[i] * h[i - 1])
        g[i] = (d[i] - a[i] * g[i - 1]) / (b[i] - a[i] * h[i - 1])

    for i in range(km - 2, 2, -1):
        u[i] = -h[i] * u[i + 1] + g[i]
    uu = u
    return uu


@nb.njit(parallel=True)
def eqvor_ADI_Temp(imax, jmax, dx, dy, vorn, dt, ux, vy, amu):
    a = np.zeros((imax))
    b = np.zeros((imax))
    c = np.zeros((imax))
    d = np.zeros((imax))
    tt = np.zeros((imax))
    vor = vorn
    d_x = amu * dt / dx**2
    d_y = amu * dt / dy**2
    # Upwinding method
    # x - sweep:
    for j in range(1, jmax - 1):
        for i in range(1, imax - 1):
            cy = vy[i, j] * dt / dx
            epy = np.sign(vy[i, j])
            sp = 1 + epy
            sm = 1 - epy
            Dx = (
                0.5 * (0.5 * cy * sp + d_y) * vorn[i, j - 1]
                + (1 - d_y - 0.5 * cy * epy) * vorn[i, j]
                + 0.5 * (-0.5 * cy * sm + d_y) * vorn[i, j + 1]
            )
            d[i] = Dx

        for i in range(imax):
            cx = ux[i, j] * dt / dx
            epx = np.sign(ux[i, j])
            sp = 1 + epx
            sm = 1 - epx
            a[i] = -0.5 * (0.5 * cx * sp + d_x)
            b[i] = 1 + d_x + 0.5 * epx * cx
            c[i] = 0.5 * (0.5 * cx * sm - d_x)

        # for flux boundary condition at left wall
        jl = 0  # flux at left wall
        b[1] = b[1] + a[1]
        d[1] = d[1] + jl * dx * a[1]
        a[1] = 0
        # for flux boundary condition at right wall
        jr = 0  # flux at right wall
        b[imax - 2] = b[imax - 2] + c[imax - 2]
        d[imax - 2] = d[imax - 2] + jr * dx * c[imax - 2]
        c[imax - 2] = 0
        for i in range(imax):
            tt[i] = vorn[i, j]

        u = tridiagonal(imax, a, b, c, d, tt)
        for i in range(1, imax - 1):
            vor[i, j] = u[i]

    # y - sweep:
    for i in range(1, imax - 1):
        for j in range(1, jmax - 1):
            cx = ux[i, j] * dt / dx
            epx = np.sign(ux[i, j])
            sp = 1 + epx
            sm = 1 - epx
            Dy = (
                0.5 * (0.5 * cx * sp + d_x) * vor[i - 1, j]
                + (1 - d_x - 0.5 * cx * epx) * vor[i, j]
                + 0.5 * (-0.5 * cx * sm + d_x) * vor[i + 1, j]
            )
            d[j] = Dy
        for j in range(jmax):
            cy = vy[i, j] * dt / dx
            epy = np.sign(vy[i, j])
            sp = 1 + epy
            sm = 1 - epy
            a[j] = -0.5 * (0.5 * cy * sp + d_y)
            b[j] = 1 + d_y + 0.5 * epy * cy
            c[j] = 0.5 * (0.5 * cy * sm - d_y)

        for j in range(1, jmax):
            tt[j] = vor[i, j]

        u = tridiagonal(jmax, a, b, c, d, tt)
        for j in range(1, jmax - 1):
            vorn[i, j] = u[j]

    # updating boundaries for numerical values/graphs
    vorn[0, :jmax] = vorn[1, :jmax]
    vorn[imax - 1, :jmax] = vorn[imax - 2, :jmax]
    vr = vorn
    return vr


@nb.njit(parallel=True)
def eqvor_ADI_upwind(imax, jmax, dx, dy, vorn, dt, ux, vy, amu):
    a = np.zeros((imax))
    b = np.zeros((imax))
    c = np.zeros((imax))
    d = np.zeros((imax))
    tt = np.zeros((imax))
    vor = vorn
    d_x = amu * dt / dx**2
    d_y = amu * dt / dy**2
    # Upwinding method
    # x - sweep:
    for j in range(1, jmax - 1):
        for i in range(1, imax - 1):
            cy = vy[i, j] * dt / dx
            epy = np.sign(vy[i, j])
            sp = 1 + epy
            sm = 1 - epy
            Dx = (
                0.5 * (0.5 * cy * sp + d_y) * vorn[i, j - 1]
                + (1 - d_y - 0.5 * cy * epy) * vorn[i, j]
                + 0.5 * (-0.5 * cy * sm + d_y) * vorn[i, j + 1]
            )
            d[i] = Dx

        for i in range(imax):
            cx = ux[i, j] * dt / dx
            epx = np.sign(ux[i, j])
            sp = 1 + epx
            sm = 1 - epx
            a[i] = -0.5 * (0.5 * cx * sp + d_x)
            b[i] = 1 + d_x + 0.5 * epx * cx
            c[i] = 0.5 * (0.5 * cx * sm - d_x)

        # for flux boundary condition at left wall
        jl = 0  # flux at left wall
        b[1] = b[1] + a[1]
        d[1] = d[1] + jl * dx * a[1]
        a[1] = 0
        # for flux boundary condition at right wall
        jr = 0  # flux at right wall
        b[imax - 2] = b[imax - 2] + c[imax - 2]
        d[imax - 2] = d[imax - 2] + jr * dx * c[imax - 2]
        c[imax - 2] = 0
        for i in range(imax):
            tt[i] = vorn[i, j]

        u = tridiagonal(imax, a, b, c, d, tt)
        for i in range(1, imax - 1):
            vor[i, j] = u[i]

    # y - sweep:
    for i in range(1, imax - 1):
        for j in range(1, jmax - 1):
            cx = ux[i, j] * dt / dx
            epx = np.sign(ux[i, j])
            sp = 1 + epx
            sm = 1 - epx
            Dy = (
                0.5 * (0.5 * cx * sp + d_x) * vor[i - 1, j]
                + (1 - d_x - 0.5 * cx * epx) * vor[i, j]
                + 0.5 * (-0.5 * cx * sm + d_x) * vor[i + 1, j]
            )
            d[j] = Dy

        for j in range(jmax):
            cy = vy[i, j] * dt / dx
            epy = np.sign(vy[i, j])
            sp = 1 + epy
            sm = 1 - epy
            a[j] = -0.5 * (0.5 * cy * sp + d_y)
            b[j] = 1 + d_y + 0.5 * epy * cy
            c[j] = 0.5 * (0.5 * cy * sm - d_y)

        # for flux boundary condition at the bottom wall
        jb = 0  # flux at bottom wall
        b[jmax - 2] = b[jmax - 2] + c[jmax - 2]
        d[jmax - 2] = d[jmax - 2] + jb * dx * c[imax - 2]
        c[imax - 2] = 0
        for j in range(jmax):
            tt[j] = vor[i, j]

        u = tridiagonal(jmax, a, b, c, d, tt)
        for j in range(1, jmax - 1):
            vorn[i, j] = u[j]

    # updating boundaries for numerical values/graphs
    vorn[:imax, jmax - 1] = vorn[:imax, jmax - 2]
    vorn[0, :jmax] = vorn[1, :jmax]
    vorn[imax - 1, :jmax] = vorn[imax - 2, :jmax]
    vr = vorn
    return vr


def round_divmod(b, a):
    n, d = np.frompyfunc(lambda x: decimal.Decimal(x).as_integer_ratio(), 1, 2)(
        a.astype("U")
    )
    n, d = n.astype(int), d.astype(int)
    q, r = np.divmod(b * d, n)
    return q, r / d


def draw_fig(c, Tt, namee, itrr):
    # Tt = T

    XX, YY = np.meshgrid(np.arange(im), np.arange(jm))
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    # imagesc.plot(c.T)
    plt.imshow(np.flipud(c.T), cmap="jet")
    plt.axis([0, (im - 1), 0, (jm - 1)])
    plt.title("(a)-CO2 Concentration)", fontsize=10)
    plt.ylabel("Z direction", fontsize=10)
    plt.xlabel("X direction", fontsize=10)

    plt.subplot(1, 2, 2)
    # imagesc.plot(T.T)
    # plt.pcolormesh(XX.T,YY.T,Tt,cmap = 'jet')
    plt.imshow(np.flipud(Tt.T), cmap="jet")
    plt.axis([0, (im - 1), 0, (jm - 1)])
    plt.title("(b)-Temperature)", fontsize=10)
    plt.ylabel("Z direction", fontsize=10)
    plt.xlabel("X direction", fontsize=10)
    plt.suptitle(namee, fontsize=20)
    namez = "Iteration_ " + str(itrr) + ".png"
    plt.savefig(namez)
    plt.clf()
    plt.close()


print(".........................START OF CODE........................")
im = 33  # grid size in x direction
im1 = im - 1
jm = 33  # grid size in z direction
jm1 = jm - 1
dx = 1.0 / (im - 1)
dz = 1.0 / (jm - 1)
beta = dx / dz
dt = 1 * 1e-06  # dimensionless time step size
itermx = 2000  # maximum no. of iterations (days)
km = 100  # mean permeability
# aa = importdata('PERM_35.xlsx')  # read heterogeneous permeability data in mD
aa = np.ones((im, jm))
aa = km * aa  # for homogeneous distribution

phi = 0.2  # porosity
k = 4.0e-13  # permeability (Darcy, 9.869233e-13 m**2)
k_T = 0.657  # thermal conductivity (or, 0.353)
Cp = 4043  # specific heat at constant pressure (J/Kg/K) (or, 3420)
D = 1e-09  # Diffusion coefficient (m**2/s)
alpha = 3.7e-7  # Thermal diffusivity (m**2/s)
mu = 9.95 * 1e-04  # kg/m.s  # viscosity
g = 9.81  # m/s**2
H = 100  # m      # reservoir depth

beta_C = 8.44 * 1e-04  # m3/kg
beta_T = -1.205 * 1e-04  # /K
rho0 = 1000.00  # kg/m3 (or, 1185.00)
delC = 3  # kg/m3
delT = 20  # K
dtime = H * H * dt / D  # dimensional time step (sec)
time_max = itermx * dtime / 86400 / 365.25  # maximum time(years)

Ra = rho0 * beta_C * delC * g * k * H / (phi * D * mu)
RaC = (
    rho0 * beta_C * delC * g * k * H / (phi * D * mu)
)  # Ra for stream function (solutal)
RaT = (
    rho0 * beta_T * delT * g * k * H / (alpha * mu)
)  # Ra for stream function (temperature)
Pr = 1 / (k_T / (rho0 * Cp * D))  # Pr for temperature (Energy) equation
Le = alpha / phi / D  # Lewis number
N = RaC / RaT / Le  # Buoyancy ratio

# Pr = 0.0062
# Ra = 10000                 # Ra number
# phi = 0.42                  # porosity
# k = 1200.0                 # permeability (Darcy)
# D = 2*1e-09                 # Diffusion coefficient (m**2/s)

# non-dimensionalising of permeability
Kx = aa / km  # Dimensional
Kz = Kx
kx = Kx  # non-dimensional
kz = Kz

# allocating sizes
strfun = 0 * np.ones((im, jm))
ux = 0 * np.ones((im, jm))
uz = 0 * np.ones((im, jm))
c = 0 * np.ones((im, jm))  # non-dimensional
T = 0 * np.ones((im, jm))
amu = 1.0  # do not change

a = np.zeros((im))
a[-1] = 0
b = np.zeros((im))
b[-1] = 0
cc = np.zeros((im))
d = np.zeros((im))
tt = np.zeros((im))
x = 0 * np.zeros((im))
x[0] = 0
# boundary conditions (concentration)
c[:im, 0] = 1
for i in prange(1, im):
    x[i] = x[i - 1] + dx

for i in prange(im):
    c[i, 1] = 1 + 0.01 * np.sin(2 * np.pi * x[i] / (1 / 24))

cn = c
for i in prange(im):
    T[i, 1] = 1 + 0.01 * np.sin(2 * np.pi * x[i] / (1 / 24))

Tn = T

# calculation starts:
iterr = 0
uxall = []
uzall = []
call = []
tall = []

for itr in range(itermx):
    cn = eqvor_ADI_upwind(im, jm, dx, dz, c, dt, ux, uz, amu)
    errc = np.sum(np.sum(np.abs(np.abs(cn) - np.abs(c))))
    c = cn
    Tn = eqvor_ADI_Temp(im, jm, dx, dz, Tn, dt, ux * phi, uz * phi, 1 / Pr)
    for i in range(im):
        # print(i)
        jmm = jm
        jmm = jmm - 1
        # print(jmm)
        for j in range(jm):
            # print(j)
            T[i, j] = Tn[i, jmm]
            jmm = jmm - 1

    ## solving stream function
    it = 0
    for ii in range(2000):
        ## Jacobian iteration
        err2 = 0
        for i in range(1, im1):
            for j in range(1, jm1):
                strfunnew = (
                    1
                    / 2
                    / (kx[i, j] + kz[i, j] * beta**2)
                    * (
                        kx[i, j] * (strfun[i - 1, j] + strfun[i + 1, j])
                        + beta**2 * kz[i, j] * (strfun[i, j - 1] + strfun[i, j + 1])
                        - 0.5 * RaC * (c[i + 1, j] - c[i - 1, j]) * dx
                        - 0.5 * Le * RaT * (T[i + 1, j] - T[i - 1, j]) * dx
                    )
                )
                err2 = err2 + np.abs((strfun[i, j] - strfunnew) / strfun[i, j])
                strfun[i, j] = strfunnew

        ## checking convergence of stream function at each time iteration
        it = it + 1
        if err2 < 0.0001:
            break
        elif it >= 2000:
            print("too many iterations...")
            break

    ## updating velocities from stream function
    for i in range(1, im1):
        for j in range(1, jm1):
            ux[i, j] = -(strfun[i, j + 1] - strfun[i, j - 1]) / 2 / dz
            uz[i, j] = (strfun[i + 1, j] - strfun[i - 1, j]) / 2 / dx

    ## time step size calculation
    Umax = np.sqrt(np.max(np.max(np.abs(ux))) ** 2 + np.max(np.max(np.abs(uz))) ** 2)
    dtmax = 0.1 * (dx / Umax)
    dt = min(1.05 * dt, dtmax)

    # Steady state checking
    iterr = iterr + 1
    # ratio1 = err1/np.sum(np.sum(np.abs(c)))
    # if total iteration number is geater than the maximum iteration
    # number allowed, program is terminated.
    if iterr > itermx:
        print("solution not converged..")
        break

    # # if error is less than pre-specified value, solution is converged.
    # # otherwise, next iteration begins.
    # if err1>0.00001:
    #     continue
    # else:
    #     iterr
    #     break

    if (itr % 10) == 0:
        print("Iteration--" + str((itr + 1)) + " | " + str(itermx))
        # drawnow(draw_fig)
        namee = "Iteration--" + str(itr + 1)
        draw_fig(c, T, namee, itr + 1)
    uxall.append(ux)
    uzall.append(uz)
    call.append(c)
    tall.append(T)

from PIL import Image
import glob

frames = []
imgs = sorted(glob.glob("*Iteration*"), key=os.path.getmtime)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
name_j = "Evolution" + ".gif"
frames[0].save(
    name_j, format="GIF", append_images=frames[1:], save_all=True, duration=500, loop=0
)
from glob import glob

for f3 in glob("*Iteration*"):
    os.remove(f3)


del glob
