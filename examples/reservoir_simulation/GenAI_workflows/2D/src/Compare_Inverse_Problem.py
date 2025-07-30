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
from NVRS import *
from physicsnemo.sym.models.fno import *
import shutil
import pandas as pd
import scipy.io as sio
import torch
import requests


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


oldfolder = os.getcwd()
os.chdir(oldfolder)


default = int(input("Select 1 = use default | 2 = Use user defined \n"))
if not os.path.exists("../COMPARE_RESULTS_INVERSE_PROBLEM"):
    os.makedirs("../COMPARE_RESULTS_INVERSE_PROBLEM")
else:
    shutil.rmtree("../COMPARE_RESULTS_INVERSE_PROBLEM")
    os.makedirs("../COMPARE_RESULTS_INVERSE_PROBLEM")


if default == 1:
    print(" --------------Use constrained presure residual method--------- ")
    method = 5
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


if not os.path.exists("../PACKETS"):
    os.makedirs("../PACKETS")
else:
    pass


# Varaibles needed for NVRS
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
UW = cp.float32(1)  # water viscosity in cP
UO = cp.float32(2.5)  # oil viscosity in cP
SWI = cp.float32(0.1)
SWR = cp.float32(0.1)
CFO = cp.float32(1e-5)  # oil compressibility in 1/psi
IWSw = 0.2  # initial water saturation
pini_alt = 1e3
P1 = cp.float32(pini_alt)  # Bubble point pressure psia
PB = P1
mpor, hpor = 0.05, 0.5  # minimum and maximum porosity
BW = cp.float32(BW)  # Water formation volume factor
BO = cp.float32(BO)  # Oil formation volume factor
PATM = cp.float32(14.6959)  # Atmospheric pressure in psi

# training
LUB, HUB = 1e-1, 1  # Permeability rescale
aay, bby = 50, 500  # Permeability range mD
Low_K, High_K = aay, bby


batch_size = 500  #'size of simulated labelled dtaa to run'
timmee = 100.0  # float(input ('Enter the time step interval duration for simulation (days): '))
max_t = 3000.0  # float(input ('Enter the maximum time in days for simulation(days): '))
MAXZ = 6000  # reference maximum time in days of simulation
steppi = int(max_t / timmee)
choice = 1  #  1= Non-Gaussian prior, 2 = Gaussian prior
factorr = 0.1  # from [0 1] excluding the limits for PermZ
LIR = 200  # lower injection rate
UIR = 2000  # uppwer injection rate
RE = 0.2 * DX
rwell = 200  # well radius
skin = 0  # well deformation
pwf_producer = 100
cuda = 0
input_channel = 7  # [Perm, Q,QW,Phi,dt, initial_pressure, initial_water_sat]
device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
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


# tc2 = Equivalent_time(timmee,2100,timmee,max_t)
tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
dt = np.diff(tc2)[0]  # Time-step
# 4 injector and 4 producer wells
wells = np.array(
    [1, 24, 1, 3, 3, 1, 31, 1, 1, 31, 31, 1, 7, 9, 2, 14, 12, 2, 28, 19, 2, 14, 27, 2]
)
wells = np.reshape(wells, (-1, 3), "C")


bb = os.path.isfile("../PACKETS/iglesias2.out")
if not bb:
    print("....Downloading Please hold.........")
    download_file_from_google_drive(
        "1bib2JAZfBpW4bKz5LdCmhAOtxQixkTzj", "../PACKETS/iglesias2.out"
    )
    print("...Downlaod completed.......")

Truee = np.genfromtxt("../PACKETS/iglesias2.out", dtype="float")


Truee = np.reshape(Truee, (-1,))
Ne = 1
ini = []
inip = []
inij = []
injj = 500 * np.ones((1, 4))
for i in range(Ne):
    at1 = rescale_linear(Truee, aay, bby)
    at2 = rescale_linear(Truee, mpor, hpor)
    ini.append(at1.reshape(-1, 1))
    inip.append(at2.reshape(-1, 1))
    inij.append(injj)
ini = np.hstack(ini)
inip = np.hstack(inip)
kka = np.vstack(inij)

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
)

print("")
print("Finished FVM simulations")

ini_ensemble1 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble2 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble3 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble4 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble5 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble6 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble7 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)


for kk in range(X_data1.shape[0]):
    perm = X_data1[kk, 0, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble1[kk, :, :, :] = permin

    perm = X_data1[kk, 1, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble2[kk, :, :, :] = permin

    perm = X_data1[kk, 2, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble3[kk, :, :, :] = permin

    perm = X_data1[kk, 3, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble4[kk, :, :, :] = permin

    perm = X_data1[kk, 4, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble5[kk, :, :, :] = permin

    perm = X_data1[kk, 5, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble6[kk, :, :, :] = permin

    perm = X_data1[kk, 6, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble7[kk, :, :, :] = permin

inn = {
    "perm": torch.from_numpy(ini_ensemble1).to(device, torch.float32),
    "Q": torch.from_numpy(ini_ensemble2).to(device, dtype=torch.float32),
    "Qw": torch.from_numpy(ini_ensemble3).to(device, dtype=torch.float32),
    "Phi": torch.from_numpy(ini_ensemble4).to(device, dtype=torch.float32),
    "Time": torch.from_numpy(ini_ensemble5).to(device, dtype=torch.float32),
    "Pini": torch.from_numpy(ini_ensemble6).to(device, dtype=torch.float32),
    "Swini": torch.from_numpy(ini_ensemble7).to(device, dtype=torch.float32),
}


filename = "../HM_RESULTS/BEST_RESERVOIR_MODEL/BEST_RESERVOIR_MODEL.mat"  # Ensemble generated offline
mat = sio.loadmat(filename)
TrueK = mat["permeability"]
Truephi = mat["porosity"]

Ne = 1
ini = []
inip = []
inij = []
injj = 500 * np.ones((1, 4))
for i in range(Ne):
    at1 = TrueK
    at2 = Truephi
    ini.append(at1.reshape(-1, 1))
    inip.append(at2.reshape(-1, 1))
    inij.append(injj)
ini = np.hstack(ini)
inip = np.hstack(inip)
kka = np.vstack(inij)

X_data2, data_use2 = inference_single(
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
)

print("")
print("Finished FVM simulations")

ini_ensemble1 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble2 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble3 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble4 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble5 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble6 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble7 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)


for kk in range(X_data2.shape[0]):
    perm = X_data2[kk, 0, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble1[kk, :, :, :] = permin

    perm = X_data2[kk, 1, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble2[kk, :, :, :] = permin

    perm = X_data2[kk, 2, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble3[kk, :, :, :] = permin

    perm = X_data2[kk, 3, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble4[kk, :, :, :] = permin

    perm = X_data2[kk, 4, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble5[kk, :, :, :] = permin

    perm = X_data2[kk, 5, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble6[kk, :, :, :] = permin

    perm = X_data2[kk, 6, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble7[kk, :, :, :] = permin

inn2 = {
    "perm": torch.from_numpy(ini_ensemble1).to(device, torch.float32),
    "Q": torch.from_numpy(ini_ensemble2).to(device, dtype=torch.float32),
    "Qw": torch.from_numpy(ini_ensemble3).to(device, dtype=torch.float32),
    "Phi": torch.from_numpy(ini_ensemble4).to(device, dtype=torch.float32),
    "Time": torch.from_numpy(ini_ensemble5).to(device, dtype=torch.float32),
    "Pini": torch.from_numpy(ini_ensemble6).to(device, dtype=torch.float32),
    "Swini": torch.from_numpy(ini_ensemble7).to(device, dtype=torch.float32),
}


filename = "../HM_RESULTS/MEAN_RESERVOIR_MODEL/MEAN_RESERVOIR_MODEL.mat"  # Ensemble generated offline
mat = sio.loadmat(filename)
TrueK = mat["permeability"]
Truephi = mat["porosity"]

Ne = 1
ini = []
inip = []
inij = []
injj = 500 * np.ones((1, 4))
for i in range(Ne):
    at1 = TrueK
    at2 = Truephi
    ini.append(at1.reshape(-1, 1))
    inip.append(at2.reshape(-1, 1))
    inij.append(injj)
ini = np.hstack(ini)
inip = np.hstack(inip)
kka = np.vstack(inij)

X_data3, data_use3 = inference_single(
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
)

print("")
print("Finished FVM simulations")

ini_ensemble1 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble2 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble3 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble4 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble5 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble6 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble7 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)


for kk in range(X_data3.shape[0]):
    perm = X_data3[kk, 0, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble1[kk, :, :, :] = permin

    perm = X_data3[kk, 1, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble2[kk, :, :, :] = permin

    perm = X_data3[kk, 2, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble3[kk, :, :, :] = permin

    perm = X_data3[kk, 3, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble4[kk, :, :, :] = permin

    perm = X_data3[kk, 4, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble5[kk, :, :, :] = permin

    perm = X_data3[kk, 5, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble6[kk, :, :, :] = permin

    perm = X_data3[kk, 6, :, :]
    permin = np.zeros((1, nx, ny))
    permin[0, :, :] = perm
    ini_ensemble7[kk, :, :, :] = permin

inn3 = {
    "perm": torch.from_numpy(ini_ensemble1).to(device, torch.float32),
    "Q": torch.from_numpy(ini_ensemble2).to(device, dtype=torch.float32),
    "Qw": torch.from_numpy(ini_ensemble3).to(device, dtype=torch.float32),
    "Phi": torch.from_numpy(ini_ensemble4).to(device, dtype=torch.float32),
    "Time": torch.from_numpy(ini_ensemble5).to(device, dtype=torch.float32),
    "Pini": torch.from_numpy(ini_ensemble6).to(device, dtype=torch.float32),
    "Swini": torch.from_numpy(ini_ensemble7).to(device, dtype=torch.float32),
}


print("")
print("Plotting outputs")
os.chdir("../COMPARE_RESULTS_INVERSE_PROBLEM")
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


seeTrue = Peaceman_well2(
    inn,
    data_use1[:, :steppi, :, :],
    data_use1[:, steppi:, :, :],
    MAXZ,
    0,
    1e1,
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
    nz,
)

see2 = Peaceman_well2(
    inn2,
    data_use2[:, :steppi, :, :],
    data_use2[:, steppi:, :, :],
    MAXZ,
    0,
    1e1,
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
    nz,
)

see3 = Peaceman_well2(
    inn3,
    data_use3[:, :steppi, :, :],
    data_use3[:, steppi:, :, :],
    MAXZ,
    0,
    1e1,
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
    nz,
)

seeuset = pd.DataFrame(seeTrue)
seeuset.to_csv("RSM_FVM.csv", header=spittsbig, sep=",")
seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)


seeuset = pd.DataFrame(see2)
seeuset.to_csv("aREKI_BEST_FVM.csv", header=spittsbig, sep=",")
seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)


seeuset = pd.DataFrame(see3)
seeuset.to_csv("aREKI_MEAN_FVM.csv", header=spittsbig, sep=",")
seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)


Plot_RSM_percentile2(see2, see3, seeTrue, "Compare_Inverse_Problem.png")

os.chdir(oldfolder)
print("")
print("-------------------PROGRAM EXECUTED-----------------------------------")
