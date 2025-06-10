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
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from NVRS import *
from physicsnemo.sym.models.fno import *
import shutil
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
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


oldfolder = os.getcwd()
os.chdir(oldfolder)


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


default = int(input("Select 1 = use default | 2 = Use user defined \n"))

if default == 1:
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


if not os.path.exists("../COMPARE_RESULTS_NUMERICAL"):
    os.makedirs("../COMPARE_RESULTS_NUMERICAL")
else:
    shutil.rmtree("../COMPARE_RESULTS_NUMERICAL")
    os.makedirs("../COMPARE_RESULTS_NUMERICAL")


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

chuu = None
while True:
    chuu = np.int(
        input(
            "Select model to use to test the skill of the surrogate -\n\
    1 = Model default\n\
    2 = From realisations\n: "
        )
    )
    if (chuu > 2) or (chuu < 1):
        # raise SyntaxError('please select value between 1-2')
        print("")
        print("please try again and select value between 1-2")
    else:

        break

if chuu == 1:
    bb = os.path.isfile("../PACKETS/iglesias2.out")
    if bb == False:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1bib2JAZfBpW4bKz5LdCmhAOtxQixkTzj", "../PACKETS/iglesias2.out"
        )
        print("...Downlaod completed.......")

    Truee = np.genfromtxt("../PACKETS/iglesias2.out", dtype="float")
else:
    bb1 = os.path.isfile(("../PACKETS/Ganensemble.mat"))
    if bb1 == False:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1w81M5M2S0PD9CF2761dxmiKQ5c0OFPaH",
            to_absolute_path("../PACKETS/Ganensemble.mat"),
        )
        print("...Downlaod completed.......")

    filename = "../PACKETS/Ganensemble.mat"  # Ensemble generated offline
    mat = sio.loadmat(filename)
    ini_ensemblef = mat["Z"]
    index = np.random.choice(ini_ensemblef.shape[1], 1, replace=False)

    Truee = ini_ensemblef[:, index]
    scaler1a = MinMaxScaler(feature_range=(aay, bby))
    (scaler1a.fit(Truee))
    Truee = scaler1a.transform(Truee)

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

abig = []
bbig = []
for kk in range(5):
    method = kk + 1
    typee = 0
    print("")
    if method == 1:
        print("------------------Left-preconditioned GMRES-------------------")
    elif method == 2:
        print("--------------------------spsolve-----------------------------")
    elif method == 3:
        print("---------------Left-preconditioned Conjugate gradient---------")
    elif method == 4:
        print("------------------------------LSQR-----------------------------")
    elif method == 5:
        print("-------------------------CPR----------------------------------")
    print("")
    start_time_plots1 = time.time()
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
    elapsed_time_secs = time.time() - start_time_plots1
    msg = "Reservoir simulation  took: %s secs (Wall clock time)" % timedelta(
        seconds=round(elapsed_time_secs)
    )
    print(msg)
    print("")
    abig.append(X_data1)
    bbig.append(data_use1)


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


print("")
print("Finished constructing Pytorch inputs")


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
    if bb == False:
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
    if bba == False:
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
    if bb == False:
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
    if bba == False:
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
    if bb == False:
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
    if bba == False:
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
    if bb == False:
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
    if bba == False:
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

inn = {
    "perm": torch.from_numpy(ini_ensemble1).to(device, torch.float32),
    "Q": torch.from_numpy(ini_ensemble2).to(device, dtype=torch.float32),
    "Qw": torch.from_numpy(ini_ensemble3).to(device, dtype=torch.float32),
    "Phi": torch.from_numpy(ini_ensemble4).to(device, dtype=torch.float32),
    "Time": torch.from_numpy(ini_ensemble5).to(device, dtype=torch.float32),
    "Pini": torch.from_numpy(ini_ensemble6).to(device, dtype=torch.float32),
    "Swini": torch.from_numpy(ini_ensemble7).to(device, dtype=torch.float32),
}

print("")
print("predicting with surrogate model")
ouut_p = modelP(inn)["pressure"].detach().cpu().numpy()
ouut_s = modelS(inn)["water_sat"].detach().cpu().numpy()
ouut_oil = np.ones_like(ouut_s) - ouut_s


print("")
print("Plotting outputs")
os.chdir("../COMPARE_RESULTS_NUMERICAL")
Runs = steppi
ty = np.arange(1, Runs + 1)

methodd = 5
data_use1 = bbig[methodd - 1]
for kk in range(steppi):
    progressBar = "\rPlotting Progress: " + ProgressBar(Runs - 1, kk - 1, Runs - 1)
    ShowBar(progressBar)
    time.sleep(1)
    Plot_performance(
        ouut_p[0, :, :, :],
        ouut_s[0, :, :, :],
        data_use1[0, :, :, :],
        nx,
        ny,
        "PINN_model_PyTorch.png",
        UIR,
        kk,
        dt,
        MAXZ,
        pini_alt,
        steppi,
        wells,
    )
    Plot_performance(
        ouut_p[0, :, :, :],
        ouut_s[0, :, :, :],
        data_use1[0, :, :, :],
        nx,
        ny,
        "new.png",
        UIR,
        kk,
        dt,
        MAXZ,
        pini_alt,
        steppi,
        wells,
    )

    Plot_performance_Numerical(
        bbig, nx, ny, "Jesuss.png", UIR, kk, dt, MAXZ, pini_alt, steppi, wells
    )

    Pressz = (ouut_p[0, kk, :, :][:, :, None]) * pini_alt
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii
    plot3d2(Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii)

    watsz = ouut_s[0, kk, :, :][:, :, None]
    maxii = max(watsz.ravel())
    minii = min(watsz.ravel())
    plot3d2(watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii)

    oilsz = 1 - ouut_s[0, kk, :, :][:, :, None]
    maxii = max(oilsz.ravel())
    minii = min(oilsz.ravel())
    plot3d2(oilsz, nx, ny, nz, kk, dt, MAXZ, "unie_oil", "oil_sat", maxii, minii)
progressBar = "\rPlotting Progress: " + ProgressBar(Runs - 1, kk, Runs - 1)
ShowBar(progressBar)
time.sleep(1)

print("")
print("Now - Creating GIF")
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
imgs = sorted(glob.glob("*Jesuss*"), key=os.path.getmtime)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "Evolution_numerical.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0,
)

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
for f3 in glob("*Jesuss*"):
    os.remove(f3)
for f4 in glob("*new*"):
    os.remove(f4)

for f4 in glob("*unie_pressure*"):
    os.remove(f4)

for f4 in glob("*unie_water*"):
    os.remove(f4)

for f4 in glob("*unie_oil*"):
    os.remove(f4)

# os.chdir(oldfolder)


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


see = Peaceman_well(
    inn,
    ouut_p,
    ouut_s,
    MAXZ,
    1,
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
seeTrue = Peaceman_well(
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


seeuse = pd.DataFrame(see)
seeuse.to_csv("RSM_PINO.csv", header=spittsbig, sep=",")
seeuse.drop(columns=seeuse.columns[0], axis=1, inplace=True)


seeuset = pd.DataFrame(seeTrue)
seeuset.to_csv("RSM_FVM.csv", header=spittsbig, sep=",")
seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

Plot_RSM_percentile(see, seeTrue, "Compare.png")

Numpred = []
for kk in range(5):
    progressBar = "\rPlotting Progress: " + ProgressBar(5, kk - 1, 5)
    ShowBar(progressBar)
    time.sleep(1)
    data_use1 = bbig[kk]
    seeTrue = Peaceman_well(
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
    if kk == 0:
        namei = "Left_preconditioned_GMRES"
    elif kk == 1:
        namei = "spsolve"
    elif kk == 2:
        namei = "Left_preconditioned_Conjugate_gradient"
    elif kk == 3:
        namei = "LSQR"
    elif kk == 4:
        namei = "CPR"

    seeuset = pd.DataFrame(seeTrue)
    seeuset.to_csv(namei + ".csv", header=spittsbig, sep=",")
    seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)
    Numpred.append(seeTrue)
progressBar = "\rPlotting Progress: " + ProgressBar(5, kk, 5)
ShowBar(progressBar)
time.sleep(1)

Plot_RSM_Numerical(see, Numpred, "Compare_Numerical.png")
os.chdir(oldfolder)
print("")
print("-------------------PROGRAM EXECUTED-----------------------------------")
