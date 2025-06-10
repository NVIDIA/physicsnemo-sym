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


default = int(input("Select 1 = use default | 2 = Use user defined \n"))
if not os.path.exists("../COMPARE_RESULTS"):
    os.makedirs("../COMPARE_RESULTS")
else:
    shutil.rmtree("../COMPARE_RESULTS")
    os.makedirs("../COMPARE_RESULTS")


if default == 1:
    print(" Surrogate Model chosen = PINO")
    surrogate = 2
else:
    surrogate = None
    while True:
        surrogate = int(
            input(
                "Select surrogate method type:\n1=FNO [PhysicsNeMo Implementation]\n\
2=PINO [PhysicsNeMo Implemnation]\n"
            )
        )
        if (surrogate > 2) or (surrogate < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:

            break


if default == 1:
    print(" --------------Use constrained presure residual method--------- ")
    method = 6
    typee = 0
else:

    method = None
    while True:
        method = cp.int(
            input(
                "Select inverse problem solution method for pressure & saturation \
        solve -\n\
        1 = GMRES\n\
        2 = spsolve\n\
        3 = Conjugate Gradient\n\
        4 = LSQR\n\
        5 = AMG\n\
        6 = CPR\n\
        7 = AMGX\n: "
            )
        )
        if (method > 7) or (method < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-6")
        else:

            break

    if method == 7:
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
nx, ny, nz = 40, 40, 3
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
    [1, 24, 1, 1, 1, 1, 31, 1, 1, 31, 31, 1, 7, 9, 2, 14, 12, 2, 28, 19, 2, 14, 27, 2]
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
    if (chuu > 2) or (method < 1):
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
            "1VSy2m3ocUkZnhCsorbkhcJB5ADrPxzIp", "../PACKETS/iglesias2.out"
        )
        print("...Downlaod completed.......")

    else:
        pass
    Truee = np.genfromtxt("../PACKETS/iglesias2.out", dtype="float")
else:
    bb1 = os.path.isfile(("../PACKETS/Ganensemble.mat"))
    if bb1 == False:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1KZvxypUSsjpkLogGm__56-bckEee3VJh",
            to_absolute_path("../PACKETS/Ganensemble.mat"),
        )
        print("...Downlaod completed.......")
    else:
        pass

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


X_data2 = X_data1


ini_ensemble1 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
ini_ensemble2 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
ini_ensemble3 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
ini_ensemble4 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
ini_ensemble5 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
ini_ensemble6 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
ini_ensemble7 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)


for kk in range(X_data2.shape[0]):
    perm = X_data2[kk, 0, :, :, :]
    permin = np.zeros((1, nz, nx, ny))
    for i in range(nz):
        permin[0, i, :, :] = perm[:, :, i]
    ini_ensemble1[kk, :, :, :, :] = permin

    perm = X_data2[kk, 1, :, :, :]
    permin = np.zeros((1, nz, nx, ny))
    for i in range(nz):
        permin[0, i, :, :] = perm[:, :, i]
    ini_ensemble2[kk, :, :, :, :] = permin

    perm = X_data2[kk, 2, :, :, :]
    permin = np.zeros((1, nz, nx, ny))
    for i in range(nz):
        permin[0, i, :, :] = perm[:, :, i]
    ini_ensemble3[kk, :, :, :, :] = permin

    perm = X_data2[kk, 3, :, :, :]
    permin = np.zeros((1, nz, nx, ny))
    for i in range(nz):
        permin[0, i, :, :] = perm[:, :, i]
    ini_ensemble4[kk, :, :, :, :] = permin

    perm = X_data2[kk, 4, :, :, :]
    permin = np.zeros((1, nz, nx, ny))
    for i in range(nz):
        permin[0, i, :, :] = perm[:, :, i]
    ini_ensemble5[kk, :, :, :, :] = permin

    perm = X_data2[kk, 5, :, :, :]
    permin = np.zeros((1, nz, nx, ny))
    for i in range(nz):
        permin[0, i, :, :] = perm[:, :, i]
    ini_ensemble6[kk, :, :, :, :] = permin

    perm = X_data2[kk, 6, :, :, :]
    permin = np.zeros((1, nz, nx, ny))
    for i in range(nz):
        permin[0, i, :, :] = perm[:, :, i]
    ini_ensemble7[kk, :, :, :, :] = permin


print("")
print("Finished constructing Pytorch inputs")


print("*******************Load the trained Forward models*******************")

decoder1 = ConvFullyConnectedArch([Key("z", size=32)], [Key("pressure", size=steppi)])
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
    fno_modes=16,
    dimension=3,
    padding=13,
    nr_fno_layers=4,
    decoder_net=decoder1,
)

decoder2 = ConvFullyConnectedArch([Key("z", size=32)], [Key("water_sat", size=steppi)])
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
    fno_modes=16,
    dimension=3,
    padding=13,
    nr_fno_layers=4,
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
            "1WcLz50Iz5nlBtAYYdAjpigwc7qCtxG9W",
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
            "1V-7wSyaV7Fd_tThedqlL-q7861p6ZzHj",
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
else:
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
            "1YxNkCTEWCUDyYbztFSTnEaRV2h3AR6yT",
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
            "1L1b9Jhaz-jAgFUGASqf5QY6onsiRf9VZ",
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
os.chdir("../COMPARE_RESULTS")
Runs = steppi
ty = np.arange(1, Runs + 1)
for kk in range(steppi):
    progressBar = "\rPlotting Progress: " + ProgressBar(Runs - 1, kk - 1, Runs - 1)
    ShowBar(progressBar)
    time.sleep(1)
    Plot_performance(
        ouut_p[0, :, :, :, :],
        ouut_s[0, :, :, :, :],
        data_use1[0, :, :, :, :],
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

    Pressz = Reinvent(ouut_p[0, kk, :, :, :]) * pini_alt
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii
    plot3d2(Pressz, nx, ny, nz, kk, dt, MAXZ, "unie_pressure", "Pressure", maxii, minii)

    watsz = Reinvent(ouut_s[0, kk, :, :, :])
    maxii = max(watsz.ravel())
    minii = min(watsz.ravel())
    plot3d2(watsz, nx, ny, nz, kk, dt, MAXZ, "unie_water", "water_sat", maxii, minii)

    oilsz = Reinvent(1 - ouut_s[0, kk, :, :, :])
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
imgs = sorted(glob.glob("*pressure_evolution*"), key=os.path.getmtime)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "Evolution_pressure.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0,
)


frames = []
imgs = sorted(glob.glob("*water_evolution*"), key=os.path.getmtime)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "Evolution_water.gif",
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=500,
    loop=0,
)


frames = []
imgs = sorted(glob.glob("*oil_evolution*"), key=os.path.getmtime)
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

frames[0].save(
    "Evolution_oil.gif",
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

for f3 in glob("*pressure_evolution*"):
    os.remove(f3)

for f4 in glob("*water_evolution*"):
    os.remove(f4)

for f4 in glob("*oil_evolution*"):
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
    0,
    nz,
)
seeTrue = Peaceman_well(
    inn,
    data_use1[:, :steppi, :, :, :],
    data_use1[:, steppi:, :, :, :],
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
    1,
    nz,
)


seeuse = pd.DataFrame(see)
seeuse.to_csv("RSM_PINO.csv", header=spittsbig, sep=",")
seeuse.drop(columns=seeuse.columns[0], axis=1, inplace=True)


seeuset = pd.DataFrame(seeTrue)
seeuset.to_csv("RSM_FVM.csv", header=spittsbig, sep=",")
seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

Plot_RSM_percentile(see, seeTrue, "Compare.png")

os.chdir(oldfolder)
print("")
print("-------------------PROGRAM EXECUTED-----------------------------------")
