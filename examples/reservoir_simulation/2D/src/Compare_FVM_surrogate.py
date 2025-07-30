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

import os
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from NVRS import *
from physicsnemo.sym.models.fno import *
import shutil
import pandas as pd
import scipy.io as sio
import torch
import yaml
from multiprocessing import Lock, Value
from PIL import Image
import requests
import concurrent.futures


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


def process_chunk(chunk):
    chunk_results = []
    for kk in chunk:
        result = process_step(kk)
        chunk_results.append(result)

    with lock:
        Runs = len(chunks)
        processed_chunks.value += 1
        completion_percentage = (processed_chunks.value / len(chunks)) * 100
        100 - completion_percentage
        # print(f"Processed chunk {processed_chunks.value}. {completion_percentage:.2f}% completed. {remaining_percentage:.2f}% remaining.")
        progressBar = "\rPlotting Progress: " + ProgressBar(
            Runs, processed_chunks.value, Runs
        )
        ShowBar(progressBar)
        # time.sleep(1)

    return chunk_results


def sort_key(s):
    """Extract the number from the filename for sorting."""
    return int(re.search(r"\d+", s).group())


def process_step(kk):
    f_3 = plt.figure(figsize=(20, 20), dpi=200)
    current_time = int((kk + 1) * dt * MAXZ)

    f_3 = plt.figure(figsize=(20, 20), dpi=200)

    look = ouut_p[0, kk, :, :]
    look = look * pini_alt
    lookf = cPress[0, kk, :, :]
    lookf = lookf * pini_alt
    diff1 = abs(look - lookf)
    ax1 = f_3.add_subplot(331, projection="3d")
    Plot_PhysicsNeMo(
        ax1,
        nx,
        ny,
        nz,
        look,
        N_injw,
        N_pr,
        "pressure PhysicsNeMo",
        injectors,
        producers,
    )
    ax2 = f_3.add_subplot(332, projection="3d")
    Plot_PhysicsNeMo(
        ax2, nx, ny, nz, lookf, N_injw, N_pr, "pressure Numerical", injectors, producers
    )
    ax3 = f_3.add_subplot(333, projection="3d")
    Plot_PhysicsNeMo(
        ax3, nx, ny, nz, diff1, N_injw, N_pr, "pressure diff", injectors, producers
    )
    R2p, L2p = compute_metrics(look.ravel(), lookf.ravel())

    look = ouut_s[0, kk, :, :]
    lookf = cSat[0, kk, :, :]
    diff1 = abs(look - lookf)
    ax1 = f_3.add_subplot(334, projection="3d")
    Plot_PhysicsNeMo(
        ax1, nx, ny, nz, look, N_injw, N_pr, "water PhysicsNeMo", injectors, producers
    )
    ax2 = f_3.add_subplot(335, projection="3d")
    Plot_PhysicsNeMo(
        ax2, nx, ny, nz, lookf, N_injw, N_pr, "water Numerical", injectors, producers
    )
    ax3 = f_3.add_subplot(336, projection="3d")
    Plot_PhysicsNeMo(
        ax3, nx, ny, nz, diff1, N_injw, N_pr, "water diff", injectors, producers
    )
    R2w, L2w = compute_metrics(look.ravel(), lookf.ravel())

    look = 1 - ouut_s[0, kk, :, :]
    lookf = 1 - cSat[0, kk, :, :]
    diff1 = abs(look - lookf)
    ax1 = f_3.add_subplot(337, projection="3d")
    Plot_PhysicsNeMo(
        ax1, nx, ny, nz, look, N_injw, N_pr, "oil PhysicsNeMo", injectors, producers
    )
    ax2 = f_3.add_subplot(338, projection="3d")
    Plot_PhysicsNeMo(
        ax2, nx, ny, nz, lookf, N_injw, N_pr, "oil Numerical", injectors, producers
    )
    ax3 = f_3.add_subplot(339, projection="3d")
    Plot_PhysicsNeMo(
        ax3, nx, ny, nz, diff1, N_injw, N_pr, "oil diff", injectors, producers
    )
    R2o, L2o = compute_metrics(look.ravel(), lookf.ravel())

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    tita = "Timestep --" + str(current_time) + " days"
    plt.suptitle(tita, fontsize=16)
    plt.savefig("Dynamic" + str(int(kk)))
    plt.clf()
    plt.close()

    return current_time, (R2p, L2p), (R2w, L2w), (R2o, L2o)


oldfolder = os.getcwd()
os.chdir(oldfolder)

surrogate = None
while True:
    surrogate = int(
        input(
            "Select surrogate method type:\n1=FNO [PhysicsNeMo Implementation]\n\
2=PINO [PhysicsNeMo Implementation]\n3\3=AFNO (data driven)\n4=AFNO ( data + physics driven)\n"
        )
    )
    if (surrogate > 4) or (surrogate < 1):
        # raise SyntaxError('please select value between 1-2')
        print("")
        print("please try again and select value between 1-4")
    else:
        break
if not os.path.exists("../COMPARE_RESULTS"):
    os.makedirs("../COMPARE_RESULTS")


if surrogate == 1:
    folderr = "../COMPARE_RESULTS/FNO"
    if not os.path.exists("../COMPARE_RESULTS/FNO"):
        os.makedirs("../COMPARE_RESULTS/FNO")
    else:
        shutil.rmtree("../COMPARE_RESULTS/FNO")
        os.makedirs("../COMPARE_RESULTS/FNO")

elif surrogate == 2:
    folderr = "../COMPARE_RESULTS/PINO"
    if not os.path.exists("../COMPARE_RESULTS/PINO"):
        os.makedirs("../COMPARE_RESULTS/PINO")
    else:
        shutil.rmtree("../COMPARE_RESULTS/PINO")
        os.makedirs("../COMPARE_RESULTS/PINO")
elif surrogate == 3:
    folderr = "../COMPARE_RESULTS/AFNOD"
    if not os.path.exists("../COMPARE_RESULTS/AFNOD"):
        os.makedirs("../COMPARE_RESULTS/AFNOD")
    else:
        shutil.rmtree("../COMPARE_RESULTS/AFNOD")
        os.makedirs("../COMPARE_RESULTS/AFNOD")
else:
    folderr = "../COMPARE_RESULTS/AFNOP"
    if not os.path.exists("../COMPARE_RESULTS/AFNOP"):
        os.makedirs("../COMPARE_RESULTS/AFNOP")
    else:
        shutil.rmtree("../COMPARE_RESULTS/AFNOP")
        os.makedirs("../COMPARE_RESULTS/AFNOP")

if not os.path.exists("../PACKETS"):
    os.makedirs("../PACKETS")
else:
    pass

plan = read_yaml("conf/config_FNO.yaml")
injectors = plan["custom"]["WELLSPECS"]["water_injector_wells"]
producers = plan["custom"]["WELLSPECS"]["producer_wells"]


N_injw = len(
    plan["custom"]["WELLSPECS"]["water_injector_wells"]
)  # Number of water injectors
N_pr = len(plan["custom"]["WELLSPECS"]["producer_wells"])  # Number of producers


# Varaibles needed for NVRS
nx = plan["custom"]["NVRS"]["nx"]
ny = plan["custom"]["NVRS"]["ny"]
nz = plan["custom"]["NVRS"]["nz"]
BO = plan["custom"]["NVRS"]["BO"]  # oil formation volume factor
BW = plan["custom"]["NVRS"]["BW"]  # Water formation volume factor
UW = plan["custom"]["NVRS"]["UW"]  # water viscosity in cP
UO = plan["custom"]["NVRS"]["UO"]  # oil viscosity in cP
DX = plan["custom"]["NVRS"]["DX"]  # size of pixel in x direction
DY = plan["custom"]["NVRS"]["DY"]  # sixze of pixel in y direction
DZ = plan["custom"]["NVRS"]["DZ"]  # sizze of pixel in z direction

DX = cp.float32(DX)
DY = cp.float32(DY)
UW = cp.float32(UW)  # water viscosity in cP
UO = cp.float32(UO)  # oil viscosity in cP
SWI = cp.float32(plan["custom"]["NVRS"]["SWI"])
SWR = cp.float32(plan["custom"]["NVRS"]["SWR"])
CFO = cp.float32(plan["custom"]["NVRS"]["CFO"])  # oil compressibility in 1/psi
IWSw = plan["custom"]["NVRS"]["IWSw"]  # initial water saturation
pini_alt = plan["custom"]["NVRS"]["pini_alt"]
# print(pini_alt)
P1 = cp.float32(pini_alt)  # Bubble point pressure psia
PB = P1
mpor, hpor = (
    plan["custom"]["NVRS"]["mpor"],
    plan["custom"]["NVRS"]["hpor"],
)  # minimum and maximum porosity
BW = cp.float32(BW)  # Water formation volume factor
BO = cp.float32(BO)  # Oil formation volume factor
PATM = cp.float32(plan["custom"]["NVRS"]["PATM"])  # Atmospheric pressure in psi

# training
LUB, HUB = (
    plan["custom"]["NVRS"]["LUB"],
    plan["custom"]["NVRS"]["HUB"],
)  # Permeability rescale
aay, bby = (
    plan["custom"]["NVRS"]["aay"],
    plan["custom"]["NVRS"]["bby"],
)  # Permeability range mD
Low_K, High_K = aay, bby


batch_size = plan["custom"]["NVRS"][
    "batch_size"
]  #'size of simulated labelled dtaa to run'
timmee = plan["custom"]["NVRS"][
    "timmee"
]  # float(input ('Enter the time step interval duration for simulation (days): '))
max_t = plan["custom"]["NVRS"][
    "max_t"
]  # float(input ('Enter the maximum time in days for simulation(days): '))
MAXZ = plan["custom"]["NVRS"]["MAXZ"]  # reference maximum time in days of simulation
steppi = int(max_t / timmee)
choice = 1  #  1= Non-Gaussian prior, 2 = Gaussian prior
factorr = 0.1  # from [0 1] excluding the limits for PermZ
LIR = plan["custom"]["NVRS"]["LIR"]  # lower injection rate
UIR = plan["custom"]["NVRS"]["UIR"]  # uppwer injection rate
RE = 0.2 * DX
rwell = 200  # well radius
skin = 0  # well deformation
pwf_producer = 100
cuda = 0
input_channel = 7  # [Perm, Q,QW,Phi,dt, initial_pressure, initial_water_sat]
device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
N_inj = 4
N_pr = 4


# tc2 = Equivalent_time(timmee,2100,timmee,max_t)
tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
dt = np.diff(tc2)[0]  # Time-step
# 4 injector and 4 producer wells
wells = np.array(
    [1, 24, 1, 3, 3, 1, 31, 1, 1, 31, 31, 1, 7, 9, 2, 14, 12, 2, 28, 19, 2, 14, 27, 2]
)
wells = np.reshape(wells, (-1, 3), "C")


bb = os.path.isfile(to_absolute_path("../PACKETS/Test4.mat"))
if not bb:
    print("....Downloading Please hold.........")
    download_file_from_google_drive(
        "1G4Cvg8eIObyBK0eoo7iX-0hhMTnpJktj", to_absolute_path("../PACKETS/Test4.mat")
    )
    print("...Downlaod completed.......")
    print("Load simulated labelled test data from MAT file")
    matt = sio.loadmat(to_absolute_path("../PACKETS/Test4.mat"))
    X_data11 = matt["INPUT"]
    data_use11 = matt["OUTPUT"]

else:
    print("Load simulated labelled test data from MAT file")
    matt = sio.loadmat(to_absolute_path("../PACKETS/Test4.mat"))
    X_data11 = matt["INPUT"]
    data_use11 = matt["OUTPUT"]

# index = np.random.choice(X_data11.shape[0], 1, \
#                                  replace=False)
index = 253
X_data1 = X_data11[index, :, :, :][None, :, :, :]
data_use1 = data_use11[index, :, :, :][None, :, :, :]

# print(X_data1.shape)

Ne = 1
ini_ensemble1 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble2 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble3 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble4 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble5 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble6 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
ini_ensemble7 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)

cPress = np.zeros((X_data1.shape[0], steppi, nx, ny))  # Pressure
cSat = np.zeros((X_data1.shape[0], steppi, nx, ny))  # Water saturation


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

    perm = data_use1[kk, :steppi, :, :]
    cPress[kk, :, :, :] = perm  # np.clip(perm ,1/pini_alt,1.)

    perm = data_use1[kk, steppi:, :, :]
    cSat[kk, :, :, :] = perm


print("")
print("Finished constructing Pytorch inputs")


print("*******************Load the trained Forward models*******************")

if (surrogate == 1) or (surrogate == 2):
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

if (surrogate == 3) or (surrogate == 4):
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
            "14njayJd77pKUeuCG9Jno03MbS8SaAVbn",
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
            "1fI1whx3CJSsICcvMddF2uF1hlMekWUrF",
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
            "1lmtvM8LucUU3DqSgUe7hYyYZvYS4yLnf",
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
            "10n0NgdgskQbNyTtZNG53IhttB8WCpzli",
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
            "1JGV9vNYIRUWS1Gd6bqDcG_jRbO8wEkzX",
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
            "1V_TKAqWlw3_kKaE1vnWGwWB7vlmk7t3s",
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
elif surrogate == 4:
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
            "18BrbHfgM3r8OWmpuMhZi9y9geUyAI8t4",
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
            "1aSf4QYoU0ITuQdbSNRPWbtkQz7ZQh_Bt",
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
start_time_plots1 = time.time()
ouut_p = modelP(inn)["pressure"].detach().cpu().numpy()
ouut_s = modelS(inn)["water_sat"].detach().cpu().numpy()
elapsed_time_secs = time.time() - start_time_plots1
msg = "Surrogate Reservoir simulation  took: %s secs (Wall clock time)" % timedelta(
    seconds=round(elapsed_time_secs)
)
print(msg)
print("")
ouut_oil = np.ones_like(ouut_s) - ouut_s


print("")
print("Plotting outputs")
os.chdir(folderr)
Runs = steppi
ty = np.arange(1, Runs + 1)

Time_vector = np.zeros((steppi))
Accuracy_presure = np.zeros((steppi, 2))
Accuracy_oil = np.zeros((steppi, 2))
Accuracy_water = np.zeros((steppi, 2))


lock = Lock()
processed_chunks = Value("i", 0)


NUM_CORES = 12  # specify the number of cores you want to use

# Split the range of steps into chunks
chunks = [
    list(range(i, min(i + steppi // NUM_CORES, steppi)))
    for i in range(0, steppi, steppi // NUM_CORES)
]

with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
    chunked_results = list(executor.map(process_chunk, chunks))

# Flatten the chunked results to get the ordered results
results = [result for sublist in chunked_results for result in sublist]

for kk, (current_time, acc_pressure, acc_oil, acc_water) in enumerate(results):
    Time_vector[kk] = current_time
    Accuracy_presure[kk] = acc_pressure
    Accuracy_oil[kk] = acc_oil
    Accuracy_water[kk] = acc_water

fig4 = plt.figure(figsize=(20, 20), dpi=100)
font = FontProperties()
font.set_family("Helvetica")
font.set_weight("bold")

fig4.text(
    0.5,
    0.98,
    "R2(%) Accuracy - PhysicsNeMo/Numerical(GPU)",
    ha="center",
    va="center",
    fontproperties=font,
    fontsize=11,
)
fig4.text(
    0.5,
    0.49,
    "L2(%) Accuracy - PhysicsNeMo/Numerical(GPU)",
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
    Accuracy_oil[:, 0],
    label="R2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("oil saturation", fontproperties=font)
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
    Accuracy_oil[:, 1],
    label="L2",
    marker="*",
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=0.5,
)
plt.title("oil saturation", fontproperties=font)
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

import re

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
    cPress,
    cSat,
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
seeuse.to_csv("RSM_MODULUS.csv", header=spittsbig, sep=",")
seeuse.drop(columns=seeuse.columns[0], axis=1, inplace=True)


seeuset = pd.DataFrame(seeTrue)
seeuset.to_csv("RSM_NUMERICAL.csv", header=spittsbig, sep=",")
seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

Plot_RSM_percentile(see, seeTrue, "Compare.png")

os.chdir(oldfolder)
print("")
print("-------------------PROGRAM EXECUTED-----------------------------------")
