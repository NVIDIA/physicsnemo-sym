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

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import physicsnemo
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from NVRS import *
import time
from datetime import timedelta
from utilities import preprocess_FNO_mat
from ops import dx
from physicsnemo.sym.models.fno import *
import cupy as cp
from PIL import Image

# import glob
from sklearn.model_selection import train_test_split
import scipy.io as sio
import requests


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


def CustomValidatorPlotter(
    timmee,
    max_t,
    MAXZ,
    pini_alt,
    nx,
    ny,
    wells,
    steppi,
    tc2,
    dt,
    namet,
    true_water,
    pred_water,
    true_pressure,
    pred_pressure,
):
    water_true, water_pred = (
        true_water.detach().cpu().numpy(),
        pred_water.detach().cpu().numpy(),
    )
    pressure_true, pressure_pred = (
        true_pressure.detach().cpu().numpy(),
        pred_pressure.detach().cpu().numpy(),
    )

    for itt in range(steppi):
        progressBar = "\rPlotting Progress: " + ProgressBar(
            steppi - 1, itt - 1, steppi - 1
        )
        ShowBar(progressBar)
        time.sleep(1)

        XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
        plt.figure(figsize=(12, 12), dpi=100)

        look = (pressure_pred[0, itt, :, :]) * pini_alt

        lookf = (pressure_true[0, itt, :, :]) * pini_alt

        diff1 = abs(look - lookf)

        XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
        plt.figure(figsize=(12, 12), dpi=100)
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

        look_sat = water_pred[0, itt, :, :]
        look_oil = 1 - look_sat

        lookf_sat = water_true[0, itt, :, :]
        lookf_oil = 1 - lookf_sat

        diff1_wat = abs(look_sat - lookf_sat)
        diff1_oil = abs(look_oil - lookf_oil)

        plt.subplot(3, 3, 4)
        plt.pcolormesh(XX.T, YY.T, look_sat, cmap="jet")
        plt.title("water_sat PINO", fontsize=13)
        plt.ylabel("Y", fontsize=13)
        plt.xlabel("X", fontsize=13)
        plt.axis([0, (nx - 1), 0, (ny - 1)])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        cbar1 = plt.colorbar()
        plt.clim(
            np.min(np.reshape(lookf_sat, (-1,))), np.max(np.reshape(lookf_sat, (-1,)))
        )
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
        plt.clim(
            np.min(np.reshape(lookf_oil, (-1,))), np.max(np.reshape(lookf_oil, (-1,)))
        )
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
    progressBar = "\rPlotting Progress: " + ProgressBar(steppi - 1, itt, steppi - 1)
    ShowBar(progressBar)
    time.sleep(1)


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

    torch.square(S)
    torch.square(torch.sub(torch.ones(S.shape, device=u.device), S))
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


class Labelledset:
    def __init__(self, datacc):
        self.data1 = torch.from_numpy(datacc["perm"])
        self.data2 = torch.from_numpy(datacc["Q"])
        self.data3 = torch.from_numpy(datacc["Qw"])
        self.data4 = torch.from_numpy(datacc["Phi"])
        self.data5 = torch.from_numpy(datacc["Time"])
        self.data6 = torch.from_numpy(datacc["Pini"])
        self.data7 = torch.from_numpy(datacc["Swini"])
        self.data8 = torch.from_numpy(datacc["pressure"])
        self.data9 = torch.from_numpy(datacc["water_sat"])

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        x3 = self.data3[index]
        x4 = self.data4[index]
        x5 = self.data5[index]
        x6 = self.data6[index]
        x7 = self.data7[index]
        x8 = self.data8[index]
        x9 = self.data9[index]

        cuda = 0
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
        return {
            "perm": x1.to(device, torch.float32),
            "Q": x2.to(device, torch.float32),
            "Qw": x3.to(device, torch.float32),
            "Phi": x4.to(device, torch.float32),
            "Time": x5.to(device, torch.float32),
            "Pini": x6.to(device, torch.float32),
            "Swini": x7.to(device, torch.float32),
            "pressure": x8.to(device, torch.float32),
            "water_sat": x9.to(device, torch.float32),
        }

    def __len__(self):
        return len(self.data1)


class unLabelledset:
    def __init__(self, datacc):
        self.data1 = torch.from_numpy(datacc["perm"])
        self.data2 = torch.from_numpy(datacc["Q"])
        self.data3 = torch.from_numpy(datacc["Qw"])
        self.data4 = torch.from_numpy(datacc["Phi"])
        self.data5 = torch.from_numpy(datacc["Time"])
        self.data6 = torch.from_numpy(datacc["Pini"])
        self.data7 = torch.from_numpy(datacc["Swini"])

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        x3 = self.data3[index]
        x4 = self.data4[index]
        x5 = self.data5[index]
        x6 = self.data6[index]
        x7 = self.data7[index]

        cuda = 0
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

        return {
            "perm": x1.to(device, torch.float32),
            "Q": x2.to(device, torch.float32),
            "Qw": x3.to(device, torch.float32),
            "Phi": x4.to(device, torch.float32),
            "Time": x5.to(device, torch.float32),
            "Pini": x6.to(device, torch.float32),
            "Swini": x7.to(device, torch.float32),
        }

    def __len__(self):
        return len(self.data1)


def MyLossClement(a, b):
    loss = torch.sum(torch.abs(a - b) / a.shape[0])

    # loss = ((a-b)**2).mean()
    return loss


@physicsnemo.sym.main(config_path="conf", config_name="config_PINO_original")
def run(cfg: PhysicsNeMoConfig) -> None:
    print("")
    print("------------------------------------------------------------------")
    print("")
    print("\n")
    print("|-----------------------------------------------------------------|")
    print("|                 TRAIN THE MODEL USING A 2D PINO APPROACH:       |")
    print("|-----------------------------------------------------------------|")
    print("")
    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    default = None
    while True:
        default = int(
            input("Select 1 = use default values | 2 = Use user defined values \n")
        )
        if (default > 2) or (default < 1):
            # raise SyntaxError('please select value between 1-2')
            print("")
            print("please try again and select value between 1-2")
        else:
            break

    if not os.path.exists(to_absolute_path("../PACKETS")):
        os.makedirs(to_absolute_path("../PACKETS"))
    else:
        pass

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
            1 = GMRES\n\
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

    batch_size = cfg.custom.batch_size  #'size of simulated labelled data to run'
    timmee = 100.0  # float(input ('Enter the time step interval duration for simulation (days): '))
    max_t = (
        3000.0  # float(input ('Enter the maximum time in days for simulation(days): '))
    )
    MAXZ = 6000  # reference maximum time in days of simulation
    steppi = int(max_t / timmee)
    choice = 1  #  1= Non-Gaussian prior, 2 = Gaussian prior
    factorr = 0.1  # from [0 1] excluding the limits for PermZ
    LIR = 200  # lower injection rate
    UIR = 2000  # uppwer injection rate
    input_channel = 7  # [Perm, Q,QW,Phi,dt, initial_pressure, initial_water_sat]

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
        step2 = int(5)

    # tc2 = Equivalent_time(timmee,2100,timmee,max_t)
    tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
    dt = np.diff(tc2)[0]  # Time-step
    # 4 injector and 4 producer wells
    wells = np.array(
        [
            1,
            24,
            1,
            3,
            3,
            1,
            31,
            1,
            1,
            31,
            31,
            1,
            7,
            9,
            2,
            14,
            12,
            2,
            28,
            19,
            2,
            14,
            27,
            2,
        ]
    )
    wells = np.reshape(wells, (-1, 3), "C")

    # Get prior density of permeability and porosity field

    if not os.path.exists(to_absolute_path("../Training_Images")):
        os.makedirs(to_absolute_path("../Training_Images"))
    else:
        pass

    if default == 1:
        use_pretrained = 1
    else:
        use_pretrained = 2

    if use_pretrained == 1:
        print("Use already generated ensemble from Google drive folder")
        if choice == 1:
            bb = os.path.isfile(to_absolute_path("../PACKETS/Ganensemble.mat"))
            if not bb:
                print("Get initial geology from saved Multiple-point-statistics run")

                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1w81M5M2S0PD9CF2761dxmiKQ5c0OFPaH",
                    to_absolute_path("../PACKETS/Ganensemble.mat"),
                )
                print("...Downlaod completed.......")

                filename = to_absolute_path(
                    "../PACKETS/Ganensemble.mat"
                )  # Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                scaler1a = MinMaxScaler(feature_range=(aay, bby))
                (scaler1a.fit(ini_ensemblef))
                ini_ensemblef = scaler1a.transform(ini_ensemblef)
            else:
                filename = to_absolute_path(
                    "../PACKETS/Ganensemble.mat"
                )  # Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                scaler1a = MinMaxScaler(feature_range=(aay, bby))
                (scaler1a.fit(ini_ensemblef))
                ini_ensemblef = scaler1a.transform(ini_ensemblef)
        else:
            bb = os.path.isfile(to_absolute_path("../PACKETS/Ganensemble_gauss.mat"))
            if not bb:
                print("Get initial geology from saved Two - point-statistics run")
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1Kbe3F7XRzubmDU2bPkfHo2bEmjBNIZ29",
                    to_absolute_path("../PACKETS/Ganensemble_gauss.mat"),
                )
                print("...Downlaod completed.......")

                filename = to_absolute_path(
                    "../PACKETS/Ganensemble_gauss.mat"
                )  # Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                scaler1a = MinMaxScaler(feature_range=(aay, bby))
                (scaler1a.fit(ini_ensemblef))
                ini_ensemblef = scaler1a.transform(ini_ensemblef)
            else:
                filename = to_absolute_path(
                    "../PACKETS/Ganensemble_gauss.mat"
                )  # Ensemble generated offline
                mat = sio.loadmat(filename)
                ini_ensemblef = mat["Z"]
                scaler1a = MinMaxScaler(feature_range=(aay, bby))
                (scaler1a.fit(ini_ensemblef))
                ini_ensemblef = scaler1a.transform(ini_ensemblef)
    else:
        print("Generated prior ensemble from training images")
        if choice == 1:
            print("Get initial geology from scratch using Multiple-point-statistics")
            TII = 3  # Training image
            if TII == 3:
                print("TI = 3")

                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1bib2JAZfBpW4bKz5LdCmhAOtxQixkTzj",
                    to_absolute_path("../Training_Images/iglesias2.out"),
                )
                print("...Downlaod completed.......")

                kq = np.genfromtxt(
                    to_absolute_path("../Training_Images/iglesias2.out"),
                    skip_header=0,
                    dtype="float",
                )
                kq = kq.reshape(-1, 1)
                clfy = MinMaxScaler(feature_range=(Low_K, High_K))
                (clfy.fit(kq))
                kq = np.reshape(kq, (nx, ny), "F")
                kjennq = kq
                permx = kjennq

            else:
                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1TrAVvB-XXCzwHqDdCR4BJnmoe8nPsWIF",
                    to_absolute_path("../Training_Images/TI_2.out"),
                )
                print("...Downlaod completed.......")

                kq = np.genfromtxt(
                    to_absolute_path("../Training_Images/TI_2.out"),
                    skip_header=3,
                    dtype="float",
                )
                kq = kq.reshape(-1, 1)
                clfy = MinMaxScaler(feature_range=(Low_K, High_K))
                (clfy.fit(kq))
                kq = clfy.transform(kq)
                kq = np.reshape(kq, (250, 250), "F")
                kjennq = kq.T
                permx = kjennq

            N_tot = batch_size
            see = intial_ensemble(nx, ny, nz, N_tot, permx)

            ini_ensemblef = see
            sio.savemat(
                to_absolute_path("../PACKETS/Ganensemble.mat"), {"Z": ini_ensemblef}
            )
        else:
            print("Get initial geology from scratch using Two-point-statistics run")
            ini_ensemblef = initial_ensemble_gaussian(nx, ny, nz, batch_size, aay, bby)

            scaler1a = MinMaxScaler(feature_range=(aay, bby))
            (scaler1a.fit(ini_ensemblef))
            ini_ensemblef = scaler1a.transform(ini_ensemblef)
            sio.savemat(
                to_absolute_path("../PACKETS/Ganensemble_gauss.mat"),
                {"Z": ini_ensembleee},
            )

    X_train, X_test = train_test_split(
        ini_ensemblef.T, train_size=batch_size + 1, random_state=42
    )
    X_train = X_train.T
    X_test = X_test.T

    index = batch_size
    imp = batch_size

    print("Get training data from scratch")
    index = np.random.choice(X_train.shape[1], imp, replace=False)
    X_data1 = No_Sim(
        X_train[:, index],
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
        1,
    )

    sio.savemat(to_absolute_path("../PACKETS/Training4nosim.mat"), {"INPUT": X_data1})

    cPerm = np.zeros((X_data1.shape[0], 1, nx, ny))  # Permeability
    cQ = np.zeros((X_data1.shape[0], 1, nx, ny))  # Overall source/sink term
    cQw = np.zeros((X_data1.shape[0], 1, nx, ny))  # Sink term
    cPhi = np.zeros((X_data1.shape[0], 1, nx, ny))  # Porosity
    cTime = np.zeros((X_data1.shape[0], 1, nx, ny))  # Time index
    cPini = np.zeros((X_data1.shape[0], 1, nx, ny))  # Initial pressure
    cSini = np.zeros((X_data1.shape[0], 1, nx, ny))  # Initial water saturation

    for kk in range(X_data1.shape[0]):
        perm = X_data1[kk, 0, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPerm[kk, :, :, :] = permin

        perm = X_data1[kk, 1, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQ[kk, :, :, :] = permin

        perm = X_data1[kk, 2, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQw[kk, :, :, :] = permin

        perm = X_data1[kk, 3, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPhi[kk, :, :, :] = permin

        perm = X_data1[kk, 4, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cTime[kk, :, :, :] = permin

        perm = X_data1[kk, 5, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPini[kk, :, :, :] = permin

        perm = X_data1[kk, 6, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cSini[kk, :, :, :] = permin

    sio.savemat(
        to_absolute_path("../PACKETS/simulationsnosim.mat"),
        {
            "perm": cPerm,
            "Q": cQ,
            "Qw": cQw,
            "Phi": cPhi,
            "Time": cTime,
            "Pini": cPini,
            "Swini": cSini,
        },
    )

    preprocess_FNO_mat(to_absolute_path("../PACKETS/simulationsnosim.mat"))

    index = cfg.custom.batch_size2
    imp = cfg.custom.batch_size2

    bb = os.path.isfile(to_absolute_path("../PACKETS/Training4.mat"))
    if not bb:
        if use_pretrained == 1:
            print("....Downloading Please hold.........")
            download_file_from_google_drive(
                "1I-27_S53ORRFB_hIN_41r3Ntc6PpOE40",
                to_absolute_path("../PACKETS/Training4.mat"),
            )
            print("...Downlaod completed.......")

            print("Load simulated labelled training data from MAT file")
            matt = sio.loadmat(to_absolute_path("../PACKETS/Training4.mat"))
            X_data1 = matt["INPUT"]
            data_use1 = matt["OUTPUT"]
        else:
            print("Get simulated training data from scratch")
            index = np.random.choice(X_train.shape[1], imp, replace=False)
            X_data1, data_use1 = Get_actual_few(
                X_train[:, index],
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
                1,
            )

            sio.savemat(
                to_absolute_path("../PACKETS/Training4.mat"),
                {"INPUT": X_data1, "OUTPUT": data_use1},
            )
    else:
        print("Load simulated labelled training data from MAT file")
        matt = sio.loadmat(to_absolute_path("../PACKETS/Training4.mat"))
        X_data1 = matt["INPUT"]
        data_use1 = matt["OUTPUT"]

    # index = np.random.choice(X_train.shape[1], imp, \
    #                           replace=False)
    # X_data1, data_use1 = Get_actual_few(X_train[:,index],nx,ny,nz,max_t,\
    # DX,DY,DZ,BO,BW,CFL,timmee,MAXZ,factorr,steppi,LIR,UIR,LUB,HUB,aay,bby\
    # ,mpor,hpor,dt,IWSw,PB,PATM,CFO,method,SWI,SWR,UW,UO,typee,step2,pini_alt,\
    #     input_channel,2)

    # sio.savemat(to_absolute_path("../PACKETS/Training4.mat"),{'INPUT': X_data1,'OUTPUT': data_use1})

    cPerm = np.zeros((X_data1.shape[0], 1, nx, ny))  # Permeability
    cQ = np.zeros((X_data1.shape[0], 1, nx, ny))  # Overall source/sink term
    cQw = np.zeros((X_data1.shape[0], 1, nx, ny))  # Sink term
    cPhi = np.zeros((X_data1.shape[0], 1, nx, ny))  # Porosity
    cTime = np.zeros((X_data1.shape[0], 1, nx, ny))  # Time index
    cPini = np.zeros((X_data1.shape[0], 1, nx, ny))  # Initial pressure
    cSini = np.zeros((X_data1.shape[0], 1, nx, ny))  # Initial water saturation

    cPress = np.zeros((X_data1.shape[0], steppi, nx, ny))  # Pressure
    cSat = np.zeros((X_data1.shape[0], steppi, nx, ny))  # Water saturation

    for kk in range(X_data1.shape[0]):
        perm = X_data1[kk, 0, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPerm[kk, :, :, :] = permin

        perm = X_data1[kk, 1, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQ[kk, :, :, :] = permin

        perm = X_data1[kk, 2, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQw[kk, :, :, :] = permin

        perm = X_data1[kk, 3, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPhi[kk, :, :, :] = permin

        perm = X_data1[kk, 4, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cTime[kk, :, :, :] = permin

        perm = X_data1[kk, 5, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPini[kk, :, :, :] = permin

        perm = X_data1[kk, 6, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cSini[kk, :, :, :] = permin

        perm = data_use1[kk, :steppi, :, :]
        cPress[kk, :, :, :] = perm

        perm = data_use1[kk, steppi:, :, :]
        cSat[kk, :, :, :] = perm

    sio.savemat(
        to_absolute_path("../PACKETS/simulations.mat"),
        {
            "perm": cPerm,
            "Q": cQ,
            "Qw": cQw,
            "Phi": cPhi,
            "Time": cTime,
            "Pini": cPini,
            "Swini": cSini,
            "pressure": cPress,
            "water_sat": cSat,
        },
    )

    preprocess_FNO_mat(to_absolute_path("../PACKETS/simulations.mat"))

    bb = os.path.isfile(to_absolute_path("../PACKETS/iglesias2.out"))
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1_9VRt8tEOF6IV7GvUnD7CFVM40DMHkxn",
            to_absolute_path("../PACKETS/iglesias2.out"),
        )
        print("...Downlaod completed.......")
    else:
        pass

    Truee = np.genfromtxt(to_absolute_path("../PACKETS/iglesias2.out"), dtype="float")
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
    print("Finished FVM simulation")

    ini_ensemble1 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble2 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble3 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble4 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble5 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble6 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble7 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)

    ini_ensemble8 = np.zeros((Ne, steppi, nx, ny))  # Pressure
    ini_ensemble9 = np.zeros((Ne, steppi, nx, ny))  # Water saturation

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

        perm = data_use2[kk, :steppi, :, :]
        ini_ensemble8[kk, :, :, :] = perm

        perm = data_use2[kk, steppi:, :, :]
        ini_ensemble9[kk, :, :, :] = perm

    sio.savemat(
        to_absolute_path("../PACKETS/single.mat"),
        {
            "perm": ini_ensemble1,
            "Q": ini_ensemble2,
            "Qw": ini_ensemble3,
            "Phi": ini_ensemble4,
            "Time": ini_ensemble5,
            "Pini": ini_ensemble6,
            "Swini": ini_ensemble7,
            "pressure": ini_ensemble8,
            "water_sat": ini_ensemble9,
        },
    )

    preprocess_FNO_mat(to_absolute_path("../PACKETS/single.mat"))

    index = np.random.choice(X_train.shape[1], 1, replace=False)

    zaa = X_train[:, index]
    Ne = 1
    ini = []
    inip = []
    inij = []
    injj = 500 * np.ones((1, 4))
    for i in range(Ne):
        at1 = rescale_linear(zaa, aay, bby)
        at2 = rescale_linear(zaa, mpor, hpor)
        ini.append(at1.reshape(-1, 1))
        inip.append(at2.reshape(-1, 1))
        inij.append(injj)
    ini = np.hstack(ini)
    inip = np.hstack(inip)
    kka = np.vstack(inij)

    X_data_test, data_use_test = inference_single(
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
    print("Finished FVM simulation")

    ini_ensemble1 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble2 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble3 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble4 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble5 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble6 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)
    ini_ensemble7 = np.zeros((Ne, 1, nx, ny), dtype=np.float32)

    ini_ensemble8 = np.zeros((Ne, steppi, nx, ny))  # Pressure
    ini_ensemble9 = np.zeros((Ne, steppi, nx, ny))  # Water saturation

    for kk in range(X_data_test.shape[0]):
        perm = X_data_test[kk, 0, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble1[kk, :, :, :] = permin

        perm = X_data_test[kk, 1, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble2[kk, :, :, :] = permin

        perm = X_data_test[kk, 2, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble3[kk, :, :, :] = permin

        perm = X_data_test[kk, 3, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble4[kk, :, :, :] = permin

        perm = X_data_test[kk, 4, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble5[kk, :, :, :] = permin

        perm = X_data_test[kk, 5, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble6[kk, :, :, :] = permin

        perm = X_data_test[kk, 6, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        ini_ensemble7[kk, :, :, :] = permin

        perm = data_use_test[kk, :steppi, :, :]
        ini_ensemble8[kk, :, :, :] = perm

        perm = data_use_test[kk, steppi:, :, :]
        ini_ensemble9[kk, :, :, :] = perm

    sio.savemat(
        to_absolute_path("../PACKETS/singletest.mat"),
        {
            "perm": ini_ensemble1,
            "Q": ini_ensemble2,
            "Qw": ini_ensemble3,
            "Phi": ini_ensemble4,
            "Time": ini_ensemble5,
            "Pini": ini_ensemble6,
            "Swini": ini_ensemble7,
            "pressure": ini_ensemble8,
            "water_sat": ini_ensemble9,
        },
    )

    preprocess_FNO_mat(to_absolute_path("../PACKETS/singletest.mat"))

    # [init-node]
    cuda = 0
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    # Define FNO model for forward model (pressure)
    decoder1 = ConvFullyConnectedArch(
        [Key("z", size=32)],
        [Key("pressure", size=steppi), Key("Tx", size=steppi), Key("Ty", size=steppi)],
    )
    fno_pressure = FNOArch(
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

    fno_pressure = fno_pressure.to(device)

    # Define FNO model for forward model (saturation)
    decoder2 = ConvFullyConnectedArch(
        [Key("z", size=32)],
        [
            Key("water_sat", size=steppi),
            Key("Txw", size=steppi),
            Key("Tyw", size=steppi),
        ],
    )
    fno_saturation = FNOArch(
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
    fno_saturation = fno_saturation.to(device)

    learning_rate = cfg.optimizer.lr
    gamma = 0.5
    step_size = 100

    optimizer_pressure = torch.optim.Adam(
        fno_pressure.parameters(),
        lr=learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler_pressure = torch.optim.lr_scheduler.StepLR(
        optimizer_pressure, step_size=step_size, gamma=gamma
    )

    optimizer_sat = torch.optim.Adam(
        fno_saturation.parameters(),
        lr=learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler_sat = torch.optim.lr_scheduler.StepLR(
        optimizer_sat, step_size=step_size, gamma=gamma
    )

    ##############################################################################
    #         START THE TRAINING OF THE MODEL WITH UNLABELLED DATA - OPERATOR LEARNING
    ##############################################################################
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>OPERATOR LEARNING>>>>>>>>>>>>>>>>>>>>>>>>")
    hista = []
    hist2a = []
    hist22a = []
    aha = 0
    ah2a = 0
    ah22a = 0
    costa = []
    cost2a = []
    cost22a = []
    epochs = cfg.custom.epochs  #'number of epochs to train'

    myloss = LpLoss(size_average=True)
    datacc = sio.loadmat(to_absolute_path("../PACKETS/simulationsnosim.mat"))
    dataset = unLabelledset(datacc)
    unlabelled_loader = DataLoader(dataset, batch_size=cfg.batch_size.unlabelled)

    test = sio.loadmat(to_absolute_path("../PACKETS/singletest.mat"))
    inn_test_new = {
        "perm": torch.from_numpy(test["perm"]).to(device, torch.float32),
        "Q": torch.from_numpy(test["Q"]).to(device, dtype=torch.float32),
        "Qw": torch.from_numpy(test["Qw"]).to(device, dtype=torch.float32),
        "Phi": torch.from_numpy(test["Phi"]).to(device, dtype=torch.float32),
        "Time": torch.from_numpy(test["Time"]).to(device, dtype=torch.float32),
        "Pini": torch.from_numpy(test["Pini"]).to(device, dtype=torch.float32),
        "Swini": torch.from_numpy(test["Swini"]).to(device, dtype=torch.float32),
    }

    out_test_new = {
        "pressure": torch.from_numpy(test["pressure"]).to(device, torch.float32),
        "water_sat": torch.from_numpy(test["water_sat"]).to(
            device, dtype=torch.float32
        ),
    }

    test = sio.loadmat(to_absolute_path("../PACKETS/single.mat"))
    inn_test_new1 = {
        "perm": torch.from_numpy(test["perm"]).to(device, torch.float32),
        "Q": torch.from_numpy(test["Q"]).to(device, dtype=torch.float32),
        "Qw": torch.from_numpy(test["Qw"]).to(device, dtype=torch.float32),
        "Phi": torch.from_numpy(test["Phi"]).to(device, dtype=torch.float32),
        "Time": torch.from_numpy(test["Time"]).to(device, dtype=torch.float32),
        "Pini": torch.from_numpy(test["Pini"]).to(device, dtype=torch.float32),
        "Swini": torch.from_numpy(test["Swini"]).to(device, dtype=torch.float32),
    }

    out_test_new1 = {
        "pressure": torch.from_numpy(test["pressure"]).to(device, torch.float32),
        "water_sat": torch.from_numpy(test["water_sat"]).to(
            device, dtype=torch.float32
        ),
    }

    print(" Training with " + str(batch_size) + " unlabelled members ")
    start_epoch = 1
    start_time = time.time()
    imtest_use = 1
    for epoch in range(start_epoch, epochs + 1):
        fno_pressure.train()
        fno_saturation.train()
        loss_train = 0.0
        print("Epoch " + str(epoch) + " | " + str(epochs))
        print("****************************************************************")
        for inputaa in unlabelled_loader:
            optimizer_pressure.zero_grad()
            optimizer_sat.zero_grad()

            # with torch.no_grad():
            outputa_p = fno_pressure(inputaa)["pressure"]
            outputa_s = fno_saturation(inputaa)["water_sat"]

            target2_p = fno_pressure(inn_test_new)["pressure"]
            target2_s = fno_saturation(inn_test_new)["water_sat"]

            dtrue_test2_p = out_test_new["pressure"]
            dtrue_test2_s = out_test_new["water_sat"]

            loss_test1a = MyLossClement(
                (target2_p).reshape(imtest_use, -1),
                (dtrue_test2_p).reshape(imtest_use, -1),
            )

            loss_test2a = MyLossClement(
                (target2_s).reshape(imtest_use, -1),
                (dtrue_test2_s).reshape(imtest_use, -1),
            )

            loss_testa = loss_test1a + loss_test2a

            # Field Case
            # with torch.no_grad():
            t2_p = fno_pressure(inn_test_new1)["pressure"]
            t2_s = fno_saturation(inn_test_new1)["water_sat"]

            y_p = out_test_new1["pressure"]
            y_s = out_test_new1["water_sat"]

            loss_t1a = MyLossClement(
                (t2_p).reshape(imtest_use, -1), (y_p).reshape(imtest_use, -1)
            )

            loss_t2a = MyLossClement(
                (t2_s).reshape(imtest_use, -1), (y_s).reshape(imtest_use, -1)
            )

            loss_t = loss_t1a + loss_t2a

            input_var = inputaa
            input_var["pressure"] = outputa_p
            input_var["water_sat"] = outputa_s
            input_var["Tx"] = fno_pressure(inputaa)["Tx"]
            input_var["Ty"] = fno_pressure(inputaa)["Ty"]
            input_var["Txw"] = fno_saturation(inputaa)["Txw"]
            input_var["Tyw"] = fno_saturation(inputaa)["Tyw"]

            # print(outputa_p.shape)
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

            loss_pde2 = f_loss2 + f_water2  # + (loss_ini )

            loss2 = (loss_pde2 * cfg.custom.pde_weighting) + (
                loss_testa + loss_t
            ) * cfg.custom.data_weighting * 1e1

            # loss2 = (f_loss2*cfg.custom.pde_weighting *1e-1) + \
            # (f_water2*cfg.custom.pde_weighting * 1e-1) + \
            # ((loss_test1a + loss_t1a)*cfg.custom.data_weighting *1e3) + \
            # (loss_test2a + loss_t2a)*cfg.custom.data_weighting*1e3

            model_pressure = fno_pressure
            model_saturation = fno_saturation
            loss2.backward()

            optimizer_pressure.step()
            optimizer_sat.step()

            loss_train += loss2.item()

        if (epoch % cfg.training.rec_results_freq) == 0:
            print("-----------------------Plot Results------------------------")
            CustomValidatorPlotter(
                timmee,
                max_t,
                MAXZ,
                pini_alt,
                nx,
                ny,
                wells,
                steppi,
                tc2,
                dt,
                "dynamics_simulations",
                dtrue_test2_s,
                target2_s,
                dtrue_test2_p,
                target2_p,
            )
            import glob

            frames = []
            imgs = sorted(glob.glob("*dynamics_simulations*"), key=os.path.getmtime)
            for i in imgs:
                new_frame = Image.open(i)
                frames.append(new_frame)

            frames[0].save(
                "pde_simulation_test.gif",
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=500,
                loop=0,
            )
            from glob import glob

            for f3 in glob("*dynamics_simulations*"):
                os.remove(f3)

            CustomValidatorPlotter(
                timmee,
                max_t,
                MAXZ,
                pini_alt,
                nx,
                ny,
                wells,
                steppi,
                tc2,
                dt,
                "dynamics_simulations",
                y_s,
                t2_s,
                y_p,
                t2_p,
            )
            import glob

            frames = []
            imgs = sorted(glob.glob("*dynamics_simulations*"), key=os.path.getmtime)
            for i in imgs:
                new_frame = Image.open(i)
                frames.append(new_frame)

            frames[0].save(
                "pde_simulation_field.gif",
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=500,
                loop=0,
            )
            from glob import glob

            for f3 in glob("*dynamics_simulations*"):
                os.remove(f3)
        else:
            pass

        scheduler_pressure.step()
        scheduler_sat.step()
        ahnewa = loss2.detach().cpu().numpy()
        hista.append(ahnewa)

        ahnew2a = loss_testa.detach().cpu().numpy()
        hist2a.append(ahnew2a)

        ahnew22a = loss_t.detach().cpu().numpy()
        hist22a.append(ahnew22a)

        print("TRAINING")
        if aha < ahnewa:
            print(
                "   FORWARD PROBLEM COMMENT : Loss increased by "
                + str(abs(aha - ahnewa))
            )

        elif aha > ahnewa:
            print(
                "   FORWARD PROBLEM COMMENT : Loss decreased by "
                + str(abs(aha - ahnewa))
            )

        else:
            print("   FORWARD PROBLEM COMMENT : No change in Loss ")

        print("   training loss = " + str(ahnewa))
        print("   pde loss = " + str(loss_pde2.detach().cpu().numpy()))
        print("   pressure equation loss = " + str(f_loss2.detach().cpu().numpy()))
        print("   saturation equation loss = " + str(f_water2.detach().cpu().numpy()))
        print("    ******************************   ")
        if ah2a < ahnew2a:
            print("   TEST COMMENT : Loss increased by " + str(abs(ah2a - ahnew2a)))

        elif ah2a > ahnew2a:
            print("   TEST COMMENT : Loss decreased by " + str(abs(ah2a - ahnew2a)))

        else:
            print("   TEST COMMENT : No change in Loss ")
        print("   Test loss = " + str(ahnew2a))
        print("   Test: Pressure loss = " + str(loss_test1a.detach().cpu().numpy()))
        print("   Test: saturation loss = " + str(loss_test2a.detach().cpu().numpy()))

        print("    ******************************   ")
        if ah22a < ahnew22a:
            print(
                "   TEST COMMENT FIELD : Loss increased by "
                + str(abs(ah22a - ahnew22a))
            )
        elif ah22a > ahnew22a:
            print(
                "   TEST COMMENT FIELD : Loss decreased by "
                + str(abs(ah22a - ahnew22a))
            )
        else:
            print("   TEST COMMENT FIELD : No change in Loss ")
        print("   Test loss Field = " + str(ahnew22a))
        print("   Test Field: Pressure loss = " + str(loss_t1a.detach().cpu().numpy()))
        print(
            "   Test Field: saturation loss = " + str(loss_t2a.detach().cpu().numpy())
        )

        aha = ahnewa
        ah2a = ahnew2a
        ah22a = ahnew22a

        if epoch == 1:
            best_cost = aha
        else:
            pass
        if best_cost > ahnewa:
            print("    ******************************   ")
            print("   Forward models saved")
            print("   Current best cost = " + str(best_cost))
            print("   Current epoch cost = " + str(ahnewa))
            torch.save(model_pressure.state_dict(), oldfolder + "/pressure_model.pth")
            torch.save(
                model_saturation.state_dict(), oldfolder + "/saturation_model.pth"
            )
            best_cost = ahnewa
        else:
            print("    ******************************   ")
            print("   Forward models NOT saved")
            print("   Current best cost = " + str(best_cost))
            print("   Current epoch cost = " + str(ahnewa))

        costa.append(ahnewa)
        cost2a.append(ahnew2a)
        cost22a.append(ahnew22a)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>Finished training >>>>>>>>>>>>>>>>> ")
    elapsed_time_secs = time.time() - start_time
    msg = "PDE learning Execution took: %s secs (Wall clock time)" % timedelta(
        seconds=round(elapsed_time_secs)
    )
    print(msg)

    # Define FNO model for forward model (pressure)
    decoder1 = ConvFullyConnectedArch(
        [Key("z", size=32)],
        [Key("pressure", size=steppi), Key("Tx", size=steppi), Key("Ty", size=steppi)],
    )
    fno_pressure = FNOArch(
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

    fno_pressure.load_state_dict(torch.load(oldfolder + "/pressure_model.pth"))
    fno_pressure = fno_pressure.to(device)

    # Define FNO model for forward model (saturation)
    decoder2 = ConvFullyConnectedArch(
        [Key("z", size=32)],
        [
            Key("water_sat", size=steppi),
            Key("Txw", size=steppi),
            Key("Tyw", size=steppi),
        ],
    )
    fno_saturation = FNOArch(
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

    fno_saturation.load_state_dict(torch.load(oldfolder + "/saturation_model.pth"))
    fno_saturation = fno_saturation.to(device)

    learning_rate = cfg.optimizer.lr
    gamma = 0.5
    step_size = 100

    optimizer_pressure = torch.optim.Adam(
        fno_pressure.parameters(),
        lr=learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler_pressure = torch.optim.lr_scheduler.StepLR(
        optimizer_pressure, step_size=step_size, gamma=gamma
    )

    optimizer_sat = torch.optim.Adam(
        fno_saturation.parameters(),
        lr=learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler_sat = torch.optim.lr_scheduler.StepLR(
        optimizer_sat, step_size=step_size, gamma=gamma
    )

    ##############################################################################
    #         START THE TRAINING OF THE MODEL - OPERATOR LEARNING
    ##############################################################################
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>OPERATOR LEARNING>>>>>>>>>>>>>>>>>>>>>>>>")
    hist = []
    hist2 = []
    hist22 = []
    ah = 0
    ah2 = 0
    ah22 = 0

    cost = []
    cost2 = []
    cost22 = []

    epochs = cfg.custom.epochs2  #'number of epochs to train'

    datacc = sio.loadmat(to_absolute_path("../PACKETS/simulations.mat"))
    dataset = Labelledset(datacc)
    labelled_loader = DataLoader(dataset, batch_size=cfg.batch_size.grid)

    print("")
    print(" Training with " + str(cfg.custom.batch_size2) + " labelled members ")
    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        fno_pressure.train()
        fno_saturation.train()
        # loss_train, mse = 0., 0.
        print("Epoch " + str(epoch) + " | " + str(epochs))
        print("****************************************************************")
        for inputa in labelled_loader:
            # return {'perm':x1,'Q': x2,'Qw': x3,'Phi': x4,'Time': x5,'Pini': x6,'Swini': x7\
            #     ,'pressure': x8,'water_sat': x9}
            inputin = {
                "perm": inputa["perm"],
                "Q": inputa["Q"],
                "Qw": inputa["Qw"],
                "Phi": inputa["Phi"],
                "Time": inputa["Time"],
                "Pini": inputa["Pini"],
                "Swini": inputa["Swini"],
            }

            target = {"pressure": inputa["pressure"], "water_sat": inputa["water_sat"]}

            optimizer_pressure.zero_grad()
            optimizer_sat.zero_grad()

            # with torch.no_grad():
            output_p = fno_pressure(inputin)["pressure"]
            output_s = fno_saturation(inputin)["water_sat"]

            loss_data1 = MyLossClement(
                (output_p).reshape(cfg.batch_size.grid, -1),
                (target["pressure"]).reshape(cfg.batch_size.grid, -1),
            )

            loss_data2 = MyLossClement(
                (output_s).reshape(cfg.batch_size.grid, -1),
                (target["water_sat"]).reshape(cfg.batch_size.grid, -1),
            )

            loss_data = loss_data1 + loss_data2

            input_temp = inputin
            input_temp["pressure"] = target["pressure"]
            input_temp["water_sat"] = target["water_sat"]
            input_temp["Tx"] = fno_pressure(inputin)["Tx"]
            input_temp["Ty"] = fno_pressure(inputin)["Ty"]
            input_temp["Txw"] = fno_saturation(inputin)["Txw"]
            input_temp["Tyw"] = fno_saturation(inputin)["Tyw"]

            input_temp2 = inputin
            input_temp2["pressure"] = output_p
            input_temp2["water_sat"] = output_s
            input_temp2["Tx"] = fno_pressure(inputin)["Tx"]
            input_temp2["Ty"] = fno_pressure(inputin)["Ty"]
            input_temp2["Txw"] = fno_saturation(inputin)["Txw"]
            input_temp2["Tyw"] = fno_saturation(inputin)["Tyw"]

            f_loss2a1, f_water2a1 = Black_oil(
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
                input_temp,
                device,
                myloss,
            )

            f_loss2b1, f_water2b1 = Black_oil(
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
                input_temp2,
                device,
                myloss,
            )

            f_loss2 = f_loss2a1 + f_loss2b1
            f_water2 = f_water2a1 + f_water2b1

            loss_pde3 = f_loss2 + f_water2  # + (loss_ini )

            # Test data
            # with torch.no_grad():
            target2_p = fno_pressure(inn_test_new)["pressure"]
            target2_s = fno_saturation(inn_test_new)["water_sat"]

            dtrue_test2_p = out_test_new["pressure"]
            dtrue_test2_s = out_test_new["water_sat"]

            loss_test1 = MyLossClement(
                (target2_p).reshape(1, -1), (dtrue_test2_p).reshape(1, -1)
            )

            loss_test2 = MyLossClement(
                (target2_s).reshape(1, -1), (dtrue_test2_s).reshape(1, -1)
            )

            loss_test = loss_test1 + loss_test2

            # Field Case
            # with torch.no_grad():
            t2_p = fno_pressure(inn_test_new1)["pressure"]
            t2_s = fno_saturation(inn_test_new1)["water_sat"]

            y_p = out_test_new1["pressure"]
            y_s = out_test_new1["water_sat"]

            loss_t1a = MyLossClement((t2_p).reshape(1, -1), (y_p).reshape(1, -1))

            loss_t2a = MyLossClement((t2_s).reshape(1, -1), (y_s).reshape(1, -1))

            loss_t = loss_t1a + loss_t2a

            loss = (
                (loss_data1 * cfg.custom.dataw * 1e3)
                + (loss_test1 * 1e1) * cfg.custom.testw
                + (loss_t1a * 1e1) * cfg.custom.testw
                + (f_loss2 * 1e1) * cfg.custom.pdew
                + (loss_data2 * cfg.custom.dataw)
                + (loss_test2) * cfg.custom.testw
                + (loss_t2a) * cfg.custom.testw
                + (f_water2) * cfg.custom.pdew
            )

            model_pressure = fno_pressure
            model_saturation = fno_saturation
            loss.backward()

            optimizer_pressure.step()
            optimizer_sat.step()

        loss_train += loss.item()

        if (epoch % cfg.training.rec_results_freq) == 0:
            print("Plot Results")
            CustomValidatorPlotter(
                timmee,
                max_t,
                MAXZ,
                pini_alt,
                nx,
                ny,
                wells,
                steppi,
                tc2,
                dt,
                "dynamics_simulations",
                dtrue_test2_s,
                target2_s,
                dtrue_test2_p,
                target2_p,
            )
            import glob

            frames = []
            imgs = sorted(glob.glob("*dynamics_simulations*"), key=os.path.getmtime)
            for i in imgs:
                new_frame = Image.open(i)
                frames.append(new_frame)

            frames[0].save(
                "operator_simulation_test.gif",
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=500,
                loop=0,
            )
            from glob import glob

            for f3 in glob("*dynamics_simulations*"):
                os.remove(f3)

            CustomValidatorPlotter(
                timmee,
                max_t,
                MAXZ,
                pini_alt,
                nx,
                ny,
                wells,
                steppi,
                tc2,
                dt,
                "dynamics_simulations",
                y_s,
                t2_s,
                y_p,
                t2_p,
            )

            import glob

            frames = []
            imgs = sorted(glob.glob("*dynamics_simulations*"), key=os.path.getmtime)
            for i in imgs:
                new_frame = Image.open(i)
                frames.append(new_frame)

            frames[0].save(
                "operator_simulation_field.gif",
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=500,
                loop=0,
            )
            from glob import glob

            for f3 in glob("*dynamics_simulations*"):
                os.remove(f3)
        else:
            pass

        scheduler_pressure.step()
        scheduler_sat.step()
        ahnew = loss.detach().cpu().numpy()
        hist.append(ahnew)

        ahnew2 = loss_test.detach().cpu().numpy()
        hist2.append(ahnew2)

        ahnew22 = loss_t.detach().cpu().numpy()
        hist22.append(ahnew22)

        print("TRAINING")
        if ah < ahnew:
            print(
                "   FORWARD PROBLEM COMMENT : Loss increased by " + str(abs(ah - ahnew))
            )

        elif ah > ahnew:
            print(
                "   FORWARD PROBLEM COMMENT : Loss decreased by " + str(abs(ah - ahnew))
            )

        else:
            print("   FORWARD PROBLEM COMMENT : No change in Loss ")

        print("   training loss = " + str(ahnew))
        print("   Data loss = " + str(loss_data.detach().cpu().numpy()))
        print("   Data loss : Pressure = " + str(loss_data1.detach().cpu().numpy()))
        print("   Data loss : saturation = " + str(loss_data2.detach().cpu().numpy()))
        print("   pde loss = " + str(loss_pde3.detach().cpu().numpy()))
        print(
            "   pde loss : pressure equation = " + str(f_loss2.detach().cpu().numpy())
        )
        print(
            "   pde loss : saturation equation = "
            + str(f_water2.detach().cpu().numpy())
        )
        print("    ******************************   ")
        if ah2 < ahnew2:
            print("    TEST COMMENT : Loss increased by " + str(abs(ah2 - ahnew2)))
        elif ah2 > ahnew2:
            print("    TEST COMMENT : Loss decreased by " + str(abs(ah2 - ahnew2)))
        else:
            print("    TEST COMMENT : No change in Loss ")
        print("   Test loss = " + str(ahnew2))
        print("   Test: Pressure loss = " + str(loss_test1.detach().cpu().numpy()))
        print("   Test: saturation loss = " + str(loss_test2.detach().cpu().numpy()))

        print("    ******************************   ")
        if ah22 < ahnew22:
            print(
                "    TEST COMMENT FIELD : Loss increased by " + str(abs(ah22 - ahnew22))
            )
        elif ah22 > ahnew22:
            print(
                "    TEST COMMENT FIELD : Loss decreased by " + str(abs(ah22 - ahnew22))
            )
        else:
            print("    TEST COMMENT FIELD : No change in Loss ")
        print("   Test loss Field = " + str(ahnew22))
        print("   Test Field: Pressure loss = " + str(loss_t1a.detach().cpu().numpy()))
        print(
            "   Test Field: saturation loss = " + str(loss_t2a.detach().cpu().numpy())
        )

        ah = ahnew
        ah2 = ahnew2
        ah22 = ahnew22

        if epoch == 1:
            best_cost = ah
        else:
            pass

        if best_cost > ahnew:
            print("    ******************************   ")
            print("   Forward models saved")
            print("   Current best cost = " + str(best_cost))
            print("   Current epoch cost = " + str(ahnew))
            torch.save(model_pressure.state_dict(), oldfolder + "/pressure_model.pth")
            torch.save(
                model_saturation.state_dict(), oldfolder + "/saturation_model.pth"
            )
            best_cost = ahnew
        else:
            print("    ******************************   ")
            print("   Forward models NOT saved")
            print("   Current best cost = " + str(best_cost))
            print("   Current epoch cost = " + str(ahnew))
        cost.append(ahnew)
        cost2.append(ahnew2)
        cost22.append(ahnew22)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>Finished training with ADAM optimser>>>>>>>>>>>>>>>>> ")

    elapsed_time_secs = time.time() - start_time
    msg = "Model learning Execution took: %s secs (Wall clock time)" % timedelta(
        seconds=round(elapsed_time_secs)
    )
    print(msg)

    print("")
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 3, 1)
    plt.semilogy(range(len(hist)), hist, "k-")
    plt.title("Forward problem -operator learning", fontsize=13)
    plt.ylabel("$\\phi_{n_{epoch}}$", fontsize=13)
    plt.xlabel("$n_{epoch}$", fontsize=13)

    plt.subplot(3, 3, 2)
    plt.semilogy(range(len(hist2)), hist2, "k-")
    plt.title("Testing -operator learning", fontsize=13)
    plt.ylabel("$\\phi_{n_{epoch}}$", fontsize=13)
    plt.xlabel("$n_{epoch}$", fontsize=13)

    plt.subplot(3, 3, 3)
    plt.semilogy(range(len(hist22)), hist22, "k-")
    plt.title("Testing Field -operator learning", fontsize=13)
    plt.ylabel("$\\phi_{n_{epoch}}$", fontsize=13)
    plt.xlabel("$n_{epoch}$", fontsize=13)

    plt.subplot(3, 3, 4)
    plt.semilogy(range(len(hista)), hista, "k-")
    plt.title("Forward problem -PDE Learning", fontsize=13)
    plt.ylabel("$\\phi_{n_{epoch}}$", fontsize=13)
    plt.xlabel("$n_{epoch}$", fontsize=13)

    plt.subplot(3, 3, 5)
    plt.semilogy(range(len(hist2a)), hist2a, "k-")
    plt.title("Testing -PDE learning", fontsize=13)
    plt.ylabel("$\\phi_{n_{epoch}}$", fontsize=13)
    plt.xlabel("$n_{epoch}$", fontsize=13)

    plt.subplot(3, 3, 6)
    plt.semilogy(range(len(hist22a)), hist22a, "k-")
    plt.title("Testing Field -PDE learning", fontsize=13)
    plt.ylabel("$\\phi_{n_{epoch}}$", fontsize=13)
    plt.xlabel("$n_{epoch}$", fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("MODEL LEARNING", fontsize=16)
    plt.savefig(oldfolder + "/cost_PyTorch.png")
    plt.close()
    plt.clf()


if __name__ == "__main__":
    run()
