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

import numpy as np
import os
import physicsnemo.sym
import torch
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from NVRS import *
from utilities import load_FNO_dataset2, preprocess_FNO_mat
from physicsnemo.sym.models.fno import *
import cupy as cp
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


from physicsnemo.sym.utils.io.plotter import ValidatorPlotter


class CustomValidatorPlotterP(ValidatorPlotter):
    def __init__(
        self,
        timmee,
        max_t,
        MAXZ,
        pini_alt,
        nx,
        ny,
        nz,
        steppi,
        tc2,
        dt,
        injectors,
        producers,
        N_injw,
        N_pr,
    ):
        self.timmee = timmee
        self.max_t = max_t
        self.MAXZ = MAXZ
        self.pini_alt = pini_alt
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.steppi = steppi
        self.tc2 = tc2
        self.dt = dt
        self.injectors = injectors
        self.producers = producers
        self.N_injw = N_injw
        self.N_pr = N_pr

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables

        # get and interpolate output variable
        pressure_true, pressure_pred = true_outvar["pressure"], pred_outvar["pressure"]

        # make plot

        f_big = []
        Time_vector = np.zeros((self.steppi))

        Accuracy_presure = np.zeros((self.steppi, 2))
        for itt in range(self.steppi):
            Time_vector[itt] = int((itt + 1) * self.dt * self.MAXZ)
            look = (pressure_pred[0, itt, :, :]) * self.pini_alt
            lookf = (pressure_true[0, itt, :, :]) * self.pini_alt
            diff1 = abs(look - lookf)

            XX, YY = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
            f_2 = plt.figure(figsize=(10, 10), dpi=100)
            plt.subplot(1, 3, 1)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                look,
                self.N_injw,
                self.N_pr,
                "pressure PhysicsNeMo",
                self.injectors,
                self.producers,
            )
            plt.subplot(1, 3, 2)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                lookf,
                self.N_injw,
                self.N_pr,
                "pressure Numerical",
                self.injectors,
                self.producers,
            )
            plt.subplot(1, 3, 3)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                diff1,
                self.N_injw,
                self.N_pr,
                "pressure diff",
                self.injectors,
                self.producers,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = "Timestep --" + str(int((itt + 1) * self.dt * self.MAXZ)) + " days"

            plt.suptitle(tita, fontsize=16)
            namez = "pressure_simulations" + str(int(itt))
            yes = (f_2, namez)
            f_big.append(yes)
            # plt.clf()
            plt.close()

            R2p, L2p = compute_metrics(look.ravel(), lookf.ravel())
            Accuracy_presure[itt, 0] = R2p
            Accuracy_presure[itt, 1] = L2p

            f_3 = plt.figure(figsize=(20, 20), dpi=200)
            ax1 = f_3.add_subplot(131, projection="3d")
            Plot_PhysicsNeMo(
                ax1,
                self.nx,
                self.ny,
                self.nz,
                look,
                self.N_injw,
                self.N_pr,
                "pressure PhysicsNeMo",
                self.injectors,
                self.producers,
            )
            ax2 = f_3.add_subplot(132, projection="3d")
            Plot_PhysicsNeMo(
                ax2,
                self.nx,
                self.ny,
                self.nz,
                lookf,
                self.N_injw,
                self.N_pr,
                "pressure Numerical",
                self.injectors,
                self.producers,
            )
            ax3 = f_3.add_subplot(133, projection="3d")
            Plot_PhysicsNeMo(
                ax3,
                self.nx,
                self.ny,
                self.nz,
                diff1,
                self.N_injw,
                self.N_pr,
                "pressure diff",
                self.injectors,
                self.producers,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = (
                "3D Map - Timestep --"
                + str(int((itt + 1) * self.dt * self.MAXZ))
                + " days"
            )

            plt.suptitle(tita, fontsize=20, weight="bold")

            namez = "Simulations3Dp" + str(int(itt))
            yes2 = (f_3, namez)
            f_big.append(yes2)
            # plt.clf()
            plt.close()

        fig4 = plt.figure(figsize=(10, 10), dpi=200)

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
            fontsize=16,
        )
        fig4.text(
            0.5,
            0.49,
            "L2(%) Accuracy - PhysicsNeMo/Numerical(GPU)",
            ha="center",
            va="center",
            fontproperties=font,
            fontsize=16,
        )

        # Plot R2 accuracies
        plt.subplot(2, 1, 1)
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

        # Plot L2 accuracies
        plt.subplot(2, 1, 2)
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

        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        namez = "R2L2_pressure"
        yes21 = (fig4, namez)
        f_big.append(yes21)
        # plt.clf()
        plt.close()

        return f_big


class CustomValidatorPlotterS(ValidatorPlotter):
    def __init__(
        self,
        timmee,
        max_t,
        MAXZ,
        pini_alt,
        nx,
        ny,
        nz,
        steppi,
        tc2,
        dt,
        injectors,
        producers,
        N_injw,
        N_pr,
    ):
        self.timmee = timmee
        self.max_t = max_t
        self.MAXZ = MAXZ
        self.pini_alt = pini_alt
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.steppi = steppi
        self.tc2 = tc2
        self.dt = dt
        self.injectors = injectors
        self.producers = producers
        self.N_injw = N_injw
        self.N_pr = N_pr

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables

        water_true, water_pred = true_outvar["water_sat"], pred_outvar["water_sat"]

        # make plot

        f_big = []
        Accuracy_oil = np.zeros((self.steppi, 2))
        Accuracy_water = np.zeros((self.steppi, 2))
        Time_vector = np.zeros((self.steppi))
        for itt in range(self.steppi):
            Time_vector[itt] = int((itt + 1) * self.dt * self.MAXZ)

            XX, YY = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
            f_2 = plt.figure(figsize=(12, 12), dpi=100)

            look_sat = water_pred[0, itt, :, :]  # *1e-2
            look_oil = 1 - look_sat

            lookf_sat = water_true[0, itt, :, :]  # * 1e-2
            lookf_oil = 1 - lookf_sat

            diff1_wat = abs(look_sat - lookf_sat)
            diff1_oil = abs(look_oil - lookf_oil)

            plt.subplot(2, 3, 1)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                look_sat,
                self.N_injw,
                self.N_pr,
                "water PhysicsNeMo",
                self.injectors,
                self.producers,
            )
            plt.subplot(2, 3, 2)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                lookf_sat,
                self.N_injw,
                self.N_pr,
                "water Numerical",
                self.injectors,
                self.producers,
            )
            plt.subplot(2, 3, 3)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                diff1_wat,
                self.N_injw,
                self.N_pr,
                "water diff",
                self.injectors,
                self.producers,
            )
            R2w, L2w = compute_metrics(look_sat.ravel(), lookf_sat.ravel())
            Accuracy_water[itt, 0] = R2w
            Accuracy_water[itt, 1] = L2w

            plt.subplot(2, 3, 4)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                look_oil,
                self.N_injw,
                self.N_pr,
                "oil PhysicsNeMo",
                self.injectors,
                self.producers,
            )
            plt.subplot(2, 3, 5)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                lookf_oil,
                self.N_injw,
                self.N_pr,
                "oil Numerical",
                self.injectors,
                self.producers,
            )
            plt.subplot(2, 3, 6)
            Plot_2D(
                XX,
                YY,
                plt,
                self.nx,
                self.ny,
                self.nz,
                diff1_oil,
                self.N_injw,
                self.N_pr,
                "oil diff",
                self.injectors,
                self.producers,
            )

            R2o, L2o = compute_metrics(look_oil.ravel(), lookf_oil.ravel())
            Accuracy_oil[itt, 0] = R2o
            Accuracy_oil[itt, 1] = L2o

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = "Timestep --" + str(int((itt + 1) * self.dt * self.MAXZ)) + " days"

            plt.suptitle(tita, fontsize=16)
            namez = "saturation_simulations" + str(int(itt))
            yes = (f_2, namez)
            f_big.append(yes)
            # plt.clf()
            plt.close()

            f_3 = plt.figure(figsize=(20, 20), dpi=200)
            ax1 = f_3.add_subplot(231, projection="3d")
            Plot_PhysicsNeMo(
                ax1,
                self.nx,
                self.ny,
                self.nz,
                look_sat,
                self.N_injw,
                self.N_pr,
                "water PhysicsNeMo",
                self.injectors,
                self.producers,
            )
            ax2 = f_3.add_subplot(232, projection="3d")
            Plot_PhysicsNeMo(
                ax2,
                self.nx,
                self.ny,
                self.nz,
                lookf_sat,
                self.N_injw,
                self.N_pr,
                "water Numerical",
                self.injectors,
                self.producers,
            )
            ax3 = f_3.add_subplot(233, projection="3d")
            Plot_PhysicsNeMo(
                ax3,
                self.nx,
                self.ny,
                self.nz,
                diff1_wat,
                self.N_injw,
                self.N_pr,
                "water diff",
                self.injectors,
                self.producers,
            )

            ax4 = f_3.add_subplot(234, projection="3d")
            Plot_PhysicsNeMo(
                ax4,
                self.nx,
                self.ny,
                self.nz,
                look_oil,
                self.N_injw,
                self.N_pr,
                "oil PhysicsNeMo",
                self.injectors,
                self.producers,
            )
            ax5 = f_3.add_subplot(235, projection="3d")
            Plot_PhysicsNeMo(
                ax5,
                self.nx,
                self.ny,
                self.nz,
                lookf_oil,
                self.N_injw,
                self.N_pr,
                "oil Numerical",
                self.injectors,
                self.producers,
            )
            ax6 = f_3.add_subplot(236, projection="3d")
            Plot_PhysicsNeMo(
                ax6,
                self.nx,
                self.ny,
                self.nz,
                diff1_oil,
                self.N_injw,
                self.N_pr,
                "oil diff",
                self.injectors,
                self.producers,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = (
                "3D Map - Timestep --"
                + str(int((itt + 1) * self.dt * self.MAXZ))
                + " days"
            )

            plt.suptitle(tita, fontsize=20, weight="bold")

            namez = "Simulations3Ds" + str(int(itt))
            yes2 = (f_3, namez)
            f_big.append(yes2)
            # plt.clf()
            plt.close()

        fig4 = plt.figure(figsize=(20, 20), dpi=200)

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
            fontsize=16,
        )
        fig4.text(
            0.5,
            0.49,
            "L2(%) Accuracy - PhysicsNeMo/Numerical(GPU)",
            ha="center",
            va="center",
            fontproperties=font,
            fontsize=16,
        )

        plt.subplot(2, 2, 1)
        plt.plot(
            Time_vector,
            Accuracy_water[:, 0],
            label="R2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("water_saturation", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("R2(%)", fontproperties=font)

        plt.subplot(2, 2, 2)
        plt.plot(
            Time_vector,
            Accuracy_oil[:, 0],
            label="R2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("oil_saturation", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("R2(%)", fontproperties=font)

        plt.subplot(2, 2, 3)
        plt.plot(
            Time_vector,
            Accuracy_water[:, 1],
            label="L2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("water_saturation", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("L2(%)", fontproperties=font)

        plt.subplot(2, 2, 4)
        plt.plot(
            Time_vector,
            Accuracy_oil[:, 1],
            label="L2",
            marker="*",
            markerfacecolor="red",
            markeredgecolor="red",
            linewidth=0.5,
        )
        plt.title("oil_saturation", fontproperties=font)
        plt.xlabel("Time (days)", fontproperties=font)
        plt.ylabel("L2(%)", fontproperties=font)

        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        namez = "R2L2_saturations"
        yes21 = (fig4, namez)
        f_big.append(yes21)
        # plt.clf()
        plt.close()

        return f_big


@physicsnemo.sym.main(config_path="conf", config_name="config_FNO")
def run(cfg: PhysicsNeMoConfig) -> None:
    print("")
    print("------------------------------------------------------------------")
    print("")
    print("\n")
    print("|-----------------------------------------------------------------|")
    print("|                 TRAIN THE MODEL USING A 2D FNO APPROACH:        |")
    print("|-----------------------------------------------------------------|")
    print("")

    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    if not os.path.exists(to_absolute_path("../PACKETS")):
        os.makedirs(to_absolute_path("../PACKETS"))
    else:
        pass

    # Varaibles needed for NVRS
    nx = cfg.custom.NVRS.nx
    ny = cfg.custom.NVRS.ny
    BO = cfg.custom.NVRS.BO  # oil formation volume factor
    BW = cfg.custom.NVRS.BW  # Water formation volume factor
    UW = cfg.custom.NVRS.UW  # water viscosity in cP
    UO = cfg.custom.NVRS.UO  # oil viscosity in cP
    DX = cfg.custom.NVRS.DX  # size of pixel in x direction
    DY = cfg.custom.NVRS.DY  # sixze of pixel in y direction

    DX = cp.float32(DX)
    DY = cp.float32(DY)
    UW = cp.float32(UW)  # water viscosity in cP
    UO = cp.float32(UO)  # oil viscosity in cP
    cp.float32(cfg.custom.NVRS.SWI)
    cp.float32(cfg.custom.NVRS.SWR)
    cp.float32(cfg.custom.NVRS.CFO)  # oil compressibility in 1/psi
    pini_alt = cfg.custom.NVRS.pini_alt
    cp.float32(pini_alt)  # Bubble point pressure psia
    BW = cp.float32(BW)  # Water formation volume factor
    BO = cp.float32(BO)  # Oil formation volume factor
    cp.float32(cfg.custom.NVRS.PATM)  # Atmospheric pressure in psi

    # training
    aay, bby = cfg.custom.NVRS.aay, cfg.custom.NVRS.bby  # Permeability range mD
    _Low_K, _High_K = aay, bby

    timmee = (
        cfg.custom.NVRS.timmee
    )  # float(input ('Enter the time step interval duration for simulation (days): '))
    max_t = (
        cfg.custom.NVRS.max_t
    )  # float(input ('Enter the maximum time in days for simulation(days): '))
    MAXZ = cfg.custom.NVRS.MAXZ  # reference maximum time in days of simulation
    steppi = int(max_t / timmee)

    len(cfg.custom.WELLSPECS.water_injector_wells)  # Number of water injectors
    len(cfg.custom.WELLSPECS.producer_wells)  # Number of producers

    # tc2 = Equivalent_time(timmee,2100,timmee,max_t)
    tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
    np.diff(tc2)[0]  # Time-step
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

    bb = os.path.isfile(to_absolute_path("../PACKETS/Training4.mat"))
    if not bb:
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
        print("Load simulated labelled training data from MAT file")
        matt = sio.loadmat(to_absolute_path("../PACKETS/Training4.mat"))

        X_data1 = matt["INPUT"]
        data_use1 = matt["OUTPUT"]

    bb = os.path.isfile(to_absolute_path("../PACKETS/Test4.mat"))
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1G4Cvg8eIObyBK0eoo7iX-0hhMTnpJktj",
            to_absolute_path("../PACKETS/Test4.mat"),
        )
        print("...Downlaod completed.......")
        print("Load simulated labelled test data from MAT file")
        matt = sio.loadmat(to_absolute_path("../PACKETS/Test4.mat"))

        X_data2 = matt["INPUT"]
        data_use2 = matt["OUTPUT"]

    else:
        print("Load simulated labelled test data from MAT file")
        matt = sio.loadmat(to_absolute_path("../PACKETS/Test4.mat"))

        X_data2 = matt["INPUT"]
        data_use2 = matt["OUTPUT"]

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
        cPress[kk, :, :, :] = perm  # np.clip(perm ,1/pini_alt,1.)

        perm = data_use1[kk, steppi:, :, :]
        cSat[kk, :, :, :] = perm

    sio.savemat(
        to_absolute_path("../PACKETS/simulationstrain.mat"),
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

    preprocess_FNO_mat(to_absolute_path("../PACKETS/simulationstrain.mat"))

    cPerm = np.zeros((X_data2.shape[0], 1, nx, ny))  # Permeability
    cQ = np.zeros((X_data2.shape[0], 1, nx, ny))  # Overall source/sink term
    cQw = np.zeros((X_data2.shape[0], 1, nx, ny))  # Sink term
    cPhi = np.zeros((X_data2.shape[0], 1, nx, ny))  # Porosity
    cTime = np.zeros((X_data2.shape[0], 1, nx, ny))  # Time index
    cPini = np.zeros((X_data2.shape[0], 1, nx, ny))  # Initial pressure
    cSini = np.zeros((X_data2.shape[0], 1, nx, ny))  # Initial water saturation

    cPress = np.zeros((X_data2.shape[0], steppi, nx, ny))  # Pressure
    cSat = np.zeros((X_data2.shape[0], steppi, nx, ny))  # Water saturation

    for kk in range(X_data2.shape[0]):
        perm = X_data2[kk, 0, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPerm[kk, :, :, :] = permin

        perm = X_data2[kk, 1, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQ[kk, :, :, :] = permin

        perm = X_data2[kk, 2, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cQw[kk, :, :, :] = permin

        perm = X_data2[kk, 3, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPhi[kk, :, :, :] = permin

        perm = X_data2[kk, 4, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cTime[kk, :, :, :] = permin

        perm = X_data2[kk, 5, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cPini[kk, :, :, :] = permin

        perm = X_data2[kk, 6, :, :]
        permin = np.zeros((1, nx, ny))
        permin[0, :, :] = perm
        cSini[kk, :, :, :] = permin

        perm = data_use2[kk, :steppi, :, :]
        cPress[kk, :, :, :] = perm  # np.clip(perm ,1/pini_alt,1.)

        perm = data_use2[kk, steppi:, :, :]
        cSat[kk, :, :, :] = perm

    sio.savemat(
        to_absolute_path("../PACKETS/simulationstest.mat"),
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

    preprocess_FNO_mat(to_absolute_path("../PACKETS/simulationstest.mat"))

    # load training/ test data
    input_keys = [
        Key("perm", scale=(3.46327e-01, 3.53179e-01)),
        Key("Q", scale=(1.94683e-03, 3.70558e-02)),
        Key("Qw", scale=(2.03866e-03, 3.70199e-02)),
        Key("Phi", scale=(1.73163e-01, 1.76590e-01)),
        Key("Time", scale=(1.66667e-02, 7.45058e-09)),
        Key("Pini", scale=(1.00000e00, 0.00000e00)),
        Key("Swini", scale=(2.00000e-01, 4.91738e-07)),
    ]
    output_keys_pressure = [Key("pressure", scale=(2.87008e-01, 1.85386e-01))]

    output_keys_saturation = [Key("water_sat", scale=(3.12903e-01, 1.79786e-01))]

    invar_train, outvar_train_pressure, outvar_train_saturation = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulationstrain.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_saturation],
        n_examples=cfg.custom.ntrain,
    )
    invar_test, outvar_test_pressure, outvar_test_saturation = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulationstest.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_saturation],
        n_examples=cfg.custom.ntest,
    )

    train_dataset_pressure = DictGridDataset(invar_train, outvar_train_pressure)
    train_dataset_saturation = DictGridDataset(invar_train, outvar_train_saturation)

    test_dataset_pressure = DictGridDataset(invar_test, outvar_test_pressure)
    test_dataset_saturation = DictGridDataset(invar_test, outvar_test_saturation)

    # [init-node]
    # Make custom Darcy residual node for PINO

    # Define FNO model for forward model (pressure)
    decoder1 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("pressure", size=steppi)]
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

    # Define FNO model for forward model (saturation)
    decoder2 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("water_sat", size=steppi)]
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

    nodes = [fno_pressure.make_node("fno_forward_model_pressure")] + [
        fno_saturation.make_node("fno_forward_model_saturation")
    ]

    # [constraint]
    # make domain
    domain = Domain()

    # add constraints to domain
    supervised_pressure = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset_pressure,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised_pressure, "supervised_pressure")

    supervised_saturation = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset_saturation,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised_saturation, "supervised_saturation")

    # [constraint]
    # add validator
    # test_pressure = GridValidator(
    #     nodes,
    #     dataset=test_dataset_pressure,
    #     batch_size=1,
    #     plotter=CustomValidatorPlotterP(timmee,max_t,MAXZ,pini_alt,nx,ny,nz,\
    #                                 steppi,tc2,dt,injectors,producers,N_injw,N_pr),
    #     requires_grad=False,
    # )

    test_pressure = GridValidator(
        nodes,
        dataset=test_dataset_pressure,
        batch_size=1,
        requires_grad=False,
    )

    domain.add_validator(test_pressure, "test_pressure")

    # test_saturation = GridValidator(
    #     nodes,
    #     dataset=test_dataset_saturation,
    #     batch_size=1,
    #     plotter=CustomValidatorPlotterS(timmee,max_t,MAXZ,pini_alt,nx,ny,nz,\
    #                                 steppi,tc2,dt,injectors,producers,N_injw,N_pr),

    #     requires_grad=False,
    # )

    test_saturation = GridValidator(
        nodes,
        dataset=test_dataset_saturation,
        batch_size=1,
        requires_grad=False,
    )
    domain.add_validator(test_saturation, "test_saturation")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
