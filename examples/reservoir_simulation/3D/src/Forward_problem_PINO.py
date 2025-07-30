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

from typing import Dict
import numpy as np
import torch
import os
import physicsnemo
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from NVRS import *
from utilities import load_FNO_dataset2, preprocess_FNO_mat
from ops import dx, ddx
from physicsnemo.sym.models.fno import *
import cupy as cp
import scipy.io as sio
import requests
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter

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


class CustomValidatorPlotterP(ValidatorPlotter):
    def __init__(
        self,
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
        self.wells = wells
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

        pressure_true, pressure_pred = true_outvar["pressure"], pred_outvar["pressure"]

        # make plot
        f_big = []
        Time_vector = np.zeros((self.steppi))
        Accuracy_presure = np.zeros((self.steppi, 2))
        for itt in range(self.steppi):
            look = (pressure_pred[0, itt, :, :, :]) * self.pini_alt

            lookf = (pressure_true[0, itt, :, :, :]) * self.pini_alt

            diff1 = abs(look - lookf)

            XX, YY = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
            f_2 = plt.figure(figsize=(12, 12), dpi=100)
            plt.subplot(3, 3, 1)
            plt.pcolormesh(XX.T, YY.T, look[0, :, :], cmap="jet")
            plt.title("Layer 1 - Pressure PINO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf[0, :, :], (-1,))),
                np.max(np.reshape(lookf[0, :, :], (-1,))),
            )
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(3, 3, 2)
            plt.pcolormesh(XX.T, YY.T, lookf[0, :, :], cmap="jet")
            plt.title(" Layer 1 - Pressure CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(3, 3, 3)
            plt.pcolormesh(XX.T, YY.T, abs(look[0, :, :] - lookf[0, :, :]), cmap="jet")
            plt.title(" Layer 1 - Pressure (CFD - PINO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(3, 3, 4)
            plt.pcolormesh(XX.T, YY.T, look[1, :, :], cmap="jet")
            plt.title("Layer 2 - Pressure PINO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf[1, :, :], (-1,))),
                np.max(np.reshape(lookf[1, :, :], (-1,))),
            )
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(3, 3, 5)
            plt.pcolormesh(XX.T, YY.T, lookf[1, :, :], cmap="jet")
            plt.title(" Layer 2 - Pressure CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(3, 3, 6)
            plt.pcolormesh(XX.T, YY.T, abs(look[1, :, :] - lookf[1, :, :]), cmap="jet")
            plt.title(" Layer 2 - Pressure (CFD - PINO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(3, 3, 7)
            plt.pcolormesh(XX.T, YY.T, look[2, :, :], cmap="jet")
            plt.title("Layer 3 - Pressure PINO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf[2, :, :], (-1,))),
                np.max(np.reshape(lookf[2, :, :], (-1,))),
            )
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(3, 3, 8)
            plt.pcolormesh(XX.T, YY.T, lookf[2, :, :], cmap="jet")
            plt.title(" Layer 3 - Pressure CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(3, 3, 9)
            plt.pcolormesh(XX.T, YY.T, abs(look[2, :, :] - lookf[2, :, :]), cmap="jet")
            plt.title(" Layer 3 - Pressure (CFD - PINO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" Pressure (psia)", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = "Timestep --" + str(int((itt + 1) * self.dt * self.MAXZ)) + " days"

            plt.suptitle(tita, fontsize=16)

            # name = namet + str(int(itt)) + '.png'
            # plt.savefig(name)
            # #plt.show()
            # plt.clf()
            namez = "pressure_simulations" + str(int(itt))
            yes = (f_2, namez)
            f_big.append(yes)

            f_3 = plt.figure(figsize=(20, 20), dpi=200)
            ax1 = f_3.add_subplot(131, projection="3d")
            Plot_PhysicsNeMo(
                ax1,
                self.nx,
                self.ny,
                self.nz,
                Reinvent(look),
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
                Reinvent(lookf),
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
                Reinvent(diff1),
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

            plt.suptitle(tita, fontsize=16)

            namez = "Simulations3Dp" + str(int(itt))
            yes2 = (f_3, namez)
            f_big.append(yes2)

            R2p, L2p = compute_metrics(look.ravel()), lookf.ravel()
            Accuracy_presure[itt, 0] = R2p
            Accuracy_presure[itt, 1] = L2p
        fig4, axs = plt.subplots(2, 1, figsize=(10, 10))

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
        for i, data in enumerate([Accuracy_presure]):
            axs[0, i].plot(
                Time_vector,
                data[:, 0],
                label="R2",
                marker="*",
                markerfacecolor="red",
                markeredgecolor="red",
                linewidth=0.5,
            )
            axs[0, i].set_title(["Pressure"][i], fontproperties=font)
            axs[0, i].set_xlabel("Time (days)", fontproperties=font)
            axs[0, i].set_ylabel("R2(%)", fontproperties=font)

        # Plot L2 accuracies
        for i, data in enumerate([Accuracy_presure]):
            axs[1, i].plot(
                Time_vector,
                data[:, 1],
                label="L2",
                marker="*",
                markerfacecolor="red",
                markeredgecolor="red",
                linewidth=0.5,
            )
            axs[1, i].set_title(["Pressure"][i], fontproperties=font)
            axs[1, i].set_xlabel("Time (days)", fontproperties=font)
            axs[1, i].set_ylabel("L2(%)", fontproperties=font)
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        namez = "R2L2_pressure"
        yes21 = (fig4, namez)
        f_big.append(yes21)
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
        wells,
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
        self.wells = wells
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
            XX, YY = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
            f_2 = plt.figure(figsize=(20, 20), dpi=100)

            look_sat = water_pred[0, itt, :, :, :]
            look_oil = 1 - look_sat

            lookf_sat = water_true[0, itt, :, :, :]
            lookf_oil = 1 - lookf_sat

            diff1_wat = abs(look_sat - lookf_sat)
            diff1_oil = abs(look_oil - lookf_oil)

            plt.subplot(6, 3, 1)
            plt.pcolormesh(XX.T, YY.T, look_sat[0, :, :], cmap="jet")
            plt.title(" Layer 1 - water_sat PINO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf_sat[0, :, :], (-1,))),
                np.max(np.reshape(lookf_sat[0, :, :], (-1,))),
            )
            cbar1.ax.set_ylabel(" water_sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 2)
            plt.pcolormesh(XX.T, YY.T, lookf_sat[0, :, :], cmap="jet")
            plt.title(" Layer 1 - water_sat CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" water sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 3)
            plt.pcolormesh(XX.T, YY.T, diff1_wat[0, :, :], cmap="jet")
            plt.title(" Layer 1- water_sat (CFD - PINO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" water sat ", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 4)
            plt.pcolormesh(XX.T, YY.T, look_sat[1, :, :], cmap="jet")
            plt.title(" Layer 2 - water_sat PINO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf_sat[1, :, :], (-1,))),
                np.max(np.reshape(lookf_sat[1, :, :], (-1,))),
            )
            cbar1.ax.set_ylabel(" water_sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 5)
            plt.pcolormesh(XX.T, YY.T, lookf_sat[1, :, :], cmap="jet")
            plt.title(" Layer 2 - water_sat CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" water sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 6)
            plt.pcolormesh(XX.T, YY.T, diff1_wat[1, :, :], cmap="jet")
            plt.title(" Layer 2- water_sat (CFD - PINO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" water sat ", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 7)
            plt.pcolormesh(XX.T, YY.T, look_sat[2, :, :], cmap="jet")
            plt.title(" Layer 3 - water_sat PINO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf_sat[2, :, :], (-1,))),
                np.max(np.reshape(lookf_sat[2, :, :], (-1,))),
            )
            cbar1.ax.set_ylabel(" water_sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 8)
            plt.pcolormesh(XX.T, YY.T, lookf_sat[2, :, :], cmap="jet")
            plt.title(" Layer 3 - water_sat CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" water sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 9)
            plt.pcolormesh(XX.T, YY.T, diff1_wat[2, :, :], cmap="jet")
            plt.title(" Layer 3- water_sat (CFD - PINO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" water sat ", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 10)
            plt.pcolormesh(XX.T, YY.T, look_oil[0, :, :], cmap="jet")
            plt.title(" Layer 1 - oil_sat PINO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf_oil[0, :, :], (-1,))),
                np.max(np.reshape(lookf_oil[0, :, :], (-1,))),
            )
            cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 11)
            plt.pcolormesh(XX.T, YY.T, lookf_oil[0, :, :], cmap="jet")
            plt.title(" Layer 1 - oil_sat CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" oil sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 12)
            plt.pcolormesh(XX.T, YY.T, diff1_oil[0, :, :], cmap="jet")
            plt.title(" Layer 1 - oil_sat (CFD - PINO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" oil sat ", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 13)
            plt.pcolormesh(XX.T, YY.T, look_oil[1, :, :], cmap="jet")
            plt.title(" Layer 2 - oil_sat PINO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf_oil[1, :, :], (-1,))),
                np.max(np.reshape(lookf_oil[1, :, :], (-1,))),
            )
            cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 14)
            plt.pcolormesh(XX.T, YY.T, lookf_oil[1, :, :], cmap="jet")
            plt.title(" Layer 2 - oil_sat CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" oil sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 15)
            plt.pcolormesh(XX.T, YY.T, diff1_oil[1, :, :], cmap="jet")
            plt.title(" Layer 2 - oil_sat (CFD - PINO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" oil sat ", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 16)
            plt.pcolormesh(XX.T, YY.T, look_oil[2, :, :], cmap="jet")
            plt.title(" Layer 3 - oil_sat PINO", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            plt.clim(
                np.min(np.reshape(lookf_oil[2, :, :], (-1,))),
                np.max(np.reshape(lookf_oil[2, :, :], (-1,))),
            )
            cbar1.ax.set_ylabel(" oil_sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 17)
            plt.pcolormesh(XX.T, YY.T, lookf_oil[2, :, :], cmap="jet")
            plt.title(" Layer 3 - oil_sat CFD", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" oil sat", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.subplot(6, 3, 18)
            plt.pcolormesh(XX.T, YY.T, diff1_oil[2, :, :], cmap="jet")
            plt.title(" Layer 3 - oil_sat (CFD - PINO)", fontsize=13)
            plt.ylabel("Y", fontsize=13)
            plt.xlabel("X", fontsize=13)
            plt.axis([0, (self.nx - 1), 0, (self.ny - 1)])
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            cbar1 = plt.colorbar()
            cbar1.ax.set_ylabel(" oil sat ", fontsize=13)
            Add_marker(plt, XX, YY, self.wells)

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = "Timestep --" + str(int((itt + 1) * self.dt * self.MAXZ)) + " days"

            plt.suptitle(tita, fontsize=16)

            # name = namet + str(int(itt)) + '.png'
            # plt.savefig(name)
            # #plt.show()
            # plt.clf()
            namez = "saturation_simulations" + str(int(itt))
            yes = (f_2, namez)
            f_big.append(yes)

            R2o, L2o = compute_metrics(look_oil.ravel()), lookf_oil.ravel()
            Accuracy_oil[itt, 0] = R2o
            Accuracy_oil[itt, 1] = L2o

            f_3 = plt.figure(figsize=(20, 20), dpi=200)
            ax1 = f_3.add_subplot(231, projection="3d")
            Plot_PhysicsNeMo(
                ax1,
                self.nx,
                self.ny,
                self.nz,
                Reinvent(look_sat),
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
                Reinvent(lookf_sat),
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
                Reinvent(diff1_sat),
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
                Reinvent(look_oil),
                self.N_injw,
                self.N_pr,
                "water PhysicsNeMo",
                self.injectors,
                self.producers,
            )
            ax5 = f_3.add_subplot(235, projection="3d")
            Plot_PhysicsNeMo(
                ax5,
                self.nx,
                self.ny,
                self.nz,
                Reinvent(lookf_oil),
                self.N_injw,
                self.N_pr,
                "water Numerical",
                self.injectors,
                self.producers,
            )
            ax6 = f_3.add_subplot(236, projection="3d")
            Plot_PhysicsNeMo(
                ax6,
                self.nx,
                self.ny,
                self.nz,
                Reinvent(diff1_oil),
                self.N_injw,
                self.N_pr,
                "water diff",
                self.injectors,
                self.producers,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            tita = (
                "3D Map - Timestep --"
                + str(int((itt + 1) * self.dt * self.MAXZ))
                + " days"
            )

            plt.suptitle(tita, fontsize=16)

            namez = "Simulations3Ds" + str(int(itt))
            yes2 = (f_3, namez)
            f_big.append(yes2)

        fig4, axs = plt.subplots(2, 2, figsize=(20, 10))

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
        for i, data in enumerate([Accuracy_water, Accuracy_oil]):
            axs[0, i].plot(
                Time_vector,
                data[:, 0],
                label="R2",
                marker="*",
                markerfacecolor="red",
                markeredgecolor="red",
                linewidth=0.5,
            )
            axs[0, i].set_title(
                ["Water_saturation", "Oil_saturation"][i], fontproperties=font
            )
            axs[0, i].set_xlabel("Time (days)", fontproperties=font)
            axs[0, i].set_ylabel("R2(%)", fontproperties=font)

        # Plot L2 accuracies
        for i, data in enumerate([Accuracy_water, Accuracy_oil]):
            axs[1, i].plot(
                Time_vector,
                data[:, 1],
                label="L2",
                marker="*",
                markerfacecolor="red",
                markeredgecolor="red",
                linewidth=0.5,
            )
            axs[1, i].set_title(
                ["Water_saturation", "Oil_saturation"][i], fontproperties=font
            )
            axs[1, i].set_xlabel("Time (days)", fontproperties=font)
            axs[1, i].set_ylabel("L2(%)", fontproperties=font)
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        namez = "R2L2_saturations"
        yes21 = (fig4, namez)
        f_big.append(yes21)
        return f_big


# [pde-loss]
# define custom class for black oil model
class Black_oil(torch.nn.Module):
    "Custom Black oil PDE definition for PINO"

    def __init__(
        self,
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
        nz,
    ):
        super().__init__()
        self.UIR = UIR
        self.UWR = UIR
        self.pini_alt = pini_alt
        self.LUB = LUB
        self.HUB = HUB
        self.aay = aay
        self.bby = bby
        self.SWI = SWI
        self.SWR = SWR
        self.UW = UW
        self.BW = BW
        self.UO = UO
        self.BO = BO
        self.MAXZ = MAXZ
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

        siniuse = sini[0, 0, 0, 0, 0]

        dtin = dt * self.MAXZ
        dxf = 1.0 / u.shape[3]

        u = u * self.pini_alt
        pini = pini * self.pini_alt
        # Pressure equation Loss
        fin = fin * self.UIR
        finwater = finwater * self.UIR
        cuda = 0
        device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

        # print(pressurey.shape)
        p_loss = torch.zeros_like(u).to(device, torch.float32)
        s_loss = torch.zeros_like(u).to(device, torch.float32)

        a = perm  # absolute permeability
        v_min, v_max = self.LUB, self.HUB
        new_min, new_max = self.aay, self.bby

        m = (new_max - new_min) / (v_max - v_min)
        b = new_min - m * v_min
        a = m * a + b

        finusew = finwater
        dta = dtin

        pressure = u
        # water_sat = sat

        prior_pressure = torch.zeros(
            sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny
        ).to(device, torch.float32)
        prior_pressure[:, 0, :, :, :] = self.pini_alt * (
            torch.ones(sat.shape[0], self.nz, self.nx, self.ny).to(
                device, torch.float32
            )
        )
        prior_pressure[:, 1:, :, :, :] = u[:, :-1, :, :, :]

        # dsp = u - prior_pressure  #dp

        prior_sat = torch.zeros(
            sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny
        ).to(device, torch.float32)
        prior_sat[:, 0, :, :, :] = siniuse * (
            torch.ones(sat.shape[0], self.nz, self.nx, self.ny).to(
                device, torch.float32
            )
        )
        prior_sat[:, 1:, :, :, :] = sat[:, :-1, :, :, :]

        dsw = sat - prior_sat  # ds
        dsw = torch.clip(dsw, 0.001, None)

        S = torch.div(
            torch.sub(prior_sat, self.SWI, alpha=1), (1 - self.SWI - self.SWR)
        )

        # Pressure equation Loss
        Mw = torch.divide(torch.square(S), (self.UW * self.BW))  # Water mobility
        Mo = torch.div(
            torch.square(torch.sub(torch.ones(S.shape, device=u.device), S)),
            (self.UO * self.BO),
        )

        Mt = Mw + Mo
        a1 = torch.mul(Mt, a)  # overall Effective permeability
        a1water = torch.mul(Mw, a)  # water Effective permeability

        # compute first dffrential
        gulpa = []
        gulp2a = []
        for m in range(sat.shape[0]):  # Batch
            inn_now = pressure[m, :, :, :, :]
            gulp = []
            gulp2 = []
            for i in range(self.nz):
                now = inn_now[:, i, :, :][:, None, :, :]
                dudx_fdma = dx(
                    now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = dx(
                    now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulp.append(dudx_fdma)
                gulp2.append(dudy_fdma)
            check = torch.stack(gulp, 2)[:, 0, :, :, :]
            check2 = torch.stack(gulp2, 2)[:, 0, :, :]
            gulpa.append(check)
            gulp2a.append(check2)
        dudx_fdm = torch.stack(gulpa, 0)
        dudy_fdm = torch.stack(gulp2a, 0)

        # Compute second diffrential
        gulpa = []
        gulp2a = []
        for m in range(sat.shape[0]):  # Batch
            inn_now = pressure[m, :, :, :, :]
            gulp = []
            gulp2 = []
            for i in range(self.nz):
                now = inn_now[:, i, :, :][:, None, :, :]
                dudx_fdma = ddx(
                    now, dx=dxf, channel=0, dim=0, order=1, padding="replication"
                )
                dudy_fdma = ddx(
                    now, dx=dxf, channel=0, dim=1, order=1, padding="replication"
                )
                gulp.append(dudx_fdma)
                gulp2.append(dudy_fdma)
            check = torch.stack(gulp, 2)[:, 0, :, :, :]
            check2 = torch.stack(gulp2, 2)[:, 0, :, :]
            gulpa.append(check)
            gulp2a.append(check2)
        dduddx_fdm = torch.stack(gulpa, 0)
        dduddy_fdm = torch.stack(gulp2a, 0)

        gulp = []
        gulp2 = []
        for i in range(self.nz):
            inn_now2 = a1[:, :, i, :, :]
            dudx_fdma = dx(
                inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
            )
            dudy_fdma = dx(
                inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
            )
            gulp.append(dudx_fdma)
            gulp2.append(dudy_fdma)
        dcdx = torch.stack(gulp, 2)
        dcdy = torch.stack(gulp2, 2)

        # Expand dcdx
        # dss = dcdx
        dsout = torch.zeros((sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)).to(
            device, torch.float32
        )
        for k in range(dcdx.shape[0]):
            see = dcdx[k, :, :, :, :]
            gulp = []
            for i in range(sat.shape[1]):
                gulp.append(see)

            checkken = torch.vstack(gulp)
            dsout[k, :, :, :, :] = checkken

        dcdx = dsout

        dsout = torch.zeros((sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)).to(
            device, torch.float32
        )
        for k in range(dcdx.shape[0]):
            see = dcdy[k, :, :, :, :]
            gulp = []
            for i in range(sat.shape[1]):
                gulp.append(see)

            checkken = torch.vstack(gulp)
            dsout[k, :, :, :, :] = checkken

        dcdy = dsout

        darcy_pressure = (
            fin
            + (dcdx * dudx_fdm)
            + (a1 * dduddx_fdm)
            + (dcdy * dudy_fdm)
            + (a1 * dduddy_fdm)
        )

        # Zero outer boundary
        # darcy_pressure = F.pad(darcy_pressure[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
        darcy_pressure = dxf * darcy_pressure * 1e-5

        p_loss = darcy_pressure

        # Saruration equation loss
        dudx = dudx_fdm
        dudy = dudy_fdm

        dduddx = dduddx_fdm
        dduddy = dduddy_fdm

        gulp = []
        gulp2 = []
        for i in range(self.nz):
            inn_now2 = a1water[:, :, i, :, :]
            dudx_fdma = dx(
                inn_now2, dx=dxf, channel=0, dim=0, order=1, padding="replication"
            )
            dudy_fdma = dx(
                inn_now2, dx=dxf, channel=0, dim=1, order=1, padding="replication"
            )
            gulp.append(dudx_fdma)
            gulp2.append(dudy_fdma)
        dadx = torch.stack(gulp, 2)
        dady = torch.stack(gulp2, 2)

        dsout = torch.zeros((sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)).to(
            device, torch.float32
        )
        for k in range(dadx.shape[0]):
            see = dadx[k, :, :, :, :]
            gulp = []
            for i in range(sat.shape[1]):
                gulp.append(see)

            checkken = torch.vstack(gulp)
            dsout[k, :, :, :, :] = checkken

        dadx = dsout

        dsout = torch.zeros((sat.shape[0], sat.shape[1], self.nz, self.nx, self.ny)).to(
            device, torch.float32
        )
        for k in range(dady.shape[0]):
            see = dady[k, :, :, :, :]
            gulp = []
            for i in range(sat.shape[1]):
                gulp.append(see)

            checkken = torch.vstack(gulp)
            dsout[k, :, :, :, :] = checkken

        dady = dsout

        flux = (dadx * dudx) + (a1water * dduddx) + (dady * dudy) + (a1water * dduddy)
        fifth = poro * (dsw / dta)
        toge = flux + finusew
        darcy_saturation = fifth - toge

        # Zero outer boundary
        # darcy_saturation = F.pad(darcy_saturation[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
        darcy_saturation = dxf * darcy_saturation * 1e-5

        s_loss = darcy_saturation

        # output_var["darcy_saturation"] = torch.mean(s_loss,dim = 0)[None,:,:,:]
        output_var = {"pressured": p_loss, "saturationd": s_loss}
        return output_var


# [pde-loss]
@physicsnemo.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: PhysicsNeMoConfig) -> None:
    print("")
    print("------------------------------------------------------------------")
    print("")
    print("\n")
    print("|-----------------------------------------------------------------|")
    print("|                 TRAIN THE MODEL USING A 3D PINO APPROACH:       |")
    print("|-----------------------------------------------------------------|")
    print("")

    wells = np.array(
        [
            1,
            24,
            1,
            1,
            1,
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

    oldfolder = os.getcwd()
    os.chdir(oldfolder)

    # Varaibles needed for NVRS
    nx = cfg.custom.NVRS.nx
    ny = cfg.custom.NVRS.ny
    nz = cfg.custom.NVRS.nz
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
    SWI = cp.float32(cfg.custom.NVRS.SWI)
    SWR = cp.float32(cfg.custom.NVRS.SWR)
    pini_alt = cfg.custom.NVRS.pini_alt
    BW = cp.float32(BW)  # Water formation volume factor
    BO = cp.float32(BO)  # Oil formation volume factor

    # training
    LUB = cfg.custom.NVRS.LUB
    HUB = cfg.custom.NVRS.HUB  # Permeability rescale
    aay, bby = cfg.custom.NVRS.aay, cfg.custom.NVRS.bby  # Permeability range mD
    # Low_K, High_K = aay,bby

    # batch_size = cfg.custom.NVRS.batch_size #'size of simulated labelled data to run'
    timmee = (
        cfg.custom.NVRS.timmee
    )  # float(input ('Enter the time step interval duration for simulation (days): '))
    max_t = (
        cfg.custom.NVRS.max_t
    )  # float(input ('Enter the maximum time in days for simulation(days): '))
    MAXZ = cfg.custom.NVRS.MAXZ  # reference maximum time in days of simulation
    steppi = int(max_t / timmee)
    UIR = cfg.custom.NVRS.UIR  # uppwer injection rate

    len(cfg.custom.WELLSPECS.water_injector_wells)  # Number of water injectors
    len(cfg.custom.WELLSPECS.producer_wells)  # Number of producers

    # tc2 = Equivalent_time(timmee,2100,timmee,max_t)
    tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
    np.diff(tc2)[0]  # Time-step

    bb = os.path.isfile(to_absolute_path("../PACKETS/Training4.mat"))
    if not bb:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1wYyREUcpp0qLhbRItG5RMPeRMxVtntDi",
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
            "1PX2XFG1-elzQItvkUERJqeOerTO2kevq",
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

    cPerm = np.zeros((X_data1.shape[0], 1, nz, nx, ny))  # Permeability
    cQ = np.zeros((X_data1.shape[0], 1, nz, nx, ny))  # Overall source/sink term
    cQw = np.zeros((X_data1.shape[0], 1, nz, nx, ny))  # Sink term
    cPhi = np.zeros((X_data1.shape[0], 1, nz, nx, ny))  # Porosity
    cTime = np.zeros((X_data1.shape[0], 1, nz, nx, ny))  # Time index
    cPini = np.zeros((X_data1.shape[0], 1, nz, nx, ny))  # Initial pressure
    cSini = np.zeros((X_data1.shape[0], 1, nz, nx, ny))  # Initial water saturation

    cPress = np.zeros((X_data1.shape[0], steppi, nz, nx, ny))  # Pressure
    cSat = np.zeros((X_data1.shape[0], steppi, nz, nx, ny))  # Water saturation

    for kk in range(X_data1.shape[0]):
        perm = X_data1[kk, 0, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cPerm[kk, :, :, :, :] = permin

        perm = X_data1[kk, 1, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cQ[kk, :, :, :, :] = permin

        perm = X_data1[kk, 2, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cQw[kk, :, :, :, :] = permin

        perm = X_data1[kk, 3, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cPhi[kk, :, :, :, :] = permin

        perm = X_data1[kk, 4, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cTime[kk, :, :, :, :] = permin

        perm = X_data1[kk, 5, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cPini[kk, :, :, :, :] = permin

        perm = X_data1[kk, 6, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cSini[kk, :, :, :, :] = permin

        perm = data_use1[kk, :steppi, :, :, :]
        perm_big = np.zeros((steppi, nz, nx, ny))
        for mum in range(steppi):
            use = perm[mum, :, :, :]
            mum1 = np.zeros((nz, nx, ny))
            for i in range(nz):
                mum1[i, :, :] = use[:, :, i]

            perm_big[mum, :, :, :] = mum1
        cPress[kk, :, :, :, :] = np.clip(perm_big, 1 / pini_alt, 2.0)

        perm = data_use1[kk, steppi:, :, :, :]
        perm_big = np.zeros((steppi, nz, nx, ny))
        for mum in range(steppi):
            use = perm[mum, :, :, :]
            mum1 = np.zeros((nz, nx, ny))
            for i in range(nz):
                mum1[i, :, :] = use[:, :, i]

            perm_big[mum, :, :, :] = mum1
        cSat[kk, :, :, :, :] = perm_big

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

    cPerm = np.zeros((X_data2.shape[0], 1, nz, nx, ny))  # Permeability
    cQ = np.zeros((X_data2.shape[0], 1, nz, nx, ny))  # Overall source/sink term
    cQw = np.zeros((X_data2.shape[0], 1, nz, nx, ny))  # Sink term
    cPhi = np.zeros((X_data2.shape[0], 1, nz, nx, ny))  # Porosity
    cTime = np.zeros((X_data2.shape[0], 1, nz, nx, ny))  # Time index
    cPini = np.zeros((X_data2.shape[0], 1, nz, nx, ny))  # Initial pressure
    cSini = np.zeros((X_data2.shape[0], 1, nz, nx, ny))  # Initial water saturation

    cPress = np.zeros((X_data2.shape[0], steppi, nz, nx, ny))  # Pressure
    cSat = np.zeros((X_data2.shape[0], steppi, nz, nx, ny))  # Water saturation

    for kk in range(X_data2.shape[0]):
        perm = X_data2[kk, 0, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cPerm[kk, :, :, :, :] = permin

        perm = X_data2[kk, 1, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cQ[kk, :, :, :, :] = permin

        perm = X_data2[kk, 2, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cQw[kk, :, :, :, :] = permin

        perm = X_data2[kk, 3, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cPhi[kk, :, :, :, :] = permin

        perm = X_data2[kk, 4, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cTime[kk, :, :, :, :] = permin

        perm = X_data2[kk, 5, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cPini[kk, :, :, :, :] = permin

        perm = X_data2[kk, 6, :, :, :]
        permin = np.zeros((1, nz, nx, ny))
        for i in range(nz):
            permin[0, i, :, :] = perm[:, :, i]
        cSini[kk, :, :, :, :] = permin

        perm = data_use2[kk, :steppi, :, :, :]
        perm_big = np.zeros((steppi, nz, nx, ny))
        for mum in range(steppi):
            use = perm[mum, :, :, :]
            mum1 = np.zeros((nz, nx, ny))
            for i in range(nz):
                mum1[i, :, :] = use[:, :, i]

            perm_big[mum, :, :, :] = mum1
        cPress[kk, :, :, :, :] = np.clip(perm_big, 1 / pini_alt, 2.0)

        perm = data_use2[kk, steppi:, :, :, :]
        perm_big = np.zeros((steppi, nz, nx, ny))
        for mum in range(steppi):
            use = perm[mum, :, :, :]
            mum1 = np.zeros((nz, nx, ny))
            for i in range(nz):
                mum1[i, :, :] = use[:, :, i]

            perm_big[mum, :, :, :] = mum1
        cSat[kk, :, :, :, :] = perm_big

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
        Key("perm", scale=(5.38467e-01, 2.29917e-01)),
        Key("Q", scale=(1.33266e-03, 3.08151e-02)),
        Key("Qw", scale=(1.39516e-03, 3.07869e-02)),
        Key("Phi", scale=(2.69233e-01, 1.14958e-01)),
        Key("Time", scale=(1.66666e-02, 1.08033e-07)),
        Key("Pini", scale=(1.00000e00, 0.00000e00)),
        Key("Swini", scale=(1.99998e-01, 2.07125e-06)),
    ]
    output_keys_pressure = [Key("pressure", scale=(1.16260e00, 5.75724e-01))]

    output_keys_saturation = [Key("water_sat", scale=(3.61902e-01, 1.97300e-01))]

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

    # add additional constraining values for darcy variable
    outvar_train_pressure["pressured"] = np.zeros_like(
        outvar_train_pressure["pressure"]
    )
    outvar_train_saturation["saturationd"] = np.zeros_like(
        outvar_train_saturation["water_sat"]
    )

    # outvar_train1_pressure["pressured"] = np.zeros_like(outvar_train1_pressure["pressure"])
    # outvar_train1_saturation["saturationd"] = np.zeros_like(outvar_train1_saturation["water_sat"])

    train_dataset_pressure = DictGridDataset(invar_train, outvar_train_pressure)
    train_dataset_saturation = DictGridDataset(invar_train, outvar_train_saturation)

    test_dataset_pressure = DictGridDataset(invar_test, outvar_test_pressure)
    test_dataset_saturation = DictGridDataset(invar_test, outvar_test_saturation)

    # [init-node]

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
        fno_modes=16,
        dimension=3,
        padding=13,
        nr_fno_layers=4,
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
        fno_modes=16,
        dimension=3,
        padding=13,
        nr_fno_layers=4,
        decoder_net=decoder2,
    )

    inputs = [
        "perm",
        "Q",
        "Qw",
        "Phi",
        "Time",
        "Pini",
        "Swini",
        "pressure",
        "water_sat",
    ]

    darcyy = Node(
        inputs=inputs,
        outputs=[
            "pressured",
            "saturationd",
        ],
        evaluate=Black_oil(
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
            nz,
        ),
        name="Darcy node",
    )
    nodes = (
        [darcyy]
        + [fno_pressure.make_node("pino_forward_model_pressure", jit=cfg.jit)]
        + [fno_saturation.make_node("pino_forward_model_saturation", jit=cfg.jit)]
    )

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
    #     batch_size=cfg.batch_size.test,
    #     plotter=CustomValidatorPlotterP(timmee,max_t,MAXZ,pini_alt,nx,ny,\
    #                                 wells,steppi,tc2,dt,dt,injectors,producers,N_injw,N_pr),
    #     requires_grad=False,
    # )

    test_pressure = GridValidator(
        nodes,
        dataset=test_dataset_pressure,
        batch_size=cfg.batch_size.test,
        requires_grad=False,
    )

    domain.add_validator(test_pressure, "test_pressure")

    # test_saturation = GridValidator(
    #     nodes,
    #     dataset=test_dataset_saturation,
    #     batch_size=cfg.batch_size.test,
    #     plotter=CustomValidatorPlotterS(timmee,max_t,MAXZ,pini_alt,nx,ny,\
    #                                 wells,steppi,tc2,dt,dt,injectors,producers,N_injw,N_pr),
    #     requires_grad=False,
    # )

    test_saturation = GridValidator(
        nodes,
        dataset=test_dataset_saturation,
        batch_size=cfg.batch_size.test,
        requires_grad=False,
    )

    domain.add_validator(test_saturation, "test_saturation")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
