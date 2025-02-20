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
import os
import physicsnemo
import torch
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from physicsnemo.sym.utils.io.plotter import GridValidatorPlotter
from NVRS import *
from utilities import load_FNO_dataset2, preprocess_FNO_mat
from physicsnemo.sym.models.fno import *
import shutil
import cupy as cp
from skimage.transform import resize
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


from physicsnemo.sym.utils.io.plotter import ValidatorPlotter


class CustomValidatorPlotterP(ValidatorPlotter):
    def __init__(self, timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt):
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

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables

        pressure_true, pressure_pred = true_outvar["pressure"], pred_outvar["pressure"]

        # make plot
        f_big = []
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
        return f_big


class CustomValidatorPlotterS(ValidatorPlotter):
    def __init__(self, timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt):
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

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables

        water_true, water_pred = true_outvar["water_sat"], pred_outvar["water_sat"]

        # make plot

        f_big = []
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
        return f_big


@physicsnemo.sym.main(config_path="conf", config_name="config_FNO")
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
    print("")
    print("------------------------------------------------------------------")
    print("")
    print("\n")
    print("|-----------------------------------------------------------------|")
    print("|                 TRAIN THE MODEL USING A 3D FNO APPROACH:        |")
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

    batch_size = 1000  #'size of simulated labelled data to run'
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
            if bb == False:
                print("Get initial geology from saved Multiple-point-statistics run")

                print("....Downloading Please hold.........")
                download_file_from_google_drive(
                    "1KZvxypUSsjpkLogGm__56-bckEee3VJh",
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
            if bb == False:
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
                    "1VSy2m3ocUkZnhCsorbkhcJB5ADrPxzIp",
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

                kq = np.reshape(kq, (nx, ny, nz), "F")

                # Truee1 = imresize(Truee, output_shape=shape)

                kq = resize(kq, (nx, ny), order=1, preserve_range=True)
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

            ini_ensembleee = see
            sio.savemat(
                to_absolute_path("../PACKETS/Ganensemble.mat"), {"Z": ini_ensembleee}
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

    bb = os.path.isfile(to_absolute_path("../PACKETS/Training4.mat"))
    if bb == False:
        if use_pretrained == 1:

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
                2,
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
    if bb == False:
        print("....Downloading Please hold.........")
        download_file_from_google_drive(
            "1VSy2m3ocUkZnhCsorbkhcJB5ADrPxzIp",
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

    ini_ensemble1 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    ini_ensemble2 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    ini_ensemble3 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    ini_ensemble4 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    ini_ensemble5 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    ini_ensemble6 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    ini_ensemble7 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)

    ini_ensemble8 = np.zeros((Ne, steppi, nz, nx, ny))  # Pressure
    ini_ensemble9 = np.zeros((Ne, steppi, nz, nx, ny))  # Water saturation

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

        perm = data_use2[kk, :steppi, :, :, :]
        perm_big = np.zeros((steppi, nz, nx, ny))
        for mum in range(steppi):
            use = perm[mum, :, :, :]
            mum1 = np.zeros((nz, nx, ny))
            for i in range(nz):
                mum1[i, :, :] = use[:, :, i]

            perm_big[mum, :, :, :] = mum1
        ini_ensemble8[kk, :, :, :, :] = np.clip(perm_big, 1 / pini_alt, 2.0)

        perm = data_use2[kk, steppi:, :, :, :]
        perm_big = np.zeros((steppi, nz, nx, ny))
        for mum in range(steppi):
            use = perm[mum, :, :, :]
            mum1 = np.zeros((nz, nx, ny))
            for i in range(nz):
                mum1[i, :, :] = use[:, :, i]

            perm_big[mum, :, :, :] = mum1
        ini_ensemble9[kk, :, :, :, :] = perm_big

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
        to_absolute_path("../PACKETS/simulations.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_saturation],
        n_examples=cfg.custom.ntrain,
    )
    invar_test, outvar_test_pressure, outvar_test_saturation = load_FNO_dataset2(
        to_absolute_path("../PACKETS/simulations.hdf5"),
        [k.name for k in input_keys],
        [k.name for k in output_keys_pressure],
        [k.name for k in output_keys_saturation],
        n_examples=cfg.custom.ntest,
    )

    input_keys1 = [
        Key("perm"),
        Key("Q"),
        Key("Qw"),
        Key("Phi"),
        Key("Time"),
        Key("Pini"),
        Key("Swini"),
    ]
    output_keys_pressure1 = [Key("pressure")]

    output_keys_saturation1 = [Key("water_sat")]

    invar_train1, outvar_train1_pressure, outvar_train1_saturation = load_FNO_dataset2(
        to_absolute_path("../PACKETS/single.hdf5"),
        [k.name for k in input_keys1],
        [k.name for k in output_keys_pressure1],
        [k.name for k in output_keys_saturation1],
        n_examples=1,
    )

    train_dataset_pressure = DictGridDataset(invar_train, outvar_train_pressure)
    train_dataset_saturation = DictGridDataset(invar_train, outvar_train_saturation)

    test_dataset_pressure = DictGridDataset(invar_test, outvar_test_pressure)
    test_dataset_saturation = DictGridDataset(invar_test, outvar_test_saturation)

    train_dataset1_pressure = DictGridDataset(invar_train1, outvar_train1_pressure)
    train_dataset1_saturation = DictGridDataset(invar_train1, outvar_train1_saturation)

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

    supervised1_pressure = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset1_pressure,
        batch_size=1,
    )
    domain.add_constraint(supervised1_pressure, "supervised1_pressure")

    supervised1_saturation = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset1_saturation,
        batch_size=1,
    )
    domain.add_constraint(supervised1_saturation, "supervised1_saturation")

    # [constraint]
    # add validator
    val_pressure = GridValidator(
        nodes,
        dataset=train_dataset1_pressure,
        batch_size=1,
        plotter=CustomValidatorPlotterP(
            timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt
        ),
        requires_grad=False,
    )
    domain.add_validator(val_pressure, "val_pressure")

    val_saturation = GridValidator(
        nodes,
        dataset=train_dataset1_saturation,
        batch_size=1,
        plotter=CustomValidatorPlotterS(
            timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt
        ),
        requires_grad=False,
    )
    domain.add_validator(val_saturation, "val_saturation")

    test_pressure = GridValidator(
        nodes,
        dataset=test_dataset_pressure,
        batch_size=cfg.batch_size.test,
        plotter=CustomValidatorPlotterP(
            timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt
        ),
        requires_grad=False,
    )
    domain.add_validator(test_pressure, "test_pressure")

    test_saturation = GridValidator(
        nodes,
        dataset=test_dataset_saturation,
        batch_size=cfg.batch_size.test,
        plotter=CustomValidatorPlotterS(
            timmee, max_t, MAXZ, pini_alt, nx, ny, wells, steppi, tc2, dt
        ),
        requires_grad=False,
    )
    domain.add_validator(test_saturation, "test_saturation")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
