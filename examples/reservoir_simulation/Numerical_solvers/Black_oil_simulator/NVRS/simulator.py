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

from __future__ import print_function

print(__doc__)
import os
from NVR import *
import pandas as pd
import scipy.io as sio
import datetime
from PIL import Image
from joblib import Parallel, delayed
import re

oldfolder = os.getcwd()
os.chdir(oldfolder)

text = """
                                                                                       
NNNNNNNN        NNNNNNNVVVVVVVV           VVVVVVVRRRRRRRRRRRRRRRRR     SSSSSSSSSSSSSSS 
N:::::::N       N::::::V::::::V           V::::::R::::::::::::::::R  SS:::::::::::::::S
N::::::::N      N::::::V::::::V           V::::::R::::::RRRRRR:::::RS:::::SSSSSS::::::S
N:::::::::N     N::::::V::::::V           V::::::RR:::::R     R:::::S:::::S     SSSSSSS
N::::::::::N    N::::::NV:::::V           V:::::V  R::::R     R:::::S:::::S            
N:::::::::::N   N::::::N V:::::V         V:::::V   R::::R     R:::::S:::::S            
N:::::::N::::N  N::::::N  V:::::V       V:::::V    R::::RRRRRR:::::R S::::SSSS         
N::::::N N::::N N::::::N   V:::::V     V:::::V     R:::::::::::::RR   SS::::::SSSSS    
N::::::N  N::::N:::::::N    V:::::V   V:::::V      R::::RRRRRR:::::R    SSS::::::::SS  
N::::::N   N:::::::::::N     V:::::V V:::::V       R::::R     R:::::R      SSSSSS::::S 
N::::::N    N::::::::::N      V:::::V:::::V        R::::R     R:::::R           S:::::S
N::::::N     N:::::::::N       V:::::::::V         R::::R     R:::::R           S:::::S
N::::::N      N::::::::N        V:::::::V        RR:::::R     R:::::SSSSSSS     S:::::S
N::::::N       N:::::::N         V:::::V         R::::::R     R:::::S::::::SSSSSS:::::S
N::::::N        N::::::N          V:::V          R::::::R     R:::::S:::::::::::::::SS 
NNNNNNNN         NNNNNNN           VVV           RRRRRRRR     RRRRRRRSSSSSSSSSSSSSSS
"""
print(text)


def inference_single(
    ini,
    inip,
    nx,
    ny,
    nz,
    max_t,
    Dx,
    Dy,
    Dz,
    BO,
    BW,
    CFL,
    timmee,
    MAXZ,
    factorr,
    steppi,
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
    step2,
    pini_alt,
    INJ,
    PROD,
    opennI,
    opennP,
):
    paramss = ini
    Ne = paramss.shape[1]

    if nz == 1:
        ct = np.zeros((Ne, 4, nx, ny), dtype=np.float32)
        cP = np.zeros((Ne, 2 * steppi, nx, ny), dtype=np.float32)
    else:
        ct = np.zeros((Ne, 4, nx, ny, nz), dtype=np.float32)
        cP = np.zeros((Ne, 2 * steppi, nx, ny, nz), dtype=np.float32)

    Ainj = np.zeros((nx, ny, nz))
    Aprod = np.zeros((nx, ny, nz))

    n_inj = len(INJ)  # Number of injectors
    n_prod = len(PROD)  # Number of producers
    for kk in range(Ne):
        if nz == 1:
            ct1 = np.zeros((4, nx, ny), dtype=np.float32)
        else:
            ct1 = np.zeros((4, nx, ny, nz), dtype=np.float32)
        print(str(kk + 1) + " | " + str(Ne))
        a = np.reshape(paramss[:, kk], (nx, ny, nz), "F")

        at1 = ini[:, kk]
        # at1 = rescale_linear_numpy_pytorch(at1, LUB, HUB,aay,bby)

        at2 = inip[:, kk]
        # at2 = rescale_linear(at2, mpor, hpor)

        at1 = np.reshape(at1, (nx, ny, nz), "F")
        at2 = np.reshape(at2, (nx, ny, nz), "F")

        if opennI == "all":
            for ay in range(n_inj):
                use = np.array(INJ[ay])
                for jj in range(nz):
                    xloc = int(use[0])
                    yloc = int(use[1])
                    Ainj[xloc, yloc, jj] = float(use[3])

        else:
            for ay in range(n_inj):
                use = np.array(INJ[ay])
                xloc = int(use[0])
                yloc = int(use[1])
                Ainj[xloc, yloc, -1] = float(use[3])

        if opennP == "all":
            for ay in range(n_prod):
                use = np.array(PROD[ay])
                for jj in range(nz):
                    xloc = int(use[0])
                    yloc = int(use[1])
                    Aprod[xloc, yloc, jj] = float(use[3])
        else:
            for ay in range(n_prod):
                use = np.array(PROD[ay])
                xloc = int(use[0])
                yloc = int(use[1])
                Aprod[xloc, yloc, -1] = float(use[3])

        quse1 = Ainj + Aprod
        quse = np.reshape(quse1, (-1, 1), "F")
        quse_water = np.reshape(Ainj, (-1, 1), "F")

        # print(A)
        A_out = Reservoir_Simulator(
            a,
            at2,
            quse.ravel(),
            quse_water.ravel(),
            nx,
            ny,
            nz,
            factorr,
            max_t,
            Dx,
            Dy,
            Dz,
            BO,
            BW,
            CFL,
            timmee,
            MAXZ,
            PB,
            PATM,
            CFO,
            IWSw,
            method,
            steppi,
            SWI,
            SWR,
            UW,
            UO,
            step2,
            pini_alt,
        )
        if nz == 1:
            ct1[0, :, :] = at1[:, :, 0]  # permeability
            ct1[1, :, :] = quse1[:, :, 0]  # /UIR # Overall f
            ct1[2, :, :] = Ainj[:, :, 0]  # /UIR# f for water injection
            ct1[3, :, :] = at2[:, :, 0]  # porosity
        else:
            ct1[0, :, :, :] = at1  # permeability
            ct1[1, :, :, :] = quse1  # /UIR # Overall f
            ct1[2, :, :, :] = Ainj  # /UIR# f for water injection
            ct1[3, :, :, :] = at2  # porosity

        if nz == 1:
            cP[kk, :, :, :] = A_out
            ct[kk, :, :, :] = ct1
        else:
            cP[kk, :, :, :, :] = A_out
            ct[kk, :, :, :, :] = ct1

        return ct, cP


def sort_key(s):
    """Extract the number from the filename for sorting."""
    return int(re.search(r"\d+", s).group())


def inference_single2(
    ini,
    inip,
    nx,
    ny,
    nz,
    max_t,
    Dx,
    Dy,
    Dz,
    BO,
    BW,
    BG,
    RS,
    CFL,
    timmee,
    MAXZ,
    factorr,
    steppi,
    aay,
    bby,
    mpor,
    hpor,
    dt,
    IWSw,
    IWSg,
    PB,
    PATM,
    CFO,
    method,
    SWI,
    SWR,
    UW,
    UO,
    UG,
    step2,
    pini_alt,
    INJ,
    PROD,
    opennI,
    opennP,
    SWOW,
    SWOG,
):
    paramss = ini
    Ne = paramss.shape[1]

    if nz == 1:
        ct = np.zeros((Ne, 4, nx, ny), dtype=np.float32)
        cP = np.zeros((Ne, 4 * steppi, nx, ny), dtype=np.float32)
    else:
        ct = np.zeros((Ne, 4, nx, ny, nz), dtype=np.float32)
        cP = np.zeros((Ne, 4 * steppi, nx, ny, nz), dtype=np.float32)

    Ainj = np.zeros((nx, ny, nz))
    Aprod = np.zeros((nx, ny, nz))

    n_inj = len(INJ)  # Number of injectors
    n_prod = len(PROD)  # Number of producers
    for kk in range(Ne):
        if nz == 1:
            ct1 = np.zeros((4, nx, ny), dtype=np.float32)
        else:
            ct1 = np.zeros((4, nx, ny, nz), dtype=np.float32)
        print(str(kk + 1) + " | " + str(Ne))
        a = np.reshape(paramss[:, kk], (nx, ny, nz), "F")

        at1 = ini[:, kk]
        # at1 = rescale_linear_numpy_pytorch(at1, LUB, HUB,aay,bby)

        at2 = inip[:, kk]
        # at2 = rescale_linear(at2, mpor, hpor)

        at1 = np.reshape(at1, (nx, ny, nz), "F")
        at2 = np.reshape(at2, (nx, ny, nz), "F")

        if opennI == "all":
            for ay in range(n_inj):
                use = np.array(INJ[ay])
                for jj in range(nz):
                    xloc = int(use[0])
                    yloc = int(use[1])
                    Ainj[xloc, yloc, jj] = float(use[3])

        else:
            for ay in range(n_inj):
                use = np.array(INJ[ay])
                xloc = int(use[0])
                yloc = int(use[1])
                Ainj[xloc, yloc, -1] = float(use[3])

        if opennP == "all":
            for ay in range(n_prod):
                use = np.array(PROD[ay])
                for jj in range(nz):
                    xloc = int(use[0])
                    yloc = int(use[1])
                    Aprod[xloc, yloc, jj] = float(use[3])
        else:
            for ay in range(n_prod):
                use = np.array(PROD[ay])
                xloc = int(use[0])
                yloc = int(use[1])
                Aprod[xloc, yloc, -1] = float(use[3])

        quse1 = Ainj + Aprod

        quse = np.reshape(quse1, (-1, 1), "F")
        quse_water = np.reshape(Ainj, (-1, 1), "F")
        quse_oil = np.reshape(Aprod, (-1, 1), "F")
        # print(A)
        A_out = Reservoir_Simulator2(
            a,
            at2,
            quse.ravel(),
            quse_water.ravel(),
            quse_oil.ravel(),
            nx,
            ny,
            nz,
            factorr,
            max_t,
            Dx,
            Dy,
            Dz,
            BO,
            BW,
            BG,
            RS,
            CFL,
            timmee,
            MAXZ,
            PB,
            PATM,
            CFO,
            IWSw,
            IWSg,
            method,
            steppi,
            SWI,
            SWR,
            UW,
            UO,
            UG,
            step2,
            pini_alt,
            SWOW,
            SWOG,
        )
        if nz == 1:
            ct1[0, :, :] = at1[:, :, 0]  # permeability
            ct1[1, :, :] = quse1[:, :, 0]  # /UIR # Overall f
            ct1[2, :, :] = Ainj[:, :, 0]  # /UIR# f for water injection
            ct1[3, :, :] = at2[:, :, 0]  # porosity
        else:
            ct1[0, :, :, :] = at1  # permeability
            ct1[1, :, :, :] = quse1  # /UIR # Overall f
            ct1[2, :, :, :] = Ainj  # /UIR# f for water injection
            ct1[3, :, :, :] = at2  # porosity

        if nz == 1:
            cP[kk, :, :, :] = A_out
            ct[kk, :, :, :] = ct1
        else:
            cP[kk, :, :, :, :] = A_out
            ct[kk, :, :, :, :] = ct1

        return ct, cP


class Simulator:
    def __init__(self, fname) -> None:
        self.plan = read_yaml(fname)

    def run(self):
        plan = self.plan

        N_components = int(plan["N_components"])
        nx = int(plan["DIMENS"]["nx"])
        ny = int(plan["DIMENS"]["ny"])
        nz = int(plan["DIMENS"]["nz"])
        BO = float(plan["PROPS"]["BO"])
        BW = float(plan["PROPS"]["BW"])
        UW = float(plan["PROPS"]["UW"])
        UO = float(plan["PROPS"]["UO"])
        DX = float(plan["GRID"]["DX"])
        DY = float(plan["GRID"]["DY"])
        DZ = float(plan["GRID"]["DZ"])
        opennI = str(plan["SCHEDULE"]["opennI"])
        opennP = str(plan["SCHEDULE"]["opennP"])

        DX = cp.float32(DX)
        DY = cp.float32(DY)
        UW = cp.float32(UW)
        UO = cp.float32(UO)
        SWI = cp.float32(plan["PROPS"]["SWI"])
        SWR = cp.float32(plan["PROPS"]["SWR"])
        CFO = cp.float32(plan["PROPS"]["CFO"])
        IWSw = float(plan["PROPS"]["S1"])
        float(plan["PROPS"]["SO1"])
        IWSg = float(plan["PROPS"]["SG1"])
        pini_alt = float(plan["PROPS"]["P1"])
        P1 = cp.float32(pini_alt)
        step2 = int(plan["NSTACK"]["n_substep"])
        PB = P1
        mpor, hpor = float(plan["MPOR"]), float(plan["HPOR"])
        aay, bby = float(plan["aay"]), float(plan["bby"])
        _Low_K, _High_K = aay, bby
        BW = cp.float32(BW)
        BO = cp.float32(BO)
        PATM = cp.float32(float(plan["PROPS"]["PATM"]))

        method = int(plan["SOLVER"]["solution_method"])
        make_animation = int(plan["SUMMARY"]["make_animation"])
        make_csv = int(plan["SUMMARY"]["make_csv"])
        make_gassman = int(plan["SUMMARY"]["make_gassman"])
        path_trueK = plan["INCLUDE"]["path_trueK"]
        path_out = plan["path_out"]

        if N_components == 3:
            cp.float32(0)  # Initial gas saturation
        if N_components == 3:
            UG = cp.float32(calc_mu_g(P1))  # gas viscosity in cP
            RS = cp.float32(calc_rs(PB, P1))  # Solution GOR
            BG = calc_bg(PB, PATM, P1)  # gas formation volume factor

        SWOW = np.array(np.vstack(plan["PROPS"]["SWOW"]), dtype=float)
        SWOG = np.array(np.vstack(plan["PROPS"]["SWOG"]), dtype=float)

        # Ug = 0.02 # Gas viscosity
        # Bg = 0.001 # Gas formation volume factor

        num_cores = multiprocessing.cpu_count()
        print(
            "\nThis computer has %d cores, which will all be utilised in parallel\n"
            % num_cores
        )

        print_section_title("Begin the main code")
        print(str(datetime.datetime.now()))
        # Creating formatted string
        if method == 1:
            mss1 = "GMRES"
        elif method == 2:
            mss1 = "spsolve"
        elif method == 3:
            mss1 == "conjugate gradient"
        elif method == 4:
            mss1 = "LSQR"
        else:
            mss1 = "CPR"
        message1 = f"Number of components = {N_components}\nNx = {nx}\nNy = {ny}\nNz = {nz}\nLinear Solver = {mss1}"

        # Print the message
        print(message1)

        print(f"Create {path_out}")
        os.makedirs(path_out, exist_ok=True)

        timmee = float(plan["REPORTING"]["time_step"])
        max_t = float(plan["REPORTING"]["time_max"])
        MAXZ = float(plan["REPORTING"]["time_limit"])
        steppi = int(max_t / timmee)

        factorr = float(plan["MULTIPLY"]["z_factor"])
        0.2 * DX

        rwell = 200  # well radius
        skin = 0  # well deformation
        pwf_producer = 100

        N_inj = len(plan["WELLSPECS"]["injector_wells"])  # Number of injectors
        N_pr = len(plan["WELLSPECS"]["producer_wells"])  # Number of producers

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
        if N_components == 3:
            CFL = 1
        if (nx * ny * nz) >= 10000:
            CFL = 1
        tc2 = Equivalent_time(timmee, MAXZ, timmee, max_t)
        dt = np.diff(tc2)[0]  # Time-step

        # CFL = 1
        len(plan["WELLSPECS"]["injector_wells"])  # Number of injectors
        len(plan["WELLSPECS"]["producer_wells"])  # Number of producers

        injectors = plan["WELLSPECS"]["injector_wells"]
        producers = plan["WELLSPECS"]["producer_wells"]

        wellsI = np.array(np.vstack(injectors)[:, :3], dtype=float)
        wellsI[:, 2] = 1
        wellsP = np.array(np.vstack(producers)[:, :3], dtype=float)
        wellsP[:, 2] = 2
        wells = np.vstack([wellsI, wellsP])

        # NecessaryI = np.vstack(injectors)[:,4:7]
        # NecessaryP = np.vstack(producers)[:,4:7]

        NecessaryI = np.array(np.vstack(injectors)[:, 4:7], dtype=float)
        NecessaryP = np.array(np.vstack(producers)[:, 4:7], dtype=float)

        MinMaxScaler(feature_range=(aay, bby))
        if CFL == 1:
            print_section_title("Begin IMPES simulation")
        if CFL == 2:
            print_section_title("Begin Fully Implicit simulation")
        Truee = np.genfromtxt(path_trueK, dtype="float")

        # Truee = np.reshape(Truee,(120,60,10))

        os.chdir(path_out)
        Truee = rescale_linear(Truee, aay, bby)
        Pressz = np.reshape(np.log10(Truee), (nx, ny, nz), "F")
        maxii = max(Pressz.ravel())
        minii = min(Pressz.ravel())
        Pressz = Pressz / maxii
        plot3d2static(
            Pressz,
            nx,
            ny,
            nz,
            "Permeability",
            "Permeability",
            maxii,
            minii,
            injectors,
            producers,
        )
        os.chdir(oldfolder)

        Ne = 1
        ini = []
        inip = []
        # injj = 500 * np.ones((1,4))
        for i in range(Ne):
            at1 = rescale_linear(Truee, aay, bby)
            at2 = rescale_linear(Truee, mpor, hpor)
            ini.append(at1.reshape(-1, 1))
            inip.append(at2.reshape(-1, 1))

        ini = np.hstack(ini)
        inip = np.hstack(inip)
        start_time_plots1 = time.time()

        if N_components == 2:
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
                step2,
                pini_alt,
                injectors,
                producers,
                opennI,
                opennP,
            )
            os.chdir(path_out)
            sio.savemat(
                "UNRST.mat",
                {
                    "permeability": X_data1[0, 0, :, :][:, :, None],
                    "porosity": X_data1[0, 3, :, :][:, :, None],
                    "Pressure": data_use1[0, :steppi, :, :][:, :, None],
                    "Water_saturation": data_use1[0, steppi:, :, :][:, :, None],
                    "Oil_saturation": 1 - data_use1[0, steppi:, :, :][:, :, None],
                },
                do_compression=True,
            )
            os.chdir(oldfolder)
        else:
            X_data1, data_use1 = inference_single2(
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
                BG,
                RS,
                CFL,
                timmee,
                MAXZ,
                factorr,
                steppi,
                aay,
                bby,
                mpor,
                hpor,
                dt,
                IWSw,
                IWSg,
                PB,
                PATM,
                CFO,
                method,
                SWI,
                SWR,
                UW,
                UO,
                UG,
                step2,
                pini_alt,
                injectors,
                producers,
                opennI,
                opennP,
                SWOW,
                SWOG,
            )
            os.chdir(path_out)
            sio.savemat(
                "UNRST.mat",
                {
                    "permeability": X_data1[0, 0, :, :, :],
                    "porosity": X_data1[0, 3, :, :, :],
                    "Pressure": data_use1[0, :steppi, :, :, :],
                    "Water_saturation": data_use1[0, steppi : 2 * steppi, :, :, :],
                    "Oil_saturation": data_use1[0, 2 * steppi : 3 * steppi, :, :, :],
                    "Gas_saturation": data_use1[0, 3 * steppi :, :, :, :],
                },
                do_compression=True,
            )
            os.chdir(oldfolder)

        elapsed_time_secs = time.time() - start_time_plots1
        msg = "Reservoir simulation  took: %s secs (Wall clock time)" % timedelta(
            seconds=round(elapsed_time_secs)
        )
        print(msg)
        print("")
        print("Finished FVM simulations")

        if N_components == 2:
            if make_animation:
                start_time_plots = time.time()
                print("")
                print("Plotting outputs")
                os.chdir(path_out)
                Runs = steppi
                np.arange(1, Runs + 1)

                if nz == 1:
                    # for kk in range(steppi):

                    Parallel(n_jobs=-1)(
                        delayed(Plot_performance)(
                            data_use1[0, :, :, :],
                            nx,
                            ny,
                            "new",
                            kk,
                            dt,
                            MAXZ,
                            steppi,
                            wells,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk, :, :][:, :, None]
                            / max((data_use1[0, kk, :, :][:, :, None]).ravel()),
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_pressure",
                            "Pressure",
                            max((data_use1[0, kk, :, :][:, :, None]).ravel()),
                            min((data_use1[0, kk, :, :][:, :, None]).ravel()),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk + steppi, :, :][:, :, None],
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_water",
                            "water_sat",
                            max((data_use1[0, kk + steppi, :, :][:, :, None]).ravel()),
                            min((data_use1[0, kk + steppi, :, :][:, :, None]).ravel()),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            (1 - data_use1[0, kk + steppi, :, :][:, :, None]),
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_oil",
                            "oil_sat",
                            max(
                                (
                                    1 - data_use1[0, kk + steppi, :, :][:, :, None]
                                ).ravel()
                            ),
                            min(
                                (
                                    1 - data_use1[0, kk + steppi, :, :][:, :, None]
                                ).ravel()
                            ),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    progressBar = "\rPlotting Progress: " + ProgressBar(
                        Runs - 1, Runs - 1, Runs - 1
                    )
                    ShowBar(progressBar)
                    time.sleep(1)

                    print("")
                    print("Now - Creating GIF")
                    import glob

                    frames1 = []
                    imgs = sorted(glob.glob("*new*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_pressure*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_water*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_oil*"), key=sort_key)
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

                    for f4 in glob("*new*"):
                        os.remove(f4)

                    for f4 in glob("*unie_pressure*"):
                        os.remove(f4)

                    for f4 in glob("*unie_water*"):
                        os.remove(f4)

                    for f4 in glob("*unie_oil*"):
                        os.remove(f4)
                else:
                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk, :, :, :][:, :, ::-1]
                            / max((data_use1[0, kk, :, :, :][:, :, ::-1]).ravel()),
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_pressure",
                            "Pressure",
                            max((data_use1[0, kk, :, :, :][:, :, ::-1]).ravel()),
                            min((data_use1[0, kk, :, :, :][:, :, ::-1]).ravel()),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk + steppi, :, :, :][:, :, ::-1],
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_water",
                            "water_sat",
                            max(
                                (data_use1[0, kk + steppi, :, :, :][:, :, ::-1]).ravel()
                            ),
                            min(
                                (data_use1[0, kk + steppi, :, :, :][:, :, ::-1]).ravel()
                            ),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            1 - data_use1[0, kk + steppi, :, :, :][:, :, ::-1],
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_oil",
                            "oil_sat",
                            max(
                                (
                                    1 - data_use1[0, kk + steppi, :, :, :][:, :, ::-1]
                                ).ravel()
                            ),
                            min(
                                (
                                    1 - data_use1[0, kk + steppi, :, :, :][:, :, ::-1]
                                ).ravel()
                            ),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    progressBar = "\rPlotting Progress: " + ProgressBar(
                        Runs - 1, Runs - 1, Runs - 1
                    )
                    ShowBar(progressBar)
                    time.sleep(1)

                    print("")
                    print("Now - Creating GIF")
                    import glob

                    frames = []
                    imgs = sorted(glob.glob("*unie_pressure*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_water*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_oil*"), key=sort_key)
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

                    for f4 in glob("*unie_pressure*"):
                        os.remove(f4)

                    for f4 in glob("*unie_water*"):
                        os.remove(f4)

                    for f4 in glob("*unie_oil*"):
                        os.remove(f4)

            os.chdir(oldfolder)
            elapsed_time_secs = time.time() - start_time_plots
            msg = (
                "pressure and saturation plots took: %s secs (Wall clock time)"
                % timedelta(seconds=round(elapsed_time_secs))
            )
            print(msg)

            if make_gassman:
                start_time_gassman = time.time()
                print("")
                print("Plotting outputs")
                os.chdir(path_out)
                Runs = steppi
                np.arange(1, Runs + 1)

                if nz == 1:
                    # for kk in range(steppi):
                    # progressBar = "\rPlotting Progress: " + ProgressBar(Runs-1, kk-1, Runs-1)
                    # ShowBar(progressBar)
                    # time.sleep(1)

                    # Pressz = data_use1[0,kk,:,:][:,:,None]
                    # watsz = data_use1[0,kk + steppi,:,:][:,:,None]
                    # oilsz = (( 1- data_use1[0,kk + steppi,:,:][:,:,None]))

                    # PORO = np.reshape(inip,(nx,ny,nz),'F')
                    # Ip =Gassmann(cp.asarray(np.reshape(inip,(nx,ny,nz),'F'),) ,cp.asarray(data_use1[0,kk,:,:][:,:,None]),
                    # cp.asarray((( 1- data_use1[0,kk + steppi,:,:][:,:,None]))),nx,ny,nz)

                    Parallel(n_jobs=-1)(
                        delayed(Plot_impedance)(
                            (
                                Gassmann(
                                    cp.asarray(
                                        np.reshape(inip, (nx, ny, nz), "F"),
                                    ),
                                    cp.asarray(data_use1[0, kk, :, :][:, :, None]),
                                    cp.asarray(
                                        (
                                            1
                                            - data_use1[0, kk + steppi, :, :][
                                                :, :, None
                                            ]
                                        )
                                    ),
                                    nx,
                                    ny,
                                    nz,
                                )
                            ).get(),
                            nx,
                            ny,
                            "newI",
                            kk,
                            dt,
                            MAXZ,
                            steppi,
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    progressBar = "\rPlotting Progress: " + ProgressBar(
                        Runs - 1, Runs - 1, Runs - 1
                    )
                    ShowBar(progressBar)
                    time.sleep(1)

                    print("")
                    print("Now - Creating GIF")
                    import glob

                    frames1 = []
                    imgs = sorted(glob.glob("*newI*"), key=sort_key)
                    for i in imgs:
                        new_frame = Image.open(i)
                        frames1.append(new_frame)

                    frames1[0].save(
                        "Evolution2_impededance.gif",
                        format="GIF",
                        append_images=frames1[1:],
                        save_all=True,
                        duration=500,
                        loop=0,
                    )

                    from glob import glob

                    for f4 in glob("*newI*"):
                        os.remove(f4)

                else:
                    # for kk in range(steppi):
                    # progressBar = "\rPlotting Progress: " + ProgressBar(Runs-1, kk-1, Runs-1)
                    # ShowBar(progressBar)
                    # time.sleep(1)

                    # Pressz = data_use1[0,kk,:,:,:]
                    # watsz = data_use1[0,kk + steppi,:,:,:]
                    # oilsz = (( 1- data_use1[0,kk + steppi,:,:,:]))

                    # PORO = np.reshape(inip,(nx,ny,nz),'F')

                    # Ip = Gassmann(cp.asarray(np.reshape(inip,(nx,ny,nz),'F'),) ,
                    # cp.asarray(data_use1[0,kk,:,:,:]),
                    # cp.asarray((( 1- data_use1[0,kk + steppi,:,:,:]))),nx,ny,nz)

                    Parallel(n_jobs=-1)(
                        delayed(Plot_impedance)(
                            (
                                Gassmann(
                                    cp.asarray(
                                        np.reshape(inip, (nx, ny, nz), "F"),
                                    ),
                                    cp.asarray(data_use1[0, kk, :, :, :]),
                                    cp.asarray(
                                        (1 - data_use1[0, kk + steppi, :, :, :])
                                    ),
                                    nx,
                                    ny,
                                    nz,
                                )
                            ).get(),
                            nx,
                            ny,
                            "newI",
                            kk,
                            dt,
                            MAXZ,
                            steppi,
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    progressBar = "\rPlotting Progress: " + ProgressBar(
                        Runs - 1, Runs - 1, Runs - 1
                    )
                    ShowBar(progressBar)
                    time.sleep(1)

                    print("")
                    print("Now - Creating GIF")
                    import glob

                    frames = []
                    imgs = sorted(glob.glob("*newI*"), key=sort_key)
                    for i in imgs:
                        new_frame = Image.open(i)
                        frames.append(new_frame)

                    frames[0].save(
                        "Evolution2_impededance.gif",
                        format="GIF",
                        append_images=frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0,
                    )

                    from glob import glob

                    for f4 in glob("*newI*"):
                        os.remove(f4)

            os.chdir(oldfolder)
            elapsed_time_secs = time.time() - start_time_gassman
            msg = "Seismic plots took: %s secs (Wall clock time)" % timedelta(
                seconds=round(elapsed_time_secs)
            )
            print(msg)

            if make_csv:
                start_time_peaceman = time.time()
                os.chdir(path_out)
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

                if nz == 1:
                    seeTrue = Peaceman_well(
                        X_data1,
                        data_use1[:, :steppi, :, :],
                        data_use1[:, steppi:, :, :],
                        MAXZ,
                        0,
                        1e1,
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
                        NecessaryI,
                        NecessaryP,
                    )
                else:
                    seeTrue = Peaceman_well(
                        X_data1,
                        data_use1[:, :steppi, :, :, :],
                        data_use1[:, steppi:, :, :, :],
                        MAXZ,
                        0,
                        1e1,
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
                        NecessaryI,
                        NecessaryP,
                    )

                seeuset = pd.DataFrame(seeTrue)
                seeuset.to_csv("RSM_FVM.csv", header=spittsbig, sep=",")
                seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

                Plot_RSM_percentile(seeTrue, "Compare.png")

                os.chdir(oldfolder)
                elapsed_time_secs = time.time() - start_time_peaceman
                msg = (
                    "Well rates and pressure computation took: %s secs (Wall clock time)"
                    % timedelta(seconds=round(elapsed_time_secs))
                )

        else:  # 3phase flow
            if make_animation:
                start_time_plots = time.time()
                print("")
                print("Plotting outputs")
                os.chdir(path_out)
                Runs = steppi
                np.arange(1, Runs + 1)

                if nz == 1:
                    # for kk in range(steppi):

                    Parallel(n_jobs=-1)(
                        delayed(Plot_performance2)(
                            data_use1[0, :, :, :],
                            nx,
                            ny,
                            "new",
                            kk,
                            dt,
                            MAXZ,
                            steppi,
                            wells,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk, :, :][:, :, None]
                            / max((data_use1[0, kk, :, :][:, :, None]).ravel()),
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_pressure",
                            "Pressure",
                            max((data_use1[0, kk, :, :][:, :, None]).ravel()),
                            min((data_use1[0, kk, :, :][:, :, None]).ravel()),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk + steppi, :, :][:, :, None],
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_water",
                            "water_sat",
                            max((data_use1[0, kk + steppi, :, :][:, :, None]).ravel()),
                            min((data_use1[0, kk + steppi, :, :][:, :, None]).ravel()),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk + 2 * steppi, :, :][:, :, None],
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_oil",
                            "oil_sat",
                            max(
                                (
                                    data_use1[0, kk + 2 * steppi, :, :][:, :, None]
                                ).ravel()
                            ),
                            min(
                                (
                                    data_use1[0, kk + 2 * steppi, :, :][:, :, None]
                                ).ravel()
                            ),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            (data_use1[0, kk + 3 * steppi, :, :][:, :, None]),
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_gas",
                            "gas_sat",
                            max(
                                (
                                    data_use1[0, kk + 3 * steppi, :, :][:, :, None]
                                ).ravel()
                            ),
                            min(
                                (
                                    data_use1[0, kk + 3 * steppi, :, :][:, :, None]
                                ).ravel()
                            ),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    progressBar = "\rPlotting Progress: " + ProgressBar(
                        Runs - 1, Runs - 1, Runs - 1
                    )
                    ShowBar(progressBar)
                    time.sleep(1)

                    print("")
                    print("Now - Creating GIF")
                    import glob

                    frames1 = []
                    imgs = sorted(glob.glob("*new*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_pressure*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_water*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_oil*"), key=sort_key)
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

                    frames = []
                    imgs = sorted(glob.glob("*unie_gas*"), key=sort_key)
                    for i in imgs:
                        new_frame = Image.open(i)
                        frames.append(new_frame)

                    frames[0].save(
                        "Evolution_gas_3D.gif",
                        format="GIF",
                        append_images=frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0,
                    )

                    from glob import glob

                    for f3 in glob("*PINN_model_PyTorch*"):
                        os.remove(f3)

                    for f4 in glob("*new*"):
                        os.remove(f4)

                    for f4 in glob("*unie_pressure*"):
                        os.remove(f4)

                    for f4 in glob("*unie_water*"):
                        os.remove(f4)

                    for f4 in glob("*unie_oil*"):
                        os.remove(f4)

                    for f4 in glob("*unie_gas*"):
                        os.remove(f4)

                else:
                    # for kk in range(steppi):

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk, :, :, :][:, :, ::-1]
                            / max((data_use1[0, kk, :, :, :][:, :, ::-1]).ravel()),
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_pressure",
                            "Pressure",
                            max((data_use1[0, kk, :, :, :][:, :, ::-1]).ravel()),
                            min((data_use1[0, kk, :, :, :][:, :, ::-1]).ravel()),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk + steppi, :, :, :][:, :, ::-1],
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_water",
                            "water_sat",
                            max(
                                (data_use1[0, kk + steppi, :, :, :][:, :, ::-1]).ravel()
                            ),
                            min(
                                (data_use1[0, kk + steppi, :, :, :][:, :, ::-1]).ravel()
                            ),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk + 2 * steppi, :, :, :][:, :, ::-1],
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_oil",
                            "oil_sat",
                            max(
                                (
                                    data_use1[0, kk + 2 * steppi, :, :, :][:, :, ::-1]
                                ).ravel()
                            ),
                            min(
                                (
                                    data_use1[0, kk + 2 * steppi, :, :, :][:, :, ::-1]
                                ).ravel()
                            ),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    Parallel(n_jobs=-1)(
                        delayed(plot3d2)(
                            data_use1[0, kk + 3 * steppi, :, :, :][:, :, ::-1],
                            nx,
                            ny,
                            nz,
                            kk,
                            dt,
                            MAXZ,
                            "unie_gas",
                            "gas_sat",
                            max(
                                (
                                    data_use1[0, kk + 3 * steppi, :, :, :][:, :, ::-1]
                                ).ravel()
                            ),
                            min(
                                (
                                    data_use1[0, kk + 3 * steppi, :, :, :][:, :, ::-1]
                                ).ravel()
                            ),
                            injectors,
                            producers,
                        )
                        for kk in range(steppi)
                    )

                    progressBar = "\rPlotting Progress: " + ProgressBar(
                        Runs - 1, Runs - 1, Runs - 1
                    )
                    ShowBar(progressBar)
                    time.sleep(1)

                    print("")
                    print("Now - Creating GIF")
                    import glob

                    frames = []
                    imgs = sorted(glob.glob("*unie_pressure*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_water*"), key=sort_key)
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
                    imgs = sorted(glob.glob("*unie_oil*"), key=sort_key)
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

                    frames = []
                    imgs = sorted(glob.glob("*unie_gas*"), key=sort_key)
                    for i in imgs:
                        new_frame = Image.open(i)
                        frames.append(new_frame)

                    frames[0].save(
                        "Evolution_gas_3D.gif",
                        format="GIF",
                        append_images=frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0,
                    )

                    from glob import glob

                    for f4 in glob("*unie_pressure*"):
                        os.remove(f4)

                    for f4 in glob("*unie_water*"):
                        os.remove(f4)

                    for f4 in glob("*unie_oil*"):
                        os.remove(f4)

                    for f4 in glob("*unie_gas*"):
                        os.remove(f4)
            os.chdir(oldfolder)
            elapsed_time_secs = time.time() - start_time_plots
            msg = (
                "pressure and saturation plots took: %s secs (Wall clock time)"
                % timedelta(seconds=round(elapsed_time_secs))
            )
            print(msg)

            if make_csv:
                start_time_peaceman = time.time()
                os.chdir(path_out)
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
                    "P1 - WGPR(SCF/DAY)",
                    "P2 - WGPR(SCF/DAY)",
                    "P3 - WGPR(SCF/DAY)",
                    "P4 - WGPR(SCF/DAY)",
                    "P1 - WWCT(%)",
                    "P2 - WWCT(%)",
                    "P3 - WWCT(%)",
                    "P4 - WWCT(%)",
                ]

                if nz == 1:
                    seeTrue = Peaceman_well2(
                        X_data1,
                        data_use1[:, :steppi, :, :],
                        data_use1[:, steppi : 2 * steppi, :, :],
                        data_use1[:, 2 * steppi : 3 * steppi, :, :],
                        data_use1[:, 3 * steppi : 4 * steppi, :, :],
                        MAXZ,
                        0,
                        1e1,
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
                        UG,
                        BG,
                        pwf_producer,
                        dt,
                        N_inj,
                        N_pr,
                        nz,
                        NecessaryI,
                        NecessaryP,
                        SWOW,
                        SWOG,
                        PB,
                    )

                else:
                    seeTrue = Peaceman_well2(
                        X_data1,
                        data_use1[:, :steppi, :, :, :],
                        data_use1[:, steppi : 2 * steppi, :, :, :],
                        data_use1[:, 2 * steppi : 3 * steppi, :, :, :],
                        data_use1[:, 3 * steppi : 4 * steppi, :, :, :],
                        MAXZ,
                        0,
                        1e1,
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
                        UG,
                        BG,
                        pwf_producer,
                        dt,
                        N_inj,
                        N_pr,
                        nz,
                        NecessaryI,
                        NecessaryP,
                        SWOW,
                        SWOG,
                        PB,
                    )

                seeuset = pd.DataFrame(seeTrue)
                seeuset.to_csv("RSM_FVM.csv", header=spittsbig, sep=",")
                seeuset.drop(columns=seeuset.columns[0], axis=1, inplace=True)

                Plot_RSM_percentile2(seeTrue, "Compare.png")

                os.chdir(oldfolder)
                elapsed_time_secs = time.time() - start_time_peaceman
                msg = (
                    "Well rates and pressure computation took: %s secs (Wall clock time)"
                    % timedelta(seconds=round(elapsed_time_secs))
                )
                print(msg)

        print("")
        print("-------------------PROGRAM EXECUTED-----------------------------------")


if __name__ == "__main__":
    print(__doc__)

    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        sys.exit("Provide Yaml scenario! python NVRS/simulator.py Run.yaml")

    sim = Simulator(fname)
    sim.run()
