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
import warnings
import sys
import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from sympy import Symbol, Eq, Heaviside, sqrt
from sympy.logic.boolalg import And
import numpy as np
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_3d import Box
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from physicsnemo.sym.domain.inferencer import VoxelInferencer
from physicsnemo.sym.utils.io.plotter import InferencerPlotter
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.electromagnetic import PEC, SommerfeldBC, MaxwellFreqReal

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # params for domain
    length = 1
    height = 1
    width = 1

    len_slab = 0.2
    eps0 = 1.0
    eps1 = 1.5
    eps_sympy = sqrt(
        eps0
        + (Heaviside(y + len_slab / 2) - Heaviside(y - len_slab / 2))
        * (Heaviside(z + len_slab / 2) - Heaviside(z - len_slab / 2))
        * (eps1 - eps0)
    )

    wave_number = 16.0  # wave_number = freq/c
    file_path = "../validation/2Dwaveguideport.csv"
    if not os.path.exists(to_absolute_path(file_path)):
        warnings.warn(
            f"Directory {file_path} does not exist. Cannot continue. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )
        sys.exit()

    mapping = {"x": "x", "y": "y", **{"u" + str(k): "u" + str(k) for k in range(6)}}
    data_var = csv_to_dict(
        to_absolute_path("../validation/2Dwaveguideport.csv"), mapping
    )
    waveguide_port_invar_numpy = {
        "x": np.zeros_like(data_var["x"]) - 0.5,
        "y": data_var["x"],
        "z": data_var["y"],
    }
    waveguide_port_outvar_numpy = {"uz": data_var["u0"]}

    # define geometry
    rec = Box(
        (-width / 2, -length / 2, -height / 2), (width / 2, length / 2, height / 2)
    )
    # make list of nodes to unroll graph on
    hm = MaxwellFreqReal(k=wave_number * eps_sympy)
    pec = PEC()
    pml = SommerfeldBC()
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("ux"), Key("uy"), Key("uz")],
        frequencies=(
            "axis,diagonal",
            [i / 2.0 for i in range(int(wave_number * np.sqrt(eps1)) * 2 + 1)],
        ),
        frequencies_params=(
            "axis,diagonal",
            [i / 2.0 for i in range(int(wave_number * np.sqrt(eps1)) * 2 + 1)],
        ),
        cfg=cfg.arch.modified_fourier,
    )
    nodes = (
        hm.make_nodes()
        + pec.make_nodes()
        + pml.make_nodes()
        + [wave_net.make_node(name="wave_network")]
    )

    waveguide_domain = Domain()

    wall_PEC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"PEC_x": 0.0, "PEC_y": 0.0, "PEC_z": 0.0},
        batch_size=cfg.batch_size.PEC,
        lambda_weighting={"PEC_x": 100.0, "PEC_y": 100.0, "PEC_z": 100.0},
        criteria=And(~Eq(x, -width / 2), ~Eq(x, width / 2)),
        fixed_dataset=False,
    )

    waveguide_domain.add_constraint(wall_PEC, "PEC")

    Waveguide_port = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=waveguide_port_invar_numpy,
        outvar=waveguide_port_outvar_numpy,
        batch_size=cfg.batch_size.Waveguide_port,
        lambda_weighting={"uz": np.full_like(waveguide_port_invar_numpy["x"], 0.5)},
    )
    waveguide_domain.add_constraint(Waveguide_port, "Waveguide_port")

    ABC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "SommerfeldBC_real_x": 0.0,
            "SommerfeldBC_real_y": 0.0,
            "SommerfeldBC_real_z": 0.0,
        },
        batch_size=cfg.batch_size.ABC,
        lambda_weighting={
            "SommerfeldBC_real_x": 10.0,
            "SommerfeldBC_real_y": 10.0,
            "SommerfeldBC_real_z": 10.0,
        },
        criteria=Eq(x, width / 2),
        fixed_dataset=False,
    )
    waveguide_domain.add_constraint(ABC, "ABC")

    Interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "Maxwell_Freq_real_x": 0,
            "Maxwell_Freq_real_y": 0.0,
            "Maxwell_Freq_real_z": 0.0,
        },
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "Maxwell_Freq_real_x": 1.0 / wave_number**2,
            "Maxwell_Freq_real_y": 1.0 / wave_number**2,
            "Maxwell_Freq_real_z": 1.0 / wave_number**2,
        },
        fixed_dataset=False,
    )
    waveguide_domain.add_constraint(Interior, "Interior")

    # add inferencer data
    slab_inference = VoxelInferencer(
        bounds=[[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
        npoints=[128, 128, 128],
        nodes=nodes,
        output_names=["ux", "uy", "uz"],
    )
    waveguide_domain.add_inferencer(slab_inference, "Inf" + str(int(wave_number)))

    # make solver
    slv = Solver(cfg, waveguide_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
