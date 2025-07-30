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

from sympy import Symbol, Eq, Heaviside, sqrt
from sympy.logic.boolalg import Or
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from physicsnemo.sym.domain.inferencer import VoxelInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.wave_equation import HelmholtzEquation
from physicsnemo.sym.eq.pdes.navier_stokes import GradNormal

x, y = Symbol("x"), Symbol("y")


# helper function for computing laplacian eigen values
def Laplacian_1D_eig(a, b, N, eps=lambda x: np.ones_like(x), k=3):
    n = N - 2
    h = (b - a) / (N - 1)

    L = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    L = -L / h**2

    x = np.linspace(a, b, num=N)
    M = diags([eps(x[1:-1])], [0])

    eigvals, eigvecs = eigsh(L, k=k, M=M, which="SM")
    eigvecs = np.vstack((np.zeros((1, k)), eigvecs, np.zeros((1, k))))
    norm_eigvecs = np.linalg.norm(eigvecs, axis=0)
    eigvecs /= norm_eigvecs
    return eigvals.astype(np.float32), eigvecs.astype(np.float32), x.astype(np.float32)


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # params for domain
    height = 2
    width = 2

    len_slab = 0.6
    eps0 = 1.0
    eps1 = 2.0

    def eps_numpy(y):
        return np.where(
            np.logical_and(y > (height - len_slab) / 2, y < (height + len_slab) / 2),
            eps1,
            eps0,
        )

    eps_sympy = sqrt(
        eps0
        + (
            Heaviside(y - (height - len_slab) / 2)
            - Heaviside(y - (height + len_slab) / 2)
        )
        * (eps1 - eps0)
    )
    eigvals, eigvecs, yv = Laplacian_1D_eig(0, height, 1000, eps=eps_numpy, k=3)
    yv = yv.reshape((-1, 1))
    wave_number = 16.0  # wave_number = freq/c
    waveguide_port_invar_numpy = {"x": np.zeros_like(yv), "y": yv}
    waveguide_port_outvar_numpy = {"u": 10 * eigvecs[:, 0:1]}

    # define geometry
    rec = Rectangle((0, 0), (width, height))
    # make list of nodes to unroll graph on
    hm = HelmholtzEquation(u="u", k=wave_number * eps_sympy, dim=2)
    gn = GradNormal(T="u", dim=2, time=False)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
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
        hm.make_nodes() + gn.make_nodes() + [wave_net.make_node(name="wave_network")]
    )

    waveguide_domain = Domain()

    PEC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.PEC,
        lambda_weighting={"u": 100.0},
        criteria=Or(Eq(y, 0), Eq(y, height)),
    )

    waveguide_domain.add_constraint(PEC, "PEC")

    Waveguide_port = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=waveguide_port_invar_numpy,
        outvar=waveguide_port_outvar_numpy,
        batch_size=cfg.batch_size.Waveguide_port,
        lambda_weighting={"u": np.full_like(yv, 0.5)},
    )
    waveguide_domain.add_constraint(Waveguide_port, "Waveguide_port")

    ABC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"normal_gradient_u": 0.0},
        batch_size=cfg.batch_size.ABC,
        lambda_weighting={"normal_gradient_u": 10.0},
        criteria=Eq(x, width),
    )
    waveguide_domain.add_constraint(ABC, "ABC")

    Interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"helmholtz": 0.0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={
            "helmholtz": 1.0 / wave_number**2,
        },
    )
    waveguide_domain.add_constraint(Interior, "Interior")

    # add inferencer data
    slab_inference = VoxelInferencer(
        bounds=[[0, 2], [0, 2]],
        npoints=[256, 256],
        nodes=nodes,
        output_names=["u"],
    )
    waveguide_domain.add_inferencer(slab_inference, "Inf" + str(int(wave_number)))

    # make solver
    slv = Solver(cfg, waveguide_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
