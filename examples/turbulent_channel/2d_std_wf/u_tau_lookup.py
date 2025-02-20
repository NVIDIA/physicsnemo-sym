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

import torch
import numpy as np
from sympy import Symbol, Eq, sin, cos, Min, Max, Abs, log, exp
from scipy import optimize

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node


@physicsnemo.sym.main(config_path="conf_u_tau_lookup", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    u = np.linspace(1e-3, 50, num=100)
    y = np.linspace(1e-3, 0.5, num=100)

    U, Y = np.meshgrid(u, y)

    U = np.reshape(U, (U.size,))
    Y = np.reshape(Y, (Y.size,))

    Re = 590
    nu = 1 / Re

    def f(u_tau, y, u):
        return u_tau * np.log(9.793 * y * u_tau / nu) - u * 0.4187

    def fprime(u_tau, y, u):
        return 1 + np.log(9.793 * y * u_tau / nu)

    u_tau = []
    for i in range(len(U)):
        u_tau_calc = optimize.newton(
            f,
            1.0,
            fprime=fprime,
            args=(Y[i], U[i]),
            tol=1.48e-08,
            maxiter=200,
            fprime2=None,
        )
        u_tau.append(u_tau_calc)

    # save tabulations to a csv file
    results = np.concatenate(
        (
            np.reshape(U, (len(U), 1)),
            np.reshape(Y, (len(Y), 1)),
            np.reshape(u_tau, (len(u_tau), 1)),
        ),
        axis=1,
    )
    np.savetxt("u_tau.csv", results, delimiter=",")

    invar = {"u_in": np.reshape(U, (len(U), 1)), "y_in": np.reshape(Y, (len(Y), 1))}
    outvar = {"u_tau_out": np.reshape(u_tau, (len(u_tau), 1))}

    u = np.random.uniform(1e-3, 50, size=100)
    y = np.random.uniform(1e-3, 0.5, size=100)

    U, Y = np.meshgrid(u, y)

    U = np.reshape(U, (U.size,))
    Y = np.reshape(Y, (Y.size,))

    u_tau_val = []
    for i in range(len(U)):
        u_tau_calc = optimize.newton(
            f,
            1.0,
            fprime=fprime,
            args=(Y[i], U[i]),
            tol=1.48e-08,
            maxiter=200,
            fprime2=None,
        )
        u_tau_val.append(u_tau_calc)

    # save tabulations to a csv file
    results = np.concatenate(
        (
            np.reshape(U, (len(U), 1)),
            np.reshape(Y, (len(Y), 1)),
            np.reshape(u_tau, (len(u_tau), 1)),
        ),
        axis=1,
    )
    np.savetxt("u_tau_val.csv", results, delimiter=",")

    invar_val = {"u_in": np.reshape(U, (len(U), 1)), "y_in": np.reshape(Y, (len(Y), 1))}
    outvar_val = {"u_tau_out": np.reshape(u_tau_val, (len(u_tau_val), 1))}

    # make list of nodes to unroll graph on
    u_tau_net = instantiate_arch(
        input_keys=[Key("u_in"), Key("y_in")],
        output_keys=[Key("u_tau_out")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = [u_tau_net.make_node(name="u_tau_network")]

    # make domain
    domain = Domain()

    train = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=invar,
        outvar=outvar,
        batch_size=10000,
    )
    domain.add_constraint(train, "LogLawLoss")

    # add validation
    validator = PointwiseValidator(nodes=nodes, invar=invar_val, true_outvar=outvar_val)
    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
