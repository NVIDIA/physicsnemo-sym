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
import sys
import warnings

import numpy as np
from sympy import Symbol, Function, Number

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.wave_equation import WaveEquation
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter


# Read in npz files generated using finite difference simulator Devito
def read_wf_data(time, dLen):
    file_path = "Training_data"
    if not os.path.exists(to_absolute_path(file_path)):
        warnings.warn(
            f"Directory {file_path} does not exist. Cannot continue. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/Modulus_sym_examples_supplemental_materials"
        )
        sys.exit()
    wf_filename = to_absolute_path(f"Training_data/wf_{int(time):04d}ms.npz")
    wave = np.load(wf_filename)["arr_0"].astype(np.float32)
    mesh_y, mesh_x = np.meshgrid(
        np.linspace(0, dLen, wave.shape[0]),
        np.linspace(0, dLen, wave.shape[1]),
        indexing="ij",
    )
    invar = {}
    invar["x"] = np.expand_dims(mesh_y.astype(np.float32).flatten(), axis=-1)
    invar["y"] = np.expand_dims(mesh_x.astype(np.float32).flatten(), axis=-1)
    invar["t"] = np.full_like(invar["x"], time * 0.001)
    outvar = {}
    outvar["u"] = np.expand_dims(wave.flatten(), axis=-1)
    return invar, outvar


# define open boundary conditions
class OpenBoundary(PDE):
    """
    Open boundary condition for wave problems
    Ref: http://hplgit.github.io/wavebc/doc/pub/._wavebc_cyborg002.html

    Parameters
    ==========
    u : str
        The dependent variable.
    c : float, Sympy Symbol/Expr, str
        Wave speed coefficient. If `c` is a str then it is
        converted to Sympy Function of form 'c(x,y,z,t)'.
        If 'c' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    dim : int
        Dimension of the wave equation (1, 2, or 3). Default is 2.
    time : bool
        If time-dependent equations or not. Default is True.
    """

    name = "OpenBoundary"

    def __init__(self, u="u", c="c", dim=3, time=True):
        # set params
        self.u = u
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # normal
        normal_x, normal_y, normal_z = (
            Symbol("normal_x"),
            Symbol("normal_y"),
            Symbol("normal_z"),
        )

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)

        # set equations
        self.equations = {}
        self.equations["open_boundary"] = (
            u.diff(t)
            + normal_x * c * u.diff(x)
            + normal_y * c * u.diff(y)
            + normal_z * c * u.diff(z)
        )


class WavePlotter(ValidatorPlotter):
    "Define custom validator plotting class"

    def __call__(self, invar, true_outvar, pred_outvar):
        # only plot x,y dimensions
        invar = {k: v for k, v in invar.items() if k in ["x", "y"]}
        fs = super().__call__(invar, true_outvar, pred_outvar)
        return fs


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    2d acoustic wave propagation at a domain of 2kmx2km, with a single Ricker source at the middle of the 2D domain
    """

    # override defaults
    cfg.arch.fully_connected.layer_size = 128

    # define PDEs
    we = WaveEquation(u="u", c="c", dim=2, time=True)
    ob = OpenBoundary(u="u", c="c", dim=2, time=True)

    # define networks and nodes
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    speed_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("c")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        we.make_nodes(detach_names=["c"])
        + ob.make_nodes(detach_names=["c"])
        + [
            wave_net.make_node(name="wave_network"),
            speed_net.make_node(name="speed_network"),
        ]
    )

    # define geometry
    dLen = 2  # km
    rec = Rectangle((0, 0), (dLen, dLen))

    # define sympy domain variables
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")

    # define time range
    time_length = 1
    time_range = {t: (0.15, time_length)}

    # define target velocity model
    # 2.0 km/s at the bottom and 1.0 km/s at the top using tanh function
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(0, 2, 512), np.linspace(0, 2, 512), indexing="ij"
    )
    wave_speed_invar = {}
    wave_speed_invar["x"] = np.expand_dims(mesh_x.flatten(), axis=-1)
    wave_speed_invar["y"] = np.expand_dims(mesh_y.flatten(), axis=-1)
    wave_speed_outvar = {}
    wave_speed_outvar["c"] = np.tanh(80 * (wave_speed_invar["y"] - 1.0)) / 2 + 1.5

    # make domain
    domain = Domain()

    # add velocity constraint
    velocity = PointwiseConstraint.from_numpy(
        nodes=nodes, invar=wave_speed_invar, outvar=wave_speed_outvar, batch_size=1024
    )
    domain.add_constraint(velocity, "Velocity")

    # add initial timesteps constraints
    batch_size = 1024
    for i, ms in enumerate(np.linspace(150, 300, 4)):
        timestep_invar, timestep_outvar = read_wf_data(ms, dLen)
        lambda_weighting = {}
        lambda_weighting["u"] = np.full_like(timestep_invar["x"], 10.0 / batch_size)
        timestep = PointwiseConstraint.from_numpy(
            nodes,
            timestep_invar,
            timestep_outvar,
            batch_size,
            lambda_weighting=lambda_weighting,
        )
        domain.add_constraint(timestep, f"BC{i:04d}")

    # add interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"wave_equation": 0},
        batch_size=4096,
        bounds={x: (0, dLen), y: (0, dLen)},
        lambda_weighting={"wave_equation": 0.0001},
        parameterization=time_range,
    )
    domain.add_constraint(interior, "Interior")

    # add open boundary constraint
    edges = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"open_boundary": 0},
        batch_size=1024,
        lambda_weighting={"open_boundary": 0.01 * time_length},
        parameterization=time_range,
    )
    domain.add_constraint(edges, "Edges")

    # add validators
    for i, ms in enumerate(np.linspace(350, 950, 13)):
        val_invar, val_true_outvar = read_wf_data(ms, dLen)
        validator = PointwiseValidator(
            nodes=nodes,
            invar=val_invar,
            true_outvar=val_true_outvar,
            batch_size=1024,
            plotter=WavePlotter(),
        )
        domain.add_validator(validator, f"VAL_{i:04d}")
    validator = PointwiseValidator(
        nodes=nodes,
        invar=wave_speed_invar,
        true_outvar=wave_speed_outvar,
        batch_size=1024,
        plotter=WavePlotter(),
    )
    domain.add_validator(validator, "Velocity")

    slv = Solver(cfg, domain)

    slv.solve()


if __name__ == "__main__":
    run()
