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
PDE configs for here

Architecture params need to be updated to more primative focused
This file is largely a place folder right now
"""

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class PDEConf:
    _target_: str = MISSING


@dataclass
class AdvectionDiffusionConf(PDEConf):
    _target_: str = "physicsnemo.sym.eq.pdes.advection_diffusion.AdvectionDiffusion"
    T: str = "T"
    D: str = "D"
    Q: int = 0
    rho: str = "rho"
    dim: int = 3
    time: bool = False


@dataclass
class DiffusionConf(PDEConf):
    _target_: str = "physicsnemo.sym.eq.pdes.diffusion.Diffusion"
    T: str = "T"
    D: str = "D"
    Q: int = 0
    dim: int = 3
    time: bool = False


@dataclass
class MaxwellFreqRealConf(PDEConf):
    _target_: str = "physicsnemo.sym.eq.pdes.electromagnetic.MaxwellFreqReal"
    ux: str = "ux"
    uy: str = "uy"
    uz: str = "uz"
    k: float = 1.0


@dataclass
class EnergyFluidConf(PDEConf):
    _target_: str = "physicsnemo.sym.eq.pdes.energy_equation.EnergyFluid"
    cp: str = "cp"
    kappa: str = "kappa"
    rho: str = "rho"
    nu: str = "nu"
    visc_heating: bool = False
    dim: int = 3
    time: bool = False


@dataclass
class LinearElasticityConf(PDEConf):
    _target_: str = "physicsnemo.sym.eq.pdes.linear_elasticity.LinearElasticity"
    E = 10
    nu = 0.3
    lambda_ = None
    mu = None
    rho: int = 1
    dim: int = 3
    time: bool = False


@dataclass
class LinearElasticityPlaneConf(PDEConf):
    _target_: str = (
        "physicsnemo.sym.eq.pdes.linear_elasticity.LinearElasticityPlaneStress"
    )
    E = 10
    nu = 0.3
    lambda_ = None
    mu = None
    rho: int = 1
    time: bool = False


@dataclass
class NavierStokesConf(PDEConf):
    _target_: str = "physicsnemo.sym.eq.pdes.navier_stokes.NavierStokes"
    nu = MISSING
    rho: float = 1
    dim: int = 3
    time: bool = True


@dataclass
class ZeroEquationConf(PDEConf):
    _target_: str = "physicsnemo.sym.eq.pdes.turbulence_zero_eq.ZeroEquation"
    nu = MISSING
    rho: float = 1
    dim: int = 3
    time: bool = True


@dataclass
class WaveEquationConf(PDEConf):
    _target_: str = "physicsnemo.sym.eq.pdes.wave_equation.WaveEquation"
    u = "u"
    c = "c"
    dim: int = 3
    time: bool = True


@dataclass
class HelmholtzEquationConf(PDEConf):
    _target_: str = "physicsnemo.sym.eq.pdes.wave_equation.HelmholtzEquation"
    u = MISSING
    K = MISSING
    dim: int = 3


def register_pde_configs() -> None:
    # TODO: Allow multiple pdes via multiple config groups
    # https://hydra.cc/docs/next/patterns/select_multiple_configs_from_config_group/
    cs = ConfigStore.instance()
    cs.store(
        group="pde",
        name="advection-diffusion",
        node=AdvectionDiffusionConf,
    )

    cs.store(
        group="pde",
        name="diffusion",
        node=DiffusionConf,
    )

    cs.store(
        group="pde",
        name="maxwell-real",
        node=MaxwellFreqRealConf,
    )

    cs.store(
        group="pde",
        name="energy-fluid",
        node=EnergyFluidConf,
    )

    cs.store(
        group="pde",
        name="linear-elasticity",
        node=LinearElasticityConf,
    )

    cs.store(
        group="pde",
        name="linear-elasticity-plane",
        node=LinearElasticityPlaneConf,
    )

    cs.store(
        group="pde",
        name="navier-stokes",
        node=NavierStokesConf,
    )

    cs.store(
        group="pde",
        name="zero-eq-turbulence",
        node=ZeroEquationConf,
    )

    cs.store(
        group="pde",
        name="wave",
        node=WaveEquationConf,
    )

    cs.store(
        group="pde",
        name="helmholtz",
        node=HelmholtzEquationConf,
    )
