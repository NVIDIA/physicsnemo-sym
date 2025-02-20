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

"""Advection diffusion equation
Reference:
https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation
"""
from sympy import Symbol, Function, Number

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.node import Node


class AdvectionDiffusion(PDE):
    """
    Advection diffusion equation

    Parameters
    ==========
    T : str
        The dependent variable.
    D : float, Sympy Symbol/Expr, str
        Diffusivity. If `D` is a str then it is
        converted to Sympy Function of form 'D(x,y,z,t)'.
        If 'D' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    Q : float, Sympy Symbol/Expr, str
        The source term. If `Q` is a str then it is
        converted to Sympy Function of form 'Q(x,y,z,t)'.
        If 'Q' is a Sympy Symbol or Expression then this
        is substituted into the equation. Default is 0.
    rho : float, Sympy Symbol/Expr, str
        The density. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes.
    dim : int
        Dimension of the diffusion equation (1, 2, or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is False.
    mixed_form: bool
        If True, use the mixed formulation of the wave equation.

    Examples
    ========
    >>> ad = AdvectionDiffusion(D=0.1, rho=1.)
    >>> ad.pprint()
      advection_diffusion: u*T__x + v*T__y + w*T__z - 0.1*T__x__x - 0.1*T__y__y - 0.1*T__z__z
    >>> ad = AdvectionDiffusion(D='D', rho=1, dim=2, time=True)
    >>> ad.pprint()
      advection_diffusion: -D*T__x__x - D*T__y__y + u*T__x + v*T__y - D__x*T__x - D__y*T__y + T__t
    """

    name = "AdvectionDiffusion"

    def __init__(
        self, T="T", D="D", Q=0, rho="rho", dim=3, time=False, mixed_form=False
    ):
        # set params
        self.T = T
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

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

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)

        # Temperature
        assert type(T) == str, "T needs to be string"
        T = Function(T)(*input_variables)

        # Diffusivity
        if type(D) is str:
            D = Function(D)(*input_variables)
        elif type(D) in [float, int]:
            D = Number(D)

        # Source
        if type(Q) is str:
            Q = Function(Q)(*input_variables)
        elif type(Q) in [float, int]:
            Q = Number(Q)

        # Density
        if type(rho) is str:
            rho = Function(rho)(*input_variables)
        elif type(rho) in [float, int]:
            rho = Number(rho)

        # set equations
        self.equations = {}
        advection = (
            rho * u * (T.diff(x)) + rho * v * (T.diff(y)) + rho * w * (T.diff(z))
        )
        if not self.mixed_form:
            diffusion = (
                (rho * D * T.diff(x)).diff(x)
                + (rho * D * T.diff(y)).diff(y)
                + (rho * D * T.diff(z)).diff(z)
            )
            self.equations["advection_diffusion_" + self.T] = (
                T.diff(t) + advection - diffusion - Q
            )

        elif self.mixed_form:
            T_x = Function(self.T + "_x")(*input_variables)
            T_y = Function(self.T + "_y")(*input_variables)
            if self.dim == 3:
                T_z = Function(self.T + "_z")(*input_variables)
            else:
                T_z = Number(0)

            diffusion = (
                (rho * D * T_x).diff(x)
                + (rho * D * T_y).diff(y)
                + (rho * D * T_z).diff(z)
            )
            self.equations["compatibility_" + self.T + "_x"] = T.diff(x) - T_x
            self.equations["compatibility_" + self.T + "_y"] = T.diff(y) - T_y
            self.equations["compatibility_" + self.T + "_z"] = T.diff(z) - T_z
            self.equations["compatibility_" + self.T + "_xy"] = T_x.diff(y) - T_y.diff(
                x
            )
            self.equations["compatibility_" + self.T + "_xz"] = T_x.diff(z) - T_z.diff(
                x
            )
            self.equations["compatibility_" + self.T + "_yz"] = T_y.diff(z) - T_z.diff(
                y
            )
            if self.dim == 2:
                self.equations.pop("compatibility_" + self.T + "_z")
                self.equations.pop("compatibility_" + self.T + "_xz")
                self.equations.pop("compatibility_" + self.T + "_yz")
            self.equations["advection_diffusion_" + self.T] = (
                T.diff(t) + advection - diffusion - Q
            )
