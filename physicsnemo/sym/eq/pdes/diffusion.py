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

"""Diffusion equation
"""

from sympy import Symbol, Function, Number

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.node import Node


class Diffusion(PDE):
    """
    Diffusion equation

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
    dim : int
        Dimension of the diffusion equation (1, 2, or 3).
        Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the diffusion equations.

    Examples
    ========
    >>> diff = Diffusion(D=0.1, Q=1, dim=2)
    >>> diff.pprint()
      diffusion_T: T__t - 0.1*T__x__x - 0.1*T__y__y - 1
    >>> diff = Diffusion(T='u', D='D', Q='Q', dim=3, time=False)
    >>> diff.pprint()
      diffusion_u: -D*u__x__x - D*u__y__y - D*u__z__z - Q - D__x*u__x - D__y*u__y - D__z*u__z
    """

    name = "Diffusion"

    def __init__(self, T="T", D="D", Q=0, dim=3, time=True, mixed_form=False):
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

        # set equations
        self.equations = {}

        if not self.mixed_form:
            self.equations["diffusion_" + self.T] = (
                T.diff(t)
                - (D * T.diff(x)).diff(x)
                - (D * T.diff(y)).diff(y)
                - (D * T.diff(z)).diff(z)
                - Q
            )
        elif self.mixed_form:
            T_x = Function("T_x")(*input_variables)
            T_y = Function("T_y")(*input_variables)
            if self.dim == 3:
                T_z = Function("T_z")(*input_variables)
            else:
                T_z = Number(0)

            self.equations["diffusion_" + self.T] = (
                T.diff(t)
                - (D * T_x).diff(x)
                - (D * T_y).diff(y)
                - (D * T_z).diff(z)
                - Q
            )
            self.equations["compatibility_T_x"] = T.diff(x) - T_x
            self.equations["compatibility_T_y"] = T.diff(y) - T_y
            self.equations["compatibility_T_z"] = T.diff(z) - T_z
            self.equations["compatibility_T_xy"] = T_x.diff(y) - T_y.diff(x)
            self.equations["compatibility_T_xz"] = T_x.diff(z) - T_z.diff(x)
            self.equations["compatibility_T_yz"] = T_y.diff(z) - T_z.diff(y)
            if self.dim == 2:
                self.equations.pop("compatibility_T_z")
                self.equations.pop("compatibility_T_xz")
                self.equations.pop("compatibility_T_yz")


class DiffusionInterface(PDE):
    """
    Matches the boundary conditions at an interface

    Parameters
    ==========
    T_1, T_2 : str
        Dependent variables to match the boundary conditions at the interface.
    D_1, D_2 : float
        Diffusivity at the interface.
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    Example
    ========
    >>> diff = DiffusionInterface('theta_s', 'theta_f', 0.1, 0.05, dim=2)
    >>> diff.pprint()
      diffusion_interface_dirichlet_theta_s_theta_f: -theta_f + theta_s
      diffusion_interface_neumann_theta_s_theta_f: -0.05*normal_x*theta_f__x
      + 0.1*normal_x*theta_s__x - 0.05*normal_y*theta_f__y
      + 0.1*normal_y*theta_s__y
    """

    name = "DiffusionInterface"

    def __init__(self, T_1, T_2, D_1, D_2, dim=3, time=True):
        # set params
        self.T_1 = T_1
        self.T_2 = T_2
        self.D_1 = D_1
        self.D_2 = D_2
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
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

        # variables to match the boundary conditions (example Temperature)
        T_1 = Function(T_1)(*input_variables)
        T_2 = Function(T_2)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["diffusion_interface_dirichlet_" + self.T_1 + "_" + self.T_2] = (
            T_1 - T_2
        )
        flux_1 = self.D_1 * (
            normal_x * T_1.diff(x) + normal_y * T_1.diff(y) + normal_z * T_1.diff(z)
        )
        flux_2 = self.D_2 * (
            normal_x * T_2.diff(x) + normal_y * T_2.diff(y) + normal_z * T_2.diff(z)
        )
        self.equations["diffusion_interface_neumann_" + self.T_1 + "_" + self.T_2] = (
            flux_1 - flux_2
        )
