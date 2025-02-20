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

"""Maxwell's equation
"""

from sympy import Symbol, Function, Number
import numpy as np

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.node import Node

# helper functions computing curl
def _curl(v):
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    dim = len(v)
    if dim == 3:
        vx, vy, vz = v
        return [
            vz.diff(y) - vy.diff(z),
            vx.diff(z) - vz.diff(x),
            vy.diff(x) - vx.diff(y),
        ]
    elif dim == 2:
        vx, vy = v
        return [vy.diff(x) - vx.diff(y)]
    elif dim == 1:
        return [v[0].diff(y), -v[0].diff(x)]
    else:
        raise Exception("Input dimension for Curl operator must be 1, 2 or 3!")


# helper functions computing cross product
def _cross(a, b):
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    dim = len(a)
    if dim == 3:
        a1, a2, a3 = a
        b1, b2, b3 = b
        return [a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1]
    elif dim == 2:
        a1, a2 = a
        b1, b2 = b
        return [a1 * b2 - a2 * b1]
    else:
        raise Exception("Input dimension for cross product must be 2 or 3!")


class MaxwellFreqReal(PDE):
    """
    Frequency domain Maxwell's equation

    Parameters
    ==========
    ux : str
       Ex
    uy : str
       Ey
    uz : str
       Ez
    k : float, Sympy Symbol/Expr, str
      Wave number. If `k` is a str then it is
      converted to Sympy Function of form 'k(x,y,z,t)'.
      If 'k' is a Sympy Symbol or Expression then this
      is substituted into the equation.
    mixed_form: bool
        If True, use the mixed formulation of the diffusion equations.
    """

    name = "MaxwellFreqReal"

    def __init__(self, ux="ux", uy="uy", uz="uz", k=1.0, mixed_form=False):
        # set params
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.mixed_form = mixed_form

        if self.mixed_form:
            raise NotImplementedError(
                "Maxwell's equation is not implemented in mixed form"
            )

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # wave speed coefficient
        if isinstance(k, str):
            k = Function(k)(*input_variables)
        elif isinstance(k, (float, int)):
            k = Number(k)

        # E field
        assert isinstance(ux, str), "uz needs to be string"
        ux = Function(ux)(*input_variables)
        assert isinstance(uy, str), "uy needs to be string"
        uy = Function(uy)(*input_variables)
        assert isinstance(uz, str), "uz needs to be string"
        uz = Function(uz)(*input_variables)

        # compute del X (del X E)
        c2ux, c2uy, c2uz = _curl(_curl([ux, uy, uz]))

        # set equations
        self.equations = {}
        self.equations["Maxwell_Freq_real_x"] = c2ux - k**2 * ux
        self.equations["Maxwell_Freq_real_y"] = c2uy - k**2 * uy
        self.equations["Maxwell_Freq_real_z"] = c2uz - k**2 * uz


class SommerfeldBC(PDE):
    """
    Frequency domain ABC, Sommerfeld radiation condition
    Only for real part
    Equation: 'n x _curl(E) = 0'

    Parameters
    ==========
    ux : str
       Ex
    uy : str
       Ey
    uz : str
       Ez
    """

    name = "SommerfeldBC"

    def __init__(self, ux="ux", uy="uy", uz="uz"):
        # set params
        self.ux = ux
        self.uy = uy
        self.uz = uz

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")

        # make input variables, t is wave number
        input_variables = {"x": x, "y": y, "z": z}

        # E field
        assert isinstance(ux, str), "uz needs to be string"
        ux = Function(ux)(*input_variables)
        assert isinstance(uy, str), "uy needs to be string"
        uy = Function(uy)(*input_variables)
        assert isinstance(uz, str), "uz needs to be string"
        uz = Function(uz)(*input_variables)

        # compute cross product of curl for sommerfeld bc
        n = [normal_x, normal_y, normal_z]
        u = [ux, uy, uz]
        bcs = _cross(n, _curl(u))

        # set equations
        self.equations = {}
        self.equations["SommerfeldBC_real_x"] = bcs[0]
        self.equations["SommerfeldBC_real_y"] = bcs[1]
        self.equations["SommerfeldBC_real_z"] = bcs[2]


class PEC(PDE):
    """
    Perfect Electric Conduct BC for

    Parameters
    ==========
    ux : str
       Ex
    uy : str
       Ey
    uz : str
       Ez
    dim : int
        Dimension of the equations (2, or 3). Default is 3.
    """

    name = "PEC_3D"

    def __init__(self, ux="ux", uy="uy", uz="uz", dim=3):
        # set params
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.dim = dim

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")

        # make input variables, t is wave number
        input_variables = {"x": x, "y": y, "z": z}

        # E field
        assert isinstance(ux, str), "uz needs to be string"
        ux = Function(ux)(*input_variables)
        assert isinstance(uy, str), "uy needs to be string"
        uy = Function(uy)(*input_variables)
        if self.dim == 3:
            assert isinstance(uz, str), "uz needs to be string"
            uz = Function(uz)(*input_variables)

        # compute cross of electric field
        if self.dim == 2:
            n = [normal_x, normal_y]
            u = [ux, uy]
        elif self.dim == 3:
            n = [normal_x, normal_y, normal_z]
            u = [ux, uy, uz]
        else:
            raise ValueError("'dim' needs to be 2 or 3")
        bcs = _cross(n, u)

        # set equations
        self.equations = {}
        self.equations["PEC_x"] = bcs[0]
        if self.dim == 3:
            self.equations["PEC_y"] = bcs[1]
            self.equations["PEC_z"] = bcs[2]
