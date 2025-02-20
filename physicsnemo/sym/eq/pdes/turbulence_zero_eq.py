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

"""Zero Equation Turbulence model
References:
https://www.eureka.im/954.html
https://knowledge.autodesk.com/support/cfd/learn-explore/caas/CloudHelp/cloudhelp/2019/ENU/SimCFD-Learning/files/GUID-BBA4E008-8346-465B-9FD3-D193CF108AF0-htm.html
"""

from sympy import Symbol, Function, sqrt, Number, Min

from physicsnemo.sym.eq.pde import PDE


class ZeroEquation(PDE):
    """
    Zero Equation Turbulence model

    Parameters
    ==========
    nu : float
        The kinematic viscosity of the fluid.
    max_distance : float
        The maximum wall distance in the flow field.
    rho : float, Sympy Symbol/Expr, str
        The density. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation. Default is 1.
    dim : int
        Dimension of the Zero Equation Turbulence model (2 or 3).
        Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    Example
    ========
    >>> zeroEq = ZeroEquation(nu=0.1, max_distance=2.0, dim=2)
    >>> kEp.pprint()
      nu: sqrt((u__y + v__x)**2 + 2*u__x**2 + 2*v__y**2)
      *Min(0.18, 0.419*normal_distance)**2 + 0.1
    """

    name = "ZeroEquation"

    def __init__(
        self, nu, max_distance, rho=1, dim=3, time=True
    ):  # TODO add density into model
        # set params
        self.dim = dim
        self.time = time

        # model coefficients
        self.max_distance = max_distance
        self.karman_constant = 0.419
        self.max_distance_ratio = 0.09

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # density
        if type(rho) is str:
            rho = Function(rho)(*input_variables)
        elif type(rho) in [float, int]:
            rho = Number(rho)

        # wall distance
        normal_distance = Function("sdf")(*input_variables)

        # mixing length
        mixing_length = Min(
            self.karman_constant * normal_distance,
            self.max_distance_ratio * self.max_distance,
        )
        G = (
            2 * u.diff(x) ** 2
            + 2 * v.diff(y) ** 2
            + 2 * w.diff(z) ** 2
            + (u.diff(y) + v.diff(x)) ** 2
            + (u.diff(z) + w.diff(x)) ** 2
            + (v.diff(z) + w.diff(y)) ** 2
        )

        # set equations
        self.equations = {}
        self.equations["nu"] = nu + rho * mixing_length**2 * sqrt(G)
