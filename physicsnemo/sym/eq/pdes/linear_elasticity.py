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

"""Equations related to linear elasticity"""

from sympy import Symbol, Function, Number

from physicsnemo.sym.eq.pde import PDE


class LinearElasticity(PDE):
    """
    Linear elasticity equations.
    Use either (E, nu) or (``lambda_``, mu) to define the material properties.

    Parameters
    ==========
    E : float, Sympy Symbol/Expr, str
        The Young's modulus
    nu : float, Sympy Symbol/Expr, str
        The Poisson's ratio
    ``lambda_`` : float, Sympy Symbol/Expr, str
        Lamé's first parameter
    mu: float, Sympy Symbol/Expr, str
        Lamé's second parameter (shear modulus)
    rho: float, Sympy Symbol/Expr, str
        Mass density.
    dim : int
        Dimension of the linear elasticity (2 or 3). Default is 3.

    Example
    ========
    >>> elasticity_equations = LinearElasticity(E=10, nu=0.3, dim=2)
    >>> elasticity_equations.pprint()
      navier_x: -13.4615384615385*u__x__x - 3.84615384615385*u__y__y - 9.61538461538461*v__x__y
      navier_y: -3.84615384615385*v__x__x - 13.4615384615385*v__y__y - 9.61538461538461*u__x__y
      stress_disp_xx: -sigma_xx + 13.4615384615385*u__x + 5.76923076923077*v__y
      stress_disp_yy: -sigma_yy + 5.76923076923077*u__x + 13.4615384615385*v__y
      stress_disp_xy: -sigma_xy + 3.84615384615385*u__y + 3.84615384615385*v__x
      equilibrium_x: -sigma_xx__x - sigma_xy__y
      equilibrium_y: -sigma_xy__x - sigma_yy__y
      traction_x: normal_x*sigma_xx + normal_y*sigma_xy
      traction_y: normal_x*sigma_xy + normal_y*sigma_yy
    """

    name = "LinearElasticity"

    def __init__(
        self, E=None, nu=None, lambda_=None, mu=None, rho=1, dim=3, time=False
    ):
        # set params
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
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # displacement componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        sigma_xx = Function("sigma_xx")(*input_variables)
        sigma_yy = Function("sigma_yy")(*input_variables)
        sigma_xy = Function("sigma_xy")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
            sigma_zz = Function("sigma_zz")(*input_variables)
            sigma_xz = Function("sigma_xz")(*input_variables)
            sigma_yz = Function("sigma_yz")(*input_variables)
        else:
            w = Number(0)
            sigma_zz = Number(0)
            sigma_xz = Number(0)
            sigma_yz = Number(0)

        # material properties
        if lambda_ is None:
            if isinstance(nu, str):
                nu = Function(nu)(*input_variables)
            elif isinstance(nu, (float, int)):
                nu = Number(nu)
            if isinstance(E, str):
                E = Function(E)(*input_variables)
            elif isinstance(E, (float, int)):
                E = Number(E)
            lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
        else:
            if isinstance(lambda_, str):
                lambda_ = Function(lambda_)(*input_variables)
            elif isinstance(lambda_, (float, int)):
                lambda_ = Number(lambda_)
            if isinstance(mu, str):
                mu = Function(mu)(*input_variables)
            elif isinstance(mu, (float, int)):
                mu = Number(mu)
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # set equations
        self.equations = {}

        # Stress equations
        self.equations["stress_disp_xx"] = (
            lambda_ * (u.diff(x) + v.diff(y) + w.diff(z))
            + 2 * mu * u.diff(x)
            - sigma_xx
        )
        self.equations["stress_disp_yy"] = (
            lambda_ * (u.diff(x) + v.diff(y) + w.diff(z))
            + 2 * mu * v.diff(y)
            - sigma_yy
        )
        self.equations["stress_disp_zz"] = (
            lambda_ * (u.diff(x) + v.diff(y) + w.diff(z))
            + 2 * mu * w.diff(z)
            - sigma_zz
        )
        self.equations["stress_disp_xy"] = mu * (u.diff(y) + v.diff(x)) - sigma_xy
        self.equations["stress_disp_xz"] = mu * (u.diff(z) + w.diff(x)) - sigma_xz
        self.equations["stress_disp_yz"] = mu * (v.diff(z) + w.diff(y)) - sigma_yz

        # Equations of equilibrium
        self.equations["equilibrium_x"] = rho * ((u.diff(t)).diff(t)) - (
            sigma_xx.diff(x) + sigma_xy.diff(y) + sigma_xz.diff(z)
        )
        self.equations["equilibrium_y"] = rho * ((v.diff(t)).diff(t)) - (
            sigma_xy.diff(x) + sigma_yy.diff(y) + sigma_yz.diff(z)
        )
        self.equations["equilibrium_z"] = rho * ((w.diff(t)).diff(t)) - (
            sigma_xz.diff(x) + sigma_yz.diff(y) + sigma_zz.diff(z)
        )

        # Traction equations
        self.equations["traction_x"] = (
            normal_x * sigma_xx + normal_y * sigma_xy + normal_z * sigma_xz
        )
        self.equations["traction_y"] = (
            normal_x * sigma_xy + normal_y * sigma_yy + normal_z * sigma_yz
        )
        self.equations["traction_z"] = (
            normal_x * sigma_xz + normal_y * sigma_yz + normal_z * sigma_zz
        )

        # Navier equations
        self.equations["navier_x"] = (
            rho * ((u.diff(t)).diff(t))
            - (lambda_ + mu) * (u.diff(x) + v.diff(y) + w.diff(z)).diff(x)
            - mu * ((u.diff(x)).diff(x) + (u.diff(y)).diff(y) + (u.diff(z)).diff(z))
        )
        self.equations["navier_y"] = (
            rho * ((v.diff(t)).diff(t))
            - (lambda_ + mu) * (u.diff(x) + v.diff(y) + w.diff(z)).diff(y)
            - mu * ((v.diff(x)).diff(x) + (v.diff(y)).diff(y) + (v.diff(z)).diff(z))
        )
        self.equations["navier_z"] = (
            rho * ((w.diff(t)).diff(t))
            - (lambda_ + mu) * (u.diff(x) + v.diff(y) + w.diff(z)).diff(z)
            - mu * ((w.diff(x)).diff(x) + (w.diff(y)).diff(y) + (w.diff(z)).diff(z))
        )

        if self.dim == 2:
            self.equations.pop("navier_z")
            self.equations.pop("stress_disp_zz")
            self.equations.pop("stress_disp_xz")
            self.equations.pop("stress_disp_yz")
            self.equations.pop("equilibrium_z")
            self.equations.pop("traction_z")


class LinearElasticityPlaneStress(PDE):
    """
    Linear elasticity plane stress equations.
    Use either (E, nu) or (``lambda_`` , mu) to define the material properties.

    Parameters
    ==========
    E : float, Sympy Symbol/Expr, str
        The Young's modulus
    nu : float, Sympy Symbol/Expr, str
        The Poisson's ratio
    ``lambda_`` : float, Sympy Symbol/Expr, str
        Lamé's first parameter.
    mu: float, Sympy Symbol/Expr, str
        Lamé's second parameter (shear modulus)
    rho: float, Sympy Symbol/Expr, str
        Mass density.

    Example
    ========
    >>> plane_stress_equations = LinearElasticityPlaneStress(E=10, nu=0.3)
    >>> plane_stress_equations.pprint()
      stress_disp_xx: -sigma_xx + 10.989010989011*u__x + 3.2967032967033*v__y
      stress_disp_yy: -sigma_yy + 3.2967032967033*u__x + 10.989010989011*v__y
      stress_disp_xy: -sigma_xy + 3.84615384615385*u__y + 3.84615384615385*v__x
      equilibrium_x: -sigma_xx__x - sigma_xy__y
      equilibrium_y: -sigma_xy__x - sigma_yy__y
      traction_x: normal_x*sigma_xx + normal_y*sigma_xy
      traction_y: normal_x*sigma_xy + normal_y*sigma_yy
    """

    name = "LinearElasticityPlaneStress"

    def __init__(self, E=None, nu=None, lambda_=None, mu=None, rho=1, time=False):
        # set params
        self.time = time

        # coordinates
        x, y = Symbol("x"), Symbol("y")
        normal_x, normal_y = Symbol("normal_x"), Symbol("normal_y")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "t": t}
        if not self.time:
            input_variables.pop("t")

        # displacement componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        sigma_xx = Function("sigma_xx")(*input_variables)
        sigma_yy = Function("sigma_yy")(*input_variables)
        sigma_xy = Function("sigma_xy")(*input_variables)

        # material properties
        if lambda_ is None:
            if isinstance(nu, str):
                nu = Function(nu)(*input_variables)
            elif isinstance(nu, (float, int)):
                nu = Number(nu)
            if isinstance(E, str):
                E = Function(E)(*input_variables)
            elif isinstance(E, (float, int)):
                E = Number(E)
            lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
        else:
            if isinstance(lambda_, str):
                lambda_ = Function(lambda_)(*input_variables)
            elif isinstance(lambda_, (float, int)):
                lambda_ = Number(lambda_)
            if isinstance(mu, str):
                mu = Function(mu)(*input_variables)
            elif isinstance(mu, (float, int)):
                mu = Number(mu)
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # set equations
        self.equations = {}

        # Stress equations
        w_z = -lambda_ / (lambda_ + 2 * mu) * (u.diff(x) + v.diff(y))
        self.equations["stress_disp_xx"] = (
            lambda_ * (u.diff(x) + v.diff(y) + w_z) + 2 * mu * u.diff(x) - sigma_xx
        )
        self.equations["stress_disp_yy"] = (
            lambda_ * (u.diff(x) + v.diff(y) + w_z) + 2 * mu * v.diff(y) - sigma_yy
        )
        self.equations["stress_disp_xy"] = mu * (u.diff(y) + v.diff(x)) - sigma_xy

        # Equations of equilibrium
        self.equations["equilibrium_x"] = rho * ((u.diff(t)).diff(t)) - (
            sigma_xx.diff(x) + sigma_xy.diff(y)
        )
        self.equations["equilibrium_y"] = rho * ((v.diff(t)).diff(t)) - (
            sigma_xy.diff(x) + sigma_yy.diff(y)
        )

        # Traction equations
        self.equations["traction_x"] = normal_x * sigma_xx + normal_y * sigma_xy
        self.equations["traction_y"] = normal_x * sigma_xy + normal_y * sigma_yy
