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

"""Equations related to Navier Stokes Equations
"""

from sympy import Symbol, Function, Number

from physicsnemo.sym.amp import AmpManager
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.node import Node


class NavierStokes(PDE):
    """
    Compressible Navier Stokes equations
    Reference:
    https://turbmodels.larc.nasa.gov/implementrans.html

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the Navier-Stokes equations.

    Examples
    ========
    >>> ns = NavierStokes(nu=0.01, rho=1, dim=2)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
      momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
    >>> ns = NavierStokes(nu='nu', rho=1, dim=2, time=False)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: -nu*u__x__x - nu*u__y__y + u*u__x + v*u__y - 2*nu__x*u__x - nu__y*u__y - nu__y*v__x + p__x
      momentum_y: -nu*v__x__x - nu*v__y__y + u*v__x + v*v__y - nu__x*u__y - nu__x*v__x - 2*nu__y*v__y + p__y
    """

    name = "NavierStokes"

    def __init__(self, nu, rho=1, dim=3, time=True, mixed_form=False):
        # set params
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

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

        # pressure
        p = Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
            # When AMP scalers are enabled, and we are using zero equation
            # models with explicit nu expressions, we have to use a separate
            # derivative scaler for nu because the dynamic range of nu is
            # much smaller than the other second-order derivatives.
            if AmpManager().scaler_enabled:
                AmpManager().register_special_term(nu, max_scale=2**20)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # dynamic viscosity
        mu = rho * nu

        # set equations
        self.equations = {}
        self.equations["continuity"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        )

        if not self.mixed_form:
            curl = Number(0) if rho.diff(x) == 0 else u.diff(x) + v.diff(y) + w.diff(z)
            self.equations["momentum_x"] = (
                (rho * u).diff(t)
                + (
                    u * ((rho * u).diff(x))
                    + v * ((rho * u).diff(y))
                    + w * ((rho * u).diff(z))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u.diff(x)).diff(x)
                - (mu * u.diff(y)).diff(y)
                - (mu * u.diff(z)).diff(z)
                - (mu * (curl).diff(x))
                - mu.diff(x) * u.diff(x)
                - mu.diff(y) * v.diff(x)
                - mu.diff(z) * w.diff(x)
            )
            self.equations["momentum_y"] = (
                (rho * v).diff(t)
                + (
                    u * ((rho * v).diff(x))
                    + v * ((rho * v).diff(y))
                    + w * ((rho * v).diff(z))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v.diff(x)).diff(x)
                - (mu * v.diff(y)).diff(y)
                - (mu * v.diff(z)).diff(z)
                - (mu * (curl).diff(y))
                - mu.diff(x) * u.diff(y)
                - mu.diff(y) * v.diff(y)
                - mu.diff(z) * w.diff(y)
            )
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w).diff(x))
                    + v * ((rho * w).diff(y))
                    + w * ((rho * w).diff(z))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w.diff(x)).diff(x)
                - (mu * w.diff(y)).diff(y)
                - (mu * w.diff(z)).diff(z)
                - (mu * (curl).diff(z))
                - mu.diff(x) * u.diff(z)
                - mu.diff(y) * v.diff(z)
                - mu.diff(z) * w.diff(z)
            )

            if self.dim == 2:
                self.equations.pop("momentum_z")

        elif self.mixed_form:
            u_x = Function("u_x")(*input_variables)
            u_y = Function("u_y")(*input_variables)
            u_z = Function("u_z")(*input_variables)
            v_x = Function("v_x")(*input_variables)
            v_y = Function("v_y")(*input_variables)
            v_z = Function("v_z")(*input_variables)

            if self.dim == 3:
                w_x = Function("w_x")(*input_variables)
                w_y = Function("w_y")(*input_variables)
                w_z = Function("w_z")(*input_variables)
            else:
                w_x = Number(0)
                w_y = Number(0)
                w_z = Number(0)
                u_z = Number(0)
                v_z = Number(0)

            curl = Number(0) if rho.diff(x) == 0 else u_x + v_y + w_z
            self.equations["momentum_x"] = (
                (rho * u).diff(t)
                + (
                    u * ((rho * u.diff(x)))
                    + v * ((rho * u.diff(y)))
                    + w * ((rho * u.diff(z)))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u_x).diff(x)
                - (mu * u_y).diff(y)
                - (mu * u_z).diff(z)
                - (mu * (curl).diff(x))
                - mu.diff(x) * u.diff(x)
                - mu.diff(y) * v.diff(x)
                - mu.diff(z) * w.diff(x)
            )
            self.equations["momentum_y"] = (
                (rho * v).diff(t)
                + (
                    u * ((rho * v.diff(x)))
                    + v * ((rho * v.diff(y)))
                    + w * ((rho * v.diff(z)))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v_x).diff(x)
                - (mu * v_y).diff(y)
                - (mu * v_z).diff(z)
                - (mu * (curl).diff(y))
                - mu.diff(x) * u.diff(y)
                - mu.diff(y) * v.diff(y)
                - mu.diff(z) * w.diff(y)
            )
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w.diff(x)))
                    + v * ((rho * w.diff(y)))
                    + w * ((rho * w.diff(z)))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w_x).diff(x)
                - (mu * w_y).diff(y)
                - (mu * w_z).diff(z)
                - (mu * (curl).diff(z))
                - mu.diff(x) * u.diff(z)
                - mu.diff(y) * v.diff(z)
                - mu.diff(z) * w.diff(z)
            )
            self.equations["compatibility_u_x"] = u.diff(x) - u_x
            self.equations["compatibility_u_y"] = u.diff(y) - u_y
            self.equations["compatibility_u_z"] = u.diff(z) - u_z
            self.equations["compatibility_v_x"] = v.diff(x) - v_x
            self.equations["compatibility_v_y"] = v.diff(y) - v_y
            self.equations["compatibility_v_z"] = v.diff(z) - v_z
            self.equations["compatibility_w_x"] = w.diff(x) - w_x
            self.equations["compatibility_w_y"] = w.diff(y) - w_y
            self.equations["compatibility_w_z"] = w.diff(z) - w_z
            self.equations["compatibility_u_xy"] = u_x.diff(y) - u_y.diff(x)
            self.equations["compatibility_u_xz"] = u_x.diff(z) - u_z.diff(x)
            self.equations["compatibility_u_yz"] = u_y.diff(z) - u_z.diff(y)
            self.equations["compatibility_v_xy"] = v_x.diff(y) - v_y.diff(x)
            self.equations["compatibility_v_xz"] = v_x.diff(z) - v_z.diff(x)
            self.equations["compatibility_v_yz"] = v_y.diff(z) - v_z.diff(y)
            self.equations["compatibility_w_xy"] = w_x.diff(y) - w_y.diff(x)
            self.equations["compatibility_w_xz"] = w_x.diff(z) - w_z.diff(x)
            self.equations["compatibility_w_yz"] = w_y.diff(z) - w_z.diff(y)

            if self.dim == 2:
                self.equations.pop("momentum_z")
                self.equations.pop("compatibility_u_z")
                self.equations.pop("compatibility_v_z")
                self.equations.pop("compatibility_w_x")
                self.equations.pop("compatibility_w_y")
                self.equations.pop("compatibility_w_z")
                self.equations.pop("compatibility_u_xz")
                self.equations.pop("compatibility_u_yz")
                self.equations.pop("compatibility_v_xz")
                self.equations.pop("compatibility_v_yz")
                self.equations.pop("compatibility_w_xy")
                self.equations.pop("compatibility_w_xz")
                self.equations.pop("compatibility_w_yz")


class IncompressibleNavierStokes(PDE):
    """
    Incompressible Navier Stokes equations
    Reference:
    https://web.stanford.edu/class/me469b/handouts/incompressible.pdf

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float
        The density of the fluid. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    Examples
    ========
    >>> inc_ns = IncompressibleNavierStokes(nu=0.01, rho=1, dim=2)
    >>> inc_ns.pprint()
    """

    name = "IncompressibleNavierStokes"

    def __init__(self, nu, rho=1, dim=3, time=True):
        # set params
        self.dim = dim
        self.time = time

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

        # pressure
        p = Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
            # When AMP scalers are enabled, and we are using zero equation
            # models with explicit nu expressions, we have to use a separate
            # derivative scaler for nu because the dynamic range of nu is
            # much smaller than the other second-order derivatives.
            if AmpManager().scaler_enabled:
                AmpManager().register_special_term(nu, max_scale=2**20)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # dynamic viscosity
        mu = rho * nu

        tau_xx__x = 2 * mu * u.diff(x).diff(x) + 2 * mu.diff(x) * u.diff(x)
        tau_xy__y = mu * (u.diff(y).diff(y) + v.diff(x).diff(y)) + mu.diff(y) * (
            u.diff(y) + v.diff(x)
        )
        tau_xz__z = mu * (u.diff(z).diff(z) + w.diff(x).diff(z)) + mu.diff(z) * (
            u.diff(z) + w.diff(x)
        )
        tau_xy__x = mu * (u.diff(y).diff(x) + v.diff(x).diff(x)) + mu.diff(x) * (
            u.diff(y) + v.diff(x)
        )
        tau_yy__y = 2 * mu * v.diff(y).diff(y) + 2 * mu.diff(y) * v.diff(y)
        tau_yz__z = mu * (v.diff(z).diff(z) + w.diff(y).diff(z)) + mu.diff(z) * (
            v.diff(z) + w.diff(y)
        )
        tau_xz__x = mu * (u.diff(z).diff(x) + w.diff(x).diff(x)) + mu.diff(x) * (
            u.diff(z) + w.diff(x)
        )
        tau_yz__y = mu * (v.diff(z).diff(y) + w.diff(y).diff(y)) + mu.diff(y) * (
            v.diff(z) + w.diff(y)
        )
        tau_zz__z = 2 * mu * w.diff(z).diff(z) + 2 * mu.diff(z) * w.diff(z)

        # set equations
        self.equations = {}
        self.equations["continuity"] = u.diff(x) + v.diff(y) + w.diff(z)
        self.equations["momentum_x"] = (
            rho * (u * u.diff(x) + v * u.diff(y) + w * u.diff(z))
            + p.diff(x)
            - tau_xx__x
            - tau_xy__y
            - tau_xz__z
        )
        self.equations["momentum_y"] = (
            rho * (u * v.diff(x) + v * v.diff(y) + w * v.diff(z))
            + p.diff(y)
            - tau_xy__x
            - tau_yy__y
            - tau_yz__z
        )
        self.equations["momentum_z"] = (
            rho * (u * w.diff(x) + v * w.diff(y) + w * w.diff(z))
            + p.diff(z)
            - tau_xz__x
            - tau_yz__y
            - tau_zz__z
        )

        if self.dim == 2:
            self.equations.pop("momentum_z")


class GradNormal(PDE):
    """
    Implementation of the gradient boundary condition

    Parameters
    ==========
    T : str
        The dependent variable.
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    Examples
    ========
    >>> gn = ns = GradNormal(T='T')
    >>> gn.pprint()
      normal_gradient_T: normal_x*T__x + normal_y*T__y + normal_z*T__z
    """

    name = "GradNormal"

    def __init__(self, T, dim=3, time=True):
        self.T = T
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")

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

        # variables to set the gradients (example Temperature)
        T = Function(T)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["normal_gradient_" + self.T] = (
            normal_x * T.diff(x) + normal_y * T.diff(y) + normal_z * T.diff(z)
        )


class Curl(PDE):
    """
    del cross vector operator

    Parameters
    ==========
    vector : tuple of 3 Sympy Exprs, floats, ints or strings
        This will be the vector to take the curl of.
    curl_name : tuple of 3 strings
        These will be the output names of the curl operations.

    Examples
    ========
    >>> c = Curl((0,0,'phi'), ('u','v','w'))
    >>> c.pprint()
      u: phi__y
      v: -phi__x
      w: 0
    """

    name = "Curl"

    def __init__(self, vector, curl_name=["u", "v", "w"]):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # vector
        v_0 = vector[0]
        v_1 = vector[1]
        v_2 = vector[2]

        # make funtions
        if type(v_0) is str:
            v_0 = Function(v_0)(*input_variables)
        elif type(v_0) in [float, int]:
            v_0 = Number(v_0)
        if type(v_1) is str:
            v_1 = Function(v_1)(*input_variables)
        elif type(v_1) in [float, int]:
            v_1 = Number(v_1)
        if type(v_2) is str:
            v_2 = Function(v_2)(*input_variables)
        elif type(v_2) in [float, int]:
            v_2 = Number(v_2)

        # curl
        curl_0 = v_2.diff(y) - v_1.diff(z)
        curl_1 = v_0.diff(z) - v_2.diff(x)
        curl_2 = v_1.diff(x) - v_0.diff(y)

        # set equations
        self.equations = {}
        self.equations[curl_name[0]] = curl_0
        self.equations[curl_name[1]] = curl_1
        self.equations[curl_name[2]] = curl_2


class CompressibleIntegralContinuity(PDE):
    """
    Compressible Integral Continuity

    Parameters
    ==========
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressibility. Default is 1.
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    name = "CompressibleIntegralContinuity"

    def __init__(self, rho=1, vec=["u", "v", "w"]):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        self.dim = len(vec)
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")

        # normal
        normal = [Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")]

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # make input variables
        self.equations = {}
        self.equations["integral_continuity"] = 0
        for v, n in zip(vec, normal):
            self.equations["integral_continuity"] += Symbol(v) * n * rho


class FluxContinuity(PDE):
    """
    Flux Continuity for arbitrary variable. Includes advective and diffusive flux

    Parameters
    ==========
    T : str
        The dependent variable.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressibility. Default is 1.
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    name = "FluxContinuity"

    def __init__(self, T="T", D="D", rho=1, vec=["u", "v", "w"]):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        self.dim = len(vec)
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")

        # normal
        normal = [Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")]

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # diffusion coefficient
        if isinstance(D, str):
            D = Function(D)(*input_variables)
        elif isinstance(D, (float, int)):
            D = Number(D)

        # variables to set the flux (example Temperature)
        T = Function(T)(*input_variables)

        gradient = [T.diff(x), T.diff(y), T.diff(z)]

        # make input variables
        self.equations = {}
        self.equations[str(T) + "_flux"] = 0
        for v, n, g in zip(vec, normal, gradient):
            self.equations[str(T) + "_flux"] += (
                Symbol(v) * n * rho * T - rho * D * n * g
            )
