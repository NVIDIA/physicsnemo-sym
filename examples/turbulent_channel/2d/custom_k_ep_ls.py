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

from sympy import Symbol, Function, Number, log, Abs, simplify

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.node import Node


class kEpsilonInit(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        p = Function("p")(*input_variables)
        k = Function("k")(*input_variables)
        ep = Function("ep")(*input_variables)

        # flow initialization
        C_mu = 0.09
        u_avg = 21  # Approx average velocity
        Re_d = (
            u_avg * 1 / nu
        )  # Reynolds number based on centerline and channel hydraulic dia
        l = 0.038 * 2  # Approx turbulent length scale
        I = 0.16 * Re_d ** (
            -1 / 8
        )  # Turbulent intensity for a fully developed pipe flow

        u_init = u_avg
        v_init = 0
        p_init = 0
        k_init = 1.5 * (u_avg * I) ** 2
        ep_init = (C_mu ** (3 / 4)) * (k_init ** (3 / 2)) / l

        # set equations
        self.equations = {}
        self.equations["u_init"] = u - u_init
        self.equations["v_init"] = v - v_init
        self.equations["p_init"] = p - p_init
        self.equations["k_init"] = k - k_init
        self.equations["ep_init"] = ep - ep_init


class kEpsilon(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        p = Function("p")(*input_variables)
        k = Function("k")(*input_variables)
        ep = Function("ep")(*input_variables)

        # Model constants
        sig_k = 1.0
        sig_ep = 1.3
        C_ep1 = 1.44
        C_ep2 = 1.92
        C_mu = 0.09
        E = 9.793

        # Turbulent Viscosity
        nu_t = C_mu * (k**2) / (ep + 1e-4)

        # Turbulent Production Term
        P_k = nu_t * (
            2 * (u.diff(x)) ** 2
            + 2 * (v.diff(y)) ** 2
            + (u.diff(y)) ** 2
            + (v.diff(x)) ** 2
            + 2 * u.diff(y) * v.diff(x)
        )

        # set equations
        self.equations = {}
        self.equations["continuity"] = simplify(u.diff(x) + v.diff(y))
        self.equations["momentum_x"] = simplify(
            u * u.diff(x)
            + v * u.diff(y)
            + p.diff(x)
            - ((nu + nu_t) * u.diff(x)).diff(x)
            - ((nu + nu_t) * u.diff(y)).diff(y)
        )
        self.equations["momentum_y"] = simplify(
            u * v.diff(x)
            + v * v.diff(y)
            + p.diff(y)
            - ((nu + nu_t) * v.diff(x)).diff(x)
            - ((nu + nu_t) * v.diff(y)).diff(y)
        )
        self.equations["k_equation"] = simplify(
            u * k.diff(x)
            + v * k.diff(y)
            - ((nu + nu_t / sig_k) * k.diff(x)).diff(x)
            - ((nu + nu_t / sig_k) * k.diff(y)).diff(y)
            - P_k
            + ep
        )
        self.equations["ep_equation"] = simplify(
            u * ep.diff(x)
            + v * ep.diff(y)
            - ((nu + nu_t / sig_ep) * ep.diff(x)).diff(x)
            - ((nu + nu_t / sig_ep) * ep.diff(y)).diff(y)
            - (C_ep1 * P_k - C_ep2 * ep) * ep / (k + 1e-3)
        )


class kEpsilonLSWF(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        k = Function("k")(*input_variables)
        ep = Function("ep")(*input_variables)

        # normals
        normal_x = -1 * Symbol(
            "normal_x"
        )  # Multiply by -1 to flip the direction of normal
        normal_y = -1 * Symbol(
            "normal_y"
        )  # Multiply by -1 to flip the direction of normal

        # wall distance
        normal_distance = Function("normal_distance")(*input_variables)

        # Model constants
        C_mu = 0.09
        E = 9.793
        C_k = -0.36
        B_k = 8.15
        karman_constant = 0.4187

        # Turbulent Viscosity
        nu_t = C_mu * (k**2) / (ep + 1e-4)

        u_tau = (C_mu**0.25) * (k**0.5)
        y_plus = u_tau * normal_distance / nu
        u_plus = log(Abs(E * y_plus)) / karman_constant

        ep_true = (C_mu ** (3 / 4)) * (k ** (3 / 2)) / karman_constant / normal_distance

        u_parallel_to_wall = [
            u - (u * normal_x + v * normal_y) * normal_x,
            v - (u * normal_x + v * normal_y) * normal_y,
        ]
        du_parallel_to_wall_dx = [
            u.diff(x) - (u.diff(x) * normal_x + v.diff(x) * normal_y) * normal_x,
            v.diff(x) - (u.diff(x) * normal_x + v.diff(x) * normal_y) * normal_y,
        ]
        du_parallel_to_wall_dy = [
            u.diff(y) - (u.diff(y) * normal_x + v.diff(y) * normal_y) * normal_x,
            v.diff(y) - (u.diff(y) * normal_x + v.diff(y) * normal_y) * normal_y,
        ]

        du_dsdf = [
            du_parallel_to_wall_dx[0] * normal_x + du_parallel_to_wall_dy[0] * normal_y,
            du_parallel_to_wall_dx[1] * normal_x + du_parallel_to_wall_dy[1] * normal_y,
        ]

        wall_shear_stress_true_x = (
            u_tau * u_parallel_to_wall[0] * karman_constant / log(Abs(E * y_plus))
        )
        wall_shear_stress_true_y = (
            u_tau * u_parallel_to_wall[1] * karman_constant / log(Abs(E * y_plus))
        )

        wall_shear_stress_x = (nu + nu_t) * du_dsdf[0]
        wall_shear_stress_y = (nu + nu_t) * du_dsdf[1]

        u_normal_to_wall = u * normal_x + v * normal_y
        u_normal_to_wall_true = 0

        u_parallel_to_wall_mag = (
            u_parallel_to_wall[0] ** 2 + u_parallel_to_wall[1] ** 2
        ) ** 0.5
        u_parallel_to_wall_true = u_plus * u_tau

        k_normal_gradient = normal_x * k.diff(x) + normal_y * k.diff(y)
        k_normal_gradient_true = 0

        # set equations
        self.equations = {}
        self.equations["velocity_wall_normal_wf"] = (
            u_normal_to_wall - u_normal_to_wall_true
        )
        self.equations["velocity_wall_parallel_wf"] = (
            u_parallel_to_wall_mag - u_parallel_to_wall_true
        )
        self.equations["ep_wf"] = ep - ep_true
        self.equations["wall_shear_stress_x_wf"] = (
            wall_shear_stress_x - wall_shear_stress_true_x
        )
        self.equations["wall_shear_stress_y_wf"] = (
            wall_shear_stress_y - wall_shear_stress_true_y
        )
