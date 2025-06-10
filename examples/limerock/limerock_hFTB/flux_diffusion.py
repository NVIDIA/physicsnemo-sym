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

from sympy import tanh, Symbol, Function

from physicsnemo.sym.eq.pde import PDE


class FluxDiffusion(PDE):
    name = "FluxDiffusion"

    def __init__(self, D=0.01):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # Flux of Temperature
        flux_theta_s_x = Function("flux_theta_s_x")(*input_variables)
        flux_theta_s_y = Function("flux_theta_s_y")(*input_variables)
        flux_theta_s_z = Function("flux_theta_s_z")(*input_variables)

        # set equations
        self.equations = {}
        self.equations["diffusion_theta_s"] = -(
            (D * flux_theta_s_x).diff(x)
            + (D * flux_theta_s_y).diff(y)
            + (D * flux_theta_s_z).diff(z)
        )
        self.equations["compatibility_theta_s_x_y"] = D * (
            flux_theta_s_x.diff(y) - flux_theta_s_y.diff(x)
        )
        self.equations["compatibility_theta_s_x_z"] = D * (
            flux_theta_s_x.diff(z) - flux_theta_s_z.diff(x)
        )
        self.equations["compatibility_theta_s_y_z"] = D * (
            flux_theta_s_y.diff(z) - flux_theta_s_z.diff(y)
        )


class FluxIntegrateDiffusion(PDE):
    name = "IntegrateDiffusion"

    def __init__(self):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # Temperature
        theta_s = Function("theta_s")(*input_variables)
        flux_theta_s_x = Function("flux_theta_s_x")(*input_variables)
        flux_theta_s_y = Function("flux_theta_s_y")(*input_variables)
        flux_theta_s_z = Function("flux_theta_s_z")(*input_variables)

        # set equations
        self.equations = {}
        self.equations["integrate_diffusion_theta_s_x"] = (
            theta_s.diff(x) - flux_theta_s_x
        )
        self.equations["integrate_diffusion_theta_s_y"] = (
            theta_s.diff(y) - flux_theta_s_y
        )
        self.equations["integrate_diffusion_theta_s_z"] = (
            theta_s.diff(z) - flux_theta_s_z
        )


class FluxGradNormal(PDE):
    def __init__(self):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # variables to set the gradients (example Temperature)
        flux_theta_s_x = Function("flux_theta_s_x")(*input_variables)
        flux_theta_s_y = Function("flux_theta_s_y")(*input_variables)
        flux_theta_s_z = Function("flux_theta_s_z")(*input_variables)

        # set equations
        self.equations = {}
        self.equations["normal_gradient_flux_theta_s"] = (
            normal_x * flux_theta_s_x
            + normal_y * flux_theta_s_y
            + normal_z * flux_theta_s_z
        )


class FluxRobin(PDE):
    def __init__(self, theta_f_conductivity, theta_s_conductivity, h):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        normal_x = Symbol("normal_x")
        normal_y = Symbol("normal_y")
        normal_z = Symbol("normal_z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # variables to set the gradients (example Temperature)
        theta_s = Function("theta_s")(*input_variables)
        flux_theta_s_x = Function("flux_theta_s_x")(*input_variables)
        flux_theta_s_y = Function("flux_theta_s_y")(*input_variables)
        flux_theta_s_z = Function("flux_theta_s_z")(*input_variables)
        theta_f = Function("theta_f_prev_step")(*input_variables)

        # set equations
        flux_theta_f = -theta_f_conductivity * (
            normal_x * theta_f.diff(x)
            + normal_y * theta_f.diff(y)
            + normal_z * theta_f.diff(z)
        )
        ambient_theta_f = theta_f - (flux_theta_f / h)
        flux_theta_s = -theta_s_conductivity * (
            normal_x * flux_theta_s_x
            + normal_y * flux_theta_s_y
            + normal_z * flux_theta_s_z
        )
        self.equations = {}
        self.equations["robin_theta_s"] = (
            flux_theta_s - h * (theta_s - ambient_theta_f)
        ) / theta_s_conductivity


class Dirichlet(PDE):
    def __init__(self, lhs="theta_s", rhs="theta_f"):
        # save name for u
        self.lhs = lhs
        self.rhs = rhs

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # functions
        rhs = Function(rhs)(*input_variables)
        lhs = Function(lhs)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["dirichlet_" + self.rhs + "_" + self.lhs] = rhs - lhs
