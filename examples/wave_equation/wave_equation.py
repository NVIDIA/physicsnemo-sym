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

"""Wave equation
Reference: https://en.wikipedia.org/wiki/Wave_equation
"""

from sympy import Symbol, Function, Number
from physicsnemo.sym.eq.pde import PDE


class WaveEquation1D(PDE):
    """
    Wave equation 1D
    The equation is given as an example for implementing
    your own PDE. A more universal implementation of the
    wave equation can be found by
    `from physicsnemo.sym.eq.pdes.wave_equation import WaveEquation`.

    Parameters
    ==========
    c : float, string
        Wave speed coefficient. If a string then the
        wave speed is input into the equation.
    """

    name = "WaveEquation1D"

    def __init__(self, c=1.0):
        # coordinates
        x = Symbol("x")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "t": t}

        # make u function
        u = Function("u")(*input_variables)

        # wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)

        # set equations
        self.equations = {}
        self.equations["wave_equation"] = u.diff(t, 2) - (c**2 * u.diff(x)).diff(x)
