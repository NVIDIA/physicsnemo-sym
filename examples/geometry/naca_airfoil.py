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

import math
import matplotlib.pyplot as plt
import numpy as np
from sympy import Number, Symbol, Heaviside, atan, sin, cos, sqrt
import os

from physicsnemo.sym.geometry.primitives_2d import Polygon
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk


# Naca implementation modified from https://stackoverflow.com/questions/31815041/plotting-a-naca-4-series-airfoil
# https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
def camber_line(x, m, p, c):
    cl = []
    for xi in x:
        cond_1 = Heaviside(xi, 0) * Heaviside((c * p) - xi, 0)
        cond_2 = Heaviside(-xi, 0) + Heaviside(xi - (c * p), 0)
        v_1 = m * (xi / p**2) * (2.0 * p - (xi / c))
        v_2 = m * ((c - xi) / (1 - p) ** 2) * (1.0 + (xi / c) - 2.0 * p)
        cl.append(cond_1 * v_1 + cond_2 * v_2)
    return cl


def dyc_over_dx(x, m, p, c):
    dd = []
    for xi in x:
        cond_1 = Heaviside(xi) * Heaviside((c * p) - xi)
        cond_2 = Heaviside(-xi) + Heaviside(xi - (c * p))
        v_1 = ((2.0 * m) / p**2) * (p - xi / c)
        v_2 = (2.0 * m) / (1 - p**2) * (p - xi / c)
        dd.append(atan(cond_1 * v_1 + cond_2 * v_2))
    return dd


def thickness(x, t, c):
    th = []
    for xi in x:
        term1 = 0.2969 * (sqrt(xi / c))
        term2 = -0.1260 * (xi / c)
        term3 = -0.3516 * (xi / c) ** 2
        term4 = 0.2843 * (xi / c) ** 3
        term5 = -0.1015 * (xi / c) ** 4
        th.append(5 * t * c * (term1 + term2 + term3 + term4 + term5))
    return th


def naca4(x, m, p, t, c=1):
    th = dyc_over_dx(x, m, p, c)
    yt = thickness(x, t, c)
    yc = camber_line(x, m, p, c)
    line = []
    for xi, thi, yti, yci in zip(x, th, yt, yc):
        line.append((xi - yti * sin(thi), yci + yti * cos(thi)))
    x.reverse()
    th.reverse()
    yt.reverse()
    yc.reverse()
    for xi, thi, yti, yci in zip(x, th, yt, yc):
        line.append((xi + yti * sin(thi), yci - yti * cos(thi)))
    return line


if __name__ == "__main__":
    # make parameters for naca airfoil
    m = 0.02
    p = 0.4
    t = 0.12
    c = 1.0

    # make naca geometry
    x = [x for x in np.linspace(0, 0.2, 10)] + [x for x in np.linspace(0.2, 1.0, 10)][
        1:
    ]  # higher res in front
    line = naca4(x, m, p, t, c)[:-1]
    geo = Polygon(line)

    # sample different parameters
    s = geo.sample_boundary(nr_points=100000)
    var_to_polyvtk(s, "naca_boundary")
    s = geo.sample_interior(nr_points=100000)
    var_to_polyvtk(s, "naca_interior")
