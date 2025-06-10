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

import numpy as np

from physicsnemo.sym.utils.sympy import np_lambdify


def _compute_outvar(invar, outvar_sympy):
    outvar = {}
    for key in outvar_sympy.keys():
        outvar[key] = np_lambdify(outvar_sympy[key], {**invar})(**invar)
    return outvar


def _compute_lambda_weighting(invar, outvar, lambda_weighting_sympy):
    lambda_weighting = {}
    if lambda_weighting_sympy is None:
        for key in outvar.keys():
            lambda_weighting[key] = np.ones_like(next(iter(invar.values())))
    else:
        for key in outvar.keys():
            lambda_weighting[key] = np_lambdify(
                lambda_weighting_sympy[key], {**invar, **outvar}
            )(**invar, **outvar)
    return lambda_weighting
