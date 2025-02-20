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

import torch
import numpy as np
from sympy import Symbol, sin

from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.dataset import (
    DictImportanceSampledPointwiseIterableDataset,
)
from physicsnemo.sym.domain.constraint.utils import _compute_outvar
from physicsnemo.sym.geometry.parameterization import Bounds


def test_DictImportanceSampledPointwiseIterableDataset():
    "sample sin function on a rectangle with importance measure sqrt(x**2 + y**2) and check its integral is zero"

    torch.manual_seed(123)
    np.random.seed(123)

    # make rectangle
    rec = Rectangle((-0.5, -0.5), (0.5, 0.5))

    # sample interior
    invar = rec.sample_interior(
        100000,
        bounds=Bounds({Symbol("x"): (-0.5, 0.5), Symbol("y"): (-0.5, 0.5)}),
    )

    # compute outvar
    outvar = _compute_outvar(invar, {"u": sin(2 * np.pi * Symbol("x") / 0.5)})

    # create importance measure
    def importance_measure(invar):
        return ((invar["x"] ** 2 + invar["y"] ** 2) ** (0.5)) + 0.01

    # make importance dataset
    dataset = DictImportanceSampledPointwiseIterableDataset(
        invar=invar,
        outvar=outvar,
        batch_size=10000,
        importance_measure=importance_measure,
    )

    # sample importance dataset
    invar, outvar, lambda_weighting = next(iter(dataset))

    # check integral calculation
    assert np.isclose(torch.sum(outvar["u"] * invar["area"]), 0.0, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":

    test_DictImportanceSampledPointwiseIterableDataset()
