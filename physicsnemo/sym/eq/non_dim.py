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
from typing import Dict

from physicsnemo.sym import quantity
from physicsnemo.sym.node import Node


class NonDimensionalizer:
    """
    Used for Non-dimensionalizion and normalization of physical quantities

    Parameters
    ----------
    length_scale : quantity
        length scale. Defaults to quantity(1.0, "m").
    time_scale : quantity
        time scale. Defaults to quantity(1.0, "s").
    mass_scale : quantity
        mass scale. Defaults to quantity(1.0, "kg").
    temperature_scale : quantity
        temperature scale. Defaults to quantity(1.0, "K").
    current_scale : quantity
        current scale. Defaults to quantity(1.0, "A").
    substance_scale : quantity
        substance scale. Defaults to quantity(1.0, "mol").
    luminosity_scale : quantity
        luminosity scale. Defaults to quantity(1.0, "cd").
    """

    def __init__(
        self,
        length_scale=quantity(1.0, "m"),
        time_scale=quantity(1.0, "s"),
        mass_scale=quantity(1.0, "kg"),
        temperature_scale=quantity(1.0, "K"),
        current_scale=quantity(1.0, "A"),
        substance_scale=quantity(1.0, "mol"),
        luminosity_scale=quantity(1.0, "cd"),
    ):
        self._print_scale(length_scale, "length")
        self._print_scale(time_scale, "time")
        self._print_scale(mass_scale, "mass")
        self._print_scale(temperature_scale, "temperature")
        self._print_scale(current_scale, "current")
        self._print_scale(substance_scale, "substance")
        self._print_scale(luminosity_scale, "luminosity")
        self.scale_dict = {
            "[length]": length_scale.to_base_units(),
            "[time]": time_scale.to_base_units(),
            "[mass]": mass_scale.to_base_units(),
            "[temperature]": temperature_scale.to_base_units(),
            "[current]": current_scale.to_base_units(),
            "[substance]": substance_scale.to_base_units(),
            "[luminosity]": luminosity_scale.to_base_units(),
        }

    def ndim(self, qty, return_unit=False):
        """
        Non-dimensionalize and normalize physical quantities

        Parameters
        ----------
        qty : quantity
            Physical quantity
        return_unit : bool
            If True, returns the non-dimensionalized and normalized value in for of a quantity with a "dimensionless" unit. If False, only returns the non-dimensionalized and normalized value
        """

        qty.ito_base_units()
        for key, value in dict(qty.dimensionality).items():
            qty /= self.scale_dict[key] ** value
        if dict(qty.dimensionality):
            raise RuntimeError("Error in non-dimensionalization")
        if return_unit:
            return qty
        else:
            return qty.magnitude

    def dim(self, invar, unit, return_unit=False):
        """
        Scales back a non-dimensionalized quantity or value to a quantity with a desired unit

        Parameters
        ----------
        invar : Any(quantity, float)
            Non-dimensionalized value or quantity
        unit:
            The target physical unit for the value or quantity
        return_unit : bool
            If True, returns the scaled value in for of a quantity with a unit. If False, only returns the scaled value
        """

        try:
            if dict(invar.dimensionality):
                raise RuntimeError("Error in dimensionalization")
        except:
            pass
        try:
            qty = quantity(invar, "")
        except:
            qty = invar
        dummy_qty = quantity(1, unit)
        dummy_qty.ito_base_units()
        for key, value in dict(dummy_qty.dimensionality).items():
            qty *= self.scale_dict[key] ** value
        qty.ito(unit)
        if return_unit:
            return qty
        else:
            return qty.magnitude

    def _print_scale(self, scale, name):
        """
        Print scales only if the default values are changed
        """
        if scale.magnitude != 1.0:
            print(f"{name} scale is {scale}")


class Scaler:
    """
    generates a PhysicsNeMo Node for scaling back non-dimensionalized and normalized quantities

    Parameters
    ----------
    invar : List[str]
        List of non-dimensionalized variable names to be scaled back
    outvar : List[str]
        List of names for the scaled variables.
    outvar_unit : List[str]
        List of unots for the scaled variables.
    non_dimensionalizer = NonDimensionalizer
        PhysicsNeMo non-dimensionalizer object
    """

    def __init__(self, invar, outvar, outvar_unit, non_dimensionalizer):
        self.invar = invar
        self.outvar = outvar
        self.outvar_unit = outvar_unit
        self.non_dimensionalizer = non_dimensionalizer

    def make_node(self):
        """
        generates a PhysicsNeMo Node
        """

        return [
            Node(
                inputs=self.invar,
                outputs=self.outvar,
                evaluate=_Scale(
                    self.invar, self.outvar, self.outvar_unit, self.non_dimensionalizer
                ),
            )
        ]


class _Scale(torch.nn.Module):
    """
    Scales back non-dimensionalized and normalized quantities

    Parameters
    ----------
    invar : List[str]
        List of non-dimensionalized variable names to be scaled back
    outvar : List[str]
        List of names for the scaled variables.
    outvar_unit : List[str]
        List of unots for the scaled variables.
    non_dimensionalizer = NonDimensionalizer
        PhysicsNeMo non-dimensionalizer object
    """

    def __init__(self, invar, outvar, outvar_unit, non_dimensionalizer):
        super().__init__()
        self.invar = invar
        self.outvar = outvar
        self.outvar_unit = outvar_unit
        self.non_dimensionalizer = non_dimensionalizer

    def forward(self, invar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outvar = {}
        for i, key in enumerate(self.invar):
            outvar[self.outvar[i]] = self.non_dimensionalizer.dim(
                invar[key], self.outvar_unit[i]
            )
        return outvar
