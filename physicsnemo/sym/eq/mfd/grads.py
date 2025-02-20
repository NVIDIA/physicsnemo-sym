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

from physicsnemo.sym.key import Key
from typing import Dict, List

Tensor = torch.Tensor


class FirstDerivSecondOrder(torch.nn.Module):
    """Module to compute first derivative with 2nd order accuracy"""

    def __init__(self, var: str, indep_var: str, out_name: str) -> None:
        super().__init__()
        self.var = var
        self.indep_var = indep_var
        self.out_name = out_name

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        # [0.5, -0.5]
        outputs[self.out_name] = (0.5 / dx) * inputs[
            f"{self.var}>>{self.indep_var}::1"
        ] + (-0.5 / dx) * inputs[f"{self.var}>>{self.indep_var}::-1"]
        return outputs


class FirstDerivFourthOrder(torch.nn.Module):
    """Module to compute first derivative with 4th order accuracy"""

    def __init__(self, var: str, indep_var: str, out_name: str) -> None:
        super().__init__()
        self.var = var
        self.indep_var = indep_var
        self.out_name = out_name

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        # [-1.0 / 12.0, 8.0 / 12.0, -8.0 / 12.0, 1.0 / 12.0]
        outputs[self.out_name] = (
            (-1.0 / (dx * 12.0)) * inputs[f"{self.var}>>{self.indep_var}::2"]
            + (8.0 / (dx * 12.0)) * inputs[f"{self.var}>>{self.indep_var}::1"]
            + (-8.0 / (dx * 12.0)) * inputs[f"{self.var}>>{self.indep_var}::-1"]
            + (1.0 / (dx * 12.0)) * inputs[f"{self.var}>>{self.indep_var}::-2"]
        )
        return outputs


class SecondDerivSecondOrder(torch.nn.Module):
    """Module to compute second derivative with 2nd order accuracy"""

    def __init__(self, var: str, indep_var: str, out_name: str) -> None:
        super().__init__()
        self.var = var
        self.indep_var = indep_var
        self.out_name = out_name

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        # [1.0, -2.0, 1.0]
        outputs[self.out_name] = (
            (1.0 / (dx**2)) * inputs[f"{self.var}>>{self.indep_var}::1"]
            + (-2.0 / (dx**2)) * inputs[f"{self.var}"]
            + (1.0 / (dx**2)) * inputs[f"{self.var}>>{self.indep_var}::-1"]
        )
        return outputs


class SecondDerivFourthOrder(torch.nn.Module):
    """Module to compute second derivative with 4th order accuracy"""

    def __init__(self, var: str, indep_var: str, out_name: str) -> None:
        super().__init__()
        self.var = var
        self.indep_var = indep_var
        self.out_name = out_name

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        # [-1/12, 4/3, -5/2, 4/3, -1/12]
        outputs[self.out_name] = (
            (-1.0 / (12.0 * dx**2)) * inputs[f"{self.var}>>{self.indep_var}::2"]
            + (4.0 / (3.0 * dx**2)) * inputs[f"{self.var}>>{self.indep_var}::1"]
            + (-5.0 / (2.0 * dx**2)) * inputs[f"{self.var}"]
            + (4.0 / (3.0 * dx**2)) * inputs[f"{self.var}>>{self.indep_var}::-1"]
            + (-1.0 / (12.0 * dx**2)) * inputs[f"{self.var}>>{self.indep_var}::-2"]
        )
        return outputs


class MixedSecondDerivSecondOrder(torch.nn.Module):
    """Module to compute second mixed derivative with 2nd order accuracy

    Ref: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119083405.app1
    """

    def __init__(self, var: str, indep_vars: List[str], out_name: str) -> None:
        super().__init__()
        self.var = var
        self.indep_vars = indep_vars
        self.indep_vars.sort()
        self.out_name = out_name

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        outputs[self.out_name] = (
            (0.25 / (dx**2))
            * inputs[f"{self.var}>>{self.indep_vars[0]}::1&&{self.indep_vars[1]}::1"]
            + (-0.25 / (dx**2))
            * inputs[f"{self.var}>>{self.indep_vars[0]}::-1&&{self.indep_vars[1]}::1"]
            + (-0.25 / (dx**2))
            * inputs[f"{self.var}>>{self.indep_vars[0]}::1&&{self.indep_vars[1]}::-1"]
            + (0.25 / (dx**2))
            * inputs[f"{self.var}>>{self.indep_vars[0]}::-1&&{self.indep_vars[1]}::-1"]
        )
        return outputs


class ThirdDerivSecondOrder(torch.nn.Module):
    """Module to compute third derivative with 2nd order accuracy"""

    def __init__(self, var: str, indep_var: str, out_name: str) -> None:
        super().__init__()

        self.var = var
        self.indep_var = indep_var
        self.out_name = out_name

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        # [1/2, -1.0, 1.0, -1/2]
        outputs[self.out_name] = (
            (0.5 / (dx**3)) * inputs[f"{self.var}>>{self.indep_var}::2"]
            + (-1.0 / (dx**3)) * inputs[f"{self.var}>>{self.indep_var}::1"]
            + (1.0 / (dx**3)) * inputs[f"{self.var}>>{self.indep_var}::-1"]
            + (-0.5 / (dx**3)) * inputs[f"{self.var}>>{self.indep_var}::-2"]
        )
        return outputs


class FourthDerivSecondOrder(torch.nn.Module):
    """Module to compute fourth derivative with 2nd order accuracy"""

    def __init__(self, var: str, indep_var: str, out_name: str) -> None:
        super().__init__()
        self.var = var
        self.indep_var = indep_var
        self.out_name = out_name

    def forward(self, inputs: Dict[str, Tensor], dx: float) -> Dict[str, Tensor]:
        outputs = {}
        # [1.0, -4.0, 6.0, -4.0, 1.0]
        outputs[self.out_name] = (
            (1.0 / (dx**4)) * inputs[f"{self.var}>>{self.indep_var}::2"]
            + (-4.0 / (dx**4)) * inputs[f"{self.var}>>{self.indep_var}::1"]
            + (6.0 / (dx**4)) * inputs[f"{self.var}"]
            + (-4.0 / (dx**4)) * inputs[f"{self.var}>>{self.indep_var}::-1"]
            + (1.0 / (dx**4)) * inputs[f"{self.var}>>{self.indep_var}::-2"]
        )
        return outputs


class DerivBase(torch.nn.Module):
    """Base class for use of MFD derivatives in PhysicsNeMo Sym"""

    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__()
        self.derivative_keys = derivative_keys
        self.dx = dx
        self.order = order

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()

    @property
    def stencil(self) -> List[str]:
        """Returns list of stencil strings for this derivative

        Returns
        -------
        List[str]
            List of stencil strings

        Example
        -------
        Central 2nd derivative will return: `['x::1','x::0','x::-1']`
        """
        return list(self._stencil)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass that calculates the finite difference gradient

        Parameters
        ----------
        inputs : Dict[str, Tensor]
            Input tensor dictionary, should include points in FD stencil

        Returns
        -------
        Dict[str, Tensor]
            Output gradients
        """
        outputs = {}
        for module in self._eval:
            outputs.update(module(inputs, self.dx))
        return outputs


class FirstDeriv(DerivBase):
    """Wrapper class that initializes the required stencil and first derivative module"""

    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__(derivative_keys, dx, order, jit)
        assert (
            len(derivative_keys) == 0 or order == 2 or order == 4
        ), "Second and forth order first derivatives supported"
        for key in derivative_keys:
            assert (
                len(key.derivatives) == 1
            ), f"Key with {len(key.derivatives)} derivs supplied to first order deriv"

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()
        for key in self.derivative_keys:
            indep_vars = key.derivatives
            if order == 2:
                self._stencil = self._stencil.union(
                    {f"{indep_vars[0]}::-1", f"{indep_vars[0]}::1"}
                )
                assert (
                    len(key.derivatives) == 1
                ), f"Key must have one derivative for first derivative calc"
                eval_list.append(
                    FirstDerivSecondOrder(key.name, str(key.derivatives[0]), str(key))
                )
            elif order == 4:
                self._stencil = self._stencil.union(
                    {
                        f"{indep_vars[0]}::-2",
                        f"{indep_vars[0]}::-1",
                        f"{indep_vars[0]}::1",
                        f"{indep_vars[0]}::2",
                    }
                )
                assert (
                    len(key.derivatives) == 1
                ), f"Key must have one derivative for first derivative calc"
                eval_list.append(
                    FirstDerivFourthOrder(key.name, str(key.derivatives[0]), str(key))
                )

        self._eval = torch.nn.ModuleList(eval_list)


class SecondDeriv(DerivBase):
    """Wrapper class that initializes the required stencil and second derivative module"""

    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__(derivative_keys, dx, order, jit)
        assert (
            len(derivative_keys) == 0 or order == 2 or order == 4
        ), "Second and forth order second derivatives supported"
        for key in derivative_keys:
            assert (
                len(key.derivatives) == 2
            ), f"Key with {len(key.derivatives)} deriv keys supplied to second deriv"

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()
        for key in self.derivative_keys:
            indep_vars = key.derivatives
            if indep_vars[0] == indep_vars[1]:
                if order == 2:
                    self._stencil = self._stencil.union(
                        {
                            f"{indep_vars[0]}::-1",
                            f"{indep_vars[0]}::0",
                            f"{indep_vars[0]}::1",
                        }
                    )
                    assert (
                        len(key.derivatives) == 2
                    ), f"Key must have two derivatives for second derivative calc"
                    assert (
                        key.derivatives[0] == key.derivatives[1]
                    ), f"Derivatives keys should be the same"
                    eval_list.append(
                        SecondDerivSecondOrder(
                            key.name, str(key.derivatives[0]), str(key)
                        )
                    )
                elif order == 4:
                    self._stencil = self._stencil.union(
                        {
                            f"{indep_vars[0]}::-2",
                            f"{indep_vars[0]}::-1",
                            f"{indep_vars[0]}::0",
                            f"{indep_vars[0]}::1",
                            f"{indep_vars[0]}::2",
                        }
                    )
                    assert (
                        len(key.derivatives) == 2
                    ), f"Key must have two derivatives for second derivative calc"
                    assert (
                        key.derivatives[0] == key.derivatives[1]
                    ), f"Derivatives keys should be the same"
                    eval_list.append(
                        SecondDerivFourthOrder(
                            key.name, str(key.derivatives[0]), str(key)
                        )
                    )
            # Mixed derivative
            else:
                if order == 2:
                    indep_vars = [str(var) for var in indep_vars]
                    indep_vars.sort()  # Avoid redundent points like (z::-1&&y::1 and y::1&&z::-1)
                    self._stencil = self._stencil.union(
                        {
                            f"{indep_vars[0]}::-1&&{indep_vars[1]}::-1",
                            f"{indep_vars[0]}::1&&{indep_vars[1]}::-1",
                            f"{indep_vars[0]}::-1&&{indep_vars[1]}::1",
                            f"{indep_vars[0]}::1&&{indep_vars[1]}::1",
                        }
                    )
                    assert (
                        len(key.derivatives) == 2
                    ), f"Key must have two derivatives for second derivative calc"
                    eval_list.append(
                        MixedSecondDerivSecondOrder(
                            key.name,
                            [str(key.derivatives[0]), str(key.derivatives[1])],
                            str(key),
                        )
                    )
                elif order == 4:
                    raise NotImplementedError(
                        "Fourth order mixed second derivatives not supported"
                    )

        self._eval = torch.nn.ModuleList(eval_list)


class ThirdDeriv(DerivBase):
    """Wrapper class that initializes the required stencil and third derivative module"""

    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__(derivative_keys, dx, order, jit)
        assert (
            len(derivative_keys) == 0 or order == 2
        ), "Second order third derivatives supported"
        for key in derivative_keys:
            assert (
                len(key.derivatives) == 3
            ), f"Key with {len(key.derivatives)} deriv keys supplied to third deriv"
            assert (
                key.derivatives[0] == key.derivatives[1] == key.derivatives[2]
            ), f"Mixed third derivatives not supported"

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()
        for key in self.derivative_keys:
            indep_vars = key.derivatives
            if order == 2:
                self._stencil = self._stencil.union(
                    {
                        f"{indep_vars[0]}::-2",
                        f"{indep_vars[0]}::-1",
                        f"{indep_vars[0]}::1",
                        f"{indep_vars[0]}::2",
                    }
                )
                assert (
                    len(key.derivatives) == 3
                ), f"Key must have three derivatives for third derivative calc"
                assert (
                    key.derivatives[0] == key.derivatives[1] == key.derivatives[2]
                ), f"Derivatives keys should be the same"
                eval_list.append(
                    ThirdDerivSecondOrder(key.name, str(key.derivatives[0]), str(key))
                )

        self._eval = torch.nn.ModuleList(eval_list)


class FourthDeriv(DerivBase):
    """Wrapper class that initializes the required stencil and fourth derivative module"""

    def __init__(
        self, derivative_keys: List[Key], dx: float, order: int = 2, jit: bool = True
    ) -> None:
        super().__init__(derivative_keys, dx, order, jit)
        assert (
            len(derivative_keys) == 0 or order == 2
        ), "Second order forth derivatives supported"
        for key in derivative_keys:
            assert (
                len(key.derivatives) == 4
            ), f"Key with {len(key.derivatives)} deriv keys supplied to forth deriv"
            assert (
                key.derivatives[0]
                == key.derivatives[1]
                == key.derivatives[2]
                == key.derivatives[3]
            ), f"Mixed forth derivatives not supported"

        # Create stencil set of points we need
        eval_list = []
        self._stencil = set()
        for key in self.derivative_keys:
            indep_vars = key.derivatives
            if order == 2:
                self._stencil = self._stencil.union(
                    {
                        f"{indep_vars[0]}::-2",
                        f"{indep_vars[0]}::-1",
                        f"{indep_vars[0]}::0",
                        f"{indep_vars[0]}::1",
                        f"{indep_vars[0]}::2",
                    }
                )
                assert (
                    len(key.derivatives) == 4
                ), f"Key must have three derivatives for forth derivative calc"
                assert (
                    key.derivatives[0]
                    == key.derivatives[1]
                    == key.derivatives[2]
                    == key.derivatives[3]
                ), f"Derivatives keys should be the same"
                eval_list.append(
                    FourthDerivSecondOrder(key.name, str(key.derivatives[0]), str(key))
                )

        self._eval = torch.nn.ModuleList(eval_list)
