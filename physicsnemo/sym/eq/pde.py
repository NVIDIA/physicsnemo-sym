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

""" base class for PDEs
"""

from sympy import (
    Symbol,
    Function,
    init_printing,
    pprint,
    latex,
    preview,
    Matrix,
    Eq,
    Basic,
)
from typing import Dict, Tuple, List, Union

from physicsnemo.sym.node import Node
from physicsnemo.sym.constants import diff_str
from physicsnemo.sym.key import Key


class PDE(object):
    """base class for all partial differential equations"""

    name = "PDE"

    def __init__(self):
        super().__init__()
        self.equations = Variables()

    def pprint(self, print_latex=False):
        """
        Print differential equation.

        Parameters
        ----------
        print_latex : bool
            If True print the equations in Latex. Else, just
            print as text.
        """
        init_printing(use_latex=True)
        for key, value in self.equations.items():
            print(str(key) + ": " + str(value))
        if print_latex:
            preview(
                Matrix(
                    [
                        Eq(Function(name, real=True), eq)
                        for name, eq in self.equations.items()
                    ]
                ),
                mat_str="cases",
                mat_delim="",
            )

    def subs(self, x, y):
        for name, eq in self.equations.items():
            self.equations[name] = eq.subs(x, y).doit()

    def make_nodes(
        self,
        create_instances: int = 1,
        freeze_terms: Dict[str, List[int]] = {},
        detach_names: List[str] = [],
    ):
        """
        Make a list of nodes from PDE.

        Parameters
        ----------
        create_instances : int
            This will create various instances of the same equations
        freeze_terms : Dict[str, List[int]]
            This will freeze the terms in appropiate equation
        detach_names : List[str]
            This will detach the inputs of the resulting node.

        Returns
        -------
        nodes : List[Node]
            Makes a separate node for every equation.
        """
        nodes = []
        if create_instances == 1:
            if bool(freeze_terms):
                print(
                    "Freezing of terms is not supported when create_instance = 1. No terms will be frozen!"
                )
                freeze_terms = {}  # override with an empty dict
            for name, eq in self.equations.items():
                nodes.append(Node.from_sympy(eq, str(name), freeze_terms, detach_names))
        else:
            # look for empty lists in freeze_terms dict
            for k in list(freeze_terms):
                if not freeze_terms[k]:
                    freeze_terms.pop(k)
            for i in range(create_instances):
                for name, eq in self.equations.items():
                    if str(name) + "_" + str(i) in freeze_terms.keys():
                        nodes.append(
                            Node.from_sympy(
                                eq,
                                str(name) + "_" + str(i),
                                freeze_terms[str(name) + "_" + str(i)],
                                detach_names,
                            )
                        )
                    else:
                        # set the freeze terms to an empty list
                        print(
                            "No freeze terms found for instance: "
                            + str(name)
                            + "_"
                            + str(i)
                            + ", setting to empty"
                        )
                        nodes.append(
                            Node.from_sympy(
                                eq,
                                str(name) + "_" + str(i),
                                [],
                                detach_names,
                            )
                        )
        return nodes
