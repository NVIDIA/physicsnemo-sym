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

"""PhysicsNeMo nodes"""

from sympy import Add
from .constants import diff_str
from .key import Key


class Node:
    """
    Base class for all nodes used to unroll computational graph in PhysicsNeMo.

    Parameters
    ----------
    inputs : List[Union[str, Key]]
        Names of inputs to node. For example, `inputs=['x', 'y']`.
    outputs : List[Union[str, Key]]
        Names of outputs to node. For example, `inputs=['u', 'v', 'p']`.
    evaluate : Pytorch Function
        A pytorch function that takes in a dictionary of tensors whose keys are the above `inputs`.
    name : str
        Name of node for print statements and debugging.
    optimize : bool
        If true then any trainable parameters contained in the node will be optimized by the `Trainer`.
    """

    def __init__(self, inputs, outputs, evaluate, name="Node", optimize=False):
        super().__init__()
        if isinstance(inputs, str):
            inputs = [inputs]
        self._inputs = Key.convert_list([x for x in inputs if diff_str not in str(x)])
        self._outputs = Key.convert_list(outputs)
        self._derivatives = Key.convert_list([x for x in inputs if diff_str in str(x)])
        self.evaluate = evaluate
        self._name = name
        self._optimize = optimize

        # set evaluate saveable to false if doesn't exist
        if not hasattr(self.evaluate, "saveable"):
            self.evaluate.saveable = False

        # check that model has name if optimizable
        if self._optimize:
            assert hasattr(self.evaluate, "name"), (
                "Optimizable nodes require model to have unique name"
            )

    @classmethod
    def from_sympy(cls, eq, out_name, freeze_terms=[], detach_names=[]):
        """
        generates a PhysicsNeMo Node from a SymPy equation

        Parameters
        ----------
        eq : Sympy Symbol/Exp
          the equation to convert to a PhysicsNeMo Node. The
          inputs to this node consist of all Symbols,
          Functions, and derivatives of Functions. For example,
          `f(x,y) + f(x,y).diff(x) + k` will be converted
          to a node whose input is [`f,f__x,k`].
        out_name : str
          This will be the name of the output for the node.
        freeze_terms : List[int]
          The terms that need to be frozen
        detach_names : List[str]
          This will detach the inputs of the resulting node.

        Returns
        -------
        node : Node
        """

        from physicsnemo.sym.utils.sympy.torch_printer import (
            _subs_derivatives,
            SympyToTorch,
        )

        # sub all functions and derivatives with symbols
        sub_eq = _subs_derivatives(eq)

        # construct PhysicsNeMo node
        if bool(freeze_terms):
            print(
                "the terms "
                + str(freeze_terms)
                + " will be frozen in the equation "
                + str(out_name)
                + ": "
                + str(Add.make_args(sub_eq))
            )
            print("Verify before proceeding!")
        else:
            pass
        evaluate = SympyToTorch(sub_eq, out_name, freeze_terms, detach_names)
        inputs = Key.convert_list(evaluate.keys)
        outputs = Key.convert_list([out_name])
        node = cls(inputs, outputs, evaluate, name="Sympy Node: " + out_name)
        return node

    @property
    def name(self):
        return self._name

    @property
    def outputs(self):
        """
        Returns
        -------
        outputs : List[str]
            Outputs of node.
        """

        return self._outputs

    @property
    def inputs(self):
        """
        Returns
        -------
        inputs : List[str]
            Inputs of node.
        """

        return self._inputs

    @property
    def derivatives(self):
        """
        Returns
        -------
        derivatives : List[str]
            Derivative inputs of node.
        """

        return self._derivatives

    @property
    def optimize(self):
        return self._optimize

    def __str__(self):
        return (
            "node: "
            + self.name
            + "\n"
            + "evaluate: "
            + str(self.evaluate.__class__.__name__)
            + "\n"
            + "inputs: "
            + str(self.inputs)
            + "\n"
            + "derivatives: "
            + str(self.derivatives)
            + "\n"
            + "outputs: "
            + str(self.outputs)
            + "\n"
            + "optimize: "
            + str(self.optimize)
        )
