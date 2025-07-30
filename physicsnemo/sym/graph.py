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

"""Helper functions for unrolling computational graph"""

from copy import copy
import torch
import logging
from typing import Dict, List, Optional

from .amp import DerivScalers
from .models.arch import Arch, FuncArch
from .node import Node
from .key import Key
from .eq.derivatives import Derivative
from .manager import JitManager, GraphManager

logger = logging.getLogger(__name__)


class Graph(torch.nn.Module):
    """
    Torch Module that is constructed by unrolling a computational graph given
    desired inputs, outputs, and evaluatable nodes.

    Examples
    ========
    Here is a simple example of using `Graph` to unroll a two node graph.
    >>> import torch
    >>> from sympy import Symbol
    >>> from physicsnemo.sym.node import Node
    >>> from physicsnemo.sym.key import Key
    >>> from physicsnemo.sym.graph import Graph
    >>> node_1 = Node.from_sympy(Symbol('x') + Symbol('y'), 'u')
    >>> node_2 = Node.from_sympy(Symbol('u') + 1.0, 'v')
    >>> graph = Graph([node_1, node_2], [Key('x'), Key('y')], [Key('v')])
    >>> graph.forward({'x': torch.tensor([1.0]), 'y': torch.tensor([2.0])})
    {'v': tensor([4.])}

    Parameters
    ----------
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    invar : List[Key]
        List of inputs to graph.
    req_names : List[Key]
        List of required outputs of graph.
    diff_nodes : List[Node]
        List of specialty nodes to compute derivatives.
        By default this is not needed.
    func_arch : bool, Optional
        If True, find the computable derivatives that are part of the Jacobian and
        Hessian of the neural network. They will be calculated during the forward
        pass using FuncArch.
        If None (default), will use the GraphManager to get the global flag
        (default is False), which could be configured in the hydra config with key
        `graph.func_arch`.
    func_arch_allow_partial_hessian : bool, Optional
        If True, allow evaluating partial hessian to save some unnecessary computations.
        For example, when the input is x, outputs are [u, p], and the needed derivatives
        are `[u__x, p__x, u__x__x]`, func_arch needs to evaluate the full hessian rows
        to be able to extract jacobian `p__x`. When this flag is on, func_arch will
        only output `[u__x, u__x__x]`, and `p__x` will be evaluated later by the autograd.
        If None (default), will use the GraphManager to get the global flag
        (default is True), which could be configured in the hydra config with key
        `graph.func_arch_allow_partial_hessian`.
    """

    def __init__(
        self,
        nodes: List[Node],
        invar: List[Key],
        req_names: List[Key],
        diff_nodes: List[Node] = [],
        func_arch: Optional[bool] = None,
        func_arch_allow_partial_hessian: Optional[bool] = None,
    ):
        super().__init__()

        # get configs from the graph manager
        graph_manager = GraphManager()
        func_arch = func_arch if func_arch is not None else graph_manager.func_arch
        func_arch_allow_partial_hessian = (
            func_arch_allow_partial_hessian
            if func_arch_allow_partial_hessian is not None
            else graph_manager.func_arch_allow_partial_hessian
        )

        self.req_names = req_names
        self.computable_names = set(_computable_names(nodes, invar))

        # check if graph can be computed
        req_names_no_diff = [Key(x.name) for x in req_names]
        if not set(req_names_no_diff).issubset(self.computable_names):
            _print_graph_unroll_error(nodes, invar, req_names)
            raise RuntimeError("Failed Unrolling Graph")

        # compute only necessary nodes for req_names
        # Walk backwards from the output nodes in the graph and keep adding required inputs
        # until all inputs are available in invar
        nodes = copy(nodes)
        necessary_nodes = []
        needed_names = [Key(x.name, derivatives=x.derivatives) for x in req_names] + [
            Key(x.name) for x in req_names
        ]
        while True:
            finished = True
            for i, node in enumerate(nodes):
                if not set(node.outputs).isdisjoint(set(needed_names)):
                    # Make needed names include derivatives!
                    needed_names += (
                        node.inputs
                        + [
                            Key(x.name, derivatives=x.derivatives)
                            for x in node.derivatives
                        ]
                        + [Key(x.name) for x in node.derivatives]
                    )
                    # needed_names.update(node.inputs() + [Key(x.name) for x in node.derivatives()])
                    necessary_nodes.append(node)
                    nodes.pop(i)
                    finished = False
            if finished:
                break

        # Convert arch node intto func_arch node if we find computable derivatives and the Arch
        # instance has supports_func_arch == True
        needed_names = set(needed_names)
        if func_arch:
            for i, node in enumerate(necessary_nodes):
                # `jit_mode_arch` is forced to be `only_activation` when func_arch is enabled,
                # so all Arch instances will not be `RecursiveScriptModules` and we are good
                # to transform it into FuncArch
                if isinstance(node.evaluate, Arch):
                    if node.evaluate.supports_func_arch:
                        computable_derivatives = (
                            node.evaluate._find_computable_deriv_with_func_arch(
                                needed_names, func_arch_allow_partial_hessian
                            )
                        )
                        if len(computable_derivatives):
                            node_name = necessary_nodes[i].name
                            necessary_nodes[i] = FuncArch(
                                node.evaluate, computable_derivatives
                            ).make_node(node_name)
                            logger.info(
                                f"{node_name} has been converted to a FuncArch node."
                            )
                    else:
                        logger.warning(
                            f"Arch {type(node.evaluate)} currently does not support FuncArch"
                        )

        # unroll graph with only necessary nodes
        # Store node evaluation order to use at runtime
        self.node_evaluation_order = []
        outvar = copy(invar)
        while True:
            # compute all nodes that don't need derivative calls
            while True:
                finished = True
                for i, node in enumerate(necessary_nodes):
                    if set(node.inputs + node.derivatives).issubset(set(outvar)):
                        self.node_evaluation_order.append(node)
                        outvar += node.outputs
                        necessary_nodes.pop(i)
                        finished = False
                if finished:
                    break
            # compute derivative calls all at once
            needed_derivatives = []
            for node in necessary_nodes:
                needed_derivatives += node.derivatives
            needed_derivatives += [x for x in req_names if x.derivatives]
            needed_derivatives = [
                diff for diff in needed_derivatives if diff not in outvar
            ]  # remove already computed diffs
            if len(needed_derivatives) > 0:
                # check if solution in diff nodes
                try_auto_diff = True
                for dn in diff_nodes:
                    if (not set(dn.outputs).isdisjoint(set(needed_derivatives))) and (
                        set(dn.inputs).issubset(set(outvar))
                    ):
                        # input_variables = Variables.subset(outvar, dn.inputs())
                        # outvar.update(dn.evaluate(input_variables))
                        self.node_evaluation_order.append(dn)
                        outvar += dn.outputs
                        try_auto_diff = False

                # compute first derivatives only
                if try_auto_diff:
                    # Variables.differentiate(outvar, outvar, needed_derivatives)
                    dnode = Derivative.make_node(
                        outvar,
                        needed_derivatives,
                        jit=(JitManager().enabled and JitManager().autograd_nodes),
                    )
                    self.node_evaluation_order.append(dnode)
                    outvar += dnode.outputs

            # check if finished
            if set(req_names).issubset(set(outvar)):
                # return Variables({key: value for key, value in outvar.items() if key in req_names})
                break

        self.evaluation_order = torch.nn.ModuleList(
            [n.evaluate for n in self.node_evaluation_order]
        )
        self.node_names: List[str] = [n.name for n in self.node_evaluation_order]
        self.optimizer_list = torch.nn.ModuleList(
            [n.evaluate for n in self.node_evaluation_order if n.optimize]
        )

        if graph_manager.debug:
            print(self)

    def forward(self, invar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outvar = invar
        for i, e in enumerate(self.evaluation_order):
            torch.cuda.nvtx.range_push(self.node_names[i])
            outvar.update(e(outvar))
            torch.cuda.nvtx.range_pop()
        outvar = {
            key: value for key, value in outvar.items() if Key(key) in self.req_names
        }
        return outvar

    def setup_deriv_scaler(self, deriv_scalers: DerivScalers, name: str = ""):
        """
        Setup derivative scalers for Derivative node and FuncArch node in the
        graph.
        """
        for i, node in enumerate(self.node_evaluation_order):
            if isinstance(node.evaluate, Derivative):
                node.evaluate.setup_deriv_scaler(deriv_scalers)
            if isinstance(node.evaluate, FuncArch):
                node.evaluate.setup_deriv_scaler(deriv_scalers)
            # If node.evaluate is a jitted Derivative instance, we do not have
            # access to the original Derivative instance and could not setup
            # derivative scalers, so we raise an error here.
            if (
                isinstance(node.evaluate, torch.jit._script.RecursiveScriptModule)
                and node.evaluate.original_name == "Derivative"
            ):
                raise RuntimeError(
                    "AMP FP16 does not work for TorchScripted Derivative node"
                )

    def __str__(self):
        repr = "=" * 100 + "\n"
        for node in self.node_evaluation_order:
            repr += "-" * 50 + "\n"
            repr += str(node) + "\n"
        return repr


def _print_graph_unroll_error(nodes, invar, req_names):
    print("####################################")
    print("could not unroll graph!")
    print(
        "This is probably because you are asking to compute a value that is not an output of any node"
    )
    print("####################################")
    print("invar: " + str(list(invar)))
    print("requested var: " + str(req_names))
    print("computable var: " + str(_computable_names(nodes, invar)))
    print("####################################")
    print("Nodes in graph: ")
    for node in nodes:
        print(node)
    print("####################################")


def _computable_names(nodes, invar):
    nodes = copy(nodes)
    computable_names = copy(invar)
    while True:
        finished = True
        for i, node in enumerate(nodes):
            if set(node.inputs).issubset(set(computable_names)):
                computable_names += node.outputs
                nodes.pop(i)
                finished = False
        if finished:
            return computable_names
