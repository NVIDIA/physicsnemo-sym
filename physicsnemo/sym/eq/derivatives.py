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

import itertools
import torch
import logging

from physicsnemo.sym.amp import DerivScaler, DerivScalers, AmpManager, AmpMode
from physicsnemo.sym.constants import diff
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.mfd import FirstDeriv, SecondDeriv, ThirdDeriv, FourthDeriv

from typing import Dict, List, Optional, Union, Callable

Tensor = torch.Tensor
logger = logging.getLogger(__name__)


# ==== Autodiff ====
@torch.jit.script
def gradient_autodiff(y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    TorchScript function to compute the gradient of a tensor wrt multiple inputs
    """
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y, device=y.device)]
    grad = torch.autograd.grad(
        [
            y,
        ],
        x,
        grad_outputs=grad_outputs,
        create_graph=True,
        allow_unused=True,
    )
    if grad is None:
        grad = [torch.zeros_like(xx) for xx in x]
    assert grad is not None
    grad = [g if g is not None else torch.zeros_like(x[i]) for i, g in enumerate(grad)]
    return grad


class Derivative(torch.nn.Module):
    """
    Module to compute derivatives using backward automatic differentiation
    """

    def __init__(self, bwd_derivative_dict: Dict[Key, List[Key]]):
        """
        Constructor of the Derivative class.

        Parameters
        ----------
        inputs : List[Key]
            A list of keys of the available variables to compute the required variables
            This list should contain both the variables that need to be differentiated
            and the variables to differentiate with respect to.
        derivatives : List[Key]
            A list of keys of the required derivatives
        """
        super().__init__()

        self.gradient_dict: Dict[str, Dict[str, int]] = {
            str(k): {str(w): w.size for w in v} for k, v in bwd_derivative_dict.items()
        }
        self.gradient_names: Dict[str, List[str]] = {
            k: [diff(k, der) for der in v.keys()] for k, v in self.gradient_dict.items()
        }
        self.nvtx_str: str = f"Auto-Diff Node: {list(self.gradient_dict.keys())}"

        self.scaler_enabled: bool = False
        self.deriv_scalers: Dict[str, DerivScaler] = {}

    @staticmethod
    def prepare_input(
        input_variables: Dict[str, torch.Tensor], mask: List[str]
    ) -> List[torch.Tensor]:
        return [input_variables[x] for x in mask]

    @staticmethod
    def dict_output(
        output_tensors: List[torch.Tensor], sizes: List[str], var_name: str
    ) -> Dict[str, torch.Tensor]:
        return {diff(var_name, name): output_tensors[i] for i, name in enumerate(sizes)}

    # GradScaler and DerivScalers are shared python objects which could/should not
    # be torchscripted.
    @torch.jit.ignore
    def _scale(self, var: torch.Tensor, var_name: str) -> torch.Tensor:
        return self.deriv_scalers[var_name].scale(var)

    @torch.jit.ignore
    def _unscale(self, grad: List[torch.Tensor], var_name: str) -> List[torch.Tensor]:
        return self.deriv_scalers[var_name].unscale_deriv(grad)

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_var = {}
        for var_name, grad_sizes in self.gradient_dict.items():
            var = input_var[var_name]
            grad_var = self.prepare_input(input_var, grad_sizes.keys())

            if self.scaler_enabled:
                var = self._scale(var, var_name)

            # Calculate gradient using Autodiff.
            with torch.cuda.amp.autocast(enabled=False):
                # Autocast needs to be disabled because using Autograd within the autocast region
                # is not recommended and could have unpredictable behavior. Autocast relies on
                # thread-local configurations, and since Autograd runs in a different thread, it
                # could use a different/default AMP dtype than the one specified in the forward thread.
                # Disabling autocast here will allow Autograd to run the same dtype as the forward pass.
                grad = gradient_autodiff(var, grad_var)

            if self.scaler_enabled:
                grad = self._unscale(grad, var_name)

            grad_dict = {
                name: grad[i] for i, name in enumerate(self.gradient_names[var_name])
            }
            output_var.update(grad_dict)
        return output_var

    @classmethod
    def make_node(cls, inputs: List[Key], derivatives: List[Key], name=None, jit=False):
        derivatives = [d for d in derivatives if d not in inputs]
        bwd_derivative_dict = _derivative_dict(inputs, derivatives, forward=False)
        output_derivatives = []
        for key, value in bwd_derivative_dict.items():
            output_derivatives += [
                Key(key.name, key.size, key.derivatives + [x]) for x in value
            ]

        evaluate = cls(bwd_derivative_dict)
        nvtx_str = evaluate.nvtx_str
        if jit:
            evaluate = torch.jit.script(evaluate)

        derivative_node = Node(
            inputs,
            output_derivatives,
            evaluate,
            name=(nvtx_str if name is None else str(name)),
        )
        return derivative_node

    def setup_deriv_scaler(self, deriv_scalers: DerivScalers):
        self.scaler_enabled = True
        amp_manager = AmpManager()

        # Set scaler for each term, they could share and map to the same scaler.
        if amp_manager.mode == AmpMode.PER_ORDER_SCALER:
            # Use the maximum derivative order for this node. For example if we
            # have gradient_dict as {"u__x", "nu"} to calculate u__x__x and
            # nu__x, then order is 2.
            order = max(len(g.split("__")) for g in self.gradient_dict)
            for var_name in self.gradient_dict:
                # use a separate scaler if it is in special_terms
                if var_name in amp_manager.special_terms:
                    self.deriv_scalers[var_name] = deriv_scalers.get_scaler(var_name)
                else:
                    self.deriv_scalers[var_name] = deriv_scalers.get_scaler(order)
        elif amp_manager.mode == AmpMode.PER_TERM_SCALER:
            for var_name in self.gradient_dict:
                self.deriv_scalers[var_name] = deriv_scalers.get_scaler(var_name)
        else:
            raise ValueError(f"amp_manager.mode has wrong type: {amp_manager.mode}")


def _derivative_dict(var, derivatives, forward=False):
    needed = derivatives
    while True:  # break apart diff to see if first order needed
        break_loop = True
        for n in needed:
            l_n = Key(n.name, n.size, n.derivatives[:-1])
            if (len(n.derivatives) > 1) and l_n not in needed and l_n not in var:
                needed.append(l_n)
                break_loop = False
        if break_loop:
            break
    current = var
    diff_dict = {}
    for c, n in itertools.product(current, needed):
        c_under_n = Key(n.name, n.size, n.derivatives[0 : len(c.derivatives)])
        if (c == c_under_n) and (len(n.derivatives) == len(c.derivatives) + 1):
            if forward:
                if n.derivatives[len(c.derivatives)] not in diff_dict:
                    diff_dict[n.derivatives[len(c.derivatives)]] = set()
                diff_dict[n.derivatives[len(c.derivatives)]].add(c)
            else:
                if c not in diff_dict:
                    diff_dict[c] = set()
                diff_dict[c].add(n.derivatives[len(c.derivatives)])
    diff_dict = {key: list(value) for key, value in diff_dict.items()}
    return diff_dict


# ==== Meshless finite derivs ====
class MeshlessFiniteDerivative(torch.nn.Module):
    """
    Module to compute derivatives using meshless finite difference

    Parameters
    ----------
    model : torch.nn.Module
        Forward torch module for calculating stencil values
    derivatives : List[Key]
        List of derivative keys to calculate
    dx : Union[float, Callable]
        Spatial discretization of all axis, can be function with parameter `count` which is
        the number of forward passes for dynamically adjusting dx
    order : int, optional
        Order of derivative, by default 2
    max_batch_size : Union[int, None], optional
        Max batch size of stencil calucations, by default uses batch size of inputs
    double_cast : bool, optional
        Cast fields to double precision to calculate derivatives, by default True
    jit : bool, optional
        Use torch script for finite deriv calcs, by default True

    """

    def __init__(
        self,
        model: torch.nn.Module,
        derivatives: List[Key],
        dx: Union[float, Callable],
        order: int = 2,
        max_batch_size: Union[int, None] = None,
        double_cast: bool = True,
        input_keys: Union[List[Key], None] = None,
    ):
        super().__init__()

        self.model = model
        self._dx = dx
        self.double_cast = double_cast
        self.max_batch_size = max_batch_size
        self.input_keys = input_keys
        self.count = 0

        self.derivatives = {1: [], 2: [], 3: [], 4: []}
        for key in derivatives:
            try:
                self.derivatives[len(key.derivatives)].append(key)
            except:
                raise NotImplementedError(
                    f"{len(key.derivatives)}th derivatives not supported"
                )

        self.first_deriv = FirstDeriv(self.derivatives[1], self.dx, order=order)
        self.second_deriv = SecondDeriv(self.derivatives[2], self.dx, order=order)
        self.third_deriv = ThirdDeriv(self.derivatives[3], self.dx, order=order)
        self.fourth_deriv = FourthDeriv(self.derivatives[4], self.dx, order=order)

    @torch.jit.ignore()
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.count += 1
        dx = self.dx
        self.first_deriv.dx = dx
        self.second_deriv.dx = dx
        self.third_deriv.dx = dx
        self.fourth_deriv.dx = dx
        torch.cuda.nvtx.range_push("Calculating meshless finite derivatives")

        # Assemble global stencil
        global_stencil = []
        for deriv in [
            self.first_deriv,
            self.second_deriv,
            self.third_deriv,
            self.fourth_deriv,
        ]:
            stencil_list = deriv.stencil
            # Remove centered stencil points if already in input dictionary
            for i, point in enumerate(stencil_list):
                if point.split("::")[1] == str(0) and point.split("::")[0] in inputs:
                    stencil_list.pop(i)
            global_stencil.extend(stencil_list)
        global_stencil = list(set(global_stencil))

        # Number of stencil points to fit into a forward pass
        input_batch_size = next(iter(inputs.values())).size(0)
        if self.max_batch_size is None:
            num_batch = 1
        else:
            num_batch = max([self.max_batch_size, input_batch_size]) // input_batch_size
        # Stencil forward passes
        index = 0
        finite_diff_inputs = inputs.copy()
        while index < len(global_stencil):
            torch.cuda.nvtx.range_push("Running stencil forward pass")
            # Batch up stencil inputs
            stencil_batch = [global_stencil[index]]
            index += 1
            for j in range(1, min([len(global_stencil) - (index - 1), num_batch])):
                stencil_batch.append(global_stencil[index])
                index += 1

            model_inputs = self._get_stencil_input(inputs, stencil_batch)

            # Model forward
            outputs = self.model(model_inputs)

            # Dissassemble batched inputs
            for key, value in outputs.items():
                outputs[key] = torch.split(value.view(-1, len(stencil_batch)), 1, dim=1)
            for i, stencil_str in enumerate(stencil_batch):
                for key, value in outputs.items():
                    finite_diff_inputs[f"{key}>>{stencil_str}"] = value[i]
            torch.cuda.nvtx.range_pop()

        # Calc finite diff grads
        torch.cuda.nvtx.range_push("Calc finite difference")
        if self.double_cast:  # Cast tensors to doubles for finite diff calc
            for key, value in finite_diff_inputs.items():
                finite_diff_inputs[key] = value.double()

        outputs_first = self.first_deriv(finite_diff_inputs)
        outputs_second = self.second_deriv(finite_diff_inputs)
        outputs_third = self.third_deriv(finite_diff_inputs)
        outputs_fourth = self.fourth_deriv(finite_diff_inputs)

        outputs = inputs
        if self.double_cast:
            dtype = torch.get_default_dtype()
            for key, value in outputs_first.items():
                outputs_first[key] = value.type(dtype)
            for key, value in outputs_second.items():
                outputs_second[key] = value.type(dtype)
            for key, value in outputs_third.items():
                outputs_third[key] = value.type(dtype)
            for key, value in outputs_fourth.items():
                outputs_fourth[key] = value.type(dtype)
        outputs = {
            **inputs,
            **outputs_first,
            **outputs_second,
            **outputs_third,
            **outputs_fourth,
        }
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
        return outputs

    @property
    def dx(self):
        if hasattr(self._dx, "__call__"):
            return self._dx(self.count)
        else:
            return self._dx

    def _get_stencil_input(
        self, inputs: Dict[str, Tensor], stencil_strs: List[str]
    ) -> Dict[str, Tensor]:
        """Creates a copy of the inputs tensor and adjusts its values based on
        the stencil str.

        Parameters
        ----------
        inputs : Dict[str, Tensor]
            Input tensor dictionary
        stencil_strs : List[str]
            batch list of stencil string from derivative class

        Returns
        -------
        Dict[str, Tensor]
            Modified input tensor dictionary

        Example
        -------
        A stencil string `x::1` will modify inputs['x'] = inputs['x'] + dx
        A stencil string `y::-1,z::1` will modify inputs['y'] = inputs['y'] - dx, inputs['z'] = inputs['z'] + dx
        """
        if self.input_keys is None:
            outputs = inputs.copy()
        else:
            outputs = {str(key): inputs[str(key)].clone() for key in self.input_keys}

        for key, value in outputs.items():
            outputs[key] = value.repeat(1, len(stencil_strs))

        for i, stencil_str in enumerate(stencil_strs):
            # Loop through points
            for point in stencil_str.split("&&"):
                var_name = point.split("::")[0]
                spacing = int(point.split("::")[1])
                outputs[var_name][:, i] = outputs[var_name][:, i] + spacing * self.dx

        for key, value in outputs.items():
            outputs[key] = value.view(-1, 1)

        return outputs

    @classmethod
    def make_node(
        cls,
        node_model: Union[Node, torch.nn.Module],
        derivatives: List[Key],
        dx: Union[float, Callable],
        order: int = 2,
        max_batch_size: Union[int, None] = None,
        name: str = None,
        double_cast: bool = True,
        input_keys: Union[List[Key], List[str], None] = None,
    ):
        """Makes a meshless finite derivative node.


        Parameters
        ----------
        node_model : Union[Node, torch.nn.Module]
            Node or torch.nn.Module for computing FD stencil values.
            Part of the inputs to this model should consist of the independent
            variables and output the functional value
        derivatives : List[Key]
            List of derivatives to be computed
        dx : Union[float, Callable]
            Spatial discretization for finite diff calcs, can be function
        order : int, optional
            Order of accuracy of finite diff calcs, by default 2
        max_batch_size : Union[int, None], optional
            Maximum batch size to used with the stenicl foward passes, by default None
        name : str, optional
            Name of node, by default None
        double_cast : bool, optional
            Cast tensors to double precision for derivatives, by default True
        input_keys : Union[List[Key], List[str], None], optional
            List of input keys to be used for input of forward model.
            Should be used if node_model is not a :obj:`Node`, by default None
        """

        # We have two sets of input keys:
        # input_keys: which are the list of inputs to the model for stencil points
        # mfd_input_keys: input keys for the MFD node
        if input_keys is None:
            input_keys = []
            mfd_input_keys = []
        else:
            input_keys = [str(key) for key in input_keys]
            mfd_input_keys = [str(key) for key in input_keys]

        for derivative in derivatives:
            mfd_input_keys.append(derivative.name)
            for dstr in derivative.derivatives:
                mfd_input_keys.append(dstr.name)
                input_keys.append(dstr.name)

        if isinstance(node_model, Node):
            model = node_model.evaluate
            input_keys = input_keys + [str(key) for key in node_model.inputs]
        else:
            model = node_model
        # Remove duplicate keys
        mfd_input_keys = Key.convert_list(list(set(mfd_input_keys)))
        input_keys = Key.convert_list(list(set(input_keys)))

        evaluate = cls(
            model,
            derivatives,
            dx=dx,
            order=order,
            max_batch_size=max_batch_size,
            double_cast=double_cast,
            input_keys=input_keys,
        )

        derivative_node = Node(
            mfd_input_keys,
            derivatives,
            evaluate,
            name=(
                "Meshless-Finite-Derivative Node" + "" if name is None else f": {name}"
            ),
        )
        return derivative_node
