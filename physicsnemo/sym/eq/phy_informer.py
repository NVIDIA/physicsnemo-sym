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

import copy
import logging
from typing import List, Optional, Union

import numpy as np
import torch
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.spatial_grads.spatial_grads import (
    GradientCalculator,
    compute_connectivity_tensor,
)
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PhysicsInformer:
    """
    A utility to compute the residual of a Partial Differential Equation (PDE).
    Given the equations and `required_outputs`, this utility constructs the
    computational graph, including computing of the derivatives to output the residuals.

    This utility computes the spatial grads automatically. Currently the spatial grads
    are computed using "autodiff", "meshless_finite_difference", "finite_difference",
    "spectral", and "least_squares" methods. All the other gradients (such as
    gradients w.r.t. time) will have to be manually included in the `input_dict` to the
    forward call.

    Parameters
    ----------
    required_outputs : List[str]
        Required keys in the output dictionary. To find the available outputs of a PDE,
        you can use the `.pprint()` method.
    equations : PDE
        Equation to use for computing the residual. The equation must be in the form of
        PhysicsNeMo Sym's PDE. For more details,
        refer: https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/features/nodes.html#equations.
        Custom PDEs are also supported.
        For details refer: https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/features/nodes.html#custom-pdes
    grad_method : str
        Gradient method to use. Currently below methods are supported, which can be
        selected based on the model output format:

        - ``autodiff``: The spatial gradients are computed using automatic
          differentiation. Ideal for networks dealing with point-clouds and
          fully-differentiable networks. The `.forward` call requires input dict with
          the relevant variables in ``[N, 1]`` shape along with entry for "coordinates"
          in ``[N, m]`` shape where m is the dimensionality of the input
          (1/2/3 based on 1D, 2D and 3D).
          Note: the coordinates tensor must have ``requires_grad`` set to ``True`` and the
          model outputs need to be connected to the coordinates in the computational
          graph.

        - ``meshless_finite_difference``: The spatial gradients are computed using
          meshless finite difference. Ideal for use with point-clouds.
          For details refer: https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/features/performance.html#meshless-finite-derivatives.
          The `.forward` call requires input dict with the relevant variables in
          ``[N, 1]`` shape along with the same variables executed at the stencil points.
          Stencil points are defined by the following convention:

              "u>>x::1": u(i+1, j)
              "u>>x::-1": u(i-1, j)
              "u>>x::1&&y::1": u(i+1, j+1)
              "u>>x::-1&&y::-1": u(i-1, j-1)
              etc.

        - ``finite_difference``: The spatial gradients are computed using finite
          difference assuming regular grid. Ideal for use with regular grids / images.
          The `.forward` call requires input dict with the relevant variables in
          ``[N, 1, H, W, D]`` for 3D, ``[N, 1, H, W]`` for 2D and ``[N, 1, H]`` for 1D.

        - ``spectral``: The spatial gradients are computed using FFTs. Note: this can
          lead to boundary artifacts for non-periodic signals. Ideal for use with
          regular grids / images.
          The `.forward` call requires input dict with the relevant variables in
          ``[N, 1, H, W, D]`` for 3D, ``[N, 1, H, W]`` for 2D and ``[N, 1, H]`` for 1D.

        - ``least_squares``: The spatial gradients are computed using Least Squares
          technique. Ideal for use with mesh based representations (i.e. unstructured
          grids). All values are computed at the nodes. The `.forward` call requires
          input dict with the relevant variables in ``[N, 1]`` shape along with entry
          for "coordinates" in ``[N, m]`` shape where m is the dimensionality of the
          input (1/2/3 based on 1D, 2D and 3D), "node_ids", "edges" and
          "connectivity_tensor". The "node_ids" and "edges" can directly derived from
          the graph representation (for example for dgl graph, by running
          ``graph.nodes()`` and ``graph.edges()``). For computing connectivity tensor,
          refer: ``physicsnemo.sym.eq.spatial_grads.spatial_grads.compute_connectivity_tensor``
    fd_dx : Union[float, List[float]], optional
        dx to be used for meshless finite difference and regular finite difference
        calculation. If float, the same value is used across all dimensions,
        by default 0.001
    bounds : List[float], optional
        bounds to be used for spectral derivatives, by default [2 * np.pi, 2 * np.pi, 2 * np.pi]
    compute_connectivity : bool, optional
        Whether to compute the connectivity tensor during forward pass (only applies for
        least squares method), by default True. Set to false if this can be computed as
        a part of the dataloader.
    device : Optional[str], optional
        The device to use for computation. Options are "cuda" or "cpu". If not
        specified, the computation defaults to "cpu".

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
    >>> from physicsnemo.sym.eq.phy_informer import PhysicsInformer
    >>> ns = NavierStokes(nu=0.1, rho=1.0, dim=2, time=True)
    >>> phy_inf = PhysicsInformer(
    ... required_outputs=["continuity", "momentum_x"],
    ... equations=ns,
    ... grad_method="finite_difference"
    ... )
    >>> tensor = torch.rand(1, 1, 10, 10)   # [N, 1, H, W]
    >>> sorted(phy_inf.required_inputs)
    ...
    ['p', 'u', 'u__t', 'v']
    >>> out_dict = phy_inf.forward({"u": tensor, "v": tensor, "u__t": tensor, "p": tensor})
    >>> out_dict.keys()
    dict_keys(['continuity', 'momentum_x'])
    >>> out_dict["continuity"].shape
    torch.Size([1, 1, 10, 10])
    """

    def __init__(
        self,
        required_outputs: List[str],
        equations: PDE,
        grad_method: str,
        fd_dx: Union[
            float, List[float]
        ] = 0.001,  # only applies for FD and Meshless FD. Ignored for the rest
        bounds: List[float] = [
            2 * np.pi,
            2 * np.pi,
            2 * np.pi,
        ],  # only applies for FD and Meshless FD. Ignored for the rest
        compute_connectivity: bool = True,  # only applies for least squares. Ignored for the rest
        device: Optional[str] = None,
    ):
        self.required_outputs = required_outputs
        self.equations = equations
        self.dim = equations.dim
        self.grad_method = grad_method
        self.fd_dx = fd_dx
        self.bounds = bounds
        self.compute_connectivity = compute_connectivity
        self.device = device if device is not None else torch.device("cpu")
        self.grad_calc = GradientCalculator(device=self.device)
        self.nodes = self.equations.make_nodes()

        self.require_mixed_derivs = False

        self.graph = self._create_graph()

    @property
    def required_inputs(self):
        """Find the required inputs"""
        node_outputs = [str(n.outputs[0]) for n in self.nodes]
        node_inputs = set()

        for node in self.required_outputs:
            if node not in node_outputs:
                raise ValueError(
                    f"{node} does not appear in the equation outputs provided. "
                    + f"Please choose from {node_outputs}"
                )

        fd, sd, others = self._extract_derivatives()

        for input in fd | sd | others:
            node_inputs.add(input)

        for node in self.nodes:
            if str(node.outputs[0]) in self.required_outputs and node.inputs:
                node_inputs.update(map(str, node.inputs))

        node_inputs = list(node_inputs)

        if self.grad_method == "meshless_finite_difference":
            node_inputs = self._expand_for_meshless_fd(node_inputs)
        elif self.grad_method == "autodiff":
            node_inputs.append("coordinates")
        elif self.grad_method == "least_squares":
            if self.compute_connectivity:
                node_inputs.extend(["coordinates", "nodes", "edges"])
            else:
                node_inputs.extend(
                    ["coordinates", "nodes", "edges", "connectivity_tensor"]
                )

        # print(f"To compute the required {self.required_outputs}, using {self.grad_method} method, {node_inputs} will be required. Please provide them during the forward call")
        return node_inputs

    def _expand_for_meshless_fd(self, node_inputs):
        """Add input keys specific to MFD"""
        node_inputs_new = copy.deepcopy(node_inputs)
        for node in node_inputs:
            mfd_vars = [
                f"{node}>>x::1",
                f"{node}>>x::-1",
                f"{node}>>y::1",
                f"{node}>>y::-1",
                f"{node}>>z::1",
                f"{node}>>z::-1",
            ]
            node_inputs_new.extend(mfd_vars[: 2 * self.dim])
        return node_inputs_new

    def _create_graph(self):
        """Create the computational graph"""
        first_deriv, second_deriv, _ = self._extract_derivatives()

        input_keys_sym = [Key(k) for k in self.required_inputs]
        output_keys_sym = [Key(k) for k in self.required_outputs]

        diff_nodes = self._create_diff_nodes(first_deriv, dim=self.dim, order=1)
        diff_nodes += self._create_diff_nodes(second_deriv, dim=self.dim, order=2)

        return Graph(
            self.nodes, input_keys_sym, output_keys_sym, diff_nodes=diff_nodes
        ).to(self.device)

    def _extract_derivatives(self):
        """Extract the derivatives from the provided PDE"""
        first_deriv, second_deriv, other_derivs = set(), set(), set()

        for node in self.nodes:
            if str(node.outputs[0]) in self.required_outputs:
                for derr in node.derivatives:
                    self._process_derivative(
                        derr, first_deriv, second_deriv, other_derivs
                    )

        first_deriv_consolidated = {i.split("__")[0] for i in first_deriv}
        second_deriv_consolidated = {i.split("__")[0] for i in second_deriv}

        return first_deriv_consolidated, second_deriv_consolidated, other_derivs

    def _process_derivative(self, derr, first_deriv, second_deriv, other_derivs):
        """Helper to process and find the valid derivative nodes"""
        if str(derr).count("__") > 2:
            raise ValueError("Only second order PDEs are supported presently")

        allowed_derr_vars = ["x", "y", "z"]
        for var in str(derr).split("__")[1:]:
            if var not in allowed_derr_vars:
                logging.warning(
                    f"Detected derivative w.r.t {var}. "
                    + f"Note, derivatives w.r.t only {allowed_derr_vars} vars are "
                    + f"computed automatically. The {str(derr)} will have to be "
                    + "provided as an input during the forward call."
                )
                other_derivs.add(str(derr))

        if (
            str(derr).count("__") == 2
            and str(derr).split("__")[1] != str(derr).split("__")[2]
        ):
            self.require_mixed_derivs = True

        if str(derr).count("__") == 1:
            first_deriv.add(str(derr))
        elif str(derr).count("__") == 2:
            second_deriv.add(str(derr))

    def _create_diff_nodes(self, derivatives, dim, order):
        """Create various custom derivative nodes"""
        diff_nodes = []
        for derr_var in derivatives:
            node = self._create_diff_node(derr_var, dim, order)
            if node:
                diff_nodes.append(node)
        return diff_nodes

    def _create_diff_node(self, derr_var, dim, order):
        """Select appropriate derivative node based on grad_method"""
        methods = {
            "finite_difference": self._fd_gradient_module,
            "spectral": self._spectral_gradient_module,
            "least_squares": self._ls_gradient_module,
            "autodiff": self._autodiff_gradient_module,
            "meshless_finite_difference": self._meshless_fd_gradient_module,
        }

        if self.grad_method in methods:
            return Node(
                [derr_var],
                self._derivative_keys(
                    derr_var, dim, order, return_mixed_derivs=self.require_mixed_derivs
                ),
                methods[self.grad_method](derr_var, dim, order),
            )

    def _derivative_keys(self, derr_var, dim, order, return_mixed_derivs=False):
        """Helper to set the output keys"""
        base_keys = ["__x", "__y", "__z"]
        base_keys = [base_keys[i] for i in range(dim)]
        output_keys = [f"{derr_var}{k * order}" for k in base_keys]
        if return_mixed_derivs:
            if order == 2:
                if dim == 2:
                    output_keys.append(f"{derr_var}__x__y")
                    output_keys.append(f"{derr_var}__y__x")
                if dim == 3:
                    output_keys.append(f"{derr_var}__x__y")
                    output_keys.append(f"{derr_var}__y__x")
                    output_keys.append(f"{derr_var}__x__z")
                    output_keys.append(f"{derr_var}__z__x")
                    output_keys.append(f"{derr_var}__y__z")
                    output_keys.append(f"{derr_var}__z__y")
        return output_keys

    def _fd_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            dx=self.fd_dx,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def _spectral_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            ell=self.bounds,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def _ls_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def _autodiff_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def _meshless_fd_gradient_module(self, derr_var, dim, order):
        return_mixed_derivs = False
        if order == 2 and self.require_mixed_derivs:
            return_mixed_derivs = True
        return self.grad_calc.get_gradient_module(
            self.grad_method,
            derr_var,
            dx=self.fd_dx,
            dim=dim,
            order=order,
            return_mixed_derivs=return_mixed_derivs,
        )

    def forward(self, inputs):
        """Forward pass"""
        if self.grad_method == "least_squares":
            if self.compute_connectivity:
                connectivity_tensor = compute_connectivity_tensor(
                    inputs["nodes"], inputs["edges"]
                )
                inputs["connectivity_tensor"] = connectivity_tensor

        return self.graph.forward(inputs)
