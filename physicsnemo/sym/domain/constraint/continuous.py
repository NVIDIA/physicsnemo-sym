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

"""Continuous type constraints"""

import torch
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from typing import Dict, List, Union, Tuple, Callable
import sympy as sp
import logging

from .constraint import Constraint
from .utils import _compute_outvar, _compute_lambda_weighting
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.loss import Loss, PointwiseLossNorm, IntegralLossNorm
from physicsnemo.sym.distributed import DistributedManager

from physicsnemo.sym.geometry import Geometry
from physicsnemo.sym.geometry.parameterization import Parameterization

from physicsnemo.sym.dataset import (
    DictPointwiseDataset,
    ListIntegralDataset,
    ContinuousPointwiseIterableDataset,
    ContinuousIntegralIterableDataset,
    DictImportanceSampledPointwiseIterableDataset,
    DictVariationalDataset,
)

Tensor = torch.Tensor
logger = logging.getLogger(__name__)


class PointwiseConstraint(Constraint):
    """
    Base class for all Pointwise Constraints
    """

    def save_batch(self, filename):
        # sample batch
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        invar = Constraint._set_device(invar, device=self.device, requires_grad=True)
        true_outvar = Constraint._set_device(true_outvar, device=self.device)
        lambda_weighting = Constraint._set_device(lambda_weighting, device=self.device)

        # If using DDP, strip out collective stuff to prevent deadlocks
        # This only works either when one process alone calls in to save_batch
        # or when multiple processes independently save data
        if hasattr(self.model, "module"):
            modl = self.model.module
        else:
            modl = self.model

        # compute pred outvar
        pred_outvar = modl(invar)

        # rename values and save batch to vtk file TODO clean this up after graph unroll stuff
        named_lambda_weighting = {
            "lambda_" + key: value for key, value in lambda_weighting.items()
        }
        named_true_outvar = {"true_" + key: value for key, value in true_outvar.items()}
        named_pred_outvar = {"pred_" + key: value for key, value in pred_outvar.items()}
        save_var = {
            **{key: value for key, value in invar.items()},
            **named_true_outvar,
            **named_pred_outvar,
            **named_lambda_weighting,
        }
        save_var = {
            key: value.cpu().detach().numpy() for key, value in save_var.items()
        }
        var_to_polyvtk(save_var, filename)

    def load_data(self):
        # get train points from dataloader
        invar, true_outvar, lambda_weighting = next(self.dataloader)

        self._input_vars = Constraint._set_device(
            invar, device=self.device, requires_grad=True
        )
        self._target_vars = Constraint._set_device(true_outvar, device=self.device)
        self._lambda_weighting = Constraint._set_device(
            lambda_weighting, device=self.device
        )

    def load_data_static(self):
        if self._input_vars is None:
            # Default loading if vars not allocated
            self.load_data()
        else:
            # get train points from dataloader
            invar, true_outvar, lambda_weighting = next(self.dataloader)
            # Set grads to false here for inputs, static var has allocation already
            input_vars = Constraint._set_device(
                invar, device=self.device, requires_grad=False
            )
            target_vars = Constraint._set_device(true_outvar, device=self.device)
            lambda_weighting = Constraint._set_device(
                lambda_weighting, device=self.device
            )

            for key in input_vars.keys():
                self._input_vars[key].data.copy_(input_vars[key])
            for key in target_vars.keys():
                self._target_vars[key].copy_(target_vars[key])
            for key in lambda_weighting.keys():
                self._lambda_weighting[key].copy_(lambda_weighting[key])

    def forward(self):
        # compute pred outvar
        self._output_vars = self.model(self._input_vars)

    def loss(self, step: int) -> Dict[str, torch.Tensor]:
        if self._output_vars is None:
            logger.warning("Calling loss without forward call")
            return {}

        losses = self._loss(
            self._input_vars,
            self._output_vars,
            self._target_vars,
            self._lambda_weighting,
            step,
        )

        return losses

    @classmethod
    def from_numpy(
        cls,
        nodes: List[Node],
        invar: Dict[str, np.ndarray],
        outvar: Dict[str, np.ndarray],
        batch_size: int,
        lambda_weighting: Dict[str, np.ndarray] = None,
        loss: Loss = PointwiseLossNorm(),
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):
        """
        Create custom pointwise constraint from numpy arrays.

        Parameters
        ----------
        nodes : List[Node]
            List of PhysicsNeMo Nodes to unroll graph with.
        invar : Dict[str, np.ndarray (N, 1)]
            Dictionary of numpy arrays as input.
        outvar : Dict[str, np.ndarray (N, 1)]
            Dictionary of numpy arrays to enforce constraint on.
        batch_size : int
            Batch size used in training.
        lambda_weighting : Dict[str, np.ndarray (N, 1)]
            Dictionary of numpy arrays to pointwise weight losses.
            Default is ones.
        loss : Loss
            PhysicsNeMo Loss module that defines the loss type, (e.g. L2, L1, ...).
        shuffle : bool, optional
            Randomly shuffle examples in dataset every epoch, by default True
        drop_last : bool, optional
            Drop last mini-batch if dataset not fully divisible but batch_size, by default False
        num_workers : int
            Number of worker used in fetching data.
        """

        if "area" not in invar:
            invar["area"] = np.ones_like(next(iter(invar.values())))
        # TODO: better area definition?
        # no need to lambdify: outvar / lambda_weighting already contain np arrays

        # make point dataset
        dataset = DictPointwiseDataset(
            invar=invar,
            outvar=outvar,
            lambda_weighting=lambda_weighting,
        )

        return cls(
            nodes=nodes,
            dataset=dataset,
            loss=loss,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )


class PointwiseBoundaryConstraint(PointwiseConstraint):
    """
    Pointwise Constraint applied to boundary/perimeter/surface of geometry.
    For example, in 3D this will create a constraint on the surface of the
    given geometry.

    Parameters
    ----------
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    geometry : Geometry
        PhysicsNeMo `Geometry` to apply the constraint with.
    outvar : Dict[str, Union[int, float, sp.Basic]]
        A dictionary of SymPy Symbols/Expr, floats or int.
        This is used to describe the constraint. For example,
        ``outvar={'u': 0}`` would specify ``'u'`` to be zero everywhere
        on the constraint.
    batch_size : int
        Batch size used in training.
    criteria : Union[sp.Basic, True]
        SymPy criteria function specifies to only apply constraint to areas
        that satisfy this criteria. For example, if
        ``criteria=sympy.Symbol('x')>0`` then only areas that have positive
        ``'x'`` values will have the constraint applied to them.
    lambda_weighting :  Dict[str, Union[int, float, sp.Basic]] = None
        The spatial pointwise weighting of the constraint. For example,
        ``lambda_weighting={'lambda_u': 2.0*sympy.Symbol('x')}`` would
        apply a pointwise weighting to the loss of ``2.0 * x``.
    parameterization : Union[Parameterization, None], optional
        This allows adding parameterization or additional inputs.
    fixed_dataset : bool = True
        If True then the points sampled for this constraint are done right
        when initialized and fixed. If false then the points are continually
        resampled.
    compute_sdf_derivatives: bool, optional
        Compute SDF derivatives when sampling geometery
    importance_measure : Union[Callable, None] = None
        A callable function that computes a scalar importance measure. This
        importance measure is then used in the constraint when sampling
        points. Areas with higher importance are sampled more frequently
        according to Monte Carlo importance sampling,
        https://en.wikipedia.org/wiki/Monte_Carlo_integration.
    batch_per_epoch : int = 1000
        If ``fixed_dataset=True`` then the total number of points generated
        to apply constraint on is ``total_nr_points=batch_per_epoch*batch_size``.
    quasirandom : bool = False
        If true then sample the points using the Halton sequence.
    num_workers : int
        Number of worker used in fetching data.
    loss : Loss
        PhysicsNeMo Loss module that defines the loss type, (e.g. L2, L1, ...).
    shuffle : bool, optional
        Randomly shuffle examples in dataset every epoch, by default True
    """

    def __init__(
        self,
        nodes: List[Node],
        geometry: Geometry,
        outvar: Dict[str, Union[int, float, sp.Basic]],
        batch_size: int,
        criteria: Union[sp.Basic, Callable, None] = None,
        lambda_weighting: Dict[str, Union[int, float, sp.Basic]] = None,
        parameterization: Union[Parameterization, None] = None,
        fixed_dataset: bool = True,
        importance_measure: Union[Callable, None] = None,
        batch_per_epoch: int = 1000,
        quasirandom: bool = False,
        num_workers: int = 0,
        loss: Loss = PointwiseLossNorm(),
        shuffle: bool = True,
    ):
        # assert that not using importance measure with continuous dataset
        assert not ((not fixed_dataset) and (importance_measure is not None)), (
            "Using Importance measure with continuous dataset is not supported"
        )

        # if fixed dataset then sample points and fix for all of training
        if fixed_dataset:
            # sample boundary
            invar = geometry.sample_boundary(
                batch_size * batch_per_epoch,
                criteria=criteria,
                parameterization=parameterization,
                quasirandom=quasirandom,
            )

            # compute outvar
            outvar = _compute_outvar(invar, outvar)

            # set lambda weighting
            lambda_weighting = _compute_lambda_weighting(
                invar, outvar, lambda_weighting
            )

            # make point dataset
            if importance_measure is None:
                invar["area"] *= batch_per_epoch  # TODO find better way to do this
                dataset = DictPointwiseDataset(
                    invar=invar,
                    outvar=outvar,
                    lambda_weighting=lambda_weighting,
                )
            else:
                dataset = DictImportanceSampledPointwiseIterableDataset(
                    invar=invar,
                    outvar=outvar,
                    batch_size=batch_size,
                    importance_measure=importance_measure,
                    lambda_weighting=lambda_weighting,
                    shuffle=shuffle,
                )

        # else sample points every batch
        else:
            # invar function
            def invar_fn():
                return geometry.sample_boundary(
                    batch_size,
                    criteria=criteria,
                    parameterization=parameterization,
                    quasirandom=quasirandom,
                )

            # outvar function
            def outvar_fn(invar):
                return _compute_outvar(invar, outvar)

            # lambda weighting function
            def lambda_weighting_fn(invar, outvar):
                return _compute_lambda_weighting(invar, outvar, lambda_weighting)

            # make point dataloader
            dataset = ContinuousPointwiseIterableDataset(
                invar_fn=invar_fn,
                outvar_fn=outvar_fn,
                lambda_weighting_fn=lambda_weighting_fn,
            )

        # initialize constraint
        super().__init__(
            nodes=nodes,
            dataset=dataset,
            loss=loss,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
        )


class PointwiseInteriorConstraint(PointwiseConstraint):
    """
    Pointwise Constraint applied to interior of geometry.
    For example, in 3D this will create a constraint on the interior
    volume of the given geometry.

    Parameters
    ----------
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    geometry : Geometry
        PhysicsNeMo Geometry to apply the constraint with.
    outvar : Dict[str, Union[int, float, sp.Basic]]
        A dictionary of SymPy Symbols/Expr, floats or int.
        This is used to describe the constraint. For example,
        ``outvar={'u': 0}`` would specify ``'u'`` to be zero everywhere
        in the constraint.
    batch_size : int
        Batch size used in training.
    bounds : Dict[sp.Basic, Tuple[float, float]] = None
        Bounds of the given geometry,
        (e.g. ``bounds={sympy.Symbol('x'): (0, 1), sympy.Symbol('y'): (0, 1)}``).
    criteria : Union[sp.basic, True]
        SymPy criteria function specifies to only apply constraint to areas
        that satisfy this criteria. For example, if
        ``criteria=sympy.Symbol('x')>0`` then only areas that have positive
        ``'x'`` values will have the constraint applied to them.
    lambda_weighting :  Dict[str, Union[int, float, sp.Basic]] = None
        The spatial pointwise weighting of the constraint. For example,
        ``lambda_weighting={'lambda_u': 2.0*sympy.Symbol('x')}`` would
        apply a pointwise weighting to the loss of ``2.0 * x``.
    parameterization: Union[Parameterization, None] = {}
        This allows adding parameterization or additional inputs.
    fixed_dataset : bool = True
        If True then the points sampled for this constraint are done right
        when initialized and fixed. If false then the points are continually
        resampled.
    importance_measure : Union[Callable, None] = None
        A callable function that computes a scalar importance measure. This
        importance measure is then used in the constraint when sampling
        points. Areas with higher importance are sampled more frequently
        according to Monte Carlo importance sampling,
        https://en.wikipedia.org/wiki/Monte_Carlo_integration.
    batch_per_epoch : int = 1000
        If ``fixed_dataset=True`` then the total number of points generated
        to apply constraint on is ``total_nr_points=batch_per_epoch*batch_size``.
    quasirandom : bool = False
        If true then sample the points using the Halton sequence.
    num_workers : int
        Number of worker used in fetching data.
    loss : Loss
        PhysicsNeMo Loss module that defines the loss type, (e.g. L2, L1, ...).
    shuffle : bool, optional
        Randomly shuffle examples in dataset every epoch, by default True
    """

    def __init__(
        self,
        nodes: List[Node],
        geometry: Geometry,
        outvar: Dict[str, Union[int, float, sp.Basic]],
        batch_size: int,
        bounds: Dict[sp.Basic, Tuple[float, float]] = None,
        criteria: Union[sp.Basic, Callable, None] = None,
        lambda_weighting: Dict[str, Union[int, float, sp.Basic]] = None,
        parameterization: Union[Parameterization, None] = None,
        fixed_dataset: bool = True,
        compute_sdf_derivatives: bool = False,
        importance_measure: Union[Callable, None] = None,
        batch_per_epoch: int = 1000,
        quasirandom: bool = False,
        num_workers: int = 0,
        loss: Loss = PointwiseLossNorm(),
        shuffle: bool = True,
    ):
        # assert that not using importance measure with continuous dataset
        assert not ((not fixed_dataset) and (importance_measure is not None)), (
            "Using Importance measure with continuous dataset is not supported"
        )

        # if fixed dataset then sample points and fix for all of training
        if fixed_dataset:
            # sample interior
            invar = geometry.sample_interior(
                batch_size * batch_per_epoch,
                bounds=bounds,
                criteria=criteria,
                parameterization=parameterization,
                quasirandom=quasirandom,
                compute_sdf_derivatives=compute_sdf_derivatives,
            )

            # compute outvar
            outvar = _compute_outvar(invar, outvar)

            # set lambda weighting
            lambda_weighting = _compute_lambda_weighting(
                invar, outvar, lambda_weighting
            )

            # make point dataset
            if importance_measure is None:
                invar["area"] *= batch_per_epoch  # TODO find better way to do this
                dataset = DictPointwiseDataset(
                    invar=invar,
                    outvar=outvar,
                    lambda_weighting=lambda_weighting,
                )
            else:
                dataset = DictImportanceSampledPointwiseIterableDataset(
                    invar=invar,
                    outvar=outvar,
                    batch_size=batch_size,
                    importance_measure=importance_measure,
                    lambda_weighting=lambda_weighting,
                    shuffle=shuffle,
                )

        # else sample points every batch
        else:
            # invar function
            def invar_fn():
                return geometry.sample_interior(
                    batch_size,
                    bounds=bounds,
                    criteria=criteria,
                    parameterization=parameterization,
                    quasirandom=quasirandom,
                    compute_sdf_derivatives=compute_sdf_derivatives,
                )

            # outvar function
            def outvar_fn(invar):
                return _compute_outvar(invar, outvar)

            # lambda weighting function
            def lambda_weighting_fn(invar, outvar):
                return _compute_lambda_weighting(invar, outvar, lambda_weighting)

            # make point dataloader
            dataset = ContinuousPointwiseIterableDataset(
                invar_fn=invar_fn,
                outvar_fn=outvar_fn,
                lambda_weighting_fn=lambda_weighting_fn,
            )

        # initialize constraint
        super().__init__(
            nodes=nodes,
            dataset=dataset,
            loss=loss,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
        )


class IntegralConstraint(Constraint):
    """
    Base class for all Integral Constraints
    """

    def save_batch(self, filename):
        pass
        # sample batch
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        invar = Constraint._set_device(invar, device=self.device, requires_grad=True)

        # rename values and save batch to vtk file TODO clean this up after graph unroll stuff
        for i in range(self.batch_size):
            save_var = {
                key: value[i].cpu().detach().numpy() for key, value in invar.items()
            }
            var_to_polyvtk(save_var, filename + "_batch_" + str(i))

    def load_data(self):
        # get train points from dataloader
        invar, true_outvar, lambda_weighting = next(self.dataloader)

        self._input_vars = Constraint._set_device(
            invar, device=self.device, requires_grad=True
        )
        self._target_vars = Constraint._set_device(true_outvar, device=self.device)
        self._lambda_weighting = Constraint._set_device(
            lambda_weighting, device=self.device
        )

    def load_data_static(self):
        if self._input_vars is None:
            # Default loading if vars not allocated
            self.load_data()
        else:
            # get train points from dataloader
            invar, true_outvar, lambda_weighting = next(self.dataloader)
            # Set grads to false here for inputs, static var has allocation already
            input_vars = Constraint._set_device(
                invar, device=self.device, requires_grad=False
            )
            target_vars = Constraint._set_device(true_outvar, device=self.device)
            lambda_weighting = Constraint._set_device(
                lambda_weighting, device=self.device
            )

            for key in input_vars.keys():
                self._input_vars[key].data.copy_(input_vars[key])
            for key in target_vars.keys():
                self._target_vars[key].copy_(target_vars[key])
            for key in lambda_weighting.keys():
                self._lambda_weighting[key].copy_(lambda_weighting[key])

    @property
    def output_vars(self) -> Dict[str, Tensor]:
        return self._output_vars

    @output_vars.setter
    def output_vars(self, data: Dict[str, Tensor]):
        self._output_vars = {}
        for output in self.output_names:
            self._output_vars[str(output)] = data[str(output)]

    def forward(self):
        # compute pred outvar
        self._output_vars = self.model(self._input_vars)

    def loss(self, step: int) -> Dict[str, torch.Tensor]:
        if self._output_vars is None:
            logger.warning("Calling loss without forward call")
            return {}

        # split for individual integration
        list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting = (
            [],
            [],
            [],
            [],
        )
        for i in range(self.batch_size):
            list_invar.append(
                {key: value[i] for key, value in self._input_vars.items()}
            )
            list_pred_outvar.append(
                {key: value[i] for key, value in self._output_vars.items()}
            )
            list_true_outvar.append(
                {key: value[i] for key, value in self._target_vars.items()}
            )
            list_lambda_weighting.append(
                {key: value[i] for key, value in self._lambda_weighting.items()}
            )

        # compute integral losses
        losses = self._loss(
            list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting, step
        )
        return losses


class IntegralBoundaryConstraint(IntegralConstraint):
    """
    Integral Constraint applied to boundary/perimeter/surface of geometry.
    For example, in 3D this will create a constraint on the surface of the
    given geometry.

    Parameters
    ----------
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    geometry : Geometry
        PhysicsNeMo `Geometry` to apply the constraint with.
    outvar : Dict[str, Union[int, float, sp.Basic]]
        A dictionary of SymPy Symbols/Expr, floats or int.
        This is used to describe the constraint. For example,
        ``outvar={'u': 0}`` would specify the integral of ``'u'``
        to be zero.
    batch_size : int
        Number of integrals to apply.
    integral_batch_size :  int
        Batch sized used in the Monte Carlo integration to compute
        the integral.
    criteria : Union[sp.basic, True]
        SymPy criteria function specifies to only integrate areas
        that satisfy this criteria. For example, if
        ``criteria=sympy.Symbol('x')>0`` then only areas that have positive
        ``'x'`` values will be integrated.
    lambda_weighting :  Dict[str, Union[int, float, sp.Basic]] = None
        The weighting of the constraint. For example,
        ``lambda_weighting={'lambda_u': 2.0}`` would
        weight the integral constraint by ``2.0``.
    parameterization : Union[Parameterization, None]
        This allows adding parameterization or additional inputs.
    fixed_dataset : bool = True
        If True then the points sampled for this constraint are done right
        when initialized and fixed. If false then the points are continually
        resampled.
    batch_per_epoch : int = 100
        If ``fixed_dataset=True`` then the total number of integrals generated
        to apply constraint on is ``total_nr_integrals=batch_per_epoch*batch_size``.
    quasirandom : bool = False
        If true then sample the points using the Halton sequence.
    num_workers : int
        Number of worker used in fetching data.
    loss : Loss
        PhysicsNeMo Loss module that defines the loss type, (e.g. L2, L1, ...).
    shuffle : bool, optional
        Randomly shuffle examples in dataset every epoch, by default True
    """

    def __init__(
        self,
        nodes: List[Node],
        geometry: Geometry,
        outvar: Dict[str, Union[int, float, sp.Basic]],
        batch_size: int,
        integral_batch_size: int,
        criteria: Union[sp.Basic, Callable, None] = None,
        lambda_weighting: Dict[str, Union[int, float, sp.Basic]] = None,
        parameterization: Union[Parameterization, None] = None,
        fixed_dataset: bool = True,
        batch_per_epoch: int = 100,
        quasirandom: bool = False,
        num_workers: int = 0,
        loss: Loss = IntegralLossNorm(),
        shuffle: bool = True,
    ):
        # convert dict to parameterization if needed
        if parameterization is None:
            parameterization = geometry.parameterization
        elif isinstance(parameterization, dict):
            parameterization = Parameterization(parameterization)

        # Fixed number of integral examples
        if fixed_dataset:
            # sample geometry to generate integral batchs
            list_invar = []
            list_outvar = []
            list_lambda_weighting = []
            for i in range(batch_size * batch_per_epoch):
                # sample parameter ranges
                if parameterization:
                    specific_param_ranges = parameterization.sample(1)
                else:
                    specific_param_ranges = {}

                # sample boundary
                invar = geometry.sample_boundary(
                    integral_batch_size,
                    criteria=criteria,
                    parameterization=Parameterization(
                        {
                            sp.Symbol(key): float(value)
                            for key, value in specific_param_ranges.items()
                        }
                    ),
                    quasirandom=quasirandom,
                )

                # compute outvar
                if (
                    not specific_param_ranges
                ):  # TODO this can be removed after a np_lambdify rewrite
                    specific_param_ranges = {"_": next(iter(invar.values()))[0:1]}
                outvar_star = _compute_outvar(specific_param_ranges, outvar)

                # set lambda weighting
                lambda_weighting_star = _compute_lambda_weighting(
                    specific_param_ranges, outvar, lambda_weighting
                )

                # store samples
                list_invar.append(invar)
                list_outvar.append(outvar_star)
                list_lambda_weighting.append(lambda_weighting_star)

            # make dataset of integral planes
            dataset = ListIntegralDataset(
                list_invar=list_invar,
                list_outvar=list_outvar,
                list_lambda_weighting=list_lambda_weighting,
            )
        # Continuous sampling
        else:
            # sample parameter ranges
            if parameterization:

                def param_ranges_fn():
                    return parameterization.sample(1)

            else:

                def param_ranges_fn():
                    return {}

            # invar function
            def invar_fn(param_range):
                return geometry.sample_boundary(
                    integral_batch_size,
                    criteria=criteria,
                    parameterization=Parameterization(
                        {
                            sp.Symbol(key): float(value)
                            for key, value in param_range.items()
                        }
                    ),
                    quasirandom=quasirandom,
                )

            # outvar function
            def outvar_fn(param_range):
                return _compute_outvar(param_range, outvar)

            # lambda weighting function
            def lambda_weighting_fn(param_range, outvar):
                return _compute_lambda_weighting(param_range, outvar, lambda_weighting)

            # make dataset of integral planes
            dataset = ContinuousIntegralIterableDataset(
                invar_fn=invar_fn,
                outvar_fn=outvar_fn,
                batch_size=batch_size,
                lambda_weighting_fn=lambda_weighting_fn,
                param_ranges_fn=param_ranges_fn,
            )

        self.batch_size = batch_size

        # initialize constraint
        super().__init__(
            nodes=nodes,
            dataset=dataset,
            loss=loss,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
        )


class VariationalConstraint(Constraint):
    """
    Base class for all Variational Constraints.

    B(u, v, g, dom) = \\int_{dom} (F(u, v) - g*v) dx = 0,
    where F is an operator, g is a given function/data,
    v is the test function.
    loss of variational = B1(u1, v1, g1, dom1) + B2(u2, v2, g2, dom2) + ...
    """

    def __init__(
        self,
        nodes: List[Node],
        datasets: Dict[str, DictVariationalDataset],
        batch_sizes: Dict[str, int],
        loss: Loss = PointwiseLossNorm(),
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):
        # Get DDP manager
        self.manager = DistributedManager()
        self.device = self.manager.device
        if not drop_last and self.manager.cuda_graphs:
            logger.info("drop_last must be true when using cuda graphs")
            drop_last = True

        # make dataloader from dataset
        self.data_loaders = {}
        invar_keys = []
        outvar_keys = []
        for name in datasets:
            self.data_loaders[name] = iter(
                Constraint.get_dataloader(
                    dataset=datasets[name],
                    batch_size=batch_sizes[name],
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=num_workers,
                )
            )
            invar_keys = invar_keys + datasets[name].invar_keys
            outvar_keys = outvar_keys + datasets[name].outvar_keys

        # construct model from nodes
        self.model = Graph(
            nodes,
            Key.convert_list(list(set(invar_keys))),
            Key.convert_list(list(set(outvar_keys))),
        )
        self.manager = DistributedManager()
        self.device = self.manager.device
        self.model.to(self.device)
        if self.manager.distributed:
            # https://pytorch.org/docs/master/notes/cuda.html#id5
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.manager.local_rank],
                    output_device=self.device,
                    broadcast_buffers=self.manager.broadcast_buffers,
                    find_unused_parameters=self.manager.find_unused_parameters,
                    process_group=self.manager.group(
                        "data_parallel"
                    ),  # None by default
                )
            torch.cuda.current_stream().wait_stream(s)

        self._input_names = Key.convert_list(list(set(invar_keys)))
        self._output_names = Key.convert_list(list(set(outvar_keys)))

        self._input_vars = None
        self._target_vars = None
        self._lambda_weighting = None

        # put loss on device
        self._loss = loss.to(self.device)

    def save_batch(self, filename):
        # sample batch
        for name, data_loader in self.data_loaders.items():
            invar = Constraint._set_device(
                next(data_loader), device=self.device, requires_grad=True
            )

            # If using DDP, strip out collective stuff to prevent deadlocks
            # This only works either when one process alone calls in to save_batch
            # or when multiple processes independently save data
            if hasattr(self.model, "module"):
                modl = self.model.module
            else:
                modl = self.model

            # compute pred outvar
            outvar = modl(invar)

            named_outvar = {
                "pred_" + key: value.cpu().detach().numpy()
                for key, value in outvar.items()
            }
            save_var = {
                **{key: value.cpu().detach().numpy() for key, value in invar.items()},
                **named_outvar,
            }
            var_to_polyvtk(save_var, filename + "_" + name)

    def load_data(self):
        self._input_vars = {}
        self._output_vars = {}
        for name, data_loader in self.data_loaders.items():
            # get train points from dataloader
            invar = next(data_loader)

            self._input_vars[name] = Constraint._set_device(
                invar, device=self.device, requires_grad=True
            )

    def load_data_static(self):
        if self._input_vars is None:
            # Default loading if vars not allocated
            self.load_data()
        else:
            for name, data_loader in self.data_loaders.items():
                # get train points from dataloader
                invar = next(data_loader)
                # Set grads to false here for inputs, static var has allocation already
                input_vars = Constraint._set_device(
                    invar, device=self.device, requires_grad=False
                )

                for key in input_vars.keys():
                    self._input_vars[name][key].data.copy_(input_vars[key])

                self._input_vars[name] = Constraint._set_device(
                    invar, device=self.device, requires_grad=True
                )

    def forward(self):
        # compute pred outvar
        for name in self._input_vars.keys():
            self._output_vars[name] = self.model(self._input_vars[name])

    def loss(self, step):
        # compute loss
        losses = self._loss(
            list(self._input_vars.values()), list(self._output_vars.values()), step
        )
        return losses


class VariationalDomainConstraint(VariationalConstraint):
    """
    Simple Variational Domain Constraint with a single geometry
    that represents the domain.

    TODO add comprehensive doc string after refactor
    """

    def __init__(
        self,
        nodes: List[Node],
        geometry: Geometry,
        outvar_names: List[str],
        boundary_batch_size: int,
        interior_batch_size: int,
        interior_bounds: Dict[sp.Basic, Tuple[float, float]] = None,
        boundary_criteria: Union[sp.Basic, Callable, None] = None,
        interior_criteria: Union[sp.Basic, Callable, None] = None,
        parameterization: Union[Parameterization, None] = None,
        batch_per_epoch: int = 1000,
        quasirandom: bool = False,
        num_workers: int = 0,
        loss: Loss = PointwiseLossNorm(),
        shuffle: bool = True,
    ):
        # sample boundary
        invar = geometry.sample_boundary(
            boundary_batch_size * batch_per_epoch,
            criteria=boundary_criteria,
            parameterization=parameterization,
            quasirandom=quasirandom,
        )
        invar["area"] *= batch_per_epoch

        # make variational boundary dataset
        dataset_boundary = DictVariationalDataset(
            invar=invar,
            outvar_names=outvar_names,
        )

        # sample interior
        invar = geometry.sample_interior(
            interior_batch_size * batch_per_epoch,
            bounds=interior_bounds,
            criteria=interior_criteria,
            parameterization=parameterization,
            quasirandom=quasirandom,
        )
        invar["area"] *= batch_per_epoch

        # make variational interior dataset
        dataset_interior = DictVariationalDataset(
            invar=invar,
            outvar_names=outvar_names,
        )

        datasets = {"boundary": dataset_boundary, "interior": dataset_interior}
        batch_sizes = {"boundary": boundary_batch_size, "interior": interior_batch_size}

        # initialize constraint
        super().__init__(
            nodes=nodes,
            datasets=datasets,
            batch_sizes=batch_sizes,
            loss=loss,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
        )


class DeepONetConstraint(PointwiseConstraint):
    """
    Base DeepONet Constraint class for all DeepONets
    """

    def save_batch(self, filename):
        # sample batch
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        invar = Constraint._set_device(invar, device=self.device, requires_grad=True)
        true_outvar = Constraint._set_device(true_outvar, device=self.device)
        lambda_weighting = Constraint._set_device(lambda_weighting, device=self.device)

        # If using DDP, strip out collective stuff to prevent deadlocks
        # This only works either when one process alone calls in to save_batch
        # or when multiple processes independently save data
        if hasattr(self.model, "module"):
            modl = self.model.module
        else:
            modl = self.model

        # compute pred outvar
        pred_outvar = modl(invar)

        # rename values and save batch to vtk file TODO clean this up after graph unroll stuff
        named_lambda_weighting = {
            "lambda_" + key: value for key, value in lambda_weighting.items()
        }
        named_true_outvar = {"true_" + key: value for key, value in true_outvar.items()}
        named_pred_outvar = {"pred_" + key: value for key, value in pred_outvar.items()}
        save_var = {
            **{key: value for key, value in invar.items()},
            **named_true_outvar,
            **named_pred_outvar,
            **named_lambda_weighting,
        }
        save_var = {
            key: value.cpu().detach().numpy() for key, value in save_var.items()
        }
        np.savez_compressed(filename + ".npz", **save_var)

    @classmethod
    def from_numpy(
        cls,
        nodes: List[Node],
        invar: Dict[str, np.ndarray],
        outvar: Dict[str, np.ndarray],
        batch_size: int,
        lambda_weighting: Dict[str, np.ndarray] = None,
        loss: Loss = PointwiseLossNorm(),
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):
        """
        Create custom DeepONet constraint from numpy arrays.

        Parameters
        ----------
        nodes : List[Node]
            List of PhysicsNeMo Nodes to unroll graph with.
        invar : Dict[str, np.ndarray (N, 1)]
            Dictionary of numpy arrays as input.
        outvar : Dict[str, np.ndarray (N, 1)]
            Dictionary of numpy arrays to enforce constraint on.
        batch_size : int
            Batch size used in training.
        lambda_weighting : Dict[str, np.ndarray (N, 1)]
            Dictionary of numpy arrays to pointwise weight losses.
            Default is ones.
        loss : Loss
            PhysicsNeMo Loss module that defines the loss type, (e.g. L2, L1, ...).
        shuffle : bool, optional
            Randomly shuffle examples in dataset every epoch, by default True
        drop_last : bool, optional
            Drop last mini-batch if dataset not fully divisible but batch_size, by default False
        num_workers : int
            Number of worker used in fetching data.
        """

        # make point dataset
        dataset = DictPointwiseDataset(
            invar=invar,
            outvar=outvar,
            lambda_weighting=lambda_weighting,
        )

        return cls(
            nodes=nodes,
            dataset=dataset,
            loss=loss,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
