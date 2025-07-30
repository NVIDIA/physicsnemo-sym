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
import torch.nn as nn
from torch import Tensor

from typing import Dict, List
from torch.autograd import Function


class LossL2(Function):
    @staticmethod
    def forward(
        ctx,
        pred_outvar: Tensor,
        true_outvar: Tensor,
        lambda_weighting: Tensor,
        area: Tensor,
    ):
        ctx.save_for_backward(pred_outvar, true_outvar, lambda_weighting, area)
        loss = pde_cpp.l2_loss_forward(pred_outvar, true_outvar, lambda_weighting, area)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        pred_outvar, true_outvar, lambda_weighting, area = ctx.saved_tensors

        outputs = pde_cpp.l2_loss_backward(
            grad_output, pred_outvar, true_outvar, lambda_weighting, area
        )
        return outputs[0], None, None, None


class Loss(nn.Module):
    """
    Base class for all loss functions
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError("Subclass of Loss needs to implement this")


class PointwiseLossNorm(Loss):
    """
    L-p loss function for pointwise data
    Computes the p-th order loss of each output tensor

    Parameters
    ----------
    ord : int
        Order of the loss. For example, `ord=2` would be the L2 loss.
    """

    def __init__(self, ord: int = 2):
        super().__init__()
        self.ord: int = ord

    @staticmethod
    def _loss(
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
        ord: float,
    ) -> Dict[str, Tensor]:
        losses = {}
        for key, value in pred_outvar.items():
            l = lambda_weighting[key] * torch.abs(
                pred_outvar[key] - true_outvar[key]
            ).pow(ord)
            if "area" in invar.keys():
                l *= invar["area"]
            losses[key] = l.sum()
        return losses

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        return PointwiseLossNorm._loss(
            invar, pred_outvar, true_outvar, lambda_weighting, step, self.ord
        )


class IntegralLossNorm(Loss):
    """
    L-p loss function for integral data
    Computes the p-th order loss of each output tensor

    Parameters
    ----------
    ord : int
        Order of the loss. For example, `ord=2` would be the L2 loss.
    """

    def __init__(self, ord: int = 2):
        super().__init__()
        self.ord: int = ord

    @staticmethod
    def _loss(
        list_invar: List[Dict[str, Tensor]],
        list_pred_outvar: List[Dict[str, Tensor]],
        list_true_outvar: List[Dict[str, Tensor]],
        list_lambda_weighting: List[Dict[str, Tensor]],
        step: int,
        ord: float,
    ) -> Dict[str, Tensor]:
        # compute integral losses
        losses = {key: 0 for key in list_pred_outvar[0].keys()}
        for invar, pred_outvar, true_outvar, lambda_weighting in zip(
            list_invar, list_pred_outvar, list_true_outvar, list_lambda_weighting
        ):
            for key in pred_outvar.keys():
                losses[key] += (
                    lambda_weighting[key]
                    * torch.abs(
                        true_outvar[key] - (invar["area"] * pred_outvar[key]).sum()
                    ).pow(ord)
                ).sum()
        return losses

        losses = {}
        for key, value in pred_outvar.items():
            l = lambda_weighting[key] * torch.abs(
                pred_outvar[key] - true_outvar[key]
            ).pow(ord)
            if "area" in invar.keys():
                l *= invar["area"]
            losses[key] = l.sum()
        return losses

    def forward(
        self,
        list_invar: List[Dict[str, Tensor]],
        list_pred_outvar: List[Dict[str, Tensor]],
        list_true_outvar: List[Dict[str, Tensor]],
        list_lambda_weighting: List[Dict[str, Tensor]],
        step: int,
    ) -> Dict[str, Tensor]:
        return IntegralLossNorm._loss(
            list_invar,
            list_pred_outvar,
            list_true_outvar,
            list_lambda_weighting,
            step,
            self.ord,
        )


class DecayedLossNorm(Loss):
    """
    Base class for decayed loss norm
    """

    def __init__(
        self,
        start_ord: int = 2,
        end_ord: int = 1,
        decay_steps: int = 1000,
        decay_rate: float = 0.95,
    ):
        super().__init__()
        self.start_ord: int = start_ord
        self.end_ord: int = end_ord
        self.decay_steps: int = decay_steps
        self.decay_rate: int = decay_rate

    def ord(self, step):
        return self.start_ord - (self.start_ord - self.end_ord) * (
            1.0 - self.decay_rate ** (step / self.decay_steps)
        )


class DecayedPointwiseLossNorm(DecayedLossNorm):
    """
    Loss function for pointwise data where the norm of
    the loss is decayed from a start value to an end value.

    Parameters
    ----------
    start_ord : int
        Order of the loss when current iteration is zero.
    end_ord : int
        Order of the loss to decay to.
    decay_steps : int
        Number of steps to take for each `decay_rate`.
    decay_rate :
        The rate of decay from `start_ord` to `end_ord`. The current ord
        will be given by `ord = start_ord - (start_ord - end_ord) * (1.0 - decay_rate**(current_step / decay_steps))`.
    """

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        return PointwiseLossNorm._loss(
            invar, pred_outvar, true_outvar, lambda_weighting, step, self.ord(step)
        )


class DecayedIntegralLossNorm(DecayedLossNorm):
    """
    Loss function for integral data where the norm of
    the loss is decayed from a start value to an end value.

    Parameters
    ----------
    start_ord : int
        Order of the loss when current iteration is zero.
    end_ord : int
        Order of the loss to decay to.
    decay_steps : int
        Number of steps to take for each `decay_rate`.
    decay_rate :
        The rate of decay from `start_ord` to `end_ord`. The current ord
        will be given by `ord = start_ord - (start_ord - end_ord) * (1.0 - decay_rate**(current_step / decay_steps))`.
    """

    def forward(
        self,
        list_invar: List[Dict[str, Tensor]],
        list_pred_outvar: List[Dict[str, Tensor]],
        list_true_outvar: List[Dict[str, Tensor]],
        list_lambda_weighting: List[Dict[str, Tensor]],
        step: int,
    ) -> Dict[str, Tensor]:
        return IntegralLossNorm._loss(
            list_invar,
            list_pred_outvar,
            list_true_outvar,
            list_lambda_weighting,
            step,
            self.ord(step),
        )


class CausalLossNorm(Loss):
    """
    Causal loss function for pointwise data
    Computes the p-th order loss of each output tensor

    Parameters
    ----------
    ord : int
        Order of the loss. For example, `ord=2` would be the L2 loss.
    eps: float
        Causal parameter determining the slopeness of the temporal weights. "eps=1.0" would be default value.
    n_chunks: int
        Number of chunks splitting the temporal domain evenly.
    """

    def __init__(self, ord: int = 2, eps: float = 1.0, n_chunks=10):
        super().__init__()
        self.ord: int = ord
        self.eps: float = eps
        self.n_chunks: int = n_chunks

    @staticmethod
    def _loss(
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
        ord: float,
        eps: float,
        n_chunks: int,
    ) -> Dict[str, Tensor]:
        losses = {}
        for key, value in pred_outvar.items():
            l = lambda_weighting[key] * torch.abs(
                pred_outvar[key] - true_outvar[key]
            ).pow(ord)

            if "area" in invar.keys():
                l *= invar["area"]

            # batch size should be divided by the number of chunks
            if l.shape[0] % n_chunks != 0:
                raise ValueError(
                    "The batch size must be divided by the number of chunks"
                )
            # divide the loss values into chunks
            l = l.reshape(n_chunks, -1)
            l = l.sum(axis=-1)
            # compute causal temporal weights
            with torch.no_grad():
                w = torch.exp(-eps * torch.cumsum(l, dim=0))
                w = w / w[0]

            l = w * l
            losses[key] = l.sum()

        return losses

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        return CausalLossNorm._loss(
            invar,
            pred_outvar,
            true_outvar,
            lambda_weighting,
            step,
            self.ord,
            self.eps,
            self.n_chunks,
        )
