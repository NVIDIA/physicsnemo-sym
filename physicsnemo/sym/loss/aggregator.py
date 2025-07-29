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

# Import libraries
import torch
import logging
from torch import nn
from typing import Dict, List, Optional, Callable, Union

# Import from PhysicsNeMo
from physicsnemo.sym.eq.derivatives import gradient_autodiff
from physicsnemo.sym.hydra import to_absolute_path, add_hydra_run_path

logger = logging.getLogger(__name__)


class Aggregator(nn.Module):
    """
    Base class for loss aggregators
    """

    def __init__(self, params, num_losses, weights):
        super().__init__()
        self.params: List[torch.Tensor] = list(params)
        self.num_losses: int = num_losses
        self.weights: Optional[Dict[str, float]] = weights
        self.device: torch.device
        self.device = list(set(p.device for p in self.params))[0]
        self.init_loss: torch.Tensor = torch.tensor(0.0, device=self.device)

        def weigh_losses_initialize(
            weights: Optional[Dict[str, float]],
        ) -> Callable[
            [Dict[str, torch.Tensor], Optional[Dict[str, float]]],
            Dict[str, torch.Tensor],
        ]:
            if weights is None:

                def weigh_losses(
                    losses: Dict[str, torch.Tensor], weights: None
                ) -> Dict[str, torch.Tensor]:
                    return losses

            else:

                def weigh_losses(
                    losses: Dict[str, torch.Tensor], weights: Dict[str, float]
                ) -> Dict[str, torch.Tensor]:
                    for key in losses.keys():
                        if key not in weights.keys():
                            weights.update({key: 1.0})
                    losses = {key: weights[key] * losses[key] for key in losses.keys()}
                    return losses

            return weigh_losses

        self.weigh_losses = weigh_losses_initialize(self.weights)


class Sum(Aggregator):
    """
    Loss aggregation by summation
    """

    def __init__(self, params, num_losses, weights=None):
        super().__init__(params, num_losses, weights)

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Aggregates the losses by summation

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        # Add losses
        for key in losses.keys():
            loss += losses[key]
        return loss


class GradNorm(Aggregator):
    """
    GradNorm for loss aggregation
    Reference: "Chen, Z., Badrinarayanan, V., Lee, C.Y. and Rabinovich, A., 2018, July.
    Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks.
    In International Conference on Machine Learning (pp. 794-803). PMLR."
    """

    def __init__(self, params, num_losses, alpha=1.0, weights=None):
        super().__init__(params, num_losses, weights)
        self.alpha: float = alpha
        self.lmbda: torch.nn.Parameter = nn.Parameter(
            torch.zeros(num_losses, device=self.device)
        )
        self.register_buffer(
            "init_losses", torch.zeros(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using the gradNorm algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # get initial losses
        if step == 0:
            for i, key in enumerate(losses.keys()):
                self.init_losses[i] = losses[key].clone().detach()

        with torch.no_grad():
            normalizer: torch.Tensor = self.num_losses / (torch.exp(self.lmbda).sum())
            for i in range(self.num_losses):
                self.lmbda[i] = self.lmbda[i].clone() + torch.log(
                    normalizer.detach()
                )  # c*exp(x) = exp(log(c)+x)
        lmbda_exp: torch.Tensor = torch.exp(self.lmbda)

        # compute relative losses, inverse rate, and grad coefficient
        losses_stacked: torch.Tensor = torch.stack(list(losses.values()))
        with torch.no_grad():
            relative_losses: torch.Tensor = torch.div(losses_stacked, self.init_losses)
            inverse_rate: torch.Tensor = relative_losses / (relative_losses.mean())
            gradnorm_coef: torch.Tensor = torch.pow(inverse_rate, self.alpha)

        # compute gradient norm and average gradient norm
        grads_norm: torch.Tensor = torch.zeros_like(self.init_losses)
        shared_params: torch.Tensor = self.params[-2]  # TODO generalize this
        for i, key in enumerate(losses.keys()):
            grads: torch.Tensor = gradient_autodiff(losses[key], [shared_params])[0]
            grads_norm[i] = torch.norm(lmbda_exp[i] * grads.detach(), p=2)
        avg_grad: torch.Tensor = grads_norm.detach().mean()

        # compute gradnorm & model losses
        loss_gradnorm: torch.Tensor = torch.abs(
            grads_norm - avg_grad * gradnorm_coef
        ).sum()
        loss_model: torch.Tensor = (lmbda_exp.detach() * losses_stacked).sum()
        loss: torch.Tensor = loss_gradnorm + loss_model
        return loss


class ResNorm(Aggregator):
    """
    Residual normalization for loss aggregation
    Contributors: T. Nandi, D. Van Essendelft, M. A. Nabian
    """

    def __init__(self, params, num_losses, alpha=1.0, weights=None):
        super().__init__(params, num_losses, weights)
        self.alpha: float = alpha
        self.lmbda: torch.nn.Parameter = nn.Parameter(
            torch.zeros(num_losses, device=self.device)
        )
        self.register_buffer(
            "init_losses", torch.zeros(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using the ResNorm algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # get initial losses
        if step == 0:
            for i, key in enumerate(losses.keys()):
                self.init_losses[i] = losses[key].clone().detach()

        with torch.no_grad():
            normalizer: torch.Tensor = self.num_losses / (torch.exp(self.lmbda).sum())
            for i in range(self.num_losses):
                self.lmbda[i] = self.lmbda[i].clone() + torch.log(
                    normalizer.detach()
                )  # c*exp(x) = exp(log(c)+x)
        lmbda_exp: torch.Tensor = torch.exp(self.lmbda)

        # compute relative losses, inverse rate, and grad coefficient
        losses_stacked: torch.Tensor = torch.stack(list(losses.values()))
        with torch.no_grad():
            relative_losses: torch.Tensor = torch.div(losses_stacked, self.init_losses)
            inverse_rate: torch.Tensor = relative_losses / (relative_losses.mean())
            resnorm_coef: torch.Tensor = torch.pow(inverse_rate, self.alpha)

        # compute residual norm and average residual norm
        residuals: torch.Tensor = torch.zeros_like(self.init_losses)
        for i, key in enumerate(losses.keys()):
            residuals[i] = lmbda_exp[i] * losses[key].detach()
        avg_residuals: torch.Tensor = losses_stacked.detach().mean()

        # compute ResNorm & model losses
        loss_resnorm: torch.Tensor = torch.abs(
            residuals - avg_residuals * resnorm_coef
        ).sum()
        loss_model: torch.Tensor = (lmbda_exp.detach() * losses_stacked).sum()
        loss: torch.Tensor = loss_resnorm + loss_model
        return loss


class HomoscedasticUncertainty(Aggregator):
    """
    Homoscedastic task uncertainty for loss aggregation
    Reference: "Reference: Kendall, A., Gal, Y. and Cipolla, R., 2018.
    Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7482-7491)."
    """

    def __init__(self, params, num_losses, weights=None):
        super().__init__(params, num_losses, weights)
        self.log_var: torch.nn.Parameter = nn.Parameter(
            torch.zeros(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using homoscedastic task uncertainty

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        # Compute precision
        precision: torch.Tensor = torch.exp(-self.log_var)

        # Aggregate losses
        for i, key in enumerate(losses.keys()):
            loss += precision[i] * losses[key]
        loss += self.log_var.sum()
        loss /= 2.0
        return loss


class LRAnnealing(Aggregator):
    """
    Learning rate annealing for loss aggregation
    References: "Wang, S., Teng, Y. and Perdikaris, P., 2020.
    Understanding and mitigating gradient pathologies in physics-informed
    neural networks. arXiv preprint arXiv:2001.04536.", and
    "Jin, X., Cai, S., Li, H. and Karniadakis, G.E., 2021.
    NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the
    incompressible Navier-Stokes equations. Journal of Computational Physics, 426, p.109951."
    """

    def __init__(
        self,
        params,
        num_losses,
        update_freq=1,
        alpha=0.01,
        ref_key=None,
        eps=1e-8,
        weights=None,
    ):
        super().__init__(params, num_losses, weights)
        self.update_freq: int = update_freq
        self.alpha: float = alpha
        self.ref_key: Union[str, None] = ref_key
        self.eps: float = eps
        self.register_buffer(
            "lmbda_ema", torch.ones(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using the learning rate annealing algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        # Determine reference loss
        if self.ref_key is None:
            ref_idx = 0
        else:
            for i, key in enumerate(losses.keys()):
                if self.ref_key in key:
                    ref_idx = i
                    break

        # Update loss weights and aggregate losses
        if step % self.update_freq == 0:
            grads_mean: List[torch.Tensor] = []

            # Compute the mean of each loss gradients
            for key in losses.keys():
                grads: List[torch.Tensor] = gradient_autodiff(losses[key], self.params)
                grads_flattened: List[torch.Tensor] = []
                for i in range(len(grads)):
                    if grads[i] is not None:
                        grads_flattened.append(torch.abs(torch.flatten(grads[i])))
                grads_mean.append((torch.mean(torch.cat(grads_flattened))))

            # Compute the exponential moving average of weights and aggregate losses
            for i, key in enumerate(losses.keys()):
                with torch.no_grad():
                    self.lmbda_ema[i] *= 1.0 - self.alpha
                    self.lmbda_ema[i] += (
                        self.alpha * grads_mean[ref_idx] / (grads_mean[i] + self.eps)
                    )
                loss += self.lmbda_ema[i].clone() * losses[key]

        # Aggregate losses without update to loss weights
        else:
            for i, key in enumerate(losses.keys()):
                loss += self.lmbda_ema[i] * losses[key]
        return loss


class SoftAdapt(Aggregator):
    """
    SoftAdapt for loss aggregation
    Reference: "Heydari, A.A., Thompson, C.A. and Mehmood, A., 2019.
    Softadapt: Techniques for adaptive loss weighting of neural networks with multi-part loss functions.
    arXiv preprint arXiv: 1912.12355."
    """

    def __init__(self, params, num_losses, eps=1e-8, weights=None):
        super().__init__(params, num_losses, weights)
        self.eps: float = eps
        self.register_buffer(
            "prev_losses", torch.zeros(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using the original variant of the softadapt algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        # Aggregate losses by summation at step 0
        if step == 0:
            for i, key in enumerate(losses.keys()):
                loss += losses[key]
                self.prev_losses[i] = losses[key].clone().detach()

        # Aggregate losses using SoftAdapt for step > 0
        else:
            lmbda: torch.Tensor = torch.ones_like(self.prev_losses)
            lmbda_sum: torch.Tensor = torch.zeros_like(self.init_loss)
            losses_stacked: torch.Tensor = torch.stack(list(losses.values()))
            normalizer: torch.Tensor = (losses_stacked / self.prev_losses).max()
            for i, key in enumerate(losses.keys()):
                with torch.no_grad():
                    lmbda[i] = torch.exp(
                        losses[key] / (self.prev_losses[i] + self.eps) - normalizer
                    )
                    lmbda_sum += lmbda[i]
                loss += lmbda[i].clone() * losses[key]
                self.prev_losses[i] = losses[key].clone().detach()
            loss *= self.num_losses / (lmbda_sum + self.eps)
        return loss


class Relobralo(Aggregator):
    """
    Relative loss balancing with random lookback
    Reference: "Bischof, R. and Kraus, M., 2021.
    Multi-Objective Loss Balancing for Physics-Informed Deep Learning.
    arXiv preprint arXiv:2110.09813."
    """

    def __init__(
        self, params, num_losses, alpha=0.95, beta=0.99, tau=1.0, eps=1e-8, weights=None
    ):
        super().__init__(params, num_losses, weights)
        self.alpha: float = alpha
        self.beta: float = beta
        self.tau: float = tau
        self.eps: float = eps
        self.register_buffer(
            "init_losses", torch.zeros(self.num_losses, device=self.device)
        )
        self.register_buffer(
            "prev_losses", torch.zeros(self.num_losses, device=self.device)
        )
        self.register_buffer(
            "lmbda_ema", torch.ones(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using the ReLoBRaLo algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        # Aggregate losses by summation at step 0
        if step == 0:
            for i, key in enumerate(losses.keys()):
                loss += losses[key]
                self.init_losses[i] = losses[key].clone().detach()
                self.prev_losses[i] = losses[key].clone().detach()

        # Aggregate losses using ReLoBRaLo for step > 0
        else:
            losses_stacked: torch.Tensor = torch.stack(list(losses.values()))
            normalizer_prev: torch.Tensor = (
                losses_stacked / (self.tau * self.prev_losses)
            ).max()
            normalizer_init: torch.Tensor = (
                losses_stacked / (self.tau * self.init_losses)
            ).max()
            rho: torch.Tensor = torch.bernoulli(torch.tensor(self.beta))
            with torch.no_grad():
                lmbda_prev: torch.Tensor = torch.exp(
                    losses_stacked / (self.tau * self.prev_losses + self.eps)
                    - normalizer_prev
                )
                lmbda_init: torch.Tensor = torch.exp(
                    losses_stacked / (self.tau * self.init_losses + self.eps)
                    - normalizer_init
                )
                lmbda_prev *= self.num_losses / (lmbda_prev.sum() + self.eps)
                lmbda_init *= self.num_losses / (lmbda_init.sum() + self.eps)

            # Compute the exponential moving average of weights and aggregate losses
            for i, key in enumerate(losses.keys()):
                with torch.no_grad():
                    self.lmbda_ema[i] = self.alpha * (
                        rho * self.lmbda_ema[i].clone() + (1.0 - rho) * lmbda_init[i]
                    )
                    self.lmbda_ema[i] += (1.0 - self.alpha) * lmbda_prev[i]
                loss += self.lmbda_ema[i].clone() * losses[key]
                self.prev_losses[i] = losses[key].clone().detach()
        return loss


class NTK(nn.Module):
    def __init__(self, run_per_step: int = 1000, save_name: Union[str, None] = None):
        super(NTK, self).__init__()
        self.run_per_step = run_per_step
        self.if_csv_head = True

        self.save_name = (
            to_absolute_path(add_hydra_run_path(save_name)) if save_name else None
        )

        if self.save_name:
            logger.warning(
                "Cuda graphs does not work when saving NTK values to file! Set `cuda_graphs` to false."
            )

    def group_ntk(self, model, losses):
        # The item in this losses should scalar loss values after MSE, etc.
        ntk_value = dict()
        for key, loss in losses.items():
            grad = torch.autograd.grad(
                torch.sqrt(torch.abs(loss)),
                model.parameters(),
                retain_graph=True,
                allow_unused=True,
            )
            ntk_value[key] = torch.sqrt(
                torch.sum(
                    torch.stack(
                        [torch.sum(t.detach() ** 2) for t in grad if t is not None],
                        dim=0,
                    )
                )
            )
        return ntk_value

    def save_ntk(self, ntk_dict, step):
        import pandas as pd  # TODO: Remove

        output_dict = {}
        for key, value in ntk_dict.items():
            output_dict[key] = value.cpu().numpy()
        df = pd.DataFrame(output_dict, index=[step])
        df.to_csv(self.save_name + ".csv", mode="a", header=self.if_csv_head)
        self.if_csv_head = False

    def forward(self, constraints, ntk_weights, step):
        losses = dict()
        dict_constraint_losses = dict()
        ntk_sum = 0
        # Execute constraint forward passes
        for key, constraint in constraints.items():
            # TODO: Test streaming here
            torch.cuda.nvtx.range_push(f"Running Constraint {key}")
            constraint.forward()
            torch.cuda.nvtx.range_pop()

        for key, constraint in constraints.items():
            # compute losses
            constraint_losses = constraint.loss(step)
            if (step % self.run_per_step == 0) and (step > 0):
                ntk_dict = self.group_ntk(constraint.model, constraint_losses)
            else:
                ntk_dict = None
            if ntk_dict is not None:
                ntk_weights[key] = ntk_dict
            if ntk_weights.get(key) is not None:
                ntk_sum += torch.sum(
                    torch.stack(list(ntk_weights[key].values()), dim=0)
                )
            dict_constraint_losses[key] = constraint_losses

        if step == 0:  # May not work on restarts
            ntk_sum = 1.0

        if self.save_name and (step % self.run_per_step == 0) and (step > 0):
            self.save_ntk(
                {
                    d_key + "_" + k: v
                    for d_key, d in ntk_weights.items()
                    for k, v in d.items()
                },
                step,
            )

        for key, constraint_losses in dict_constraint_losses.items():
            # add together losses of like kind
            for loss_key, value in constraint_losses.items():
                if (
                    ntk_weights.get(key) is None
                    or ntk_weights[key].get(loss_key) is None
                ):
                    ntk_weight = ntk_sum / 1.0
                else:
                    ntk_weight = ntk_sum / ntk_weights[key][loss_key]
                if loss_key not in list(losses.keys()):
                    losses[loss_key] = ntk_weight * value
                else:
                    losses[loss_key] += ntk_weight * value

        return losses, ntk_weights
