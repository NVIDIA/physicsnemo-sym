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

""" PhysicsNeMo Solver
"""

import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
import torch.cuda.profiler as profiler
import torch.distributed as dist
from termcolor import colored, cprint
from copy import copy
from operator import add
from omegaconf import DictConfig, OmegaConf
import hydra
import itertools
from collections import Counter
from typing import Dict, List, Optional
import logging
from contextlib import ExitStack

from .amp import DerivScalers, GradScaler, AmpManager
from .domain.constraint import Constraint
from .domain import Domain
from .loss.aggregator import Sum
from .utils.training.stop_criterion import StopCriterion
from .constants import TF_SUMMARY, JIT_PYTORCH_VERSION
from .hydra import (
    instantiate_optim,
    instantiate_sched,
    instantiate_agg,
    add_hydra_run_path,
)
from .distributed.manager import DistributedManager


class AdamMixin:
    """Special functions for training using the standard optimizers
    Should be used with ADAM, SGD, RMSProp, etc.
    """

    def adam_compute_gradients(
        self, aggregator: nn.Module, global_optimizer_model: nn.Module, step: int
    ):
        loss, losses = 0, Counter({})

        if self.cfg.cuda_graphs and self.grad_agg_freq != 1:
            raise ValueError(
                "Gradient Aggregation with CUDA Graphs is not supported currently."
            )

        for agg_step in range(self.grad_agg_freq):
            with torch.autocast(
                self.manager.device.type,
                enabled=self.amp_manager.enabled,
                dtype=self.amp_manager.dtype,
            ):
                if agg_step != 0:  # load new data for subsequent steps
                    self.load_data()
                torch.cuda.nvtx.range_push("Loss computation")
                losses_minibatch = self.compute_losses(step)
                torch.cuda.nvtx.range_pop()
                losses_minibatch = {
                    key: value / self.grad_agg_freq
                    for key, value in losses_minibatch.items()
                }
                torch.cuda.nvtx.range_push("Loss aggregator")
                loss_minibatch = aggregator(losses_minibatch, step)
                torch.cuda.nvtx.range_pop()
                loss += loss_minibatch
            torch.cuda.nvtx.range_push("Weight gradients")
            self.scaler.scale(loss_minibatch).backward()
            torch.cuda.nvtx.range_pop()
            losses.update(losses_minibatch)

        return loss, dict(losses)

    def adam_apply_gradients(self):
        if not self.deriv_scalers.found_inf:
            # using unscale_() to enable clipping of unscaled gradients:
            self.scaler.unscale_(self.optimizer)
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.global_optimizer_model.parameters()),
                max_norm=self.grad_clip_max_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            msg = colored(
                " deriv_scalers found infs, {scale: %s, growth_tracker: %s, max_scale: %s}",
                "yellow",
            )
            self.log.info(
                self.step_str + msg,
                self.deriv_scalers.get_scale(),
                self.deriv_scalers._get_growth_tracker(),
                self.deriv_scalers.get_max_scale(),
            )
        self.deriv_scalers.update()


class AdaHessianMixin:
    """Special functions for training using the higher-order optimizer AdaHessian"""

    def adahess_compute_gradients(
        self, aggregator: nn.Module, global_optimizer_model: nn.Module, step: int
    ):
        if self.amp_manager.enabled:
            raise NotImplementedError("AMP is not supported for this optimizer.")
        # With data hessian we need to keep grad graph on back-prop to approximate
        # the hessian with. The suggested PyTorch way is to use torch.grad instead
        # of backward.
        loss, losses = 0, Counter({})
        grads = [
            torch.zeros_like(parameter)
            for parameter in list(global_optimizer_model.parameters())
        ]
        for agg_step in range(self.grad_agg_freq):
            losses_minibatch = self.compute_losses(step)
            losses_minibatch = {
                key: value / self.grad_agg_freq
                for key, value in losses_minibatch.items()
            }
            loss_minibatch = aggregator(losses_minibatch, step)

            grads_step = torch.autograd.grad(
                loss_minibatch,
                list(global_optimizer_model.parameters()),
                create_graph=True,
            )
            grads = list(map(add, grads, grads_step))

            loss += loss_minibatch
            losses.update(losses_minibatch)
        # Set gradients of models manually
        for grad, param in zip(grads, global_optimizer_model.parameters()):
            param.grad = grad

        return loss, dict(losses)

    def adahess_apply_gradients(self):
        self.adam_apply_gradients()


class BFGSMixin:
    """Special functions for training using BFGS optimizer"""

    def bfgs_compute_gradients(
        self, aggregator: nn.Module, global_optimizer_model: nn.Module, step: int
    ):
        # Dummy functioned used entirely just for logging purposes and storing
        # objects for internal BFGS updates. Gradients are not calc'd here for BFGS
        if self.amp_manager.enabled:
            raise NotImplementedError("AMP is not supported for this optimizer.")
        if self.max_steps != 0:
            self.log.warning("lbfgs optimizer selected. Setting max_steps to 0")
            self.max_steps = 0
        if self.grad_agg_freq != 1:
            self.log.warning("lbfgs optimizer selected. Setting grad_agg_freq to 1")
            self.grad_agg_freq = 1
        losses = self.compute_losses(step)
        loss = aggregator(losses, step)

        self.bfgs_step = step
        self.bfgs_aggregator = aggregator
        # Re-zero any gradients
        for param in global_optimizer_model.parameters():
            param.grad = None

        return loss, losses

    def bfgs_closure_func(self):
        self.optimizer.zero_grad()
        loss = 0
        losses = self.compute_losses(self.bfgs_step)
        loss = self.bfgs_aggregator(losses, self.bfgs_step)

        loss.backward()
        self.bfgs_optim_steps += 1
        return loss

    def bfgs_apply_gradients(self):
        assert (
            not self.bfgs_aggregator is None
        ), "Call bfgs_compute_gradients prior to this!"
        assert not self.bfgs_step is None, "Call bfgs_compute_gradients prior to this!"
        self.bfgs_optim_steps = 0
        self.log.info(f"[step: {self.bfgs_step:10d}] lbfgs optimization in running")
        self.optimizer.step(self.bfgs_closure_func)
        self.log.info(
            f"lbfgs optimization completed after {self.bfgs_optim_steps} steps"
        )


# base class for optimizing networks on loss
class Trainer(AdamMixin, AdaHessianMixin, BFGSMixin):
    """Base class for optimizing networks on losses/constraints"""

    def __init__(self, cfg: DictConfig):
        super(Trainer, self).__init__()

        # Save a local copy of the config
        self.cfg = cfg

        # set training parameters
        self._network_dir = self.cfg.network_dir
        self._initialization_network_dir = self.cfg.initialization_network_dir
        self.max_steps = self.cfg.training.max_steps
        self.grad_agg_freq = self.cfg.training.grad_agg_freq
        self.save_network_freq = self.cfg.training.save_network_freq
        self.print_stats_freq = self.cfg.training.print_stats_freq
        self.summary_freq = self.cfg.training.summary_freq
        self.grad_clip_max_norm = self.cfg.training.grad_clip_max_norm
        self.monitor_grad_clip = self.cfg.training.monitor_grad_clip
        self.stop_criterion_metric = self.cfg.stop_criterion.metric
        self.stop_criterion_min_delta = self.cfg.stop_criterion.min_delta
        self.stop_criterion_patience = self.cfg.stop_criterion.patience
        self.stop_criterion_mode = self.cfg.stop_criterion.mode
        self.stop_criterion_freq = self.cfg.stop_criterion.freq
        self.stop_criterion_strict = self.cfg.stop_criterion.strict

        self.save_filetypes = self.cfg.save_filetypes
        self.summary_histograms = self.cfg.summary_histograms

        self.apply_gradients = self._apply_gradients
        self.compute_gradients = self._compute_gradients

        # make logger
        self.log = logging.getLogger(__name__)

        # Set distributed manager
        self.manager = DistributedManager()
        self.amp_manager = AmpManager()

        # set device
        self.device = self.manager.device

        # force setting amp dtype as bfloat16 if on cpu
        if (
            self.amp_manager.enabled
            and self.amp_manager.dtype == torch.float16
            and self.manager.device.type == "cpu"
        ):
            self.amp_manager.dtype = "bfloat16"
            self.log.warning(
                "Switching amp_dtype to bfloat16, AutocastCPU only supports bfloat16"
            )

    def compute_losses(self, step: int):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def _compute_gradients(self):
        raise NotImplementedError("Config should set the compute_gradients function")

    def _apply_gradients(self):
        raise NotImplementedError("Config should set the apply_gradients function")

    def get_saveable_models(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def create_global_optimizer_model(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def load_network(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def save_checkpoint(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def setup_deriv_scaler(self, deriv_scalers: DerivScalers):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def record_constraints(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def record_validators(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    @property
    def has_validators(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def record_inferencers(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    @property
    def has_inferencers(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def record_monitors(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    @property
    def has_monitors(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def get_num_losses(self):
        raise NotImplementedError("Subclass of Constraint needs to implement this")

    def _record_constraints(self):
        data_parallel_rank = (
            self.manager.group_rank("data_parallel") if self.manager.distributed else 0
        )
        if data_parallel_rank == 0:
            rec_inferencer_start = time.time()
            self.record_constraints()
            self.log.debug(
                f"{self.step_str} saved constraint results to {self.network_dir}"
            )
            self.log.info(
                f"{self.step_str} record constraint batch time: {time.time()-rec_inferencer_start:10.3e}s"
            )

    def _record_validators(self, step):
        data_parallel_rank = (
            self.manager.group_rank("data_parallel") if self.manager.distributed else 0
        )
        if data_parallel_rank == 0:
            rec_validation_start = time.time()
            self.validator_outvar = self.record_validators(step)
            self.log.debug(
                f"{self.step_str} saved validator results to {self.network_dir}"
            )
            self.log.info(
                f"{self.step_str} record validators time: {time.time()-rec_validation_start:10.3e}s"
            )

    def _record_inferencers(self, step):
        data_parallel_rank = (
            self.manager.group_rank("data_parallel") if self.manager.distributed else 0
        )
        if data_parallel_rank == 0:
            rec_inferencer_start = time.time()
            self.record_inferencers(step)
            self.log.debug(
                f"{self.step_str} saved inferencer results to {self.network_dir}"
            )
            self.log.info(
                f"{self.step_str} record inferencers time: {time.time()-rec_inferencer_start:10.3e}s"
            )

    def _record_monitors(self, step):
        data_parallel_rank = (
            self.manager.group_rank("data_parallel") if self.manager.distributed else 0
        )
        if data_parallel_rank == 0:
            if self.has_monitors:
                rec_monitor_start = time.time()
                self.monitor_outvar = self.record_monitors(step)
                self.log.debug(
                    f"{self.step_str} saved monitor results to {self.network_dir}"
                )

            # write parameter histograms to tensorboard
            if self.summary_histograms != "off":
                for (
                    name,
                    parameter,
                ) in self.global_optimizer_model.named_parameters():
                    name = name.split(".")
                    name = ".".join(name[:-1]) + "/" + ".".join(name[-1:])

                    if self.summary_histograms == "linear":
                        self.writer.add_histogram(
                            name, parameter.detach().flatten(), step
                        )
                        if parameter.grad is not None:
                            # skip if grads contain infs/NaNs
                            if torch.any(
                                torch.logical_not(torch.isfinite(parameter.grad))
                            ).item():
                                continue
                            self.writer.add_histogram(
                                name + "_gradient",
                                parameter.grad.detach().flatten(),
                                step,
                            )
                    elif self.summary_histograms == "log2":
                        self.writer.add_histogram(
                            name + "_log2",
                            (parameter.detach().flatten().abs() + 1e-30).log2(),
                            step,
                        )
                        if parameter.grad is not None:
                            # skip if grads contain infs/NaNs
                            if torch.any(
                                torch.logical_not(torch.isfinite(parameter.grad))
                            ).item():
                                continue
                            self.writer.add_histogram(
                                name + "_gradient_log2",
                                (
                                    parameter.grad.detach().flatten().abs() + 1e-30
                                ).log2(),
                                step,
                            )

            # monitoring total gradient norm and max gradient
            if self.monitor_grad_clip:
                parameters = [
                    p
                    for p in self.global_optimizer_model.parameters()
                    if p.grad is not None
                ]
                max_grad = torch.max(
                    torch.stack([p.grad.detach().max() for p in parameters])
                )
                # total_norm: code from torch.nn.utils.clip_grad_norm_
                # https://github.com/pytorch/pytorch/blob/da764f92244985b6b9dacd68e65a4ed9b1de2e78/torch/nn/utils/clip_grad.py#L42
                total_norm = torch.norm(
                    torch.stack(
                        [
                            torch.norm(p.grad.detach(), 2.0).to(
                                parameters[0].grad.device
                            )
                            for p in parameters
                        ]
                    ),
                    2.0,
                )
                self.writer.add_scalar(
                    "Monitors/grad_max", max_grad.item(), step, new_style=True
                )
                self.writer.add_scalar(
                    "Monitors/grad_norm", total_norm.item(), step, new_style=True
                )

            if self.has_monitors:
                self.log.info(
                    f"{self.step_str} record monitor time: {time.time()-rec_monitor_start:10.3e}s"
                )

    # check if stopping criterion is met
    def _check_stopping_criterion(self, loss, losses, step):
        if self.manager.rank == 0:
            if self.stop_criterion_metric is None:
                return False
            elif step % self.stop_criterion_freq == 0:
                criterion_metric_dict = {"loss": {"loss": loss.cpu().detach().numpy()}}
                criterion_metric_dict["loss"].update(
                    {key: val.cpu().detach().numpy() for key, val in losses.items()}
                )
                if self.has_monitors:
                    criterion_metric_dict.update(
                        {
                            "monitor": {
                                key: val.cpu().detach().numpy()
                                for key, val in self.monitor_outvar.items()
                            }
                        }
                    )
                if self.has_validators:
                    criterion_metric_dict.update(
                        {
                            "validation": {
                                key: val.cpu().detach().numpy()
                                for key, val in self.validator_outvar.items()
                            }
                        }
                    )
                stop_training = self.stop_criterion.evaluate(criterion_metric_dict)
                return stop_training
            else:
                return False

    def _train_loop(
        self,
        sigterm_handler=None,
    ):  # TODO this train loop may be broken up into methods if need for future children classes

        # make directory if doesn't exist
        if self.manager.rank == 0:
            # exist_ok=True to skip creating directory that already exists
            os.makedirs(self.network_dir, exist_ok=True)

        # create global model for restoring and saving
        self.saveable_models = self.get_saveable_models()
        self.global_optimizer_model = self.create_global_optimizer_model()

        # initialize optimizer from hydra
        self.compute_gradients = getattr(
            self, self.cfg.optimizer._params_.compute_gradients
        )
        self.apply_gradients = getattr(
            self, self.cfg.optimizer._params_.apply_gradients
        )
        self.optimizer = instantiate_optim(self.cfg, model=self.global_optimizer_model)

        # initialize scheduler from hydra
        self.scheduler = instantiate_sched(self.cfg, optimizer=self.optimizer)

        # initialize aggregator from hydra
        self.aggregator = instantiate_agg(
            self.cfg,
            model=self.global_optimizer_model.parameters(),
            num_losses=self.get_num_losses(),
        )

        if self.cfg.jit:
            # Warn user if pytorch version difference
            if not torch.__version__ == JIT_PYTORCH_VERSION:
                self.log.warn(
                    f"Installed PyTorch version {torch.__version__} is not TorchScript"
                    + f" supported in PhysicsNeMo. Version {JIT_PYTORCH_VERSION} is officially supported."
                )

            self.aggregator = torch.jit.script(self.aggregator)
            if self.amp_manager.enabled:
                torch._C._jit_set_autocast_mode(True)

        if len(list(self.aggregator.parameters())) > 0:
            self.log.debug("Adding loss aggregator param group. LBFGS will not work!")
            self.optimizer.add_param_group(
                {"params": list(self.aggregator.parameters())}
            )

        # AMP scalars and deriv_scalers are only enabled when amp dtype is float16
        scaler_enabled = self.amp_manager.scaler_enabled
        self.scaler = GradScaler(
            enabled=scaler_enabled,
            growth_interval=1000,
            recover_threshold=2**7,
            recover_growth_interval=100,
            growth_factor=2.0,
        )
        self.deriv_scalers = DerivScalers(
            enabled=scaler_enabled,
            init_scale=2**0,
            max_scale=self.amp_manager.default_max_scale,  # default as 1
            growth_interval=1000,
            recover_threshold=2**-6,
            recover_growth_interval=100,
        )
        if scaler_enabled:
            self.setup_deriv_scaler(self.deriv_scalers)

        # make stop criterion
        if self.stop_criterion_metric is not None:
            self.stop_criterion = StopCriterion(
                self.stop_criterion_metric,
                self.stop_criterion_min_delta,
                self.stop_criterion_patience,
                self.stop_criterion_mode,
                self.stop_criterion_freq,
                self.stop_criterion_strict,
                self.cfg.training.rec_monitor_freq,
                self.cfg.training.rec_validation_freq,
            )

        # load network
        self.initial_step = self.load_network()

        # # make summary writer
        self.writer = SummaryWriter(
            log_dir=self.network_dir, purge_step=self.summary_freq + 1
        )
        self.summary_histograms = self.cfg["summary_histograms"]

        # write tensorboard config
        if self.manager.rank == 0:
            self.writer.add_text(
                "config", f"<pre>{str(OmegaConf.to_yaml(self.cfg))}</pre>"
            )

        # create profiler
        try:
            self.profile = self.cfg.profiler.profile
            self.profiler_start_step = self.cfg.profiler.start_step
            self.profiler_end_step = self.cfg.profiler.end_step
            if self.profiler_end_step < self.profiler_start_step:
                self.profile = False
        except:
            self.profile = False
            self.profiler_start_step = -1
            self.profiler_end_step = -1

        # Distributed barrier before starting the train loop
        if self.manager.distributed:
            dist.barrier(device_ids=[self.manager.local_rank])
        barrier_flag = False

        if self.manager.cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t = time.time()

        # termination signal handler
        if sigterm_handler is None:
            self.sigterm_handler = lambda: False
        else:
            self.sigterm_handler = sigterm_handler

        # train loop
        with ExitStack() as stack:
            if self.profile:
                # Add NVTX context if in profile mode
                self.log.warning("Running in profiling mode")
                stack.enter_context(torch.autograd.profiler.emit_nvtx())

            for step in range(self.initial_step, self.max_steps + 1):

                if self.sigterm_handler():
                    if self.manager.rank == 0:
                        self.log.info(
                            f"Training terminated by the user at iteration {step}"
                        )
                    break

                if self.profile and step == self.profiler_start_step:
                    # Start profiling
                    self.log.info("Starting profiler at step {}".format(step))
                    profiler.start()

                if self.profile and step == self.profiler_end_step:
                    # Stop profiling
                    self.log.info("Stopping profiler at step {}".format(step))
                    profiler.stop()

                torch.cuda.nvtx.range_push("Training iteration")

                self.step_str = f"[step: {step:10d}]"

                if self.cfg.cuda_graphs:
                    # If cuda graphs statically load it into defined allocations
                    self.load_data(static=True)

                    loss, losses = self._cuda_graph_training_step(step)
                else:
                    # Load all data for constraints
                    self.load_data()

                    self.global_optimizer_model.zero_grad(set_to_none=True)

                    # compute gradients
                    loss, losses = self.compute_gradients(
                        self.aggregator, self.global_optimizer_model, step
                    )

                    # take optimizer step
                    self.apply_gradients()

                    # take scheduler step
                    self.scheduler.step()

                # check for infs/NaNs in loss
                if not torch.isfinite(loss):
                    if self.amp_manager.enabled:
                        self.log.warn(f"{self.step_str} loss went to INFs/NaNs")
                    else:
                        self.log.error(f"{self.step_str} loss went to INFs/NaNs")
                        break

                # write train loss / learning rate tensorboard summaries
                if step % self.summary_freq == 0:
                    if self.manager.rank == 0:

                        # add train loss scalars
                        for key, value in losses.items():
                            if TF_SUMMARY:
                                self.writer.add_scalar(
                                    "Train_/loss_L2" + str(key),
                                    value,
                                    step,
                                    new_style=True,
                                )
                            else:
                                self.writer.add_scalar(
                                    "Train/loss_" + str(key),
                                    value,
                                    step,
                                    new_style=True,
                                )
                        if TF_SUMMARY:
                            self.writer.add_scalar(
                                "Optimzer/loss", loss, step, new_style=True
                            )
                            self.writer.add_scalar(
                                "learning_rate/lr",
                                self.scheduler.get_last_lr()[0],  # TODO: handle list
                                step,
                                new_style=True,
                            )
                        else:
                            self.writer.add_scalar(
                                "Train/loss_aggregated", loss, step, new_style=True
                            )
                            self.writer.add_scalar(
                                "Train/learning_rate",
                                self.scheduler.get_last_lr()[0],  # TODO: handle list
                                step,
                                new_style=True,
                            )

                        # track scaler and deriv_scalers states
                        if self.scaler.is_enabled():
                            self.log.info(
                                self.step_str
                                + " scaler {scale: %s, growth_tracker: %s} | "
                                "deriv_scalers {scale: %s, growth_tracker: %s, max_scale: %s}",
                                self.scaler.get_scale(),
                                self.scaler._get_growth_tracker(),
                                self.deriv_scalers.get_scale(),
                                self.deriv_scalers._get_growth_tracker(),
                                self.deriv_scalers.get_max_scale(),
                            )
                            self.writer.add_scalar(
                                "AMP/grad_scaler_log2",
                                np.log2(self.scaler.get_scale()),
                                step,
                                new_style=True,
                            )
                            for key, scale in self.deriv_scalers.get_scale().items():
                                self.writer.add_scalar(
                                    f"AMP/deriv_scaler_{key}_log2",
                                    np.log2(scale),
                                    step,
                                    new_style=True,
                                )

                    if self.manager.distributed:
                        barrier_flag = True

                # write train / inference / validation datasets to tensorboard and file
                if step % self.cfg.training.rec_constraint_freq == 0:
                    barrier_flag = True
                    self._record_constraints()

                if (step % self.cfg.training.rec_validation_freq == 0) and (
                    self.has_validators
                ):
                    barrier_flag = True
                    self._record_validators(step)

                if (step % self.cfg.training.rec_inference_freq == 0) and (
                    self.has_inferencers
                ):
                    barrier_flag = True
                    self._record_inferencers(step)

                if step % self.cfg.training.rec_monitor_freq == 0:
                    barrier_flag = True
                    self._record_monitors(step)

                # save checkpoint
                if step % self.save_network_freq == 0:
                    # Get data parallel rank so all processes in the first model parallel group
                    # can save their checkpoint. In the case without model parallelism, data_parallel_rank
                    # should be the same as the process rank itself
                    data_parallel_rank = (
                        self.manager.group_rank("data_parallel")
                        if self.manager.distributed
                        else 0
                    )
                    if data_parallel_rank == 0:
                        self.save_checkpoint(step)
                        self.log.info(
                            f"{self.step_str} saved checkpoint to {add_hydra_run_path(self.network_dir)}"
                        )
                    if self.manager.distributed:
                        barrier_flag = True

                if self.manager.distributed and barrier_flag:
                    dist.barrier(device_ids=[self.manager.local_rank])
                    barrier_flag = False

                # print loss stats
                if step % self.print_stats_freq == 0:
                    # synchronize and get end time
                    if self.manager.cuda:
                        end_event.record()
                        end_event.synchronize()
                        elapsed_time = start_event.elapsed_time(
                            end_event
                        )  # in milliseconds
                    else:
                        t_end = time.time()
                        elapsed_time = (t_end - t) * 1.0e3  # in milliseconds

                    # Reduce loss across all GPUs
                    if self.manager.distributed:
                        dist.reduce(loss, 0, op=dist.ReduceOp.AVG)
                        elapsed_time = torch.tensor(elapsed_time).to(self.device)
                        dist.reduce(elapsed_time, 0, op=dist.ReduceOp.AVG)
                        elapsed_time = elapsed_time.cpu().numpy()[()]

                    # print statement
                    print_statement = (
                        f"{self.step_str} loss: {loss.cpu().detach().numpy():10.3e}"
                    )
                    if step >= self.initial_step + self.print_stats_freq:
                        print_statement += f", time/iteration: {elapsed_time/self.print_stats_freq:10.3e} ms"
                    if self.manager.rank == 0:
                        self.log.info(print_statement)

                    if self.manager.cuda:
                        start_event.record()
                    else:
                        t = time.time()

                # check stopping criterion
                stop_training = self._check_stopping_criterion(loss, losses, step)
                if stop_training:
                    if self.manager.rank == 0:
                        self.log.info(
                            f"{self.step_str} stopping criterion is met, finished training!"
                        )
                    break

                # check max steps
                if step >= self.max_steps:
                    if self.manager.rank == 0:
                        self.log.info(
                            f"{self.step_str} reached maximum training steps, finished training!"
                        )
                    break

                torch.cuda.nvtx.range_pop()

    def _cuda_graph_training_step(self, step: int):
        # Training step method for using cuda graphs
        # Warm up
        if (step - self.initial_step) < self.cfg.cuda_graph_warmup:
            if (step - self.initial_step) == 0:
                # Default stream for warm up
                self.warmup_stream = torch.cuda.Stream()

            self.warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.warmup_stream):
                # zero optimizer gradients
                self.global_optimizer_model.zero_grad(set_to_none=True)

                # # compute gradients
                self.loss_static, self.losses_static = self.compute_gradients(
                    self.aggregator, self.global_optimizer_model, step
                )
            torch.cuda.current_stream().wait_stream(self.warmup_stream)

            # take optimizer step
            self.apply_gradients()

            # take scheduler step
            self.scheduler.step()
        # Record graph
        elif (step - self.initial_step) == self.cfg.cuda_graph_warmup:
            torch.cuda.synchronize()
            if self.manager.distributed:
                dist.barrier(device_ids=[self.manager.local_rank])

            if self.cfg.cuda_graph_warmup < 11:
                self.log.warn(
                    f"Graph warm up length ({self.cfg.cuda_graph_warmup}) should be more than 11 steps, higher suggested"
                )
            self.log.info("Attempting cuda graph building, this may take a bit...")

            self.g = torch.cuda.CUDAGraph()
            self.global_optimizer_model.zero_grad(set_to_none=True)
            # TODO: temporary workaround till this issue is fixed:
            # https://github.com/pytorch/pytorch/pull/104487#issuecomment-1638665876
            delay = os.environ.get("MODULUS_CUDA_GRAPH_CAPTURE_DELAY", "10")
            time.sleep(int(delay))
            with torch.cuda.graph(self.g):
                # compute gradients
                self.loss_static, self.losses_static = self.compute_gradients(
                    self.aggregator, self.global_optimizer_model, step
                )

            # take optimizer step
            # left out of graph for AMP compat, No perf difference
            self.apply_gradients()

            # take scheduler step
            self.scheduler.step()
        # Replay
        else:
            # Graph replay
            self.g.replay()
            # take optimizer step
            self.apply_gradients()

            self.scheduler.step()

        return self.loss_static, self.losses_static

    def _eval(
        self,
    ):

        # check the directory exists
        if not os.path.exists(self.network_dir):
            raise RuntimeError("Network checkpoint is required for eval mode.")

        # create global model for restoring and saving
        self.saveable_models = self.get_saveable_models()

        # set device
        if self.device is None:
            self.device = self.manager.device

        # load model
        self.step = self.load_step()
        self.step = self.load_model()
        self.step_str = f"[step: {self.step:10d}]"

        # make summary writer
        self.writer = SummaryWriter(
            log_dir=self.network_dir, purge_step=self.summary_freq + 1
        )
        self.summary_histograms = self.cfg["summary_histograms"]

        if self.manager.cuda:
            torch.cuda.synchronize(self.device)

        # write inference / validation datasets to tensorboard and file
        if self.has_validators:
            self._record_validators(self.step)
        if self.has_inferencers:
            self._record_inferencers(self.step)
        if self.has_monitors:
            self._record_monitors(self.step)

    def _stream(
        self,
    ):

        # check the directory exists
        if not os.path.exists(self.network_dir):
            raise RuntimeError("Network checkpoint is required for stream mode.")

        # create global model for restoring and saving
        self.saveable_models = self.get_saveable_models()

        # set device
        if self.device is None:
            self.device = self.manager.device

        # load model
        self.step = self.load_step()
        self.step = self.load_model()
        self.step_str = f"[step: {self.step:10d}]"

        if self.manager.cuda:
            torch.cuda.synchronize(self.device)

        # write streamed results to file
        return self.record_stream

    @staticmethod
    def _load_network(
        initialization_network_dir: str,
        network_dir: str,
        models: List[nn.Module],
        optimizer: Optimizer,
        aggregator: nn.Module,
        scheduler: _LRScheduler,
        scaler: GradScaler,
        deriv_scalers: DerivScalers,
        log: logging.Logger,
        manager: DistributedManager,
        device: Optional[torch.device] = None,
    ):
        # set device
        if device is None:
            device = manager.device

        # load optimizer
        step = Trainer._load_optimizer(
            network_dir,
            optimizer,
            aggregator,
            scheduler,
            scaler,
            deriv_scalers,
            log,
            device,
        )

        # load model
        step = Trainer._load_model(
            initialization_network_dir,
            network_dir,
            models,
            step,
            log,
            device,
        )
        return step

    @staticmethod
    def _load_optimizer(
        network_dir: str,
        optimizer: Optimizer,
        aggregator: nn.Module,
        scheduler: _LRScheduler,
        scaler: GradScaler,
        deriv_scalers: DerivScalers,
        log: logging.Logger,
        device: torch.device,
    ):
        manager = DistributedManager()
        model_parallel_rank = (
            manager.group_rank("model_parallel") if manager.distributed else 0
        )

        # attempt to restore optimizer
        optimizer_checkpoint_file = (
            network_dir + f"/optim_checkpoint.{model_parallel_rank}.pth"
        )
        log.info("attempting to restore from: " + add_hydra_run_path(network_dir))
        if os.path.exists(optimizer_checkpoint_file):
            try:
                checkpoint = torch.load(optimizer_checkpoint_file, map_location=device)
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                aggregator.load_state_dict(checkpoint["aggregator_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                deriv_scalers.load_state_dict(checkpoint["deriv_scalers_state_dict"])
                step = checkpoint["step"]
                success = colored("Success loading optimizer: ", "green")
                log.info(success + add_hydra_run_path(optimizer_checkpoint_file))
            except:
                fail = colored("Fail loading optimizer: ", "red")
                step = 0
                log.info(
                    fail + add_hydra_run_path(network_dir + "/optim_checkpoint.pth")
                )
        else:
            log.warning("optimizer checkpoint not found")
            step = 0
        return step

    @staticmethod
    def _load_model(
        initialization_network_dir: str,
        network_dir: str,
        models: List[nn.Module],
        step: int,
        log: logging.Logger,
        device: torch.device,
    ):
        manager = DistributedManager()
        model_parallel_rank = (
            manager.group_rank("model_parallel") if manager.distributed else 0
        )

        # attempt to restrore from initialization network dir
        if initialization_network_dir != "":
            for i_dir in initialization_network_dir.split(","):
                if os.path.exists(i_dir):
                    log.info("attempting to initialize network from " + i_dir)
                    for model in models:
                        if os.path.exists(i_dir + "/" + model.checkpoint_filename):
                            try:
                                model.load(i_dir, map_location=device)
                                success = colored("Success loading model: ", "green")
                                log.info(
                                    success + i_dir + "/" + model.checkpoint_filename
                                )
                            except:
                                fail = colored("Fail loading model: ", "red")
                                step = 0
                                log.error(
                                    fail + i_dir + "/" + model.checkpoint_filename
                                )
                        else:
                            log.warning(
                                "model "
                                + model.checkpoint_filename
                                + " not found for initialization"
                            )

        # attempt to restore models
        for model in models:
            if os.path.exists(network_dir + "/" + model.checkpoint_filename):
                try:
                    model.load(network_dir, map_location=device)
                    success = colored("Success loading model: ", "green")
                    log.info(
                        success
                        + add_hydra_run_path(
                            network_dir + "/" + model.checkpoint_filename
                        )
                    )
                except:
                    fail = colored("Fail loading model: ", "red")
                    log.info(
                        fail
                        + add_hydra_run_path(
                            network_dir + "/" + model.checkpoint_filename
                        )
                    )
            else:
                log.warning("model " + model.checkpoint_filename + " not found")
                step = 0
        return step

    @staticmethod
    def _load_step(
        network_dir: str,
        device: Optional[torch.device] = None,
    ):
        manager = DistributedManager()
        model_parallel_rank = (
            manager.group_rank("model_parallel") if manager.distributed else 0
        )

        if os.path.exists(network_dir + f"/optim_checkpoint.{model_parallel_rank}.pth"):
            try:
                checkpoint = torch.load(
                    network_dir + f"/optim_checkpoint.{model_parallel_rank}.pth",
                    map_location=device,
                )
                step = checkpoint["step"]
            except:
                step = 0
        else:
            step = 0
        return step

    @staticmethod
    def _save_checkpoint(
        network_dir: str,
        models: List[nn.Module],
        optimizer: Optimizer,
        aggregator: nn.Module,
        scheduler: _LRScheduler,
        scaler: GradScaler,
        deriv_scalers: DerivScalers,
        step: int,
    ):
        # Get model parallel rank so all processes in the first model parallel group
        # can save their checkpoint. In the case without model parallelism, model_parallel_rank
        # should be the same as the process rank itself and only rank 0 saves
        manager = DistributedManager()
        model_parallel_rank = (
            manager.group_rank("model_parallel") if manager.distributed else 0
        )

        # save models
        for model in models:
            model.save(network_dir)

        # save step, optimizer, aggregator, and scaler
        torch.save(
            {
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "aggregator_state_dict": aggregator.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "deriv_scalers_state_dict": deriv_scalers.state_dict(),
            },
            network_dir + f"/optim_checkpoint.{model_parallel_rank}.pth",
        )
