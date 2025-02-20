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

""" PhysicsNeMo Neural Differential Equation Solver
"""

import os
import numpy as np
from typing import List, Union, Tuple, Callable
from omegaconf import DictConfig
import warnings

from physicsnemo.sym.amp import DerivScalers
from physicsnemo.sym.trainer import Trainer
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.loss.aggregator import NTK


# base class for solver
class Solver(Trainer):
    """
    Base solver class for solving single domain.

    Parameters
    ----------
    cfg : DictConfig
        Hydra dictionary of configs.
    domain : Domain
        Domain to solve for.
    """

    def __init__(self, cfg: DictConfig, domain: Domain):

        # set domain
        self.domain = domain

        super(Solver, self).__init__(cfg)

        # NTK setup:
        if cfg.training.ntk.use_ntk:
            ntk = NTK(
                run_per_step=cfg.training.ntk.run_freq,
                save_name=cfg.training.ntk.save_name,
            )
            self.domain.add_ntk(ntk)

    @property
    def network_dir(self):
        return self._network_dir

    @property
    def initialization_network_dir(self):
        return self._initialization_network_dir

    def compute_losses(self, step: int):
        return self.domain.compute_losses(step)

    def get_saveable_models(self):
        return self.domain.get_saveable_models()

    def create_global_optimizer_model(self):
        return self.domain.create_global_optimizer_model()

    def load_data(self, static: bool = False):
        self.domain.load_data(static)

    def load_network(self):
        return Trainer._load_network(
            self.initialization_network_dir,
            self.network_dir,
            self.saveable_models,
            self.optimizer,
            self.aggregator,
            self.scheduler,
            self.scaler,
            self.deriv_scalers,
            self.log,
            self.manager,
            self.device,
        )

    def load_optimizer(self):
        return Trainer._load_optimizer(
            self.network_dir,
            self.optimizer,
            self.aggregator,
            self.scheduler,
            self.scaler,
            self.deriv_scalers,
            self.log,
            self.device,
        )

    def load_model(self):
        return Trainer._load_model(
            self.initialization_network_dir,
            self.network_dir,
            self.saveable_models,
            self.step,
            self.log,
            self.device,
        )

    def load_step(self):
        return Trainer._load_step(
            self.network_dir,
            self.device,
        )

    def save_checkpoint(self, step: int):
        Trainer._save_checkpoint(
            self.network_dir,
            self.saveable_models,
            self.optimizer,
            self.aggregator,
            self.scheduler,
            self.scaler,
            self.deriv_scalers,
            step,
        )

    def record_constraints(self):
        self.domain.rec_constraints(self.network_dir)

    def record_validators(self, step: int):
        return self.domain.rec_validators(
            self.network_dir, self.writer, self.save_filetypes, step
        )

    @property
    def has_validators(self):
        return bool(self.domain.validators)

    def record_inferencers(self, step: int):
        self.domain.rec_inferencers(
            self.network_dir, self.writer, self.save_filetypes, step
        )

    def record_stream(self, inferencer, name):
        return self.domain.rec_stream(
            inferencer,
            name,
            self.network_dir,
            self.step,
            self.save_results,
            self.save_filetypes,
            self.to_cpu,
        )

    @property
    def has_inferencers(self):
        return bool(self.domain.inferencers)

    def record_monitors(self, step: int):
        return self.domain.rec_monitors(self.network_dir, self.writer, step)

    @property
    def has_monitors(self):
        return bool(self.domain.monitors)

    def get_num_losses(self):
        return self.domain.get_num_losses()

    def setup_deriv_scaler(self, deriv_scalers: DerivScalers):
        self.domain.setup_deriv_scaler(deriv_scalers)

    def solve(self, sigterm_handler=None):
        if self.cfg.run_mode == "train":
            self._train_loop(sigterm_handler)
        elif self.cfg.run_mode == "eval":
            self._eval()
        else:
            raise RuntimeError("Invalid run mode")

    def train(self, sigterm_handler=None):
        self._train_loop(sigterm_handler)

    def eval(self):
        self._eval()

    def stream(self, save_results=False, to_cpu=True):
        self.save_results = save_results
        self.to_cpu = to_cpu
        return self._stream()
