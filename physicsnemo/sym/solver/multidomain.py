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

import os
import numpy as np
from typing import List, Union, Tuple, Callable
from omegaconf import DictConfig
import warnings

from physicsnemo.sym.trainer import Trainer
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.loss.aggregator import NTK
from .solver import Solver


class MultiDomainSolver(Solver):
    """
    Solver class for solving multiple domains.
    NOTE this Solver is currently experimental and not fully supported.
    """

    def __init__(self, cfg: DictConfig, domains: List[Domain]):
        # warning message for experimental
        warnings.warn(
            "This solver is currently experimental and unforeseen errors may occur."
        )

        # check that domains have different names
        assert len(set([d.name for d in domains])) == len(
            domains
        ), "domains need to have unique names, " + str([d.name for _, d in domains])

        # check not using ntk with seq solver
        assert (
            not cfg.training.ntk.use_ntk
        ), "ntk is not supported with MultiDomainSolver"

        # set number of domains per iteration
        self.domain_batch_size = cfg["domain_batch_size"]

        # set domains
        self.domains = domains

        # load rest of initializations
        Trainer.__init__(self, cfg)

    def compute_losses(self, step: int):
        batch_index = np.random.choice(len(self.domains), self.domain_batch_size)
        losses = {}
        for i in batch_index:
            # compute losses
            constraint_losses = self.domains[i].compute_losses(step)

            # add together losses of like kind
            for loss_key, value in constraint_losses.items():
                if loss_key not in list(losses.keys()):
                    losses[loss_key] = value
                else:
                    losses[loss_key] += value
        return losses

    def get_saveable_models(self):
        return self.domains[0].get_saveable_models()

    def create_global_optimizer_model(self):
        return self.domains[0].create_global_optimizer_model()

    def get_num_losses(self):
        return self.domains[0].get_num_losses()
