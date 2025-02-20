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

from physicsnemo.sym.distributed.manager import DistributedManager
from physicsnemo.sym.trainer import Trainer
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.loss.aggregator import NTK
from .solver import Solver


class SequentialSolver(Solver):
    """
    Solver class for solving a sequence of domains.
    This solver can be used to set up iterative methods
    like the hFTB conjugate heat transfer method or
    the moving time window method for transient problems.

    Parameters
    ----------
    cfg : DictConfig
        Hydra dictionary of configs.
    domains : List[Tuple[int, Domain]]
        List of Domains to sequentially solve.
        Each domain is given as a tuple where the first
        element is an int for how many times to solve
        the domain and the second element is the domain.
        For example, `domains=[(1, domain_a), (4, domain_b)]`
        would solve `domain_a` once and then solve `domain_b`
        4 times in a row.
    custom_update_operation : Union[Callable, None] = None
        A callable function to update any weights in models.
        This function will be called at the end of every
        iteration.
    """

    def __init__(
        self,
        cfg: DictConfig,
        domains: List[Tuple[int, Domain]],
        custom_update_operation: Union[Callable, None] = None,
    ):
        # check that domains have different names
        assert len(set([d.name for _, d in domains])) == len(
            domains
        ), "domains need to have unique names, " + str([d.name for _, d in domains])

        # check not using ntk with seq solver
        assert (
            not cfg.training.ntk.use_ntk
        ), "ntk is not supported with SequentialSolver"

        # set domains
        self.domains = domains

        # set update operation after solving each domain
        self.custom_update_operation = custom_update_operation

        # load rest of initializations
        Trainer.__init__(self, cfg)

        # load current index
        self.load_iteration_step()

    def load_iteration_step(self):
        try:
            iteration_step_file = open(self._network_dir + "/current_step.txt", "r")
            contents = iteration_step_file.readlines()[0]
            domain_index = int(contents.split(" ")[0])
            iteration_index = int(contents.split(" ")[1])
        except:
            domain_index = 0
            iteration_index = 0
        self.domain_index = domain_index
        self.iteration_index = iteration_index

    def save_iteration_step(self):
        iteration_step_file = open(self._network_dir + "/current_step.txt", "w")
        iteration_step_file.write(
            str(self.domain_index) + " " + str(self.iteration_index)
        )

    @property
    def domain(self):
        return self.domains[self.domain_index][1]

    @property
    def network_dir(self):
        dir_name = self._network_dir + "/" + self.domain.name
        if self.domains[self.domain_index][0] > 1:
            dir_name += "_" + str(self.iteration_index).zfill(4)
        return dir_name

    def solve(self, sigterm_handler=None):
        if self.cfg.run_mode == "train":
            # make directory if doesn't exist
            if DistributedManager().rank == 0:
                os.makedirs(self.network_dir, exist_ok=True)

            # run train loop for each domain and each index
            # solve for each domain in seq_train_domin
            for domain_index in range(self.domain_index, len(self.domains)):
                # solve for number of iterations in train_domain
                for iteration_index in range(
                    self.iteration_index, self.domains[domain_index][0]
                ):

                    # set internal domain index and iteration index
                    self.domain_index = domain_index
                    self.iteration_index = iteration_index

                    # save current iteration step
                    self.save_iteration_step()

                    # solve for domain
                    self.log.info(
                        "Solving for Domain "
                        + str(self.domain.name)
                        + ", iteration "
                        + str(self.iteration_index)
                    )
                    self._train_loop(sigterm_handler)

                    # run user defined custom update operation
                    if self.custom_update_operation is not None:
                        self.custom_update_operation()

        elif self.cfg.run_mode == "eval":
            raise NotImplementedError(
                "eval mode not implemented for sequential training"
            )
        else:
            raise RuntimeError("Invalid run mode")
