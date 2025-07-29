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
from typing import Dict

# Import from PhysicsNeMo
from physicsnemo.sym.loss.aggregator import Aggregator


class CustomSum(Aggregator):
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
            A dictionary of losses
        step : int
            Optimizer step

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        smoothness = 0.0005  # use 0.0005 to smoothen the transition over ~10k steps
        step_tensor = torch.tensor(step, dtype=torch.float32)
        decay_weight = (torch.tanh((20000 - step_tensor) * smoothness) + 1.0) * 0.5

        # Add losses
        for key in losses.keys():
            if "init" not in key:
                loss += (1 - decay_weight) * (losses[key])
            else:
                loss += decay_weight * (losses[key])
        return loss
