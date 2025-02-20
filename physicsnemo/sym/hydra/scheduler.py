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

"""
Supported PyTorch scheduler configs
"""

import torch

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class SchedulerConf:
    _target_ = MISSING


@dataclass
class ExponentialLRConf(SchedulerConf):
    _target_: str = "torch.optim.lr_scheduler.ExponentialLR"
    gamma: float = 0.99998718


@dataclass
class TFExponentialLRConf(SchedulerConf):
    _target_: str = "custom"
    _name_: str = "tf.ExponentialLR"
    decay_rate: float = 0.95
    decay_steps: int = 1000


@dataclass
class CosineAnnealingLRConf(SchedulerConf):
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: int = 1000
    eta_min: float = 0
    last_epoch: int = -1


@dataclass
class CosineAnnealingWarmRestartsConf(SchedulerConf):
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    T_0: int = 1000
    T_mult: int = 1
    eta_min: float = 0
    last_epoch: int = -1


def register_scheduler_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="scheduler",
        name="exponential_lr",
        node=ExponentialLRConf,
    )

    cs.store(
        group="scheduler",
        name="tf_exponential_lr",
        node=TFExponentialLRConf,
    )

    cs.store(
        group="scheduler",
        name="cosine_annealing",
        node=CosineAnnealingLRConf,
    )

    cs.store(
        group="scheduler",
        name="cosine_annealing_warm_restarts",
        node=CosineAnnealingWarmRestartsConf,
    )
