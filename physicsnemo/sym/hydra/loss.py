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
Supported PhysicsNeMo loss aggregator configs
"""

import torch

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from typing import Any


@dataclass
class LossConf:
    _target_: str = MISSING
    weights: Any = None


@dataclass
class AggregatorSumConf(LossConf):
    _target_: str = "physicsnemo.sym.loss.aggregator.Sum"


@dataclass
class AggregatorGradNormConf(LossConf):
    _target_: str = "physicsnemo.sym.loss.aggregator.GradNorm"
    alpha: float = 1.0


@dataclass
class AggregatorResNormConf(LossConf):
    _target_: str = "physicsnemo.sym.loss.aggregator.ResNorm"
    alpha: float = 1.0


@dataclass
class AggregatorHomoscedasticConf(LossConf):
    _target_: str = "physicsnemo.sym.loss.aggregator.HomoscedasticUncertainty"


@dataclass
class AggregatorLRAnnealingConf(LossConf):
    _target_: str = "physicsnemo.sym.loss.aggregator.LRAnnealing"
    update_freq: int = 1
    alpha: float = 0.01
    ref_key: Any = None  # Change to Union[None, str] when supported by hydra
    eps: float = 1e-8


@dataclass
class AggregatorSoftAdaptConf(LossConf):
    _target_: str = "physicsnemo.sym.loss.aggregator.SoftAdapt"
    eps: float = 1e-8


@dataclass
class AggregatorRelobraloConf(LossConf):
    _target_: str = "physicsnemo.sym.loss.aggregator.Relobralo"
    alpha: float = 0.95
    beta: float = 0.99
    tau: float = 1.0
    eps: float = 1e-8


@dataclass
class NTKConf:
    use_ntk: bool = False
    save_name: Any = None  # Union[str, None]
    run_freq: int = 1000


def register_loss_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="loss",
        name="sum",
        node=AggregatorSumConf,
    )
    cs.store(
        group="loss",
        name="grad_norm",
        node=AggregatorGradNormConf,
    )
    cs.store(
        group="loss",
        name="res_norm",
        node=AggregatorResNormConf,
    )
    cs.store(
        group="loss",
        name="homoscedastic",
        node=AggregatorHomoscedasticConf,
    )
    cs.store(
        group="loss",
        name="lr_annealing",
        node=AggregatorLRAnnealingConf,
    )
    cs.store(
        group="loss",
        name="soft_adapt",
        node=AggregatorSoftAdaptConf,
    )
    cs.store(
        group="loss",
        name="relobralo",
        node=AggregatorRelobraloConf,
    )
