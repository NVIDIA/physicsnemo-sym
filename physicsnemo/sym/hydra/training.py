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
Supported physicsnemo training paradigms
"""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, II
from typing import Any

from .loss import NTKConf


@dataclass
class TrainingConf:
    max_steps: int = MISSING
    grad_agg_freq: int = MISSING
    rec_results_freq: int = MISSING
    rec_validation_freq: int = MISSING
    rec_inference_freq: int = MISSING
    rec_monitor_freq: int = MISSING
    rec_constraint_freq: int = MISSING
    save_network_freq: int = MISSING
    print_stats_freq: int = MISSING
    summary_freq: int = MISSING
    grad_clip_max_norm: float = MISSING
    monitor_grad_clip: bool = MISSING


@dataclass
class DefaultTraining(TrainingConf):
    max_steps: int = 10000
    grad_agg_freq: int = 1
    rec_results_freq: int = 1000
    rec_validation_freq: int = II("training.rec_results_freq")
    rec_inference_freq: int = II("training.rec_results_freq")
    rec_monitor_freq: int = II("training.rec_results_freq")
    rec_constraint_freq: int = II("training.rec_results_freq")
    save_network_freq: int = 1000
    print_stats_freq: int = 100
    summary_freq: int = 1000
    grad_clip_max_norm: float = 0.5
    monitor_grad_clip: bool = True

    ntk: NTKConf = field(default_factory=NTKConf)


@dataclass
class VariationalTraining(DefaultTraining):
    test_function: str = MISSING
    use_quadratures: bool = False


@dataclass
class StopCriterionConf:
    metric: Any = MISSING
    min_delta: Any = MISSING
    patience: int = MISSING
    mode: str = MISSING
    freq: int = MISSING
    strict: bool = MISSING


@dataclass
class DefaultStopCriterion(StopCriterionConf):
    metric: Any = None
    min_delta: Any = None
    patience: int = 50000
    mode: str = "min"
    freq: int = 1000
    strict: bool = False


def register_training_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="training",
        name="default_training",
        node=DefaultTraining,
    )

    cs.store(
        group="training",
        name="variational_training",
        node=VariationalTraining,
    )
    cs.store(
        group="stop_criterion",
        name="default_stop_criterion",
        node=DefaultStopCriterion,
    )
