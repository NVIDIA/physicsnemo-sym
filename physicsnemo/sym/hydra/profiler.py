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
Profiler config
"""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI, II
from typing import Any, Union, List, Dict


@dataclass
class ProfilerConf:
    profile: bool = MISSING
    start_step: int = MISSING
    end_step: int = MISSING


@dataclass
class NvtxProfiler(ProfilerConf):
    name: str = "nvtx"
    profile: bool = False
    start_step: int = 0
    end_step: int = 100


@dataclass
class TensorBoardProfiler(ProfilerConf):
    name: str = "tensorboard"
    profile: bool = False
    start_step: int = 0
    end_step: int = 100
    warmup: int = 5
    repeat: int = 1
    filename: str = "${hydra.job.override_dirname}-${hydra.job.name}.profile"


def register_profiler_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="profiler",
        name="nvtx",
        node=NvtxProfiler,
    )
    cs.store(
        group="profiler",
        name="tensorboard",
        node=TensorBoardProfiler,
    )
