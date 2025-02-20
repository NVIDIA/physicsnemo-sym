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
Supported PhysicsNeMo AMP configs
"""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Dict, Any
from omegaconf import MISSING


@dataclass
class AmpConf:
    enabled: bool = MISSING
    mode: str = MISSING
    dtype: str = MISSING
    autocast_activation: bool = MISSING
    autocast_firstlayer: bool = MISSING
    default_max_scale_log2: int = MISSING
    # Comment out the following line solves an error for hydra
    # The Error message: Missing mandatory value: amp.custom_max_scales_log2
    # custom_max_scales_log2: Dict[Any, int] = MISSING


@dataclass
class DefaultAmpConf(AmpConf):
    enabled: bool = False
    mode: str = "per_order_scaler"  # another option is "per_term_scaler"
    dtype: str = "float16"
    # TODO set default as True once we have fused SILU support
    autocast_activation: bool = False
    autocast_firstlayer: bool = False
    default_max_scale_log2: int = 0
    custom_max_scales_log2: Dict[Any, int] = field(default_factory=lambda: {})


def register_amp_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="amp",
        name="default",
        node=DefaultAmpConf,
    )
