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
Supported PhysicsNeMo graph configs
"""

import torch

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class GraphConf:
    func_arch: bool = MISSING
    func_arch_allow_partial_hessian: bool = MISSING


@dataclass
class DefaultGraphConf(GraphConf):
    func_arch: bool = False
    func_arch_allow_partial_hessian: bool = True


def register_graph_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="graph",
        name="default",
        node=DefaultGraphConf,
    )
