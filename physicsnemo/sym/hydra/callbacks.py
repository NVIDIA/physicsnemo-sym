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
Supported PhysicsNeMo callback configs
"""

import torch
import logging

from dataclasses import dataclass
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from hydra.experimental.callback import Callback
from typing import Any
from omegaconf import DictConfig, OmegaConf

from physicsnemo.sym.amp import AmpManager
from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.manager import JitManager, JitArchMode, GraphManager

logger = logging.getLogger(__name__)


class PhysicsNeMoCallback(Callback):
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        # Update dist manager singleton with config parameters
        manager = DistributedManager()
        manager.broadcast_buffers = config.broadcast_buffers
        manager.find_unused_parameters = config.find_unused_parameters
        manager.cuda_graphs = config.cuda_graphs

        # jit manager
        jit_manager = JitManager()
        jit_manager.init(
            config.jit,
            config.jit_arch_mode,
            config.jit_use_nvfuser,
            config.jit_autograd_nodes,
        )

        # graph manager
        graph_manager = GraphManager()
        graph_manager.init(
            config.graph.func_arch,
            config.graph.func_arch_allow_partial_hessian,
            config.debug,
        )
        # The FuncArch does not work with TorchScript at all, so we raise
        # a warning and disabled it.
        if config.graph.func_arch and jit_manager.enabled:
            jit_manager.enabled = False
            logger.warning("Disabling JIT because functorch does not work with it.")

        # amp manager
        amp_manager = AmpManager()
        amp_manager.init(
            config.amp.enabled,
            config.amp.mode,
            config.amp.dtype,
            config.amp.autocast_activation,
            config.amp.autocast_firstlayer,
            config.amp.default_max_scale_log2,
            config.amp.custom_max_scales_log2,
        )

        logger.info(jit_manager)
        logger.info(graph_manager)
        logger.info(amp_manager)


DefaultCallbackConfigs = DictConfig(
    {
        "physicsnemo_callback": OmegaConf.create(
            {
                "_target_": "physicsnemo.sym.hydra.callbacks.PhysicsNeMoCallback",
            }
        )
    }
)


def register_callbacks_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="hydra/callbacks",
        name="default_callback",
        node=DefaultCallbackConfigs,
    )
