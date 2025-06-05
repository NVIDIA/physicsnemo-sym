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
PhysicsNeMo main config
"""

from platform import architecture
import torch
import logging
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir, HydraConf
from omegaconf import MISSING, SI
from typing import List, Any
from physicsnemo.sym.constants import JIT_PYTORCH_VERSION
from packaging import version

from .loss import LossConf
from .optimizer import OptimizerConf
from .pde import PDEConf
from .scheduler import SchedulerConf
from .training import TrainingConf, StopCriterionConf
from .profiler import ProfilerConf
from .hydra import default_hydra

logger = logging.getLogger(__name__)


@dataclass
class PhysicsNeMoConfig:

    # General parameters
    network_dir: str = "."
    initialization_network_dir: str = ""
    save_filetypes: str = "vtk"
    summary_histograms: str = "off"
    jit: bool = version.parse(torch.__version__) >= version.parse(JIT_PYTORCH_VERSION)
    jit_use_nvfuser: bool = True
    jit_arch_mode: str = "only_activation"
    jit_autograd_nodes: bool = False

    cuda_graphs: bool = True
    cuda_graph_warmup: int = 20
    find_unused_parameters: bool = False
    broadcast_buffers: bool = False

    device: str = ""
    debug: bool = False
    run_mode: str = "train"

    arch: Any = MISSING
    models: Any = MISSING  # List of models

    training: TrainingConf = MISSING
    stop_criterion: StopCriterionConf = MISSING
    loss: LossConf = MISSING

    optimizer: OptimizerConf = MISSING
    scheduler: SchedulerConf = MISSING

    batch_size: Any = MISSING
    profiler: ProfilerConf = MISSING

    hydra: Any = field(default_factory=lambda: default_hydra)

    # User custom parameters that are not used internally in physicsnemo
    custom: Any = MISSING


default_defaults = [
    {"training": "default_training"},
    {"amp": "default"},
    {"graph": "default"},
    {"stop_criterion": "default_stop_criterion"},
    {"profiler": "nvtx"},
    {"override hydra/job_logging": "info_logging"},
    {"override hydra/launcher": "basic"},
    {"override hydra/help": "physicsnemo_help"},
    {"override hydra/callbacks": "default_callback"},
]


@dataclass
class DefaultPhysicsNeMoConfig(PhysicsNeMoConfig):
    # Core defaults
    # Can over-ride default with "override" hydra command
    defaults: List[Any] = field(default_factory=lambda: default_defaults)


# PhysicsNeMo config for debugging
debug_defaults = [
    {"training": "default_training"},
    {"amp": "default"},
    {"graph": "default"},
    {"stop_criterion": "default_stop_criterion"},
    {"profiler": "nvtx"},
    {"override hydra/job_logging": "debug_logging"},
    {"override hydra/help": "physicsnemo_help"},
    {"override hydra/callbacks": "default_callback"},
]


@dataclass
class DebugPhysicsNeMoConfig(PhysicsNeMoConfig):
    # Core defaults
    # Can over-ride default with "override" hydra command
    defaults: List[Any] = field(default_factory=lambda: debug_defaults)

    debug: bool = True


# PhysicsNeMo config with experimental features (use caution)
experimental_defaults = [
    {"training": "default_training"},
    {"amp": "default"},
    {"graph": "default"},
    {"stop_criterion": "default_stop_criterion"},
    {"profiler": "nvtx"},
    {"override hydra/job_logging": "info_logging"},
    {"override hydra/launcher": "basic"},
    {"override hydra/help": "physicsnemo_help"},
    {"override hydra/callbacks": "default_callback"},
]


@dataclass
class ExperimentalPhysicsNeMoConfig(PhysicsNeMoConfig):
    # Core defaults
    # Can over-ride default with "override" hydra command
    defaults: List[Any] = field(default_factory=lambda: experimental_defaults)

    pde: PDEConf = MISSING


def register_physicsnemo_configs() -> None:

    if not torch.__version__ == JIT_PYTORCH_VERSION:
        logger.warning(
            f"TorchScript default is being turned off due to PyTorch version mismatch."
        )

    cs = ConfigStore.instance()
    cs.store(
        name="physicsnemo_default",
        node=DefaultPhysicsNeMoConfig,
    )
    cs.store(
        name="physicsnemo_debug",
        node=DebugPhysicsNeMoConfig,
    )
    cs.store(
        name="physicsnemo_experimental",
        node=ExperimentalPhysicsNeMoConfig,
    )
