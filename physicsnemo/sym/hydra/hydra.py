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
Hydra related configs
"""

import pathlib
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, SI
from typing import Any


@dataclass
class SimpleFormat:
    format: str = "[%(asctime)s] - %(message)s"
    datefmt: str = "%H:%M:%S"


@dataclass
class DebugFormat:
    format: str = "[%(levelname)s][%(asctime)s][%(module)s] - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


info_console_handler = {
    "class": "logging.StreamHandler",
    "formatter": "simple",
    "stream": "ext://sys.stdout",
}


@dataclass
class DefaultLogging:
    version: int = 1
    formatters: Any = field(default_factory=lambda: {"simple": SimpleFormat})
    handlers: Any = field(default_factory=lambda: {"console": info_console_handler})
    root: Any = field(default_factory=lambda: {"handlers": ["console"]})
    disable_existing_loggers: bool = False
    level: int = (
        20  # CRITICAL: 50, ERROR: 40, WARNING: 30, INFO: 20, DEBUG: 10, NOTSET: 0
    )


debug_console_handler = {
    "class": "logging.StreamHandler",
    "formatter": "debug",
    "stream": "ext://sys.stdout",
}


@dataclass
class DebugLogging:
    version: int = 1
    formatters: Any = field(default_factory=lambda: {"debug": DebugFormat})
    handlers: Any = field(default_factory=lambda: {"console": debug_console_handler})
    root: Any = field(default_factory=lambda: {"handlers": ["console"]})
    disable_existing_loggers: bool = False
    level: int = (
        0  # CRITICAL: 50, ERROR: 40, WARNING: 30, INFO: 20, DEBUG: 10, NOTSET: 0
    )


# Hydra defaults group parameters for physicsnemo
file_path = pathlib.Path(__file__).parent.resolve()
physicsnemo_help = OmegaConf.load(file_path / "help.yaml")

# Standard Hydra parameters
default_hydra = {
    "run": {"dir": SI("outputs/${hydra:job.override_dirname}/${hydra:job.name}")},
    "sweep": {"dir": "multirun", "subdir": SI("${hydra.job.override_dirname}")},
    "verbose": SI("${debug}"),
}


def register_hydra_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="hydra/job_logging",
        name="info_logging",
        node=DefaultLogging,
    )
    cs.store(
        group="hydra/job_logging",
        name="debug_logging",
        node=DebugLogging,
    )
    cs.store(
        group="hydra/help",
        name="physicsnemo_help",
        node=physicsnemo_help,
    )
