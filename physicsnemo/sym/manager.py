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

""" PhysicsNeMo Managers
"""

import logging
from typing import Dict, List, Union
from enum import Enum
import torch
from packaging import version
from physicsnemo.sym.constants import JIT_PYTORCH_VERSION

logger = logging.getLogger(__name__)


class JitArchMode(Enum):
    ALL = 0
    ONLY_ACTIVATION = 1


class JitManager(object):
    _shared_state = {}

    def __new__(cls):
        obj = super(JitManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_enabled"):
            obj._enabled = version.parse(torch.__version__) >= version.parse(
                JIT_PYTORCH_VERSION
            )
        if not hasattr(obj, "_arch_mode"):
            obj._arch_mode = JitArchMode.ONLY_ACTIVATION
        if not hasattr(obj, "_use_nvfuser"):
            obj._use_nvfuser = True
        if not hasattr(obj, "_autograd_nodes"):
            obj._autograd_nodes = False

        return obj

    @property
    def arch_mode(self):
        return self._arch_mode

    @arch_mode.setter
    def arch_mode(self, mode: str):
        if mode == "all":
            self._arch_mode = JitArchMode.ALL
        elif mode == "only_activation":
            self._arch_mode = JitArchMode.ONLY_ACTIVATION
        else:
            raise ValueError(
                f"jit arch mode should be all/only_activation, but found {mode}"
            )

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, flag):
        # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/codegen/cuda/README.md
        # enable fusing single node and prevent tiny autodiff graph are inlined/reverted
        if flag:
            torch._C._jit_set_nvfuser_single_node_mode(True)
            torch._C._debug_set_autodiff_subgraph_inlining(False)
        self._enabled = flag

    @property
    def use_nvfuser(self):
        return self._use_nvfuser

    @use_nvfuser.setter
    def use_nvfuser(self, flag):
        self._use_nvfuser = flag
        torch._C._jit_set_nvfuser_enabled(flag)
        backend = "NVFuser" if flag else "NNC"
        if self.enabled:
            logger.info(f"JIT using the {backend} TorchScript backend")

    @property
    def autograd_nodes(self):
        return self._autograd_nodes

    @autograd_nodes.setter
    def autograd_nodes(self, flag):
        self._autograd_nodes = flag

    def __repr__(self):
        return f"JitManager: {self._shared_state}"

    def init(self, enabled, arch_mode, use_nvfuser, autograd_nodes):
        self.enabled = enabled
        self.arch_mode = arch_mode
        self.use_nvfuser = use_nvfuser
        self.autograd_nodes = autograd_nodes


class GraphManager(object):
    _shared_state = {}

    def __new__(cls):
        obj = super(GraphManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_func_arch"):
            obj._func_arch = True
        # TODO we should have a debug flag in the global PhysicsNeMoManager
        # in the future
        if not hasattr(obj, "_debug"):
            obj._debug = False
        if not hasattr(obj, "_func_arch_allow_partial_hessian"):
            obj._func_arch_allow_partial_hessian = True

        return obj

    @property
    def func_arch(self):
        return self._func_arch

    @func_arch.setter
    def func_arch(self, flag):
        self._func_arch = flag

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, flag):
        self._debug = flag

    @property
    def func_arch_allow_partial_hessian(self):
        return self._func_arch_allow_partial_hessian

    @func_arch_allow_partial_hessian.setter
    def func_arch_allow_partial_hessian(self, flag):
        self._func_arch_allow_partial_hessian = flag

    def __repr__(self):
        return f"GraphManager: {self._shared_state}"

    def init(self, func_arch, func_arch_allow_partial_hessian, debug):
        self.func_arch = func_arch
        self.func_arch_allow_partial_hessian = func_arch_allow_partial_hessian
        self.debug = debug
