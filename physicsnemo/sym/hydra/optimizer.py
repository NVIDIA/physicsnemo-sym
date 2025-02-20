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
Supported optimizer configs
"""

import torch

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any
from omegaconf import MISSING


@dataclass
class OptimizerConf:
    _target_ = MISSING
    _params_: Any = field(
        default_factory=lambda: {
            "compute_gradients": "adam_compute_gradients",
            "apply_gradients": "adam_apply_gradients",
        }
    )


@dataclass
class AdamConf(OptimizerConf):
    _target_: str = "torch.optim.Adam"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class SGDConf(OptimizerConf):
    _target_: str = "torch.optim.SGD"
    lr: float = 1.0e-3
    momentum: float = 1.0e-2
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


@dataclass
class AdahessianConf(OptimizerConf):
    _target_: str = "torch_optimizer.Adahessian"
    lr: float = 1.0e-1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-4
    weight_decay: float = 0.0
    hessian_power: float = 1.0
    _params_: Any = field(
        default_factory=lambda: {
            "compute_gradients": "adahess_compute_gradients",
            "apply_gradients": "adahess_apply_gradients",
        }
    )


@dataclass
class BFGSConf(OptimizerConf):
    _target_: str = "torch.optim.LBFGS"
    lr: float = 1.0
    max_iter: int = 1000
    max_eval: Any = None
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: Any = None  # Union[None, str]
    _params_: Any = field(
        default_factory=lambda: {
            "compute_gradients": "bfgs_compute_gradients",
            "apply_gradients": "bfgs_apply_gradients",
        }
    )


@dataclass
class AdadeltaConf(OptimizerConf):
    _target_: str = "torch.optim.Adadelta"
    lr: float = 1.0
    rho: float = 0.9
    eps: float = 1e-6
    weight_decay: float = 0


@dataclass
class AdagradConf(OptimizerConf):
    _target_: str = "torch.optim.Adagrad"
    lr: float = 1.0e-2
    lr_decay: float = 0
    weight_decay: float = 0
    initial_accumulator_value: float = 0
    eps: float = 1e-10


@dataclass
class AdamWConf(OptimizerConf):
    _target_: str = "torch.optim.AdamW"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False


@dataclass
class SparseAdamConf(OptimizerConf):
    _target_: str = "torch.optim.SparseAdam"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class AdamaxConf(OptimizerConf):
    _target_: str = "torch.optim.Adamax"
    lr: float = 2.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class ASGDConf(OptimizerConf):
    _target_: str = "torch.optim.ASGD"
    lr: float = 1.0e-2
    lambd: float = 1.0e-4
    alpha: float = 0.75
    t0: float = 1000000.0
    weight_decay: float = 0


@dataclass
class NAdamConf(OptimizerConf):
    _target_: str = "torch.optim.NAdam"
    lr: float = 2.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0
    momentum_decay: float = 0.004


@dataclass
class RAdamConf(OptimizerConf):
    _target_: str = "torch.optim.RAdam"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class RMSpropConf(OptimizerConf):
    _target_: str = "torch.optim.RMSprop"
    lr: float = 1.0e-2
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0
    momentum: float = 0
    centered: bool = False


@dataclass
class RpropConf(OptimizerConf):
    _target_: str = "torch.optim.Rprop"
    lr: float = 1.0e-2
    etas: List[float] = field(default_factory=lambda: [0.5, 1.2])
    step_sizes: List[float] = field(default_factory=lambda: [1.0e-6, 50])


@dataclass
class A2GradExpConf(OptimizerConf):
    _target_: str = "torch_optimizer.A2GradExp"
    lr: float = 1e-2  # LR not support for optim, but needed to not fail schedulers
    beta: float = 10.0
    lips: float = 10.0


@dataclass
class A2GradIncConf(OptimizerConf):
    _target_: str = "torch_optimizer.A2GradInc"
    lr: float = 1e-2  # LR not support for optim, but needed to not fail schedulers
    beta: float = 10.0
    lips: float = 10.0


@dataclass
class A2GradUniConf(OptimizerConf):
    _target_: str = "torch_optimizer.A2GradUni"
    lr: float = 1e-2  # LR not support for optim, but needed to not fail schedulers
    beta: float = 10.0
    lips: float = 10.0


@dataclass
class AccSGDConf(OptimizerConf):
    _target_: str = "torch_optimizer.AccSGD"
    lr: float = 1.0e-3
    kappa: float = 1000.0
    xi: float = 10.0
    small_const: float = 0.7
    weight_decay: float = 0


@dataclass
class AdaBeliefConf(OptimizerConf):
    _target_: str = "torch_optimizer.AdaBelief"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-3
    weight_decay: float = 0
    amsgrad: bool = False
    weight_decouple: bool = False
    fixed_decay: bool = False
    rectify: bool = False


@dataclass
class AdaBoundConf(OptimizerConf):
    _target_: str = "torch_optimizer.AdaBound"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    final_lr: float = 0.1
    gamma: float = 1e-3
    eps: float = 1e-8
    weight_decay: float = 0
    amsbound: bool = False


@dataclass
class AdaModConf(OptimizerConf):
    _target_: str = "torch_optimizer.AdaMod"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    beta3: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class AdafactorConf(OptimizerConf):
    _target_: str = "torch_optimizer.Adafactor"
    lr: float = 1.0e-3
    eps2: List[float] = field(default_factory=lambda: [1e-30, 1e-3])
    clip_threshold: float = 1.0
    decay_rate: float = -0.8
    beta1: Any = None
    weight_decay: float = 0
    scale_parameter: bool = True
    relative_step: bool = True
    warmup_init: bool = False


@dataclass
class AdamPConf(OptimizerConf):
    _target_: str = "torch_optimizer.AdamP"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0
    delta: float = 0.1
    wd_ratio: float = 0.1


@dataclass
class AggMoConf(OptimizerConf):
    _target_: str = "torch_optimizer.AggMo"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.0, 0.9, 0.99])
    weight_decay: float = 0


@dataclass
class ApolloConf(OptimizerConf):
    _target_: str = "torch_optimizer.Apollo"
    lr: float = 1.0e-2
    beta: float = 0.9
    eps: float = 1e-4
    warmup: int = 0
    init_lr: float = 0.01
    weight_decay: float = 0


@dataclass
class DiffGradConf(OptimizerConf):
    _target_: str = "torch_optimizer.DiffGrad"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class LambConf(OptimizerConf):
    _target_: str = "torch_optimizer.Lamb"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class MADGRADConf(OptimizerConf):
    _target_: str = "torch_optimizer.MADGRAD"
    lr: float = 1.0e-2
    momentum: float = 0.9
    weight_decay: float = 0
    eps: float = 1e-6


@dataclass
class NovoGradConf(OptimizerConf):
    _target_: str = "torch_optimizer.NovoGrad"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0
    grad_averaging: bool = False
    amsgrad: bool = False


@dataclass
class PIDConf(OptimizerConf):
    _target_: str = "torch_optimizer.PID"
    lr: float = 1.0e-3
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 1e-2
    integral: float = 5.0
    derivative: float = 10.0


@dataclass
class QHAdamConf(OptimizerConf):
    _target_: str = "torch_optimizer.QHAdam"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    nus: List[float] = field(default_factory=lambda: [1.0, 1.0])
    weight_decay: float = 0
    decouple_weight_decay: bool = False
    eps: float = 1e-8


@dataclass
class QHMConf(OptimizerConf):
    _target_: str = "torch_optimizer.QHM"
    lr: float = 1.0e-3
    momentum: float = 0
    nu: float = 0.7
    weight_decay: float = 1e-2
    weight_decay_type: str = "grad"


@dataclass
class RangerConf(OptimizerConf):
    _target_: str = "torch_optimizer.Ranger"
    lr: float = 1.0e-3
    alpha: float = 0.5
    k: int = 6
    N_sma_threshhold: int = 5
    betas: List[float] = field(default_factory=lambda: [0.95, 0.999])
    eps: float = 1e-5
    weight_decay: float = 0


@dataclass
class RangerQHConf(OptimizerConf):
    _target_: str = "torch_optimizer.RangerQH"
    lr: float = 1.0e-3
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    nus: List[float] = field(default_factory=lambda: [0.7, 1.0])
    weight_decay: float = 0
    k: int = 6
    alpha: float = 0.5
    decouple_weight_decay: bool = False
    eps: float = 1e-8


@dataclass
class RangerVAConf(OptimizerConf):
    _target_: str = "torch_optimizer.RangerVA"
    lr: float = 1.0e-3
    alpha: float = 0.5
    k: int = 6
    n_sma_threshhold: int = 5
    betas: List[float] = field(default_factory=lambda: [0.95, 0.999])
    eps: float = 1e-5
    weight_decay: float = 0
    amsgrad: bool = True
    transformer: str = "softplus"
    smooth: int = 50
    grad_transformer: str = "square"


@dataclass
class SGDPConf(OptimizerConf):
    _target_: str = "torch_optimizer.SGDP"
    lr: float = 1.0e-3
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 1e-2
    nesterov: bool = False
    delta: float = 0.1
    wd_ratio: float = 0.1


@dataclass
class SGDWConf(OptimizerConf):
    _target_: str = "torch_optimizer.SGDW"
    lr: float = 1.0e-3
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 1e-2
    nesterov: bool = False


@dataclass
class SWATSConf(OptimizerConf):
    _target_: str = "torch_optimizer.SWATS"
    lr: float = 1.0e-1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-3
    weight_decay: float = 0
    amsgrad: bool = False
    nesterov: bool = False


@dataclass
class ShampooConf(OptimizerConf):
    _target_: str = "torch_optimizer.Shampoo"
    lr: float = 1.0e-1
    momentum: float = 0
    weight_decay: float = 0
    epsilon: float = 1e-4
    update_freq: int = 1


@dataclass
class YogiConf(OptimizerConf):
    _target_: str = "torch_optimizer.Yogi"
    lr: float = 1.0e-2
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-3
    initial_accumulator: float = 1e-6
    weight_decay: float = 0


def register_optimizer_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="optimizer",
        name="adam",
        node=AdamConf,
    )
    cs.store(
        group="optimizer",
        name="sgd",
        node=SGDConf,
    )
    cs.store(
        group="optimizer",
        name="adahessian",
        node=AdahessianConf,
    )
    cs.store(
        group="optimizer",
        name="bfgs",
        node=BFGSConf,
    )
    cs.store(
        group="optimizer",
        name="adadelta",
        node=AdadeltaConf,
    )
    cs.store(
        group="optimizer",
        name="adagrad",
        node=AdagradConf,
    )
    cs.store(
        group="optimizer",
        name="adamw",
        node=AdamWConf,
    )
    cs.store(
        group="optimizer",
        name="sparse_adam",
        node=SparseAdamConf,
    )
    cs.store(
        group="optimizer",
        name="adamax",
        node=AdamaxConf,
    )
    cs.store(
        group="optimizer",
        name="asgd",
        node=ASGDConf,
    )
    cs.store(
        group="optimizer",
        name="nadam",
        node=NAdamConf,
    )
    cs.store(
        group="optimizer",
        name="radam",
        node=RAdamConf,
    )
    cs.store(
        group="optimizer",
        name="rmsprop",
        node=RMSpropConf,
    )
    cs.store(
        group="optimizer",
        name="rprop",
        node=RpropConf,
    )
    cs.store(
        group="optimizer",
        name="a2grad_exp",
        node=A2GradExpConf,
    )
    cs.store(
        group="optimizer",
        name="a2grad_inc",
        node=A2GradIncConf,
    )
    cs.store(
        group="optimizer",
        name="a2grad_uni",
        node=A2GradUniConf,
    )
    cs.store(
        group="optimizer",
        name="accsgd",
        node=AccSGDConf,
    )
    cs.store(
        group="optimizer",
        name="adabelief",
        node=AdaBeliefConf,
    )
    cs.store(
        group="optimizer",
        name="adabound",
        node=AdaBoundConf,
    )
    cs.store(
        group="optimizer",
        name="adamod",
        node=AdaModConf,
    )
    cs.store(
        group="optimizer",
        name="adafactor",
        node=AdafactorConf,
    )
    cs.store(
        group="optimizer",
        name="adamp",
        node=AdamPConf,
    )
    cs.store(
        group="optimizer",
        name="aggmo",
        node=AggMoConf,
    )
    cs.store(
        group="optimizer",
        name="apollo",
        node=ApolloConf,
    )
    cs.store(
        group="optimizer",
        name="diffgrad",
        node=DiffGradConf,
    )
    cs.store(
        group="optimizer",
        name="lamb",
        node=LambConf,
    )
    cs.store(
        group="optimizer",
        name="madgrad",
        node=MADGRADConf,
    )
    cs.store(
        group="optimizer",
        name="novograd",
        node=NovoGradConf,
    )
    cs.store(
        group="optimizer",
        name="pid",
        node=PIDConf,
    )
    cs.store(
        group="optimizer",
        name="qhadam",
        node=QHAdamConf,
    )
    cs.store(
        group="optimizer",
        name="qhm",
        node=QHMConf,
    )
    cs.store(
        group="optimizer",
        name="ranger",
        node=RangerConf,
    )
    cs.store(
        group="optimizer",
        name="ranger_qh",
        node=RangerQHConf,
    )
    cs.store(
        group="optimizer",
        name="ranger_va",
        node=RangerVAConf,
    )
    cs.store(
        group="optimizer",
        name="sgdp",
        node=SGDPConf,
    )
    cs.store(
        group="optimizer",
        name="sgdw",
        node=SGDWConf,
    )
    cs.store(
        group="optimizer",
        name="swats",
        node=SWATSConf,
    )
    cs.store(
        group="optimizer",
        name="shampoo",
        node=ShampooConf,
    )
    cs.store(
        group="optimizer",
        name="yogi",
        node=YogiConf,
    )
