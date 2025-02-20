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
Architecture/Model configs
"""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI, II
from typing import Any, Union, List, Dict, Tuple


@dataclass
class ModelConf:
    arch_type: str = MISSING
    input_keys: Any = MISSING
    output_keys: Any = MISSING
    detach_keys: Any = MISSING
    scaling: Any = None


@dataclass
class AFNOConf(ModelConf):
    arch_type: str = "afno"
    img_shape: Tuple[int] = MISSING
    patch_size: int = 16
    embed_dim: int = 256
    depth: int = 4
    num_blocks: int = 8


@dataclass
class DistributedAFNOConf(ModelConf):
    arch_type: str = "distributed_afno"
    img_shape: Tuple[int] = MISSING
    patch_size: int = 16
    embed_dim: int = 256
    depth: int = 4
    num_blocks: int = 8
    channel_parallel_inputs: bool = False
    channel_parallel_outputs: bool = False


@dataclass
class DeepOConf(ModelConf):
    arch_type: str = "deeponet"
    # branch_net: Union[Arch, str],
    # trunk_net: Union[Arch, str],
    trunk_dim: Any = None  # Union[None, int]
    branch_dim: Any = None  # Union[None, int]


@dataclass
class FNOConf(ModelConf):
    arch_type: str = "fno"
    dimension: int = MISSING
    # decoder_net: Arch
    nr_fno_layers: int = 4
    fno_modes: Any = 16  # Union[int, List[int]]
    padding: int = 8
    padding_type: str = "constant"
    activation_fn: str = "gelu"
    coord_features: bool = True


@dataclass
class FourierConf(ModelConf):
    arch_type: str = "fourier"
    frequencies: Any = "('axis', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
    frequencies_params: Any = "('axis', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
    activation_fn: str = "silu"
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False


@dataclass
class FullyConnectedConf(ModelConf):

    arch_type: str = "fully_connected"
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    activation_fn: str = "silu"
    adaptive_activations: bool = False
    weight_norm: bool = True


@dataclass
class ConvFullyConnectedConf(ModelConf):
    arch_type: str = "conv_fully_connected"
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    activation_fn: str = "silu"
    adaptive_activations: bool = False
    weight_norm: bool = True


@dataclass
class FusedMLPConf(ModelConf):
    arch_type: str = "fused_fully_connected"
    layer_size: int = 128
    nr_layers: int = 6
    activation_fn: str = "sigmoid"


@dataclass
class FusedFourierNetConf(ModelConf):
    arch_type: str = "fused_fourier"
    layer_size: int = 128
    nr_layers: int = 6
    activation_fn: str = "sigmoid"
    n_frequencies: int = 12


@dataclass
class FusedGridEncodingNetConf(ModelConf):

    arch_type: str = "fused_hash_encoding"
    layer_size: int = 128
    nr_layers: int = 6
    activation_fn: str = "sigmoid"
    indexing: str = "Hash"
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 16
    per_level_scale: float = 2.0
    interpolation: str = "Smoothstep"


@dataclass
class MultiresolutionHashNetConf(ModelConf):
    arch_type: str = "hash_encoding"
    layer_size: int = 64
    nr_layers: int = 3
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False
    bounds: Any = "[(1.0, 1.0), (1.0, 1.0)]"
    nr_levels: int = 16
    nr_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: int = 2
    finest_resolution: int = 32


@dataclass
class HighwayFourierConf(ModelConf):
    arch_type: str = "highway_fourier"
    frequencies: Any = "('axis', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
    frequencies_params: Any = "('axis', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
    activation_fn: str = "silu"
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False
    transform_fourier_features: bool = True
    project_fourier_features: bool = False


@dataclass
class ModifiedFourierConf(ModelConf):
    arch_type: str = "modified_fourier"
    frequencies: Any = "('axis', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
    frequencies_params: Any = "('axis', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
    activation_fn: str = "silu"
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False


@dataclass
class MultiplicativeFilterConf(ModelConf):
    arch_type: str = "multiplicative_fourier"
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    activation_fn: str = "identity"
    filter_type: str = "fourier"
    weight_norm: bool = True
    input_scale: float = 10.0
    gabor_alpha: float = 6.0
    gabor_beta: float = 1.0
    normalization: Any = (
        None  # Change to Union[None, Dict[str, Tuple[float, float]]] when supported
    )


@dataclass
class MultiscaleFourierConf(ModelConf):
    arch_type: str = "multiscale_fourier"
    frequencies: Any = field(default_factory=lambda: [32])
    frequencies_params: Any = None
    activation_fn: str = "silu"
    layer_size: int = 512
    nr_layers: int = 6
    skip_connections: bool = False
    weight_norm: bool = True
    adaptive_activations: bool = False


@dataclass
class Pix2PixConf(ModelConf):
    arch_type: str = "pix2pix"
    dimension: int = MISSING
    conv_layer_size: int = 64
    n_downsampling: int = 3
    n_blocks: int = 3
    scaling_factor: int = 1
    batch_norm: bool = True
    padding_type: str = "reflect"
    activation_fn: str = "relu"


@dataclass
class SirenConf(ModelConf):
    arch_type: str = "siren"
    layer_size: int = 512
    nr_layers: int = 6
    first_omega: float = 30.0
    omega: float = 30.0
    normalization: Any = (
        None  # Change to Union[None, Dict[str, Tuple[float, float]]] when supported
    )


@dataclass
class SRResConf(ModelConf):
    arch_type: str = "super_res"
    large_kernel_size: int = 7
    small_kernel_size: int = 3
    conv_layer_size: int = 32
    n_resid_blocks: int = 8
    scaling_factor: int = 8
    activation_fn: str = "prelu"


def register_arch_configs() -> None:
    # Information regarding multiple config groups
    # https://hydra.cc/docs/next/patterns/select_multiple_configs_from_config_group/
    cs = ConfigStore.instance()

    cs.store(
        group="arch",
        name="fused_fully_connected",
        node={"fused_fully_connected": FusedMLPConf()},
    )

    cs.store(
        group="arch",
        name="fused_fourier",
        node={"fused_fourier": FusedFourierNetConf()},
    )

    cs.store(
        group="arch",
        name="fused_hash_encoding",
        node={"fused_hash_encoding": FusedGridEncodingNetConf()},
    )

    cs.store(
        group="arch",
        name="fully_connected",
        node={"fully_connected": FullyConnectedConf()},
    )

    cs.store(
        group="arch",
        name="conv_fully_connected",
        node={"conv_fully_connected": ConvFullyConnectedConf()},
    )

    cs.store(
        group="arch",
        name="fourier",
        node={"fourier": FourierConf()},
    )

    cs.store(
        group="arch",
        name="highway_fourier",
        node={"highway_fourier": HighwayFourierConf()},
    )

    cs.store(
        group="arch",
        name="modified_fourier",
        node={"modified_fourier": ModifiedFourierConf()},
    )

    cs.store(
        group="arch",
        name="multiplicative_fourier",
        node={"multiplicative_fourier": MultiplicativeFilterConf()},
    )

    cs.store(
        group="arch",
        name="multiscale_fourier",
        node={"multiscale_fourier": MultiscaleFourierConf()},
    )

    cs.store(
        group="arch",
        name="siren",
        node={"siren": SirenConf()},
    )

    cs.store(
        group="arch",
        name="hash_encoding",
        node={"hash_encoding": MultiresolutionHashNetConf()},
    )

    cs.store(
        group="arch",
        name="fno",
        node={"fno": FNOConf()},
    )

    cs.store(
        group="arch",
        name="afno",
        node={"afno": AFNOConf()},
    )

    cs.store(
        group="arch",
        name="distributed_afno",
        node={"distributed_afno": DistributedAFNOConf()},
    )

    cs.store(
        group="arch",
        name="deeponet",
        node={"deeponet": DeepOConf()},
    )

    cs.store(
        group="arch",
        name="super_res",
        node={"super_res": SRResConf()},
    )

    cs.store(
        group="arch",
        name="pix2pix",
        node={"pix2pix": Pix2PixConf()},
    )

    # Schemas for extending models
    # Info: https://hydra.cc/docs/next/patterns/extending_configs/
    cs.store(
        group="arch",
        name="fully_connected_cfg",
        node=FullyConnectedConf,
    )

    cs.store(
        group="arch",
        name="conv_fully_connected_cfg",
        node=ConvFullyConnectedConf,
    )

    cs.store(
        group="arch",
        name="fused_mlp_cfg",
        node=FusedMLPConf,
    )

    cs.store(
        group="arch",
        name="fused_fourier_net_cfg",
        node=FusedFourierNetConf,
    )

    cs.store(
        group="arch",
        name="fused_grid_encoding_net_cfg",
        node=FusedGridEncodingNetConf,
    )

    cs.store(
        group="arch",
        name="fourier_cfg",
        node=FourierConf,
    )

    cs.store(
        group="arch",
        name="highway_fourier_cfg",
        node=HighwayFourierConf,
    )

    cs.store(
        group="arch",
        name="modified_fourier_cfg",
        node=ModifiedFourierConf,
    )

    cs.store(
        group="arch",
        name="multiplicative_fourier_cfg",
        node=MultiplicativeFilterConf,
    )

    cs.store(
        group="arch",
        name="multiscale_fourier_cfg",
        node=MultiscaleFourierConf,
    )

    cs.store(
        group="arch",
        name="siren_cfg",
        node=SirenConf,
    )

    cs.store(
        group="arch",
        name="hash_net_cfg",
        node=MultiresolutionHashNetConf,
    )

    cs.store(
        group="arch",
        name="fno_cfg",
        node=FNOConf,
    )

    cs.store(
        group="arch",
        name="afno_cfg",
        node=AFNOConf,
    )

    cs.store(
        group="arch",
        name="distributed_afno_cfg",
        node=DistributedAFNOConf,
    )

    cs.store(
        group="arch",
        name="deeponet_cfg",
        node=DeepOConf,
    )

    cs.store(
        group="arch",
        name="super_res_cfg",
        node=SRResConf,
    )

    cs.store(
        group="arch",
        name="pix2pix_cfg",
        node=Pix2PixConf,
    )
