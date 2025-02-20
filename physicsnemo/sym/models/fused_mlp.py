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

from typing import Optional, Dict, Tuple, Union
from physicsnemo.sym.key import Key

import torch
import torch.nn as nn
from torch import Tensor

import logging

from physicsnemo.sym.models.activation import Activation
from physicsnemo.sym.models.arch import Arch

from typing import List

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor
    if compute_capability < 80:
        logger.warning(
            "Detected GPU architecture older than Ampere. Please check documentation for instructions to recompile and run tinycudann for your GPU"
        )
    import tinycudann as tcnn
else:
    raise ImportError("Tiny CUDA NN only supported on CUDA enabled GPUs")


class TinyCudaNNArchCore(Arch):
    """
    Fully Fused Multi Layer Perceptron (MLP) architecture.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    periodicity : Union[Dict[str, Tuple[float, float]], None] = None
        Dictionary of tuples that allows making model
        give periodic predictions on the given bounds in
        tuple. For example, `periodicity={'x': (0, 1)}` would
        make the network give periodic results for `x` on the
        interval `(0, 1)`.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int = 64
        Layer size for every hidden layer of the model.
    nr_layers : int = 2
        Number of hidden layers of the model.
    activation_fn : Activation = Activation.SIGMOID
        Activation function used by network.
    fully_fused : bool = True
        Whether to use a fully fused MLP kernel implementation
        This option is only respected if the number of neurons per layer
        is one of [16, 32, 64, 128] and is supported only on Turing+
        architectures
    encoding_config : Optional[Dict] = None
        Optional encoding configuration dictionary
        See here for specifics: https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#encodings
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        detach_keys: List[Key] = [],
        layer_size: int = 64,
        nr_layers: int = 2,
        activation_fn=Activation.SIGMOID,
        fully_fused: bool = True,
        encoding_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
            periodicity=periodicity,
        )

        # supported activations
        supported_activations = {
            Activation.RELU: "ReLU",
            # Activation.EXP : "Exponential",
            # Activation.SIN : "Sine",
            Activation.SIGMOID: "Sigmoid",
            Activation.SQUAREPLUS: "Squareplus",
            Activation.SOFTPLUS: "Softplus",
            Activation.IDENTITY: "None",
        }

        if activation_fn not in supported_activations.keys():
            raise ValueError(
                f"{activation_fn} activation is not supported for fused architectures."
            )

        if self.periodicity is not None:
            in_features = sum(
                [
                    x.size
                    for x in self.input_keys
                    if x.name not in list(periodicity.keys())
                ]
            ) + +sum(
                [
                    2 * x.size
                    for x in self.input_keys
                    if x.name in list(periodicity.keys())
                ]
            )
        else:
            in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        if fully_fused and (layer_size not in set([16, 32, 64, 128])):
            fully_fused = False
            logger.warning(
                f"Unsupported layer_size {layer_size} for FullyFusedMLP. Using CutlassMLP instead."
            )

        network_config = {
            "otype": "FullyFusedMLP" if fully_fused else "CutlassMLP",
            "activation": supported_activations[activation_fn],
            "output_activation": "None",
            "n_neurons": layer_size,
            "n_hidden_layers": nr_layers,
        }

        if encoding_config is not None:
            self._impl = tcnn.NetworkWithInputEncoding(
                in_features, out_features, encoding_config, network_config
            )
        else:
            self._impl = tcnn.Network(in_features, out_features, network_config)

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        y = self._impl(x)
        return self.prepare_output(
            y, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )

    def make_node(self, name: str, jit: bool = False, optimize: bool = True):
        if jit:
            logger.warning(
                "JIT compilation not supported for TinyCudaNNArchCore. Creating node with JIT turned off"
            )
        return super().make_node(name, False, optimize)


class FusedMLPArch(TinyCudaNNArchCore):
    """
    Fully Fused Multi Layer Perceptron (MLP) architecture.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    periodicity : Union[Dict[str, Tuple[float, float]], None] = None
        Dictionary of tuples that allows making model
        give periodic predictions on the given bounds in
        tuple. For example, `periodicity={'x': (0, 1)}` would
        make the network give periodic results for `x` on the
        interval `(0, 1)`.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int = 64
        Layer size for every hidden layer of the model.
    nr_layers : int = 2
        Number of hidden layers of the model.
    activation_fn : Activation = Activation.SIGMOID
        Activation function used by network.
    fully_fused : bool = True
        Whether to use a fully fused MLP kernel implementation
        This option is only respected if the number of neurons per layer
        is one of [16, 32, 64, 128] and is supported only on Turing+
        architectures
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        detach_keys: List[Key] = [],
        layer_size: int = 64,
        nr_layers: int = 2,
        activation_fn=Activation.SIGMOID,
        fully_fused: bool = True,
    ) -> None:
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            periodicity=periodicity,
            detach_keys=detach_keys,
            layer_size=layer_size,
            nr_layers=nr_layers,
            activation_fn=activation_fn,
            fully_fused=fully_fused,
            encoding_config=None,
        )


class FusedFourierNetArch(TinyCudaNNArchCore):
    """
    Fused Fourier Net architecture.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    periodicity : Union[Dict[str, Tuple[float, float]], None] = None
        Dictionary of tuples that allows making model
        give periodic predictions on the given bounds in
        tuple. For example, `periodicity={'x': (0, 1)}` would
        make the network give periodic results for `x` on the
        interval `(0, 1)`.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int = 64
        Layer size for every hidden layer of the model.
    nr_layers : int = 2
        Number of hidden layers of the model.
    activation_fn : Activation = Activation.SIN
        Activation function used by network.
    fully_fused : bool = True
        Whether to use a fully fused MLP kernel implementation
        This option is only respected if the number of neurons per layer
        is one of [16, 32, 64, 128] and is supported only on Turing+
        architectures
    n_frequencies : int = 12
        number of frequencies to use in the encoding
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        detach_keys: List[Key] = [],
        layer_size: int = 64,
        nr_layers: int = 2,
        activation_fn=Activation.SIGMOID,
        fully_fused: bool = True,
        n_frequencies: int = 12,
    ) -> None:

        encoding_config = {
            "otype": "Frequency",
            "n_frequencies": n_frequencies,
        }

        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            periodicity=periodicity,
            detach_keys=detach_keys,
            layer_size=layer_size,
            nr_layers=nr_layers,
            activation_fn=activation_fn,
            fully_fused=fully_fused,
            encoding_config=encoding_config,
        )


class FusedGridEncodingNetArch(TinyCudaNNArchCore):
    """
    Fused Fourier Net architecture.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    periodicity : Union[Dict[str, Tuple[float, float]], None] = None
        Dictionary of tuples that allows making model
        give periodic predictions on the given bounds in
        tuple. For example, `periodicity={'x': (0, 1)}` would
        make the network give periodic results for `x` on the
        interval `(0, 1)`.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    layer_size : int = 64
        Layer size for every hidden layer of the model.
    nr_layers : int = 2
        Number of hidden layers of the model.
    activation_fn : Activation = Activation.SIN
        Activation function used by network.
    fully_fused : bool = True
        Whether to use a fully fused MLP kernel implementation
        This option is only respected if the number of neurons per layer
        is one of [16, 32, 64, 128] and is supported only on Turing+
        architectures
    indexing : str = "Hash"
        Type of backing storage of the grids. Can be "Hash", "Tiled"
        or "Dense".
    n_levels : int = 16
        Number of levels (resolutions)
    n_features_per_level : int = 2
        Dimensionality of feature vector stored in each level's entries.
    log2_hashmap_size : int = 19
        If type is "Hash", is the base-2 logarithm of the number of
        elements in each backing hash table.
    base_resolution : int = 16
        The resolution of the coarsest level is base_resolution^input_dims.
    per_level_scale : float = 2.0
        The geometric growth factor, i.e. the factor by which the resolution
        of each grid is larger (per axis) than that of the preceeding level.
    interpolation : str = "Smoothstep"
        How to interpolate nearby grid lookups.
        Can be "Nearest", "Linear", or "Smoothstep" (for smooth derivatives).
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        detach_keys: List[Key] = [],
        layer_size: int = 64,
        nr_layers: int = 2,
        activation_fn=Activation.SIGMOID,
        fully_fused: bool = True,
        indexing: str = "Hash",
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 2.0,
        interpolation: str = "Smoothstep",
    ) -> None:

        if indexing not in ["Hash", "Tiled", "Dense"]:
            raise ValueError(f"indexing type {indexing} not supported")
        if interpolation not in ["Nearest", "Linear", "Smoothstep"]:
            raise ValueError(f"interpolation type {interpolation} not supported")

        encoding_config = {
            "otype": "Grid",
            "type": indexing,
            "n_levels": n_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2.0,
            "interpolation": interpolation,
        }

        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            periodicity=periodicity,
            detach_keys=detach_keys,
            layer_size=layer_size,
            nr_layers=nr_layers,
            activation_fn=activation_fn,
            fully_fused=fully_fused,
            encoding_config=encoding_config,
        )
