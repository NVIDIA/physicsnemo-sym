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

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import logging
import ast

from termcolor import colored
from inspect import signature, _empty
from typing import Optional, Callable, List, Dict, Union, Tuple
from physicsnemo.sym.amp import DerivScaler, DerivScalers
from physicsnemo.sym.constants import NO_OP_SCALE
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.constants import JIT_PYTORCH_VERSION
from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.manager import JitManager, JitArchMode
from physicsnemo.sym.models.activation import Activation

logger = logging.getLogger(__name__)


class Arch(nn.Module):
    """
    Base class for all neural networks
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.periodicity = periodicity
        self.saveable = True

        self.input_key_dict = {str(var): var.size for var in input_keys}
        self.output_key_dict = {str(var): var.size for var in output_keys}

        # input and output scales
        input_scales = {str(k): k.scale for k in input_keys}
        output_scales = {str(k): k.scale for k in output_keys}
        self.input_scales = (
            None
            if all([s == NO_OP_SCALE for s in input_scales.values()])
            else input_scales
        )
        self.output_scales = (
            None
            if all([s == NO_OP_SCALE for s in output_scales.values()])
            else output_scales
        )
        # Register scales tensors as buffers.
        # Buffer is allowed to be None, in this case it is a no-op in process_input function.
        self.register_buffer(
            "input_scales_tensor",
            self._get_scalers_tensor(self.input_key_dict, self.input_scales),
            persistent=False,
        )
        self.register_buffer(
            "output_scales_tensor",
            self._get_scalers_tensor(self.output_key_dict, self.output_scales),
            persistent=False,
        )

        self.detach_keys = detach_keys
        self.detach_key_dict: Dict[str, int] = {
            str(var): var.size for var in detach_keys
        }

        self.var_dim = -1

        # If no detach keys, add a dummy for TorchScript compilation
        if not self.detach_key_dict:
            dummy_str = "_"
            while dummy_str in self.input_key_dict:
                dummy_str += "_"
            self.detach_key_dict[dummy_str] = 0

    def make_node(self, name: str, jit: Optional[bool] = None, optimize: bool = True):
        """Makes neural network node for unrolling with PhysicsNeMo Graph.

        Parameters
        ----------
        name : str
            This will be used as the name of created node.
        jit : bool
            If true then compile the whole arch with jit, https://pytorch.org/docs/stable/jit.html.
            If None (default), will use the JitManager to get the global flag and mode (the default
            mode is `jit_arch_mode="only_activation"`), which could be configured in the hydra config.
            Please note that jit=true does not work with functorch and autograd trim edges.
        optimize : bool
            If true then treat parameters as optimizable.

        Examples
        --------
        Here is a simple example of creating a node from the fully connected network::

            >>> from .fully_connected import FullyConnectedArch
            >>> from physicsnemo.sym.key import Key
            >>> fc_arch = FullyConnectedArch([Key('x'), Key('y')], [Key('u')])
            >>> fc_node = fc_arch.make_node(name="fc_node")
            >>> print(fc_node)
            node: fc_node
            inputs: [x, y]
            derivatives: []
            outputs: [u]
            optimize: True

        """
        manager = DistributedManager()
        model_parallel_rank = (
            manager.group_rank("model_parallel") if manager.distributed else 0
        )

        # set name for loading and saving model
        self.name = name
        # set checkpoint filename for model
        # append model parallel rank since each process in the first model
        # parallel group will save a separate checkpoint
        self.checkpoint_filename = name + f".{model_parallel_rank}.pth"

        if jit:
            logger.warning(
                "Passing jit=true when constructing Arch Node is deprecated, "
                "please remove it as JITManager could automatically handel it."
            )
        elif jit is None:
            jit = JitManager().enabled and JitManager().arch_mode == JitArchMode.ALL

        # compile network
        if jit:
            # Warn user if pytorch version difference
            if not torch.__version__ == JIT_PYTORCH_VERSION:
                logger.warning(
                    f"Installed PyTorch version {torch.__version__} is not TorchScript"
                    + f" supported in PhysicsNeMo. Version {JIT_PYTORCH_VERSION} is officially supported."
                )

            arch = torch.jit.script(self)
            node_name = "Arch Node (jit): " + ("" if name is None else str(name))
            logger.info("Jit compiling network arch")
        else:
            arch = self
            node_name = "Arch Node: " + ("" if name is None else str(name))

        # Set save and load methods TODO this is hacky but required for jit
        arch.save = self.save
        arch.load = self.load

        # Create and return node from this network architecture
        net_node = Node(
            self.input_keys, self.output_keys, arch, name=node_name, optimize=optimize
        )
        return net_node

    def save(self, directory):
        torch.save(self.state_dict(), directory + "/" + self.checkpoint_filename)

    def load(self, directory, map_location=None):
        self.load_state_dict(
            torch.load(
                directory + "/" + self.checkpoint_filename, map_location=map_location
            )
        )

    def set_scaling(
        self,
        var_name: str,
        shift: float = 0,
        scale: float = 1,
    ):
        if var_name in self.input_key_dict:
            self.input_scales[var_name] = (shift, scale)

        if var_name in self.output_key_dict:
            self.output_scales[var_name] = (shift, scale)

        self.input_scales_tensor = self._get_scalers_tensor(
            self.input_key_dict, self.input_scales
        )
        self.output_scales_tensor = self._get_scalers_tensor(
            self.output_key_dict, self.output_scales
        )

    @staticmethod
    def _get_scalers_tensor(
        key_size_dict: Dict[str, int],
        key_scales: Union[Dict[str, Tuple[float, float]], None] = None,
    ) -> Tensor:
        if key_scales is None:
            return None
        scalers_tensor = [[], []]
        for key, size in key_size_dict.items():
            for _ in range(size):
                scalers_tensor[0].append(key_scales[key][0])
                scalers_tensor[1].append(key_scales[key][1])

        return torch.tensor(scalers_tensor)

    @staticmethod
    def prepare_input(
        input_variables: Dict[str, Tensor],
        mask: List[str],
        detach_dict: Dict[str, int],
        dim: int = 0,
        input_scales: Union[Dict[str, Tuple[float, float]], None] = None,
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
    ) -> Tensor:
        output_tensor = []
        for key in mask:
            if key in detach_dict:
                x = input_variables[key].detach()
            else:
                x = input_variables[key]
            # Scale input data
            if input_scales is not None:
                x = (x - input_scales[key][0]) / input_scales[key][1]

            append_tensor = [x]
            if periodicity is not None:
                if key in list(periodicity.keys()):
                    scaled_input = (x - periodicity[key][0]) / (
                        periodicity[key][1] - periodicity[key][0]
                    )
                    sin_tensor = torch.sin(2.0 * np.pi * scaled_input)
                    cos_tensor = torch.cos(2.0 * np.pi * scaled_input)
                    append_tensor = [sin_tensor, cos_tensor]
            output_tensor += append_tensor
        return torch.cat(output_tensor, dim=dim)

    @staticmethod
    def concat_input(
        input_variables: Dict[str, Tensor],
        mask: List[str],
        detach_dict: Union[Dict[str, int], None] = None,
        dim: int = -1,
    ) -> Tensor:
        output_tensor = []
        for key in mask:
            if detach_dict is not None and key in detach_dict:
                x = input_variables[key].detach()
            else:
                x = input_variables[key]
            output_tensor += [x]
        return torch.cat(output_tensor, dim=dim)

    @staticmethod
    def process_input(
        input_tensor: Tensor,
        input_scales_tensor: Union[Tensor, None] = None,
        periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
        input_dict: Union[Dict[str, int], None] = None,
        dim: int = -1,
    ) -> Tensor:
        if input_scales_tensor is not None:
            input_tensor = (
                input_tensor - input_scales_tensor[0]
            ) / input_scales_tensor[1]
        if periodicity is not None:
            assert input_dict is not None
            inputs = input_tensor.split(list(input_dict.values()), dim=dim)
            outputs = []
            for i, key in enumerate(input_dict.keys()):
                if key in list(periodicity.keys()):
                    scaled_input = (inputs[i] - periodicity[key][0]) / (
                        periodicity[key][1] - periodicity[key][0]
                    )
                    sin_tensor = torch.sin(2.0 * np.pi * scaled_input)
                    cos_tensor = torch.cos(2.0 * np.pi * scaled_input)
                    outputs += [sin_tensor, cos_tensor]
                else:
                    outputs += [inputs[i]]
            input_tensor = torch.cat(outputs, dim=dim)
        return input_tensor

    @staticmethod
    def prepare_slice_index(
        input_dict: Dict[str, int],
        slice_keys: List[str],
    ) -> Tensor:
        """
        Used in fourier-like architectures.

        For example:
            input_dict = {"x": 1, "y": 2, "z": 1}
            slice_keys = ["x", "z"]
            return tensor([0, 3])
        """
        index_dict = {}
        c = 0
        for key, size in input_dict.items():
            index_dict[key] = []
            for _ in range(size):
                index_dict[key].append(c)
                c += 1
        slice_index = []
        for key in slice_keys:
            slice_index += index_dict[key]
        return torch.tensor(slice_index)

    @staticmethod
    def slice_input(
        input_tensor: Tensor,
        slice_index: Tensor,
        dim: int = -1,
    ) -> Tensor:
        """
        Used in fourier-like architectures.
        """
        return input_tensor.index_select(dim, slice_index)

    @staticmethod
    def _get_normalization_tensor(
        key_size_dict: Dict[str, int],
        key_normalization: Union[Dict[str, Tuple[float, float]], None] = None,
    ) -> Tensor:
        """
        Used in siren and multiplicative_filter_net architectures.
        """
        if key_normalization is None:
            return None
        normalization_tensor = [[], []]
        for key, size in key_size_dict.items():
            for _ in range(size):
                normalization_tensor[0].append(key_normalization[key][0])
                normalization_tensor[1].append(key_normalization[key][1])
        return torch.tensor(normalization_tensor)

    @staticmethod
    def _tensor_normalize(x: Tensor, norm_tensor: Tensor) -> Tensor:
        """
        Used in siren and multiplicative_filter_net architectures.
        """
        if norm_tensor is None:
            return x

        normalized_x = (x - norm_tensor[0]) / (norm_tensor[1] - norm_tensor[0])
        normalized_x = 2 * normalized_x - 1
        return normalized_x

    @staticmethod
    def prepare_output(
        output_tensor: Tensor,
        output_var: Dict[str, int],
        dim: int = 0,
        output_scales: Union[Dict[str, Tuple[float, float]], None] = None,
    ) -> Dict[str, Tensor]:
        # create unnormalised output tensor
        output = {}
        for k, v in zip(
            output_var,
            torch.split(output_tensor, list(output_var.values()), dim=dim),
        ):
            output[k] = v
            if output_scales is not None:
                output[k] = output[k] * output_scales[k][1] + output_scales[k][0]

        return output

    @staticmethod
    def split_output(
        output_tensor: Tensor,
        output_dict: Dict[str, int],
        dim: int = -1,
    ) -> Dict[str, Tensor]:
        output = {}
        for k, v in zip(
            output_dict,
            torch.split(output_tensor, list(output_dict.values()), dim=dim),
        ):
            output[k] = v
        return output

    @staticmethod
    def process_output(
        output_tensor: Tensor,
        output_scales_tensor: Union[Tensor, None] = None,
    ) -> Tensor:
        if output_scales_tensor is not None:
            output_tensor = (
                output_tensor * output_scales_tensor[1] + output_scales_tensor[0]
            )
        return output_tensor

    def _tensor_forward(self, x: Tensor) -> None:
        r"""
        This method defines the computation performed with an input tensor
        concatenated from the input dictionary. All subclasses need to
        override this method to be able to use FuncArch.
        """
        raise NotImplementedError

    def _find_computable_deriv_with_func_arch(
        self, needed_names: List[Key], allow_partial_hessian
    ):
        """
        Given a list of names, find a list of derivatives that could be computed
        by using the FuncArch API.

        allow_partial_hessian: bool
            If allow_partial_hessian is on, allow evaluating partial hessian to save
            some unnecessary computations.
            For example, when the input is x, outputs are [u, p], and the needed
            derivatives are `[u__x, p__x, u__x__x]`, when this flag is on, FuncArch
            will only evaluate [u__x, u__x__x].
        """
        compute_derivs = {1: [], 2: []}
        # collect all computable derivatives
        for n in needed_names:
            computable = True
            # check the derivative is computable
            order = len(n.derivatives)
            if 0 < order < 3 and Key(n.name) in self.output_keys:
                for deriv in n.derivatives:
                    if deriv not in self.input_keys:
                        computable = False
                if computable:
                    compute_derivs[order].append(n)
        # Filtering out the Jacobian terms that are not required for the Hessian terms,
        # these Jacobian terms will get picked up by the regular autograd engine.
        if allow_partial_hessian and len(compute_derivs[2]):
            needed_hessian_name = set([d.name for d in compute_derivs[2]])
            compute_derivs[1] = [
                d for d in compute_derivs[1] if d.name in needed_hessian_name
            ]
        return sorted(compute_derivs[1]) + sorted(compute_derivs[2])

    @property
    @torch.jit.unused
    # We could not use @torch.jit.ignore when combining with @property
    # see https://github.com/pytorch/pytorch/issues/54688 .
    # Using @torch.jit.unused is good for us as we never call `supports_func_arch`
    # in `forward` or `_tensor_forward` method.
    def supports_func_arch(self) -> bool:
        """
        Returns whether the instantiate arch object support FuncArch API.

        We determine it by checking whether the arch object's subclass has
        overridden the `_tensor_forward` method.
        """
        return self.__class__._tensor_forward != Arch._tensor_forward

    @classmethod
    def from_config(cls, cfg: Dict):
        """Instantiates a neural network based on a model's OmegaConfig

        Nearly all parameters of a model can be specified in the Hydra config or provied
        when calling `instantiate_arch`. Additionally, model keys can be defined
        and parsed or proved manually in the `instantiate_arch` method. Parameters that
        are not primitive data types can be added explicitly or as python code as a
        string in the config.

        Parameters
        ----------
        cfg : Dict
            Config dictionary

        Returns
        -------
        Arch, Dict[str, any]
            Returns instantiated model and dictionary of parameters used to initialize it


        Example
        -------
        This is an example of a fourier network's config

        >>> arch:
        >>>     fourier:
        >>>         input_keys: [x, y] # Key('x'), Key('y')
        >>>         output_keys: ['trunk', 256] # Key('trunk', size=256)
        >>>         frequencies: "('axis', [i for i in range(5)])" # Python code gets parsed
        >>>         frequencies_params: "[0,1,2,3,4]" # Literal strings allowed
        >>>         nr_layers: 4
        >>>         layer_size: 128

        Note
        ----
        Refer to `Key.convert_config` for more details on how to define keys in the config.
        """
        model_params = signature(cls.__init__).parameters

        # Init keys if config was used to define them (string)
        if "input_keys" in cfg:
            cfg["input_keys"] = Key.convert_config(cfg["input_keys"])
        if "output_keys" in cfg:
            cfg["output_keys"] = Key.convert_config(cfg["output_keys"])
        if "detach_keys" in cfg:
            cfg["detach_keys"] = Key.convert_config(cfg["detach_keys"])

        # Activation functions
        if "activation_fn" in cfg and isinstance(cfg["activation_fn"], str):
            cfg["activation_fn"] = Activation[cfg["activation_fn"]]

        params = {}
        for key in model_params:
            parameter = model_params[key]
            if parameter.name in cfg:
                if isinstance(cfg[parameter.name], str) and not isinstance(
                    parameter.kind, str
                ):
                    try:
                        # Try eval for python code that needs to run
                        # Such as list compression
                        param_literal = ast.literal_eval(cfg[parameter.name])
                    except:
                        # Fall back... hope a string works
                        param_literal = cfg[parameter.name]
                    params[parameter.name] = param_literal
                else:
                    params[parameter.name] = cfg[parameter.name]
            # If parameter is not in the config and has no default value
            # Give a warning, because this will error
            elif parameter.default is _empty and parameter.name != "self":
                logger.warning(
                    colored(
                        f"Positional argument '{parameter.name}' not provided. Consider manually adding it to instantiate_arch() call.",
                        "yellow",
                    )
                )

        model = cls(**params)

        # Set any variable scaling
        if "scaling" in cfg and cfg["scaling"] is not None:
            for scale_dict in cfg["scaling"]:
                try:
                    name = next(iter(scale_dict))
                    shift, scale = scale_dict[name]
                    model.set_scaling(name, shift, scale)
                except:
                    logger.warning(f"Failed to set scaling with config {scale_dict}")

        return model, params


class FuncArch(nn.Module):
    """
    Base class for all neural networks using functorch functional API.
    FuncArch perform Jacobian and Hessian calculations during the forward pass.

    Parameters
    ----------
    arch : Arch
        An instantiated Arch object.
    deriv_keys : List[Key]
        A list of needed derivative keys.
    forward_func : Callable, Optional
        If provided then it will be used as the forward function instead of the
        `arch._tensor_forward` function.
    """

    def __init__(
        self, arch: Arch, deriv_keys: List[Key], forward_func: Optional[Callable] = None
    ):
        super().__init__()

        if "torch.jit" in str(type(arch)):
            raise RuntimeError(
                f"Found {type(arch)}, currently FuncArch does not work with jit."
            )
        assert isinstance(arch, Arch), (
            f"arch should be an instantiated Arch object, but found to be {type(arch)}."
        )
        assert arch.supports_func_arch, (
            f"{type(arch)} currently does not support FuncArch."
        )

        if forward_func is None:
            forward_func = arch._tensor_forward

        self.saveable = True
        self.deriv_keys = deriv_keys
        self.arch = arch

        self.input_key_dim = self._get_key_dim(arch.input_keys)
        self.output_key_dim = self._get_key_dim(arch.output_keys)
        self.deriv_key_dict, self.max_order = self._collect_derivs(
            arch.input_key_dict, arch.output_key_dict, deriv_keys
        )
        # may only need to evaluate the partial hessian or jacobian
        needed_output_keys = set(
            [Key(d.name) for d in self.deriv_key_dict[1] + self.deriv_key_dict[2]]
        )
        # keep the keys in the original order, so the mapped dims are correct
        needed_output_keys = [
            key for key in arch.output_keys if key in needed_output_keys
        ]
        # needed_output_dims is used to slice I_N to save some computation
        self.needed_output_dims = torch.tensor(
            [self.output_key_dim[key.name] for key in needed_output_keys]
        )
        # if partial hessian or jacobian, the final output shape has changed and so the
        # corresponding output key dim mapping
        self.output_key_dim = {str(var): i for i, var in enumerate(needed_output_keys)}

        in_features = sum(arch.input_key_dict.values())
        out_features = sum(arch.output_key_dict.values())

        if self.max_order == 0:
            self._tensor_forward = forward_func
        elif self.max_order == 1:
            I_N = torch.eye(out_features)[self.needed_output_dims]
            self.register_buffer("I_N", I_N, persistent=False)
            self._tensor_forward = self._jacobian_impl(forward_func)
        elif self.max_order == 2:
            I_N1 = torch.eye(out_features)[self.needed_output_dims]
            I_N2 = torch.eye(in_features)
            self.register_buffer("I_N1", I_N1, persistent=False)
            self.register_buffer("I_N2", I_N2, persistent=False)
            self._tensor_forward = self._hessian_impl(forward_func)
        else:
            raise ValueError(
                "FuncArch currently does not support "
                f"{self.max_order}th order derivative"
            )

        self.scaler_enabled: bool = False
        self.deriv_scalers: Dict[int, DerivScaler] = {}

    def forward(self, in_vars: dict[str, Tensor]) -> dict[str, Tensor]:
        x = self.arch.concat_input(
            in_vars,
            self.arch.input_key_dict.keys(),
            detach_dict=self.arch.detach_key_dict,
            dim=-1,
        )

        if self.max_order == 0:
            pred = self._tensor_forward(x)
            jacobian = None
            hessian = None
        elif self.max_order == 1:
            pred, jacobian = self._tensor_forward(x)
            (jacobian,) = self._unscale([jacobian], 1)
            hessian = None
        elif self.max_order == 2:
            pred, jacobian, hessian = self._tensor_forward(x)
            jacobian, hessian = self._unscale([jacobian, hessian], 1)
            (hessian,) = self._unscale([hessian], 2)
        else:
            raise ValueError(
                "FuncArch currently does not support "
                f"{self.max_order}th order derivative"
            )

        # prepare output, jacobian and hessian
        out = self.arch.split_output(
            pred,
            self.arch.output_key_dict,
            dim=-1,
        )
        if jacobian is not None:
            out.update(
                self.prepare_jacobian(
                    jacobian,
                    self.deriv_key_dict[1],
                    self.input_key_dim,
                    self.output_key_dim,
                )
            )
        if hessian is not None:
            out.update(
                self.prepare_hessian(
                    hessian,
                    self.deriv_key_dict[2],
                    self.input_key_dim,
                    self.output_key_dim,
                )
            )
        return out

    def make_node(self, name: str, optimize: bool = True):
        """Makes functional arch node for unrolling with PhysicsNeMo Graph.

        Parameters
        ----------
        name : str
            This will be used as the name of created node.
        optimize : bool
            If true then treat parameters as optimizable.
        """

        # set name for loading and saving model
        self.name = name
        self.checkpoint_filename = name + ".pth"
        node_name = "Functional " + ("Arch" if name is None else str(name))
        ft_arch = self

        # Set save and load methods
        ft_arch.save = self.arch.save
        ft_arch.load = self.arch.load

        # Create and return node from this network architecture
        net_node = Node(
            self.arch.input_keys,
            self.arch.output_keys + self.deriv_keys,
            ft_arch,
            name=node_name,
            optimize=optimize,
        )
        return net_node

    @staticmethod
    def _get_key_dim(keys: list[Key]):
        """
        Find the corresponding dims of the keys.
        For example: Suppose we have the following keys and corresponding size
        {x: 2, y: 1, z: 1}, the concatenate result has dim 4, and each key map to
        a dim {x: [0, 1], y: 2, z: 3}.

        TODO Currently, the keys with more than one dim are dropped because they
        have no use cases.
        """

        def exclusive_sum(sizes: List):
            return np.concatenate([[0], np.cumsum(sizes)])

        exclu_sum = exclusive_sum([k.size for k in keys])
        out = {}
        for i, k in enumerate(keys):
            if k.size == 1:
                out[str(k)] = exclu_sum[i]
        return out

    @staticmethod
    def _collect_derivs(
        input_key_dict: Dict[str, int],
        output_key_dict: Dict[str, int],
        deriv_keys: List[Key],
    ):
        deriv_key_dict = {1: [], 2: []}
        for x in deriv_keys:
            # check the derivative is computable
            assert x.name in output_key_dict, f"Cannot calculate {x}"
            assert output_key_dict[x.name] == 1, f"key ({x.name}) size must be 1"
            for deriv in x.derivatives:
                assert deriv.name in input_key_dict, f"Cannot calculate {x}"
                assert input_key_dict[deriv.name] == 1, (
                    f"key ({deriv.name}) size must be 1"
                )
            # collect each order derivatives
            order = len(x.derivatives)
            if order == 0 or order >= 3:
                raise ValueError(
                    f"FuncArch currently does not support {order}th order derivative"
                )
            else:
                deriv_key_dict[order].append(x)
        max_order = 0
        for order, keys in deriv_key_dict.items():
            if keys:
                max_order = order
        return deriv_key_dict, max_order

    def _jacobian_impl(self, forward_func):
        def jacobian_func(x, v):
            pred, vjpfunc = torch.func.vjp(forward_func, x)
            return vjpfunc(v)[0], pred

        def get_jacobian(x):
            I_N = self._scale(self.I_N, 1)
            jacobian, pred = torch.vmap(
                torch.vmap(jacobian_func, in_dims=(None, 0)), in_dims=(0, None)
            )(x, I_N)
            pred = pred[:, 0, :]
            return pred, jacobian

        return get_jacobian

    def _hessian_impl(self, forward_func):
        def hessian_func(x, v1, v2):
            def jacobian_func(x):
                pred, vjpfunc = torch.func.vjp(forward_func, x)
                return vjpfunc(v1)[0], pred

            # jvp and vjp
            (jacobian, hessian, pred) = torch.func.jvp(
                jacobian_func, (x,), (v2,), has_aux=True
            )
            # vjp and vjp is slow
            # jacobian, hessianfunc, pred = torch.func.vjp(jacobian_func, x, has_aux=True)
            # hessian = hessianfunc(v2)[0]
            return hessian, jacobian, pred

        def get_hessian(x):
            I_N1 = self._scale(self.I_N1, 1)  # used to slice hessian rows
            I_N2 = self._scale(self.I_N2, 2)  # used to slice hessian columns
            hessian, jacobian, pred = torch.vmap(
                torch.vmap(
                    torch.vmap(hessian_func, in_dims=(None, None, 0)),  # I_N2
                    in_dims=(None, 0, None),  # I_N1
                ),
                in_dims=(0, None, None),  # x
            )(x, I_N1, I_N2)
            pred = pred[:, 0, 0, :]
            jacobian = jacobian[:, :, 0, :]
            return pred, jacobian, hessian

        return get_hessian

    @torch.jit.ignore
    def _scale(self, I_N: torch.Tensor, order: int) -> torch.Tensor:
        """
        It is a no-op if amp is not enabled.
        """
        if self.scaler_enabled:
            return self.deriv_scalers[order].scale(I_N)
        else:
            return I_N

    @torch.jit.ignore
    def _unscale(self, grad: List[torch.Tensor], order: int) -> List[torch.Tensor]:
        """
        It is a no-op if amp is not enabled.
        """
        if self.scaler_enabled:
            return self.deriv_scalers[order].unscale_deriv(grad)
        else:
            return grad

    def setup_deriv_scaler(self, deriv_scalers: DerivScalers):
        self.scaler_enabled = True
        self.deriv_scalers[1] = deriv_scalers.get_scaler("func_1")
        self.deriv_scalers[2] = deriv_scalers.get_scaler("func_2")

    @staticmethod
    def prepare_jacobian(
        output_tensor: Tensor,
        deriv_keys_1st_order: List[Key],
        input_key_dim: Dict[str, int],
        output_key_dim: Dict[str, int],
    ) -> Dict[str, Tensor]:
        output = {}
        for k in deriv_keys_1st_order:
            input_dim = input_key_dim[k.derivatives[0].name]
            out_dim = output_key_dim[k.name]
            output[str(k)] = output_tensor[:, out_dim, input_dim].reshape(-1, 1)
        return output

    @staticmethod
    def prepare_hessian(
        output_tensor: Tensor,
        deriv_keys_2nd_order: List[Key],
        input_key_dim: Dict[str, int],
        output_key_dim: Dict[str, int],
    ) -> Dict[str, Tensor]:
        output = {}
        for k in deriv_keys_2nd_order:
            input_dim0 = input_key_dim[k.derivatives[0].name]
            input_dim1 = input_key_dim[k.derivatives[1].name]
            out_dim = output_key_dim[k.name]
            output[str(k)] = output_tensor[:, out_dim, input_dim0, input_dim1].reshape(
                -1, 1
            )
        return output
