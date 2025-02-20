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
import logging
from collections import defaultdict
from torch.autograd import Function
from torch.amp.grad_scaler import _refresh_per_optimizer_state
from typing import List, Dict, Any
from enum import Enum
from termcolor import colored
from . import physicsnemo_ext

Tensor = torch.Tensor
logger = logging.getLogger(__name__)


class AmpMode(Enum):
    PER_ORDER_SCALER = 0
    PER_TERM_SCALER = 1


class AmpManager(object):
    _shared_state = {}

    def __new__(cls):
        obj = super(AmpManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_enabled"):
            obj._enabled = False
        if not hasattr(obj, "_mode"):
            obj._mode = AmpMode.PER_ORDER_SCALER
        if not hasattr(obj, "_dtype"):
            obj._dtype = torch.float16
        if not hasattr(obj, "_default_max_scale"):
            obj._default_max_scale = 2**0
        if not hasattr(obj, "_autocast_activation"):
            obj._autocast_activation = False
        if not hasattr(obj, "_autocast_firstlayer"):
            obj._autocast_firstlayer = False

        if not hasattr(obj, "_special_terms"):
            obj._special_terms = []
        if not hasattr(obj, "_custom_max_scales"):
            obj._custom_max_scales = {}

        return obj

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, flag):
        self._enabled = flag

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode == "per_order_scaler":
            self._mode = AmpMode.PER_ORDER_SCALER
        elif mode == "per_term_scaler":
            self._mode = AmpMode.PER_TERM_SCALER
        else:
            raise ValueError(
                f"amp mode should be per_order_scaler/per_term_scaler, but found {mode}"
            )

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype == "float16":
            self._dtype = torch.float16
        elif dtype == "bfloat16":
            self._dtype = torch.bfloat16
        else:
            raise ValueError(f"amp dtype should be float16/bfloat16, but found {dtype}")

    @property
    def default_max_scale(self):
        return self._default_max_scale

    @default_max_scale.setter
    def default_max_scale(self, scale):
        self._default_max_scale = scale

    @property
    def autocast_activation(self):
        return self._autocast_activation

    @autocast_activation.setter
    def autocast_activation(self, flag):
        self._autocast_activation = flag

    @property
    def autocast_firstlayer(self):
        return self._autocast_firstlayer

    @autocast_firstlayer.setter
    def autocast_firstlayer(self, flag):
        self._autocast_firstlayer = flag

    @property
    def special_terms(self):
        return self._special_terms

    @staticmethod
    def register_special_term(term, max_scale=None):
        assert isinstance(term, str), "term should be a str."
        manager = AmpManager()
        if term not in manager._special_terms:
            manager._special_terms.append(term)

            # set max_scale and message
            message = f"AMP: registering '{term}' as a special term and will use a separate derivative scaler"
            if max_scale is not None:
                # will not overwrite max_scale if it has been specified in the custom_max_scales
                if term not in manager.custom_max_scales:
                    manager.custom_max_scales[term] = max_scale
                message += f", max_scale: {manager.custom_max_scales[term]}"
            logger.info(message)

    @property
    def custom_max_scales(self):
        return self._custom_max_scales

    @custom_max_scales.setter
    def custom_max_scales(self, scales):
        self._custom_max_scales = scales

    @property
    def scaler_enabled(self):
        return self._enabled and self._dtype == torch.float16

    def __repr__(self):
        return f"AmpManager: {self._shared_state}"

    def init(
        self,
        enabled: bool,
        mode: "str",
        dtype: "str",
        autocast_activation: bool,
        autocast_firstlayer: bool,
        default_max_scale_log2: int,
        custom_max_scales_log2: Dict[Any, int],
    ):
        self.enabled = enabled
        self.mode = mode
        self.dtype = dtype
        self.autocast_activation = autocast_activation
        self.autocast_firstlayer = autocast_firstlayer
        self.default_max_scale = 2**default_max_scale_log2
        self.custom_max_scales = {
            key: 2**s for key, s in custom_max_scales_log2.items()
        }


# Helper functions to make AmpManager singleton work with torchscript
@torch.jit.ignore
def amp_manager_scaler_enabled_and_disable_autocast_firstlayer() -> bool:
    return AmpManager().scaler_enabled and not AmpManager().autocast_firstlayer


@torch.jit.ignore
def amp_manager_scaler_enabled_and_disable_autocast_activation() -> bool:
    return AmpManager().scaler_enabled and not AmpManager().autocast_activation


def foreach_non_finite_check_and_unscale(found_inf, inv_scale, scaled_grads):
    grad_cat = torch.cat([g.flatten() for g in scaled_grads])
    # inplace update the found_inf tensor
    torch.logical_not(torch.all(torch.isfinite(grad_cat)).flatten(), out=found_inf)
    return [g * inv_scale for g in scaled_grads]


class GradScaler(torch.cuda.amp.GradScaler):
    def __init__(
        self,
        *args,
        max_scale=2.0**30,
        recover_threshold=2.0**6,
        recover_growth_interval=100,
        **kwargs,
    ):
        """
        Modified GradScaler that supports max_scale and "recover mode".

        Parameters
        ---
        max_scale : float
            This is the upper bound of the current scaling factor.
        recover_threshold : float
            Allowing quickly recover the scaling factor when it is below or equal
            than this threshold. It is useful for FourierNetArch.
        recover_growth_interval : int
            The growth_interval that will be used when the scaling factor is less
            or equal than the recover_threshold.
        """
        super().__init__(*args, **kwargs)
        if self._enabled:
            self._max_scale = max_scale
            self._recover_threshold = recover_threshold
            self._recover_growth_interval = recover_growth_interval
            assert (
                self._init_scale <= self._max_scale
            ), "init_scale should not be greater than max_scale"
            assert (
                self._recover_threshold <= self._max_scale
            ), "recover_threshold should not be greater than max_scale"
            assert (
                self._recover_growth_interval <= self._growth_interval
            ), "recover_growth_interval should not be greater than growth_interval"

    def update(self, new_scale=None):
        """
        Updates the scaling factor.
        Copy and paste from:
        https://github.com/pytorch/pytorch/blob/dadfe1c7bf0beee0aa64edc485059e511f56a4ca/torch/cuda/amp/grad_scaler.py#L344

        The only Modification here is the custom `_amp_update_scale_` kernel, this allows
            1. A upper bound (`max_scale`) for the scaling factor.
            2. Updating the scaling factor more frequently if it is less or equal than
                the recover_threshold.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]
            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf_combined += found_infs[i]

            if found_inf_combined.bool().item():
                logger.info(
                    colored(
                        f"grad_scaler found infs, {self.state_dict()}",
                        "yellow",
                    )
                )

            torch.ops.physicsnemo_ext._amp_update_scale_(
                _scale,
                _growth_tracker,
                found_inf_combined,
                self._growth_factor,
                self._backoff_factor,
                self._growth_interval,
                self._max_scale,
                self._recover_threshold,
                self._recover_growth_interval,
            )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def state_dict(self):
        if not self._enabled:
            return {}

        state = super().state_dict()
        state.update(
            {
                "max_scale": self._max_scale,
                "recover_threshold": self._recover_threshold,
                "recover_growth_interval": self._recover_growth_interval,
            }
        )
        return state

    def load_state_dict(self, state_dict):
        if not self._enabled:
            return

        super().load_state_dict(state_dict)
        self._max_scale = state_dict["max_scale"]
        self._recover_threshold = state_dict["recover_threshold"]
        self._recover_growth_interval = state_dict["recover_growth_interval"]

    def __getstate__(self):
        state = super().__getstate__()
        if self._enabled:
            state["_max_scale"] = self._max_scale
            state["_recover_threshold"] = self._recover_threshold
            state["_recover_growth_interval"] = self._recover_growth_interval

    def get_max_scale(self):
        r"""
        Returns a Python float containing the current max_scale.
        """
        return self._max_scale


class DerivScaler(GradScaler):
    def __init__(
        self,
        *args,
        init_scale=2.0**0,
        max_scale=2.0**0,
        recover_threshold=2.0**-6,
        recover_growth_interval=100,
        **kwargs,
    ):
        """
        DerivScaler based on the GradScaler.

        The main difference between DerivScaler and GradScaler is that GradScaler unscales
        the weight gradients (maintained by the optimizer), while DerivScaler unscales the
        data gradients (higher order derivatives).

        Parameters
        ---
        max_scale : float
            This is the upper bound of the current scaling factor.
        recover_threshold : float
            Allowing quickly recover the scaling factor when it is below or equal
            than this threshold. It is useful for FourierNetArch.
        recover_growth_interval : int
            The growth_interval that will be used when the scaling factor is less
            or equal than the recover_threshold.
        """
        super().__init__(
            *args,
            init_scale=init_scale,
            max_scale=max_scale,
            recover_threshold=recover_threshold,
            recover_growth_interval=recover_growth_interval,
            **kwargs,
        )

        if self._enabled:
            self._found_inf = None

    def _lazy_init_found_inf(self):
        self._found_inf = torch.full(
            (1,), 0.0, dtype=torch.float32, device=self._scale.device
        )

    def _reset_found_inf(self):
        self._found_inf.zero_()

    def _check_found_inf(self, funcname):
        fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration."
        assert self._found_inf is not None, (
            "Attempted {} but _found_inf is None.  ".format(funcname) + fix
        )
        return self._found_inf

    def _unscale_grads_(self):
        raise RuntimeError("Function should not be called")

    def unscale_(self):
        raise RuntimeError("Function should not be called")

    def step(self):
        raise RuntimeError("Function should not be called")

    def scale(self, *args, **kwargs):
        out = super().scale(*args, **kwargs)
        if not self._enabled:
            return out
        if self._found_inf is None:
            self._lazy_init_found_inf()
        return out

    def unscale_deriv(self, grad: List[Tensor]):
        """
        Return unscaled derivatives which are also checked for infs/NaNs.

        Parameters
        ----------
        grad : List[Tensor]
            A list of derivative tensors need to be unscaled.

        Returns
        -------
        List[Tensor]
            Unscaled derivatives
        """
        if not self._enabled:
            return grad

        self._check_scale_growth_tracker("unscale_deriv")
        self._check_found_inf("unscale_deriv")

        inv_scale = self._scale.reciprocal()
        found_inf = torch.full((1,), 0.0, dtype=torch.float32, device=inv_scale.device)
        grad = foreach_non_finite_check_and_unscale(found_inf, inv_scale, grad)

        self._found_inf += found_inf
        return grad

    @property
    def found_inf(self):
        """
        Returns a Python bool containing the current found_inf state.
        If the scaler is disabled or there are no derivative nodes in the model, return False.

        .. warning::
            :meth:`found_inf` incurs a CPU-GPU sync.

        .. warning::
            :meth:`found_inf` should be called before function `update`, since `update` will reset the
            _found_inf state.
        """
        if not self._enabled:
            return False

        if self._found_inf == None:
            logger.warning(
                "deriv_scaler.scale() never got called, "
                "please make sure there are no derivative nodes in the model."
            )
            return False

        _found_inf = self._check_found_inf("found_inf")
        return _found_inf.bool().item()

    def _get_found_inf_tensor(self):
        """
        Return the tensor of _found_inf, should only be called when scaler is enabled.
        """
        _found_inf = self._check_found_inf("_get_found_inf_tensor")
        return _found_inf

    def update(self):
        """
        Updates the scale factor.

        If self._found_inf > 0, the scale is multiplied by ``backoff_factor`` to decrease it.
        If there are ``growth_interval`` consecutive unskipped iterations, the scale is
        multiplied by ``growth_factor`` to increase it.
        If the scaling factor is less or equal to the recover_threshold, it will grow more
        frequently (using the recover_growth_interval) to avoid decreasing too small and
        stopping the training process.
        The max_scale defines an upper bound for the scaling factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")
        _found_inf = self._check_found_inf("update")

        torch.ops.physicsnemo_ext._amp_update_scale_(
            _scale,
            _growth_tracker,
            _found_inf,
            self._growth_factor,
            self._backoff_factor,
            self._growth_interval,
            self._max_scale,
            self._recover_threshold,
            self._recover_growth_interval,
        )

        # To prepare for next iteration, reset found_inf states
        self._reset_found_inf()


class DerivScalers(object):
    def __init__(
        self,
        init_scale=2.0**0,
        max_scale=2.0**0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
        recover_threshold=2.0**-6,
        recover_growth_interval=100,
    ):
        """
        A container that holds all the derivative scalers.
        """

        self._enabled = enabled
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._init_scale = init_scale
        self._max_scale = max_scale
        self._recover_threshold = recover_threshold
        self._recover_growth_interval = recover_growth_interval

        self._scalers = {}

    def get_scaler(self, key):
        """
        Return / create a deriv_scaler by key.
        """
        # create a scaler if the key is not present
        if key not in self._scalers:
            # use a custom_max_scale if available
            if key in AmpManager().custom_max_scales:
                max_scale = AmpManager().custom_max_scales[key]
                self._scalers[key] = self._create_scaler(max_scale)
            else:
                self._scalers[key] = self._create_scaler()
        return self._scalers[key]

    def _create_scaler(self, max_scale=None):
        max_scale = self._max_scale if max_scale is None else max_scale
        return DerivScaler(
            init_scale=self._init_scale,
            growth_factor=self._growth_factor,
            backoff_factor=self._backoff_factor,
            growth_interval=self._growth_interval,
            enabled=self._enabled,
            max_scale=max_scale,
            recover_threshold=self._recover_threshold,
            recover_growth_interval=self._recover_growth_interval,
        )

    @property
    def found_inf(self):
        """
        Returns True if any derivative scaler found an inf.
        If this instance is disabled or there are no derivative scalers, return False.

        .. warning::
            :meth:`found_inf` incurs a CPU-GPU sync.

        .. warning::
            :meth:`found_inf` should be called before function `update`, since `update` will reset the
            `_found_inf` state.
        """
        if not self._enabled:
            return False

        if len(self._scalers) == 0:
            logger.warning(
                "There are no scalers created in DerivScalers, "
                "please make sure there are no derivative nodes in the model."
            )
            return False

        found_infs = [s._get_found_inf_tensor() for s in self._scalers.values()]
        found_inf_combined = torch.cat(found_infs).sum()
        return found_inf_combined.bool().item()

    def update(self):
        """
        Updates all the derivative scalers.
        """
        if not self._enabled:
            return

        for s in self._scalers.values():
            s.update()

    def state_dict(self):
        """
        Returns the state dicts of all scalers.
        If this instance is not enabled, returns an empty dict.
        """
        if not self._enabled:
            return {}

        return {key: scaler.state_dict() for key, scaler in self._scalers.items()}

    def load_state_dict(self, state_dict):
        """
        Loads the scalers states.
        If this instance is disabled, :meth:`load_state_dict` is a no-op.
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of DerivScalers."
            )

        for key, state in state_dict.items():
            scaler = self._scalers[key]
            scaler.load_state_dict(state)

    def get_scale(self):
        return {key: scaler.get_scale() for key, scaler in self._scalers.items()}

    def get_max_scale(self):
        return {key: scaler.get_max_scale() for key, scaler in self._scalers.items()}

    def _get_growth_tracker(self):
        return {
            key: scaler._get_growth_tracker() for key, scaler in self._scalers.items()
        }
