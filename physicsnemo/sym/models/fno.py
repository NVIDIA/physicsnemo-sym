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

from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import logging

from physicsnemo.models.layers import (
    Conv1dFCLayer,
    Conv2dFCLayer,
    Conv3dFCLayer,
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
)
from physicsnemo.models.layers.spectral_layers import (
    calc_latent_derivatives,
    first_order_pino_grads,
    second_order_pino_grads,
)
from physicsnemo.sym.models.activation import Activation, get_activation_fn
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.models.fully_connected import ConvFullyConnectedArch
from physicsnemo.sym.key import Key

logger = logging.getLogger(__name__)


class FNO1DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: Activation = Activation.GELU,
        coord_features: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 1
        self.activation_fn = get_activation_fn(activation_fn)

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Initial lift layer
        self.lift_layer = Conv1dFCLayer(self.in_channels, self.fno_width)

        # Build Neural Fourier Operators
        for _ in range(self.nr_fno_layers):
            self.spconv_layers.append(
                SpectralConv1d(self.fno_width, self.fno_width, fno_modes[0])
            )
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor) -> Tensor:

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (left, right)
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(
                    conv(x) + w(x)
                )  # Spectral Conv + GELU causes JIT issue!
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x


class FNO2DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: Activation = Activation.GELU,
        coord_features: bool = True,
    ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes, fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2
        self.activation_fn = get_activation_fn(activation_fn)

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Initial lift layer
        self.lift_layer = Conv2dFCLayer(self.in_channels, self.fno_width)

        # Build Neural Fourier Operators
        for _ in range(self.nr_fno_layers):
            self.spconv_layers.append(
                SpectralConv2d(
                    self.fno_width, self.fno_width, fno_modes[0], fno_modes[1]
                )
            )
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.dim() == 4
        ), "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[0], 0, self.pad[1]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(
                    conv(x) + w(x)
                )  # Spectral Conv + GELU causes JIT issue!
            else:
                x = conv(x) + w(x)

        # remove padding
        x = x[..., : self.ipad[1], : self.ipad[0]]

        return x

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)


class FNO3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: Activation = Activation.GELU,
        coord_features: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes, fno_modes, fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3
        self.activation_fn = get_activation_fn(activation_fn)

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Initial lift layer
        self.lift_layer = Conv3dFCLayer(self.in_channels, self.fno_width)

        # Build Neural Fourier Operators
        for _ in range(self.nr_fno_layers):
            self.spconv_layers.append(
                SpectralConv3d(
                    self.fno_width,
                    self.fno_width,
                    fno_modes[0],
                    fno_modes[1],
                    fno_modes[2],
                )
            )
            self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor) -> Tensor:

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (left, right, top, bottom, front, back)
        x = F.pad(
            x,
            (0, self.pad[0], 0, self.pad[1], 0, self.pad[2]),
            mode=self.padding_type,
        )
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(
                    conv(x) + w(x)
                )  # Spectral Conv + GELU causes JIT issue!
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[2], : self.ipad[1], : self.ipad[0]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)


def grid_to_points1d(vars_dict: Dict[str, Tensor]):
    for var, value in vars_dict.items():
        value = torch.permute(value, (0, 2, 1))
        vars_dict[var] = value.reshape(-1, value.size(-1))
    return vars_dict


def points_to_grid1d(vars_dict: Dict[str, Tensor], shape: List[int]):
    for var, value in vars_dict.items():
        value = value.reshape(shape[0], shape[2], value.size(-1))
        vars_dict[var] = torch.permute(value, (0, 2, 1))
    return vars_dict


def grid_to_points2d(vars_dict: Dict[str, Tensor]):
    for var, value in vars_dict.items():
        value = torch.permute(value, (0, 2, 3, 1))
        vars_dict[var] = value.reshape(-1, value.size(-1))
    return vars_dict


def points_to_grid2d(vars_dict: Dict[str, Tensor], shape: List[int]):
    for var, value in vars_dict.items():
        value = value.reshape(shape[0], shape[2], shape[3], value.size(-1))
        vars_dict[var] = torch.permute(value, (0, 3, 1, 2))
    return vars_dict


def grid_to_points3d(vars_dict: Dict[str, Tensor]):
    for var, value in vars_dict.items():
        value = torch.permute(value, (0, 2, 3, 4, 1))
        vars_dict[var] = value.reshape(-1, value.size(-1))
    return vars_dict


def points_to_grid3d(vars_dict: Dict[str, Tensor], shape: List[int]):
    for var, value in vars_dict.items():
        value = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
        vars_dict[var] = torch.permute(value, (0, 4, 1, 2, 3))
    return vars_dict


class FNOArch(Arch):
    """Fourier neural operator (FNO) model.

    Note
    ----
    The FNO architecture supports options for 1D, 2D and 3D fields which can
    be controlled using the `dimension` parameter.


    Parameters
    ----------
    input_keys : List[Key]
        Input key list. The key dimension size should equal the variables channel dim.
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    decoder_net : Arch
        Pointwise decoder network, input key should be the latent variable
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    nr_fno_layers : int, optional
        Number of spectral convolution layers, by default 4
    fno_modes : Union[int, List[int]], optional
        Number of Fourier modes with learnable weights, by default 16
    padding : int, optional
        Padding size for FFT calculations, by default 8
    padding_type : str, optional
        Padding type for FFT calculations ('constant', 'reflect', 'replicate'
        or 'circular'), by default "constant"
    activation_fn : Activation, optional
        Activation function, by default Activation.GELU
    coord_features : bool, optional
        Use coordinate meshgrid as additional input feature, by default True

    Variable Shape
    --------------
    Input variable tensor shape:

    - 1D: :math:`[N, size, W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Output variable tensor shape:

    - 1D: :math:`[N, size,  W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Example
    -------
    1D FNO model

    >>> decoder = FullyConnectedArch([Key("z", size=32)], [Key("y", size=2)])
    >>> fno_1d = FNOArch([Key("x", size=2)], dimension=1, decoder_net=decoder)
    >>> model = fno_1d.make_node()
    >>> input = {"x": torch.randn(20, 2, 64)}
    >>> output = model.evaluate(input)

    2D FNO model

    >>> decoder = ConvFullyConnectedArch([Key("z", size=32)], [Key("y", size=2)])
    >>> fno_2d = FNOArch([Key("x", size=2)], dimension=2, decoder_net=decoder)
    >>> model = fno_2d.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64)}
    >>> output = model.evaluate(input)

    3D FNO model

    >>> decoder = Siren([Key("z", size=32)], [Key("y", size=2)])
    >>> fno_3d = FNOArch([Key("x", size=2)], dimension=3, decoder_net=decoder)
    >>> model = fno_3d.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64, 64)}
    >>> output = model.evaluate(input)
    """

    def __init__(
        self,
        input_keys: List[Key],
        dimension: int,
        decoder_net: Arch,
        detach_keys: List[Key] = [],
        nr_fno_layers: int = 4,
        fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn: Activation = Activation.GELU,
        coord_features: bool = True,
    ) -> None:
        super().__init__(input_keys=input_keys, output_keys=[], detach_keys=detach_keys)

        self.dimension = dimension
        self.nr_fno_layers = nr_fno_layers
        self.fno_modes = fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = activation_fn
        self.coord_features = coord_features
        # decoder net
        self.decoder_net = decoder_net

        self.calc_pino_gradients = False
        self.output_keys = self.decoder_net.output_keys
        self.output_key_dict = {str(var): var.size for var in self.output_keys}
        self.output_scales = {str(k): k.scale for k in self.output_keys}

        self.latent_key = self.decoder_net.input_keys
        self.latent_key_dict = {str(var): var.size for var in self.latent_key}
        assert (
            len(self.latent_key) == 1
        ), "FNO decoder network should only have a single input key"
        self.latent_key = str(self.latent_key[0])

        in_channels = sum(self.input_key_dict.values())
        self.fno_layer_size = sum(self.latent_key_dict.values())

        if self.dimension == 1:
            FNOModel = FNO1DEncoder
            self.grid_to_points = grid_to_points1d  # For JIT
            self.points_to_grid = points_to_grid1d  # For JIT
        elif self.dimension == 2:
            FNOModel = FNO2DEncoder
            self.grid_to_points = grid_to_points2d  # For JIT
            self.points_to_grid = points_to_grid2d  # For JIT
        elif self.dimension == 3:
            FNOModel = FNO3DEncoder
            self.grid_to_points = grid_to_points3d  # For JIT
            self.points_to_grid = points_to_grid3d  # For JIT
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D, 2D and 3D FNO implemented"
            )

        self.spec_encoder = FNOModel(
            in_channels,
            nr_fno_layers=self.nr_fno_layers,
            fno_layer_size=self.fno_layer_size,
            fno_modes=self.fno_modes,
            padding=self.padding,
            padding_type=self.padding_type,
            activation_fn=self.activation_fn,
            coord_features=self.coord_features,
        )

    def add_pino_gradients(
        self, derivatives: List[Key], domain_length: List[float] = [1.0, 1.0]
    ) -> None:
        """Adds PINO "exact" gradient calculations model outputs.

        Note
        ----
        This will constraint the FNO decoder to a two layer fully-connected model with
        Tanh activactions functions. This is done for computational efficiency since
        gradients calculations are explicit. Auto-diff is far too slow for this method.

        Parameters
        ----------
        derivatives : List[Key]
            List of derivative keys
        domain_length : List[float], optional
            Domain size of input grid. Needed for calculating the gradients of the latent
            variables. By default [1.0, 1.0]

        Raises
        ------
        ValueError
            If domain length list is not the same size as the FNO model dimenion

        Note
        ----
        For details on the "exact" gradient calculation refer to section 3.3 in:
        https://arxiv.org/pdf/2111.03794.pdf
        """
        assert (
            len(domain_length) == self.dimension
        ), "Domain length must be same length as the dimension of the model"
        self.domain_length = domain_length

        logger.warning(
            "Switching decoder to two layer FC model with Tanh activations for PINO"
        )
        self.decoder_net = ConvFullyConnectedArch(
            input_keys=self.decoder_net.input_keys,
            output_keys=self.decoder_net.output_keys,
            layer_size=self.fno_layer_size,
            nr_layers=1,
            activation_fn=Activation.TANH,
            skip_connections=False,
            adaptive_activations=False,
        )

        self.calc_pino_gradients = True
        self.first_order_pino = False
        self.second_order_pino = False
        self.derivative_keys = []

        for var in derivatives:
            dx_name = str(var).split("__")  # Split name to get original var names
            if len(dx_name) == 2:  # First order
                assert (
                    dx_name[1] in ["x", "y", "z"][: self.dimension]
                ), f"Invalid first-order derivative {str(var)} for {self.dimension}d FNO"
                self.derivative_keys.append(var)
                self.first_order_pino = True
            elif len(dx_name) == 3:
                assert (
                    dx_name[1] in ["x", "y", "z"][: self.dimension]
                    and dx_name[1] == dx_name[2]
                ), f"Invalid second-order derivative {str(var)} for {self.dimension}d FNO"
                self.derivative_keys.append(var)
                self.second_order_pino = True
            elif len(dx_name) > 3:
                raise ValueError(
                    "FNO only supports first order and laplacian second order derivatives"
                )

        # Add derivative keys into output keys
        self.output_keys_fno = self.output_keys.copy()
        self.output_key_fno_dict = {str(var): var.size for var in self.output_keys_fno}
        self.output_keys = self.output_keys + self.derivative_keys
        self.output_key_dict = {str(var): var.size for var in self.output_keys}

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=1,
            input_scales=self.input_scales,
        )
        y_latent = self.spec_encoder(x)

        y_shape = list(y_latent.size())
        y_input = {self.latent_key: y_latent}
        # Reshape to pointwise inputs if not a conv FC model
        if self.decoder_net.var_dim == -1:
            y_input = self.grid_to_points(y_input)

        y = self.decoder_net(y_input)
        # Convert back into grid
        if self.decoder_net.var_dim == -1:
            y = self.points_to_grid(y, y_shape)

        if self.calc_pino_gradients:
            output_grads = self.calc_pino_derivatives(y_latent)
            y.update(output_grads)

        return y

    @torch.jit.ignore
    def calc_pino_derivatives(self, latent: Tensor) -> Dict[str, Tensor]:
        # Calculate the gradients of latent variables
        # This is done using FFT and is the reason we need a domain size
        lat_dx, lat_ddx = calc_latent_derivatives(latent, self.domain_length)

        # Get weight matrices from decoder
        weights, biases = self.decoder_net._impl.get_weight_list()
        outputs = {}
        # calc first order derivatives
        if self.first_order_pino:
            output_dx = first_order_pino_grads(
                u=latent,
                ux=lat_dx,
                weights_1=weights[0],
                weights_2=weights[1],
                bias_1=biases[0],
            )
            # Build output dictionary manually (would normally use prepare_output)
            dims = ["x", "y", "z"]
            for d in range(len(output_dx)):  # Loop through dimensions
                for k, v in zip(
                    self.output_keys_fno,
                    torch.split(
                        output_dx[d], list(self.output_key_fno_dict.values()), dim=1
                    ),
                ):  # Loop through variables
                    if f"{k}__{dims[d]}__{dims[d]}" in self.output_key_dict.keys():
                        out_scale = self.decoder_net.output_scales[str(k)][
                            1
                        ]  # Apply out scaling to grads
                        outputs[f"{k}__{dims[d]}"] = v * out_scale

        # calc first order derivatives
        if self.second_order_pino:
            output_dxx = second_order_pino_grads(
                u=latent,
                ux=lat_dx,
                uxx=lat_ddx,
                weights_1=weights[0],
                weights_2=weights[1],
                bias_1=biases[0],
            )

            # Build output dictionary manually (would normally use prepare_output)
            dims = ["x", "y", "z"]
            for d in range(len(output_dxx)):  # Loop through dimensions
                for k, v in zip(
                    self.output_keys_fno,
                    torch.split(
                        output_dxx[d], list(self.output_key_fno_dict.values()), dim=1
                    ),
                ):  # Loop through variables
                    if f"{k}__{dims[d]}__{dims[d]}" in self.output_key_dict.keys():
                        out_scale = self.decoder_net.output_scales[str(k)][
                            1
                        ]  # Apply out scaling to grads
                        outputs[f"{k}__{dims[d]}__{dims[d]}"] = v * out_scale

        return outputs
