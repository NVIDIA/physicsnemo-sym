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

import inspect
import logging
import tarfile
import torch
import numpy as np
import gc
import sys

from typing import Dict, List, Union, Callable, Tuple
from pathlib import Path
from io import BytesIO

from physicsnemo.sym.domain.inferencer import Inferencer
from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.graph import Graph
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.models.arch import Arch
from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.dataset import DictInferencePointwiseDataset

logger = logging.getLogger("__name__")


def _get_function_argspec(func):
    if sys.version_info >= (3, 11):
        sig = inspect.signature(func)  # Use the latest function in 3.11+
        return {
            "args": [
                param
                for param in sig.parameters
                if sig.parameters[param].kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ],
            "varargs": next(
                (
                    param
                    for param in sig.parameters
                    if sig.parameters[param].kind == inspect.Parameter.VAR_POSITIONAL
                ),
                None,
            ),
            "varkw": next(
                (
                    param
                    for param in sig.parameters
                    if sig.parameters[param].kind == inspect.Parameter.VAR_KEYWORD
                ),
                None,
            ),
            "defaults": [
                param.default
                for param in sig.parameters.values()
                if param.default is not param.empty
            ],
            "kwonlyargs": [
                param
                for param in sig.parameters
                if sig.parameters[param].kind == inspect.Parameter.KEYWORD_ONLY
            ],
        }
    else:
        argspec = inspect.getfullargspec(func)  # Backward-compatible approach
        return {
            "args": argspec.args,
            "varargs": argspec.varargs,
            "varkw": argspec.varkw,
            "defaults": argspec.defaults,
            "kwonlyargs": argspec.kwonlyargs,
        }


class OVVoxelInferencer(Inferencer):
    """Voxel inferencer for Omniverse extension.
    Includes some additional utilities for OV inference control.

    Parameters
    ----------
    nodes : List[Node]
        List of PhysicsNeMo Nodes to unroll graph with.
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    mask_value : float, optional
        Value to assign masked points, by default np.nan
    requires_grad : bool, optional
        If automatic differentiation is needed for computing results, by default False
    eco : bool, optional
        Economy mode, will off load model from GPU after inference, by default False
    progress_bar : PhysicsNeMoOVProgressBar, optional
        PhysicsNeMo OV Extension progress bar for displaying inference progress, by default None
    """

    def __init__(
        self,
        nodes: List[Node],
        input_keys: List[Key],
        output_keys: List[Key],
        mask_value: float = np.nan,
        requires_grad: bool = False,
        eco: bool = False,
        progress_bar=None,
    ):
        self.requires_grad = requires_grad
        self._eco = eco
        self.mask_value = mask_value
        self.mask_index = None
        self.input_names = [key.name for key in input_keys]
        self.output_names = [key.name for key in output_keys]
        self.progress_bar = progress_bar

        # construct model from nodes
        self.model = Graph(
            nodes,
            input_keys,
            output_keys,
        )
        self.manager = DistributedManager()
        self.device = self.manager.device

    def setup_voxel_domain(
        self,
        bounds: List[List[int]],
        npoints: List[int],
        invar: Dict[str, np.array] = {},  # Additional inputs
        batch_size: int = 1024,
        mask_fn: Union[Callable, None] = None,
    ) -> None:
        """Set up voxel domain for inference

        Parameters
        ----------
        bounds : List[List[int]]
            List of domain bounds to form uniform rectangular domain
        npoints : List[int]
            Resolution of voxels in each domain
        invar : Dict[str, np.array], optional
            Additional input features, by default {}
        batch_size: int, optional
            Inference batch size, by default 1024
        mask_fn : Union[Callable, None], optional
            Masking function to remove points from inferencing, by default None
        """
        # Start by setting up the
        assert len(bounds) == len(
            npoints
        ), f"Bounds and npoints must be same length {len(bounds)}, {len(npoints)}"
        assert 0 < len(bounds) < 4, "Only 1, 2, 3 grid dimensionality allowed"
        # Pad for missing dimensions
        self.npoints = np.array(npoints + [1, 1])[:3]
        self.bounds = np.array(bounds + [[0, 0], [0, 0]])[:3]

        dx = np.linspace(self.bounds[0][0], self.bounds[0][1], self.npoints[0])
        dy = np.linspace(self.bounds[1][0], self.bounds[1][1], self.npoints[1])
        dz = np.linspace(self.bounds[2][0], self.bounds[2][1], self.npoints[2])

        # Get coordinate arrays (i,j format [x,y,z])
        xx, yy, zz = np.meshgrid(dx, dy, dz, indexing="ij")

        invar.update(
            {
                "x": np.reshape(xx, (-1, 1)),
                "y": np.reshape(yy, (-1, 1)),
                "z": np.reshape(zz, (-1, 1)),
            }
        )

        # If mask set up mask indexes
        if mask_fn is not None:
            args = _get_function_argspec(mask_fn)["args"]
            # Fall back np_lambdify does not supply argument names
            # Ideally np_lambdify should allow input names to be queried
            if len(args) == 0:
                args = list(invar.keys())  # Hope your inputs all go into the mask
            mask_input = {key: invar[key] for key in args if key in invar}
            mask = np.squeeze(mask_fn(**mask_input).astype(np.bool))
            # True points get masked while False get kept, flip for index
            self.mask_index = np.logical_not(mask)
            # Mask out to only masked points (only inference here)
            for key, value in invar.items():
                invar[key] = value[self.mask_index]

        # get dataset and dataloader
        self.dataset = DictInferencePointwiseDataset(
            invar=invar, output_names=self.output_names
        )
        self.dataloader = Constraint.get_dataloader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            distributed=False,
            infinite=False,
        )

    def query(self, memory_fraction: float = 1.0) -> Tuple[Dict[str, np.array]]:
        """Query the inference model

        Parameters
        ----------
        memory_fraction : float, optional
            Fraction of GPU memory to let PyTorch allocate, by default 1.0

        Returns:
            Tuple[Dict[str, np.array]]: Dictionary of input and output arrays
        """
        torch.cuda.set_per_process_memory_fraction(memory_fraction)

        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        predvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Eco mode on/off loads model every query
        if self.eco or not next(self.model.parameters()).is_cuda:
            self.model = self.model.to(self.device)

        # Loop through mini-batches
        for i, (invar0,) in enumerate(self.dataloader):
            # Move data to device
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )

            if self.requires_grad:
                pred_outvar = self.model.forward(invar)
            else:
                with torch.no_grad():
                    pred_outvar = self.model.forward(invar)

            invar_cpu = {key: value + [invar0[key]] for key, value in invar_cpu.items()}
            predvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach().numpy()]
                for key, value in predvar_cpu.items()
            }
            # Update progress bar if provided
            if self.progress_bar:
                self.progress_bar.inference_step(float(i + 1) / len(self.dataloader))

        # Eco mode on/off loads model every query
        if self.eco:
            logger.info("Eco inference on, moving model off GPU")
            self.model.cpu()

        # Concat mini-batch arrays
        invar = {key: np.concatenate(value) for key, value in invar_cpu.items()}
        predvar = {key: np.concatenate(value) for key, value in predvar_cpu.items()}

        # Mask outputs
        invar, predvar = self._mask_results(invar, predvar)

        # Finally reshape back into grid
        for key, value in invar.items():
            shape = list(self.npoints) + [value.shape[1]]
            invar[key] = np.reshape(value, (shape))

        for key, value in predvar.items():
            shape = list(self.npoints) + [value.shape[1]]
            predvar[key] = np.reshape(value, (shape))

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return invar, predvar

    def _mask_results(self, invar, predvar):
        # Reconstruct full array if mask was applied
        for key, value in invar.items():
            full_array = np.full(
                (self.mask_index.shape[0], value.shape[1]),
                self.mask_value,
                dtype=value.dtype,
            )
            full_array[self.mask_index] = value
            invar[key] = full_array
        for key, value in predvar.items():
            full_array = np.full(
                (self.mask_index.shape[0], value.shape[1]),
                self.mask_value,
                dtype=value.dtype,
            )
            full_array[self.mask_index] = value
            predvar[key] = full_array
        return invar, predvar

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        logger.warning(
            "OVoxInferencer is not designed to be used inside of PhysicsNeMo solver"
        )
        pass

    def load_models(self, checkpoint_dir: str):
        logging.info(f"Loading model checkpoint at {checkpoint_dir}")
        for m in self.model.evaluation_order:
            if m.saveable:
                m.load(str(checkpoint_dir))

    @property
    def eco(self):
        return self._eco

    @eco.setter
    def eco(self, e: bool):
        self._eco = e
        if e == False:
            self.model.to(self.device)
        else:
            self.model.cpu()


class OVFourCastNetInferencer(Inferencer):
    """FourCastNet inferencer for Omniverse extension.
    Includes some additional utilities for OV inference control.

    Parameters
    ----------
    afno_model : Union[Arch, torch.nn.Module]
        AFNO model object
    n_channels : int
        Number of input channels / fields
    img_shape : Tuple[int, int], optional
        Input field shape, by default (720, 1440)
    eco : bool, optional
        Economy mode, will off load model from GPU after inference, by default False
    progress_bar : PhysicsNeMoOVProgressBar, optional
        PhysicsNeMo OV Extension progress bar for displaying inference progress, by default None
    """

    def __init__(
        self,
        afno_model: Union[Arch, torch.nn.Module],
        n_channels: int,
        img_shape: Tuple[int, int] = (720, 1440),
        eco: bool = False,
        progress_bar=None,
    ):
        self._eco = eco
        self.n_channels = n_channels
        self.img_shape = img_shape
        self.progress_bar = progress_bar

        self.mu = None
        self.std = None

        # Get PyTorch model out of node if a PhysicsNeMo Node
        if hasattr(afno_model, "_impl"):
            self.model = afno_model._impl
        else:
            self.model = afno_model

        self.manager = DistributedManager()
        self.device = self.manager.device

    def load_initial_state_npy(
        self,
        file_path: str,
        tar_file_path: Union[str, None] = None,
    ) -> None:
        """Loads a FCN initial state into CPU memory stored in a npy file

        Dimensionality of the .npz file should be [in_channels, height, width]

        Parameters
        ----------
        file_path : str
            File path to .npy file
        tar_file_path : Union[str, None], optional
            Optional tar ball with .npy file inside, by default None
        """
        if tar_file_path is None:
            file_path = Path(file_path)
            assert file_path.is_file(), f"Invalid npy file path {file_path}"
            init_np = np.load(file_path)
        else:
            init_np = self.get_array_from_tar(tar_file_path, file_path)

        logger.info(f"Initial condition loaded with shape {init_np.shape}")
        # Run dimension checks
        assert init_np.ndim == 3, f"Initial state should have 3 dimensions"
        assert (
            init_np.shape[0] == self.n_channels
        ), f"Incorrect channel size; expected {self.n_channels}, got {init_np.shape[0]}"
        assert (
            init_np.shape[1] == self.img_shape[0]
            and init_np.shape[2] == self.img_shape[1]
        ), "Incorrect field/image shape"

        self.init_state = torch.Tensor(init_np).unsqueeze(0)

    def load_stats_npz(
        self,
        file_path: str,
        tar_file_path: Union[str, None] = None,
    ) -> None:
        """Loads mean and standard deviation normalization stats from npz file

        Dimensionality of stats in .npz file should be [in_channels, 1, 1]. Npz
        file should have two arrays: "mu" and "std"

        Parameters
        ----------
        file_path : str
            File path to .npz file
        tar_file_path : Union[str, None], optional
            Optional tar ball with .npy file inside, by default None
        """
        if tar_file_path is None:
            file_path = Path(file_path)
            assert file_path.is_file(), f"Invalid npz file path {file_path}"
            stat_npz = np.load(file_path)
        else:
            stat_npz = self.get_array_from_tar(tar_file_path, file_path)

        mu = stat_npz["mu"]
        std = stat_npz["std"]
        logger.info(f"Mu array loaded with shape {mu.shape}")
        logger.info(f"Std array loaded with shape {std.shape}")
        # Run dimension checks
        assert mu.ndim == 3 and std.ndim == 3, f"Mu and Std should have 3 dimensions"
        assert (
            mu.shape[0] == self.n_channels
        ), f"Incorrect channel size; expected {self.n_channels}, got {mu.shape[0]}"
        assert (
            std.shape[0] == self.n_channels
        ), f"Incorrect channel size; expected {self.n_channels}, got {std.shape[0]}"

        self.mu = torch.Tensor(mu).unsqueeze(0)
        self.std = torch.Tensor(std).unsqueeze(0)

    @torch.no_grad()
    def query(self, tsteps: int, memory_fraction: float = 1.0) -> np.array:
        """Query the inference model, only a batch size of 1 is supported

        Parameters
        ----------
        tsteps : int
            Number of timesteps to forecast
        memory_fraction : float, optional
            Fraction of GPU memory to let PyTorch allocate, by default 1.0

        Returns
        -------
        np.array
            [tsteps+1, channels, height, width] output prediction fields
        """
        torch.cuda.set_per_process_memory_fraction(memory_fraction)

        # Create output prediction tensor [Tsteps, C, H, W]
        shape = self.init_state.shape
        outputs = torch.zeros(shape[0] + tsteps, shape[1], shape[2], shape[3])
        outputs[0] = (self.init_state - self.mu) / self.std

        # Eco mode on/off loads model every query
        if self.eco or not next(self.model.parameters()).is_cuda:
            self.model = self.model.to(self.device)

        # Loop through time-steps
        for t in range(tsteps):
            # Get input time-step
            invar = outputs[t : t + 1].to(self.device)
            # Predict
            outvar = self.model.forward(invar)
            # Store
            outputs[t + 1] = outvar[0].detach().cpu()

            # Update progress bar if present
            if self.progress_bar:
                self.progress_bar.inference_step(float(t + 1) / tsteps)

        # Eco mode on/off loads model every query
        if self.eco:
            logger.info("Eco inference on, moving model off GPU")
            self.model.cpu()

        outputs = self.std * outputs + self.mu
        outputs = outputs.numpy()

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return outputs

    def get_array_from_tar(self, tar_file_path: str, np_file_path: str):
        """Loads a numpy array from tar ball, will load entire numpy array into memory

        Based on workaround: https://github.com/numpy/numpy/issues/7989

        Parameters
        ----------
        tar_file_path : str
            File path to tar ball
        np_file_path : str
            Local path of numpy file inside of tar file

        Returns
        -------
        np.array
            Extracted numpy array
        """
        tar_file_path = Path(tar_file_path)
        assert tar_file_path.is_file(), f"Invalid tar file path {tar_file_path}"
        # Open tarball
        with tarfile.open(tar_file_path, "r:gz") as tar:
            logging.info(f"Loaded tar.gz with files:")
            tar.list()

            array_file = BytesIO()
            array_file.write(tar.extractfile(np_file_path).read())
            array_file.seek(0)
        return np.load(array_file)

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        logger.warning(
            "OVFourCastNetInferencer is not designed to be used inside of PhysicsNeMo solver"
        )
        pass

    @property
    def eco(self):
        return self._eco

    @eco.setter
    def eco(self, e: bool):
        self._eco = e
        if e == False:
            self.model.to(self.device)
        else:
            self.model.cpu()
