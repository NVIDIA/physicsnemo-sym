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
import sympy as sp
import numpy as np

from typing import List, Dict, Union

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch

from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.distributed import DistributedManager
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import Constraint
from physicsnemo.sym.domain.validator import GridValidator

from physicsnemo.sym.dataset import DictGridDataset
from physicsnemo.sym.loss import PointwiseLossNorm
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.utils.io import grid_to_vtk

from jhtdb_utils import make_jhtdb_dataset
from ops import FlowOps


class SuperResolutionConstraint(Constraint):
    def __init__(
        self,
        nodes: List[Node],
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        loss_weighting: Dict[str, int],
        dx: float = 1.0,
        lambda_weighting: Dict[str, Union[np.array, sp.Basic]] = None,
        num_workers: int = 0,
    ):
        dataset = DictGridDataset(
            invar=invar, outvar=outvar, lambda_weighting=lambda_weighting
        )
        super().__init__(
            nodes=nodes,
            dataset=dataset,
            loss=PointwiseLossNorm(),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )

        self.dx = dx
        self.ops = FlowOps().to(self.device)

        self.loss_weighting = {}
        self.fields = set("U")
        for key, value in loss_weighting.items():
            if float(value) > 0:
                self.fields = set(key).union(self.fields)
                self.loss_weighting[key] = value

    def calc_flow_stats(self, data_var):
        output = {"U": data_var["U"]}
        vel_output = {}
        cont_output = {}
        vort_output = {}
        enst_output = {}
        strain_output = {}
        # compute derivatives
        if len(self.fields) > 1:
            grad_output = self.ops.get_velocity_grad(
                data_var["U"], dx=self.dx, dy=self.dx, dz=self.dx
            )
        # compute continuity
        if "continuity" in self.fields:
            cont_output = self.ops.get_continuity_residual(grad_output)
        # compute vorticity
        if "omega" in self.fields or "enstrophy" in self.fields:
            vort_output = self.ops.get_vorticity(grad_output)
        # compute enstrophy
        if "enstrophy" in self.fields:
            enst_output = self.ops.get_enstrophy(vort_output)
        # compute strain rate
        if "strain" in self.fields:
            strain_output = self.ops.get_strain_rate_mag(grad_output)

        if "dU" in self.fields:
            # Add to output dictionary
            grad_output = torch.cat(
                [
                    grad_output[key]
                    for key in [
                        "u__x",
                        "u__y",
                        "u__z",
                        "v__x",
                        "v__y",
                        "v__z",
                        "w__x",
                        "w__y",
                        "w__z",
                    ]
                ],
                dim=1,
            )
            vel_output = {"dU": grad_output}

        if "omega" in self.fields:
            vort_output = torch.cat(
                [vort_output[key] for key in ["omega_x", "omega_y", "omega_z"]], dim=1
            )
            vort_output = {"omega": vort_output}

        output.update(vel_output)
        output.update(cont_output)
        output.update(vort_output)
        output.update(enst_output)
        output.update(strain_output)
        return output

    def save_batch(self, filename):
        # sample batch
        invar, true_outvar, lambda_weighting = next(self.dataloader)
        invar0 = {key: value for key, value in invar.items()}
        invar = Constraint._set_device(invar, device=self.device, requires_grad=True)
        true_outvar = Constraint._set_device(true_outvar, device=self.device)

        # compute pred outvar
        if hasattr(self.model, "module"):
            modl = self.model.module
        else:
            modl = self.model
        pred_outvar = modl(invar)
        # Calc flow related stats
        pred_outvar = self.calc_flow_stats(pred_outvar)
        true_outvar = self.calc_flow_stats(true_outvar)

        named_true_outvar = {"true_" + key: value for key, value in true_outvar.items()}
        named_pred_outvar = {"pred_" + key: value for key, value in pred_outvar.items()}
        save_var = {**named_true_outvar, **named_pred_outvar}
        out_save_var = {
            key: value.cpu().detach().numpy() for key, value in save_var.items()
        }
        in_save_var = {
            key: value.cpu().detach().numpy() for key, value in invar0.items()
        }
        # Output both the high-res and low-res fields
        for b in range(min(4, next(iter(invar.values())).shape[0])):
            grid_to_vtk(out_save_var, filename + f"_{b}_hr", batch_index=b)
            grid_to_vtk(in_save_var, filename + f"_{b}_lr", batch_index=b)

    def load_data(self):
        # get lr and high resolution data from dataloader
        invar, target_var, _ = next(self.dataloader)
        self._input_vars = Constraint._set_device(
            invar, device=self.device, requires_grad=False
        )
        self._target_vars = Constraint._set_device(target_var, device=self.device)

    def load_data_static(self):
        if self._input_vars is None:
            # Default loading if vars not allocated
            self.load_data()
        else:
            # get train points from dataloader
            invar, target_vars, _ = next(self.dataloader)
            # Set grads to false here for inputs, static var has allocation already
            invar = Constraint._set_device(
                invar, device=self.device, requires_grad=False
            )
            target_vars = Constraint._set_device(target_vars, device=self.device)

            for key in invar.keys():
                self._input_vars[key].data.copy_(invar[key])
            for key in target_vars.keys():
                self._target_vars[key].copy_(target_vars[key])

    def forward(self):
        # compute forward pass of conv net
        self._pred_outvar = self.model(self._input_vars)

    def loss(self, step: int) -> Dict[str, torch.Tensor]:
        # Calc flow related stats
        pred_outvar = self.calc_flow_stats(self._pred_outvar)
        target_vars = self.calc_flow_stats(self._target_vars)

        # compute losses
        losses = {}
        for key in target_vars.keys():
            mean = (target_vars[key] ** 2).mean()
            losses[key] = (
                self.loss_weighting[key]
                * (((pred_outvar[key] - target_vars[key]) ** 2) / mean).mean()
            )

        return losses


class SuperResolutionValidator(GridValidator):
    def __init__(self, *args, log_iter: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_iter = log_iter
        self.device = DistributedManager().device

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        invar_cpu = {key: [] for key in self.dataset.invar_keys}
        true_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        pred_outvar_cpu = {key: [] for key in self.dataset.outvar_keys}
        # Loop through mini-batches
        for i, (invar0, true_outvar0, lambda_weighting) in enumerate(self.dataloader):
            # Move data to device (may need gradients in future, if so requires_grad=True)
            invar = Constraint._set_device(
                invar0, device=self.device, requires_grad=self.requires_grad
            )
            true_outvar = Constraint._set_device(
                true_outvar0, device=self.device, requires_grad=self.requires_grad
            )
            pred_outvar = self.forward(invar)

            # Collect minibatch info into cpu dictionaries
            invar_cpu = {
                key: value + [invar[key].cpu().detach()]
                for key, value in invar_cpu.items()
            }
            true_outvar_cpu = {
                key: value + [true_outvar[key].cpu().detach()]
                for key, value in true_outvar_cpu.items()
            }
            pred_outvar_cpu = {
                key: value + [pred_outvar[key].cpu().detach()]
                for key, value in pred_outvar_cpu.items()
            }

        # Concat mini-batch tensors
        invar_cpu = {key: torch.cat(value) for key, value in invar_cpu.items()}
        true_outvar_cpu = {
            key: torch.cat(value) for key, value in true_outvar_cpu.items()
        }
        pred_outvar_cpu = {
            key: torch.cat(value) for key, value in pred_outvar_cpu.items()
        }
        # compute losses on cpu
        losses = GridValidator._l2_relative_error(true_outvar_cpu, pred_outvar_cpu)

        # convert to numpy arrays
        invar = {k: v.numpy() for k, v in invar_cpu.items()}
        true_outvar = {k: v.numpy() for k, v in true_outvar_cpu.items()}
        pred_outvar = {k: v.numpy() for k, v in pred_outvar_cpu.items()}

        # save batch to vtk file
        named_target_outvar = {"true_" + k: v for k, v in true_outvar.items()}
        named_pred_outvar = {"pred_" + k: v for k, v in pred_outvar.items()}
        for b in range(min(4, next(iter(invar.values())).shape[0])):
            if self.log_iter:
                grid_to_vtk(
                    {**named_target_outvar, **named_pred_outvar},
                    results_dir + name + f"_{b}_hr" + f"{step:06}",
                    batch_index=b,
                )
            else:
                grid_to_vtk(
                    {**named_target_outvar, **named_pred_outvar},
                    results_dir + name + f"_{b}_hr",
                    batch_index=b,
                )
            grid_to_vtk(invar, results_dir + name + f"_{b}_lr", batch_index=b)

        # add tensorboard plots
        if self.plotter is not None:
            self.plotter._add_figures(
                name,
                results_dir,
                writer,
                step,
                invar,
                true_outvar,
                pred_outvar,
            )

        # add tensorboard scalars
        for k, loss in losses.items():
            writer.add_scalar("val/" + name + "/" + k, loss, step, new_style=True)


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # load jhtdb datasets
    invar, outvar = make_jhtdb_dataset(
        nr_samples=cfg.custom.jhtdb.n_train,
        domain_size=cfg.custom.jhtdb.domain_size,
        lr_factor=cfg.arch.super_res.scaling_factor,
        token=cfg.custom.jhtdb.access_token,
        data_dir=to_absolute_path("datasets/jhtdb_training"),
        time_range=[1, 768],
        dataset_seed=123,
    )

    invar_valid, outvar_valid = make_jhtdb_dataset(
        nr_samples=cfg.custom.jhtdb.n_valid,
        domain_size=cfg.custom.jhtdb.domain_size,
        lr_factor=cfg.arch.super_res.scaling_factor,
        token=cfg.custom.jhtdb.access_token,
        data_dir=to_absolute_path("datasets/jhtdb_valid"),
        time_range=[768, 1024],
        dataset_seed=124,
    )

    model = instantiate_arch(
        input_keys=[Key("U_lr", size=3)],
        output_keys=[Key("U", size=3)],
        cfg=cfg.arch.super_res,
    )
    nodes = [model.make_node(name="super_res")]

    # make super resolution domain
    jhtdb_domain = Domain()

    # make data driven constraint
    jhtdb_constraint = SuperResolutionConstraint(
        nodes=nodes,
        invar=invar,
        outvar=outvar,
        batch_size=cfg.batch_size.train,
        loss_weighting=cfg.custom.loss_weights,
        lambda_weighting=None,
        dx=2 * np.pi / 1024.0,
    )
    jhtdb_domain.add_constraint(jhtdb_constraint, "constraint")

    # make validator
    dataset = DictGridDataset(invar_valid, outvar_valid)
    jhtdb_validator = SuperResolutionValidator(
        dataset=dataset,
        nodes=nodes,
        batch_size=cfg.batch_size.valid,
        log_iter=False,
    )
    jhtdb_domain.add_validator(jhtdb_validator, "validator")

    # make solver
    slv = Solver(
        cfg,
        domain=jhtdb_domain,
    )

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
