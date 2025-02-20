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

"Script to carry out Fourcastnet inference"

import omegaconf
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, Sampler

from physicsnemo.sym.hydra import to_absolute_path
from physicsnemo.sym.key import Key
from physicsnemo.sym.distributed.manager import DistributedManager

from src.dataset import ERA5HDF5GridDataset
from src.fourcastnet import FourcastNetArch
from src.metrics import Metrics

logging.basicConfig(format="[%(levelname)s] - %(message)s", level=logging.INFO)

var_key_dict = {
    0: "u10",
    1: "v10",
    2: "t2m",
    3: "sp",
    4: "msl",
    5: "t850",
    6: "u1000",
    7: "v1000",
    8: "z1000",
    9: "u850",
    10: "v850",
    11: "z850",
    12: "u500",
    13: "v500",
    14: "z500",
    15: "t500",
    16: "z50",
    17: "r500",
    18: "r850",
    19: "tcwv",
}


def to_device(tensor_dict):
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in tensor_dict.items()
    }


class SubsetSequentialBatchSampler(Sampler):
    """Custom subset sequential batch sampler for inferencer"""

    def __init__(self, subset):
        self.subset = subset

    def __iter__(self):
        for i in self.subset:
            yield [i]  # batch size of 1

    def __len__(self):
        return len(self.subset)


# load configuration
cfg = omegaconf.OmegaConf.load("conf/config_FCN.yaml")
model_path = to_absolute_path("fcn_era5.pth")

# get device
device = DistributedManager().device

# load test data
test_dataset = ERA5HDF5GridDataset(
    cfg.custom.test_data_path,  # Test data location e.g. /era5/20var/test
    chans=list(range(cfg.custom.n_channels)),
    tstep=cfg.custom.tstep,
    n_tsteps=1,  # set to one for inference
    patch_size=cfg.arch.afno.patch_size,
)

m = Metrics(
    test_dataset.img_shape,
    clim_mean_path="/era5/stats/time_means.npy",  # Path to climate mean
    device=device,
)

# define input/output keys
input_keys = [Key(k, size=test_dataset.nchans) for k in test_dataset.invar_keys]
output_keys = [Key(k, size=test_dataset.nchans) for k in test_dataset.outvar_keys]

# create model
model = FourcastNetArch(
    input_keys=input_keys,
    output_keys=output_keys,
    img_shape=test_dataset.img_shape,
    patch_size=cfg.arch.afno.patch_size,
    embed_dim=cfg.arch.afno.embed_dim,
    depth=cfg.arch.afno.depth,
    num_blocks=cfg.arch.afno.num_blocks,
)

# load parameters
model.load_state_dict(torch.load(model_path))
model.to(device)
logging.info(f"Loaded model {model_path}")

# define subsets of dataset to run inference
nics = 180  # Number of 2 day correl time samples
nsteps = 25
last = len(test_dataset) - 1 - nsteps * cfg.custom.tstep

# Variable dictionary
acc_recursive = {key: [] for key in var_key_dict.values()}
rmse_recursive = {key: [] for key in var_key_dict.values()}
# Normalization stats
mu = torch.tensor(test_dataset.mu[0]).to(device)  # shape [C, 1, 1]
sd = torch.tensor(test_dataset.sd[0]).to(device)  # shape [C, 1, 1]

# run inference
with torch.no_grad():
    for ic in range(0, min([8 * nics + 1, last])):
        subset = cfg.custom.tstep * np.arange(nsteps) + ic
        if (ic + 1) % 8 == 0 or (ic + 1) % 36 == 0 or ic == 0:
            logging.info(f"Running IC at step {ic}")
            # get dataloader
            dataloader = DataLoader(
                dataset=test_dataset,
                batch_sampler=SubsetSequentialBatchSampler(subset),
                pin_memory=True,
                num_workers=1,
                worker_init_fn=test_dataset.worker_init_fn,
            )

            acc_error = torch.zeros(nsteps, test_dataset.nchans)
            rmse_error = torch.zeros(nsteps, test_dataset.nchans)
            for tstep, (invar, true_outvar, _) in enumerate(dataloader):
                if tstep % 10 == 0:
                    logging.info(f"ic: {ic} tstep: {tstep}/{nsteps}")

                # place tensors on device
                invar = to_device(invar)
                true_outvar = to_device(true_outvar)
                # 1. single step inference
                pred_outvar_single = model(invar)
                pred_single = sd * pred_outvar_single["x_t1"][0]
                # 2. recursive inference
                if tstep == 0:
                    pred_outvar_recursive = model(invar)
                else:
                    pred_outvar_recursive = model(
                        {"x_t0": pred_outvar_recursive["x_t1"]}
                    )
                # get unormalised target / prediction
                true = sd * true_outvar["x_t1"][0]
                pred_recursive = sd * pred_outvar_recursive["x_t1"][0]
                # Calc metrics
                rmse_error[tstep] = m.weighted_rmse(pred_recursive, true).detach().cpu()
                acc_error[tstep] = m.weighted_acc(pred_recursive, true).detach().cpu()

            # Save fields into dictionary
            if (ic + 1) % 8 == 0 or (ic + 1) % 36 == 0 or ic == 0:
                for i, fld in var_key_dict.items():
                    # Fields with 9 day (36) dc time
                    if fld == "z500" or fld == "t2m" or fld == "t850":
                        if (ic + 1) % 36 == 0 or ic == 0:
                            acc_recursive[fld].append(acc_error[:, i].numpy())
                            rmse_recursive[fld].append(rmse_error[:, i].numpy())
                    # Rest have regular 2 day (8) dc time
                    else:
                        if (ic + 1) % 8 == 0 or ic == 0:
                            acc_recursive[fld].append(acc_error[:, i].numpy())
                            rmse_recursive[fld].append(rmse_error[:, i].numpy())

# Field stacking
for var_dict in [acc_recursive, rmse_recursive]:
    for key, value in var_dict.items():
        print(f"{len(value)} samples for field {key}")
        var_dict[key] = np.stack(value, axis=0)

np.save("rmse_recursive", rmse_recursive)
np.save("acc_recursive", acc_recursive)
