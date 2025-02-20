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

import physicsnemo
from physicsnemo.sym.key import Key
from physicsnemo.sym.hydra import to_yaml, instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.models.afno.distributed import DistributedAFNONet
from physicsnemo.sym.distributed.manager import DistributedManager

import os
import torch

# Set model parallel size to 2
os.environ["MODEL_PARALLEL_SIZE"] = "2"


@physicsnemo.sym.main(config_path="conf", config_name="config_AFNO")
def run(cfg: PhysicsNeMoConfig) -> None:
    manager = DistributedManager()
    model_rank = manager.group_rank(name="model_parallel")
    model_size = manager.group_size(name="model_parallel")

    # Check that GPUs are available
    if not manager.cuda:
        print("WARNING: No GPUs available. Exiting...")
        return
    # Check that world_size is a multiple of model parallel size
    if manager.world_size % 2 != 0:
        print(
            "WARNING: Total world size not a multiple of model parallel size (2). Exiting..."
        )
        return

    input_keys = [Key("coeff", scale=(7.48360e00, 4.49996e00))]
    output_keys = [Key("sol", scale=(5.74634e-03, 3.88433e-03))]
    img_shape = (720, 1440)

    # make list of nodes to unroll graph on
    model = instantiate_arch(
        input_keys=input_keys,
        output_keys=output_keys,
        cfg=cfg.arch.distributed_afno,
        img_shape=img_shape,
    )
    nodes = [model.make_node(name="Distributed AFNO", jit=cfg.jit)]

    model = model.to(manager.device)
    sample = {
        str(k): torch.randn(1, k.size, *img_shape).to(manager.device)
        for k in input_keys
    }

    # Run model in a loop
    for i in range(4):
        # Forward pass
        result = model(sample)
        # Compute loss
        loss = torch.square(result["sol"]).sum()
        # Backward pass
        loss.backward()

    expected_result_shape = [1, output_keys[0].size, *img_shape]
    result_shape = list(result["sol"].shape)
    assert (
        result_shape == expected_result_shape
    ), f"Incorrect result size. Expected {expected_result_shape}, got {local_result_shape}"


if __name__ == "__main__":
    run()
