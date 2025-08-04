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

import os
import numpy as np
import torch
from torch import nn
from physicsnemo.sym.loss.aggregator import GradNorm


class FitToPoly(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones((512, 512)))
        self.b = nn.Parameter(torch.ones(512, 1))

    def forward(self, x):
        x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        losses = {
            "loss_x": (torch.relu(torch.mm(self.w, x1) + self.b - x1**2)).abs().mean(),
            "loss_y": (torch.relu(torch.mm(self.w, x2) + self.b - x2**2.0))
            .abs()
            .mean(),
            "loss_z": (torch.relu(torch.mm(self.w, x3) + self.b + x3**2.0))
            .abs()
            .mean(),
        }
        return losses


def test_loss_aggregator():
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    filename = os.path.join(
        os.path.dirname(__file__), "test_aggregator_data/GradNorm_data.npz"
    )
    configs = np.load(filename, allow_pickle=True)
    x_np = torch.tensor(configs["x_np"][()]).to(device)
    w_np, b_np, loss_np = (
        configs["w_np"][()],
        configs["b_np"][()],
        configs["loss_np"][()],
    )
    total_steps, learning_rate = (
        configs["total_steps"][()],
        configs["learning_rate"][()],
    )

    # Instantiate the optimizer, scheduler, aggregator, and loss fucntion
    loss_function = torch.jit.script(FitToPoly()).to(device)
    aggregator = GradNorm(loss_function.parameters(), 3)
    optimizer = torch.optim.SGD(loss_function.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # Training loop
    for step in range(total_steps):
        optimizer.zero_grad()
        train_losses = loss_function(x_np)
        train_loss = aggregator(train_losses, step)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

    # check outputs
    w_out = list(loss_function.parameters())[0].cpu().detach().numpy()
    b_out = list(loss_function.parameters())[1].cpu().detach().numpy()
    loss_out = train_loss.cpu().detach().numpy()
    assert np.allclose(loss_np, loss_out, rtol=1e-4, atol=1e-4)
    assert np.allclose(w_np, w_out, rtol=1e-4, atol=1e-4)
    assert np.allclose(b_np, b_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_loss_aggregator()
