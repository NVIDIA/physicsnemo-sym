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
import pytest
import torch.nn as nn

from physicsnemo.sym.amp import DerivScaler, AmpManager
from physicsnemo.sym.eq.derivatives import gradient_autodiff
from physicsnemo.sym import physicsnemo_ext  # noqa: F401


Tensor = torch.Tensor
skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="There is no GPU to run this test"
)
# ensure torch.rand() is deterministic
_ = torch.manual_seed(0)


@skip_if_no_gpu
def test_run_deriv_scaler():
    assert torch.cuda.is_available()
    device = "cuda:0"
    deriv_scaler = DerivScaler(
        init_scale=2.0**0,
        max_scale=2.0**1,
        growth_interval=2,
        recover_threshold=2.0**-1,
        recover_growth_interval=1,
    )

    x = torch.ones((100, 1), device=device)
    y = torch.ones((100, 1), device=device)
    # model: x, y -> u
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
    ).to(device)

    def run(x, y):
        x = x.detach().clone().requires_grad_()
        y = y.detach().clone().requires_grad_()
        # forward with amp autocast
        with torch.cuda.amp.autocast():
            u = model(torch.cat([x, y], dim=-1))
            # first order derivatives: u__x and u__y
            u_scale = deriv_scaler.scale(u)
            grad = gradient_autodiff(u_scale, [x, y])
            u__x, u__y = deriv_scaler.unscale_deriv(grad)
            # loss
            loss = u.sum() + u__x.sum() + u__y.sum()
        # backward
        loss.backward()
        if not deriv_scaler.found_inf:
            # scaler.step(self.optimizer)
            # scaler.update()
            pass
        deriv_scaler.update()

    # init state
    assert deriv_scaler.get_scale() == 2.0**0
    assert deriv_scaler._get_growth_tracker() == 0

    # simulates 2 consecutive unskipped iteration
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**0
    assert deriv_scaler._get_growth_tracker() == 1
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**1
    assert deriv_scaler._get_growth_tracker() == 0

    # simulates reaching to max_scale
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**1
    assert deriv_scaler._get_growth_tracker() == 1
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**1
    assert deriv_scaler._get_growth_tracker() == 0

    # simulates 3 consecutive skipped iteration
    x[0][0] = torch.nan  # inject an NAN
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**0
    assert deriv_scaler._get_growth_tracker() == 0
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**-1
    assert deriv_scaler._get_growth_tracker() == 0
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**-2
    assert deriv_scaler._get_growth_tracker() == 0

    # simulates reaching to recover threashold
    # scaler will be update more frequently
    x[0][0] = 1  # remove the NAN
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**-1
    assert deriv_scaler._get_growth_tracker() == 0
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**0
    assert deriv_scaler._get_growth_tracker() == 0
    # no longer in threshold range
    run(x, y)
    assert deriv_scaler.get_scale() == 2.0**0
    assert deriv_scaler._get_growth_tracker() == 1


@skip_if_no_gpu
def test_cuda_graph_deriv_scaler():
    assert torch.cuda.is_available()
    device = "cuda:0"
    deriv_scaler = DerivScaler(init_scale=2**0)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()

    static_x = torch.ones((100, 1), device=device)
    static_y = torch.ones((100, 1), device=device)
    # model: x, y -> u
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
    ).to(device)

    def run():
        x = static_x.detach().clone().requires_grad_()
        y = static_y.detach().clone().requires_grad_()
        # forward with amp autocast
        with torch.cuda.amp.autocast():
            u = model(torch.cat([x, y], dim=-1))
            # first order derivatives: u__x and u__y
            u_scale = deriv_scaler.scale(u)
            grad = gradient_autodiff(u_scale, [x, y])
            u__x, u__y = deriv_scaler.unscale_deriv(grad)
            # loss
            loss = u.sum() + u__x.sum() + u__y.sum()
        # backward
        loss.backward()

    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        # warmup
        run()
        # capture
        g.capture_begin()
        run()
        g.capture_end()
    torch.cuda.current_stream().wait_stream(s)

    # real data
    real_x = [torch.rand_like(static_x) for _ in range(10)]
    real_y = [torch.rand_like(static_y) for _ in range(10)]
    # inject a nan for step 4
    real_x[3][0] = torch.nan

    # train
    for i, (x, y) in enumerate(zip(real_x, real_y)):
        # Fills the graph's input memory with new data to compute on
        static_x.copy_(x)
        static_y.copy_(y)
        # replay() includes forward, backward
        g.replay()
        # deriv_scaler step and update
        if not deriv_scaler.found_inf:
            # scaler.step(self.optimizer)
            # scaler.update()
            pass
        # manually injected nan should be detected
        if i == 3:
            assert deriv_scaler.found_inf
        deriv_scaler.update()


def test_amp_manager():
    manager = AmpManager()

    # register a special_term
    manager.register_special_term("nu", 2**10)
    assert "nu" in manager.special_terms
    assert "nu" in manager.custom_max_scales
    assert manager.custom_max_scales["nu"] == 2**10

    # register one without max_scale
    manager.register_special_term("m")
    assert "m" in manager.special_terms
    assert "m" not in manager.custom_max_scales

    # scaler_enbaled
    manager.enabled = False
    assert not manager.scaler_enabled

    # scaler is not enabled for bfloat16
    manager.enabled = True
    manager.dtype = "bfloat16"
    assert not manager.scaler_enabled


@skip_if_no_gpu
def test_physicsnemo_ext_amp_update_scale():
    # modified from https://github.com/pytorch/pytorch/blob/release/1.11/test/test_cuda.py#L2079
    device = "cuda:0"
    growth = 2.0
    backoff = 0.25
    growth_interval = 2
    max_scale = 2.0**3
    recover_threshold = 2.0**1
    recover_growth_interval = 1
    scale = torch.full((1,), 4.0, dtype=torch.float, device=device)
    growth_tracker = torch.full((1,), 0.0, dtype=torch.int32, device=device)
    found_inf = torch.full((1,), 0.0, dtype=torch.float, device="cuda:0")

    def amp_update_scale():
        torch.ops.physicsnemo_ext._amp_update_scale_(
            scale,
            growth_tracker,
            found_inf,
            growth,
            backoff,
            growth_interval,
            max_scale,
            recover_threshold,
            recover_growth_interval,
        )

    # Simulates 2 consecutive unskipped iterations
    amp_update_scale()
    assert growth_tracker.item() == 1
    assert scale.item() == 2.0**2
    amp_update_scale()
    assert growth_tracker.item() == 0
    assert scale.item() == 2.0**3

    # Simulates reaching to max_scale upper bound
    for i in range(4):
        amp_update_scale()
        assert growth_tracker.item() == (i + 1) % 2
        assert scale.item() == 2.0**3

    # Simulates 2 consecutive skipped iteration
    found_inf.fill_(1.0)
    amp_update_scale()
    assert growth_tracker.item() == 0
    assert scale.item() == 2.0**1
    amp_update_scale()
    assert growth_tracker.item() == 0
    assert scale.item() == 2.0**-1

    # Simulates reaching to recover_threshold, recover_growth_interval is 1
    found_inf.fill_(0.0)
    amp_update_scale()
    assert growth_tracker.item() == 0
    assert scale.item() == 2.0**0
    amp_update_scale()
    assert growth_tracker.item() == 0
    assert scale.item() == 2.0**1
    amp_update_scale()
    assert growth_tracker.item() == 0
    assert scale.item() == 2.0**2

    # no longer in threshold range
    amp_update_scale()
    assert growth_tracker.item() == 1
    assert scale.item() == 2.0**2


def test_scaler_state_dict():
    scaler1 = DerivScaler(
        init_scale=2.0**8,
        max_scale=2.0**30,
        recover_threshold=2.0**-10,
        recover_growth_interval=1,
    )
    state = scaler1.state_dict()

    scaler2 = DerivScaler()
    scaler2.load_state_dict(state)
    assert scaler2._init_scale == 2.0**8
    assert scaler2._max_scale == 2.0**30
    assert scaler2._recover_threshold == 2.0**-10
    assert scaler2._recover_growth_interval == 1


if __name__ == "__main__":
    test_run_deriv_scaler()
    test_cuda_graph_deriv_scaler()
    test_amp_manager()
    test_physicsnemo_ext_amp_update_scale()
    test_scaler_state_dict()
