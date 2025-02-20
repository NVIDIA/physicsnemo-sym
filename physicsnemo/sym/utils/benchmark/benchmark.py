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

import time
from typing import Any, Callable, Optional
import torch
from torch.profiler import record_function, ProfilerActivity


def timeit(
    func: Callable,
    *args,
    steps: int = 100,
    warmup: int = 10,
    run_profile: bool = False,
    verbose: bool = True,
    label: Optional[str] = None,
    label_padding: int = 35,
    cpu_timing: bool = False,
):
    """
    Returns time/step in ms.
    If run_profile is True, then return (time/step in ms, a captured cuda events table)
    """
    if label is None:
        assert func.__name__, "please provide a label for this benchmark"
        label = func.__name__

    # warmup
    torch.cuda.nvtx.range_push(f"{label}_warmup")
    for _ in range(warmup):
        func(*args)
    torch.cuda.nvtx.range_pop()  # pop label_warmup

    # start timer
    if cpu_timing:
        torch.cuda.synchronize()
        start = time.time()
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    torch.cuda.nvtx.range_push(f"{label}")
    if run_profile:
        if verbose:
            print("\n" + "=" * 70 + " " + label + " " + "=" * 70)
        with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
            with record_function("run_total"):
                for i in range(steps):
                    torch.cuda.nvtx.range_push(f"{i}th_iteration")
                    func(*args)
                    torch.cuda.nvtx.range_pop()
        events = prof.key_averages()
        if verbose:
            print(
                events.table(
                    sort_by="self_cuda_time_total",
                    max_src_column_width=200,
                    row_limit=15,
                )
            )
    else:
        events = None
        for i in range(steps):
            torch.cuda.nvtx.range_push(f"{i}th_iteration")
            func(*args)
            torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()  # pop label

    # stop timer
    if cpu_timing:
        torch.cuda.synchronize()
        time_ms = ((time.time() - start) / steps) * 1000
    else:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        time_ms = start_event.elapsed_time(end_event) / steps

    if verbose:
        print(f"{label.ljust(label_padding)}: {time_ms:.3f} ms/step")

    if run_profile:
        return time_ms, events
    else:
        return time_ms


def profile(
    func: Callable,
    *args,
    **kwargs,
):
    """
    Simply a convenient wrapper of the timeit function with run_profile=True.

    Returns: (time/step in ms, a captured cuda events table)
    """
    return timeit(func, *args, run_profile=True, **kwargs)
