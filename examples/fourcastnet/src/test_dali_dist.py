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

from pathlib import Path
import tempfile
from termcolor import cprint

from physicsnemo.sym.distributed import DistributedManager

from src.test_dali_dataset import (
    create_test_data,
    test_distributed_dali_loader,
)


if __name__ == "__main__":
    DistributedManager.initialize()
    m = DistributedManager()
    if not m.distributed:
        print(
            "Please run this test in distributed mode. For example, to run on 2 GPUs:\n\n"
            "mpirun -np 2 python ./src/test_dali_dist.py\n"
        )
        raise SystemExit(1)

    with tempfile.TemporaryDirectory("-data") as data_dir:
        data_path = create_test_data(Path(data_dir))

        test_distributed_dali_loader(data_path)

    cprint("Success!", "green")
