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

import GPUtil
import os
import pytest
import argparse
from pytest import ExitCode
from termcolor import colored

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testdir", default=".")
    args = parser.parse_args()

    os.system("nvidia-smi")
    availible_gpus = GPUtil.getAvailable(limit=8)
    if len(availible_gpus) == 0:
        print(colored("No free GPUs found on DGX 4850", "red"))
        raise RuntimeError()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(availible_gpus[-1])
        print(colored(f"=== Using GPU {availible_gpus[-1]} ===", "blue"))

    retcode = pytest.main(["-x", args.testdir])

    if ExitCode.OK == retcode:
        print(colored("UNIT TESTS PASSED! :D", "green"))
    else:
        print(colored("UNIT TESTS FAILED!", "red"))
        raise ValueError(f"Pytest returned error code {retcode}")
