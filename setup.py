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

import glob
from setuptools import setup

import re
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


def cuda_extension() -> CUDAExtension:
    # Parse CUDA version string assuming SemVer (handles formats like "12.8-rc.2", "12.10", etc.)
    if torch.version.cuda:
        version_pattern = re.match(
            r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", torch.version.cuda
        )
    else:  # Occurs on CPU-only PyTorch installations
        raise RuntimeError(f"CUDA version not found: {torch.version.cuda=}")
    if not version_pattern:
        raise RuntimeError(f"Unable to parse CUDA version: {torch.version.cuda=}")

    # Extract components, defaulting to 0 if not present
    cuda_major = int(version_pattern.group(1))
    cuda_minor = int(version_pattern.group(2) or 0)

    nvcc_args = [
        "-gencode=arch=compute_75,code=sm_75",
    ]
    if cuda_major >= 11:
        nvcc_args.append("-gencode=arch=compute_80,code=sm_80")
    if (cuda_major == 11 and cuda_minor >= 1) or (cuda_major > 11):
        nvcc_args.append("-gencode=arch=compute_86,code=sm_86")
    if cuda_major >= 12:
        nvcc_args.append("-gencode=arch=compute_90,code=sm_90")
    if (cuda_major == 12 and cuda_minor >= 8) or cuda_major >= 13:
        nvcc_args.append("-gencode=arch=compute_100,code=sm_100")

    nvcc_args.append("-t=0")  # Enable multi-threaded builds
    # nvcc_args.append("--time=output.txt")

    return CUDAExtension(
        name="physicsnemo.sym.physicsnemo_ext",
        sources=glob.glob("physicsnemo/sym/csrc/*.cu"),
        extra_compile_args={"cxx": ["-std=c++14"], "nvcc": nvcc_args},
    )


setup(
    ext_modules=[cuda_extension()],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
