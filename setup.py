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

import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


def cuda_extension():
    cuda_version = float(torch.version.cuda)
    nvcc_args = [
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
    ]
    if cuda_version >= 11:
        nvcc_args.append("-gencode=arch=compute_80,code=sm_80")
    if cuda_version >= 11.1:
        nvcc_args.append("-gencode=arch=compute_86,code=sm_86")
    if cuda_version >= 12:
        nvcc_args.append("-gencode=arch=compute_90,code=sm_90")

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
