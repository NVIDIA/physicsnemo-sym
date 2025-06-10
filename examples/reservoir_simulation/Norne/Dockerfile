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

# Use the NVIDIA PhysicsNeMo image as the base image
FROM nvcr.io/nvidia/physicsnemo/physicsnemo:23.09

# Clone and build MPSlib
RUN cd /workspace && git clone https://github.com/ergosimulation/mpslib.git MPSlib && cd MPSlib && make -j 

# Install Python packages
RUN pip3 install tensorflow==2.9.1 xlsxwriter PyWavelets scikit-mps kneed pyDOE FyeldGenerator py-cpuinfo gdown gstools scikit-image ema-pytorch accelerate loky xgboost scikit-learn pandas xlsxwriter openpyxl

# Set the working directory
WORKDIR /workspace/project

COPY set_env.sh /opt/nvidia/set_env.sh

ENTRYPOINT ["/opt/nvidia/set_env.sh"]
