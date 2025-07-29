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

from tflogs_reader import check_validation_error

if __name__ == "__main__":
    check_validation_error(
        "../helmholtz/outputs/helmholtz/",
        threshold=0.3,
        save_path="./checks/helmholtz/",
    )
    check_validation_error(
        "../discontinuous_galerkin/dg/outputs/dg/",
        threshold=0.3,
        save_path="./checks/dg/",
    )
    check_validation_error(
        "../anti_derivative/outputs/physics_informed/",
        threshold=0.3,
        save_path="./checks/physics_informed/",
    )
