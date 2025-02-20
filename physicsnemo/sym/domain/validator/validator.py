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


class Validator:
    """
    Validator base class
    """

    def forward_grad(self, invar):
        pred_outvar = self.model(invar)
        return pred_outvar

    def forward_nograd(self, invar):
        with torch.no_grad():
            pred_outvar = self.model(invar)
        return pred_outvar

    def save_results(self, name, results_dir, writer, save_filetypes, step):
        raise NotImplementedError("Subclass of Validator needs to implement this")

    @staticmethod
    def _l2_relative_error(true_var, pred_var):  # TODO replace with metric classes
        new_var = {}
        for key in true_var.keys():
            new_var["l2_relative_error_" + str(key)] = torch.sqrt(
                torch.mean(torch.square(true_var[key] - pred_var[key]))
                / torch.var(true_var[key])
            )
        return new_var
