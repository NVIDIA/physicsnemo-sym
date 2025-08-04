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

"""
Code modified from https://github.com/akshaysubr/TEGAN

The following license is provided from their source,

Copyright 2020 Akshay Subramaniam, Man-Long Wong, Raunak Borker, Sravya Nimmagadda

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class FlowOps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "ddx1D",
            torch.Tensor(
                [
                    -1.0 / 60.0,
                    3.0 / 20.0,
                    -3.0 / 4.0,
                    0.0,
                    3.0 / 4.0,
                    -3.0 / 20.0,
                    1.0 / 60.0,
                ]
            ),
        )

    def ddx(self, inpt, dx, channel, dim, padding_mode="replicate"):
        var = inpt[:, channel : channel + 1, :, :, :]
        ddx3D = torch.reshape(
            self.ddx1D, shape=[1, 1] + dim * [1] + [-1] + (2 - dim) * [1]
        )
        padding = _triple(3) + _triple(3)

        output = F.conv3d(
            F.pad(var, padding, mode=padding_mode),
            ddx3D,
            stride=1,
            padding=0,
            bias=None,
        )
        output = (1.0 / dx) * output

        if dim == 0:
            output = output[
                :,
                :,
                :,
                (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
            ]
        elif dim == 1:
            output = output[
                :,
                :,
                (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                :,
                (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
            ]
        elif dim == 2:
            output = output[
                :,
                :,
                (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                (self.ddx1D.shape[0] - 1) // 2 : -(self.ddx1D.shape[0] - 1) // 2,
                :,
            ]
        return output

    def get_velocity_grad(self, inpt, dx, dy, dz, xdim=0):
        output = {}
        output["u__x"] = self.ddx(inpt, dx, channel=0, dim=xdim)
        output["u__y"] = self.ddx(inpt, dy, channel=0, dim=1)
        output["u__z"] = self.ddx(inpt, dz, channel=0, dim=(2 - xdim))

        output["v__x"] = self.ddx(inpt, dx, channel=1, dim=xdim)
        output["v__y"] = self.ddx(inpt, dy, channel=1, dim=1)
        output["v__z"] = self.ddx(inpt, dz, channel=1, dim=(2 - xdim))

        output["w__x"] = self.ddx(inpt, dx, channel=2, dim=xdim)
        output["w__y"] = self.ddx(inpt, dy, channel=2, dim=1)
        output["w__z"] = self.ddx(inpt, dz, channel=2, dim=(2 - xdim))

        return output

    def get_strain_rate_mag(self, vel_dict):
        output = {}
        output["strain"] = (
            vel_dict["u__x"] ** 2
            + vel_dict["v__y"] ** 2
            + vel_dict["w__z"] ** 2
            + 2
            * (
                (0.5 * (vel_dict["u__y"] + vel_dict["v__x"])) ** 2
                + (0.5 * (vel_dict["u__z"] + vel_dict["w__x"])) ** 2
                + (0.5 * (vel_dict["w__y"] + vel_dict["v__z"])) ** 2
            )
        )
        return output

    def get_vorticity(self, vel_dict):
        output = {}
        output["omega_x"] = vel_dict["w__y"] - vel_dict["v__z"]
        output["omega_y"] = vel_dict["u__z"] - vel_dict["w__x"]
        output["omega_z"] = vel_dict["v__x"] - vel_dict["u__y"]
        return output

    def get_enstrophy(self, vort_dict):
        output = {}
        output["enstrophy"] = (
            vort_dict["omega_x"] ** 2
            + vort_dict["omega_y"] ** 2
            + vort_dict["omega_z"] ** 2
        )
        return output

    def get_continuity_residual(self, vel_dict):
        output = {}
        output["continuity"] = vel_dict["u__x"] + vel_dict["v__y"] + vel_dict["w__z"]
        return output

    # def d2dx2(inpt, channel, dx, name=None):
    #     var = inpt[:, channel : channel + 1, :, :, :]
    #     ddx1D = tf.constant(
    #         [
    #             1.0 / 90.0,
    #             -3.0 / 20.0,
    #             3.0 / 2.0,
    #             -49.0 / 18.0,
    #             3.0 / 2.0,
    #             -3.0 / 20.0,
    #             1.0 / 90.0,
    #         ]
    #     ).to(inpt.device)
    #     ddx3D = torch.reshape(ddx1D, shape=[1, 1] + dim * [1] + [-1] + (2 - dim) * [1])
    #     output = F.conv3d(var, ddx3D, padding="valid")
    #     output = (1.0 / dx ** 2) * output
    #     return output

    # def get_TKE(inpt):
    #     TKE = torch.square(inpt[:, 0, :, :, :])
    #     TKE = TKE + tf.square(inpt[:, 1, :, :, :])
    #     TKE = TKE + tf.square(inpt[:, 2, :, :, :])
    #     TKE = 0.5 * TKE
    #     TKE = tf.expand_dims(TKE, axis=1)
    #     return TKE
