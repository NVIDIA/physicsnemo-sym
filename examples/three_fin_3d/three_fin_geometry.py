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


from sympy import Symbol


from physicsnemo.sym.geometry import Parameterization
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane

# define sympy varaibles to parametize domain curves
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
x_pos = Symbol("x_pos")

# parametric variation
fin_height_m, fin_height_s = Symbol("fin_height_m"), Symbol("fin_height_s")
fin_length_m, fin_length_s = Symbol("fin_length_m"), Symbol("fin_length_s")
fin_thickness_m, fin_thickness_s = Symbol("fin_thickness_m"), Symbol("fin_thickness_s")
height_m_range = (0.0, 0.6)
height_s_range = (0.0, 0.6)
length_m_range = (0.5, 1.0)
length_s_range = (0.5, 1.0)
thickness_m_range = (0.05, 0.15)
thickness_s_range = (0.05, 0.15)
param_ranges = {
    fin_height_m: height_m_range,
    fin_height_s: height_s_range,
    fin_length_m: length_m_range,
    fin_length_s: length_s_range,
    fin_thickness_m: thickness_m_range,
    fin_thickness_s: thickness_s_range,
}
fixed_param_ranges = {
    fin_height_m: 0.4,
    fin_height_s: 0.4,
    fin_length_m: 1.0,
    fin_length_s: 1.0,
    fin_thickness_m: 0.1,
    fin_thickness_s: 0.1,
}

# geometry params for domain
channel_origin = (-2.5, -0.5, -0.5)
channel_dim = (5.0, 1.0, 1.0)
heat_sink_base_origin = (-1.0, -0.5, -0.3)
heat_sink_base_dim = (1.0, 0.2, 0.6)
fin_origin = (heat_sink_base_origin[0] + 0.5 - fin_length_s / 2, -0.3, -0.3)
fin_dim = (fin_length_s, fin_height_s, fin_thickness_s)  # two side fins
total_fins = 2  # two side fins
flow_box_origin = (-1.1, -0.5, -0.5)
flow_box_dim = (1.6, 1.0, 1.0)
source_origin = (-0.7, -0.5, -0.1)
source_dim = (0.4, 0.0, 0.2)
source_area = 0.08


# define geometry
class ThreeFin(object):
    def __init__(self, parameterized: bool = False):
        # set param ranges
        if parameterized:
            pr = Parameterization(param_ranges)
            self.pr = param_ranges
        else:
            pr = Parameterization(fixed_param_ranges)
            self.pr = fixed_param_ranges

        # channel
        self.channel = Channel(
            channel_origin,
            (
                channel_origin[0] + channel_dim[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            parameterization=pr,
        )

        # three fin heat sink
        heat_sink_base = Box(
            heat_sink_base_origin,
            (
                heat_sink_base_origin[0] + heat_sink_base_dim[0],  # base of heat sink
                heat_sink_base_origin[1] + heat_sink_base_dim[1],
                heat_sink_base_origin[2] + heat_sink_base_dim[2],
            ),
            parameterization=pr,
        )
        (
            fin_origin[0] + fin_dim[0] / 2,
            fin_origin[1] + fin_dim[1] / 2,
            fin_origin[2] + fin_dim[2] / 2,
        )
        fin = Box(
            fin_origin,
            (
                fin_origin[0] + fin_dim[0],
                fin_origin[1] + fin_dim[1],
                fin_origin[2] + fin_dim[2],
            ),
            parameterization=pr,
        )
        gap = (heat_sink_base_dim[2] - fin_dim[2]) / (
            total_fins - 1
        )  # gap between fins
        fin_2 = fin.translate([0, 0, gap])
        fin = fin + fin_2
        three_fin = heat_sink_base + fin

        # parameterized center fin
        center_fin_origin = (
            heat_sink_base_origin[0] + 0.5 - fin_length_m / 2,
            fin_origin[1],
            -fin_thickness_m / 2,
        )
        center_fin_dim = (fin_length_m, fin_height_m, fin_thickness_m)
        center_fin = Box(
            center_fin_origin,
            (
                center_fin_origin[0] + center_fin_dim[0],
                center_fin_origin[1] + center_fin_dim[1],
                center_fin_origin[2] + center_fin_dim[2],
            ),
            parameterization=pr,
        )
        self.three_fin = three_fin + center_fin

        # entire geometry
        self.geo = self.channel - self.three_fin

        # low and high resultion geo away and near the heat sink
        flow_box = Box(
            flow_box_origin,
            (
                flow_box_origin[0] + flow_box_dim[0],  # base of heat sink
                flow_box_origin[1] + flow_box_dim[1],
                flow_box_origin[2] + flow_box_dim[2],
            ),
        )
        self.lr_geo = self.geo - flow_box
        self.hr_geo = self.geo & flow_box
        lr_bounds_x = (channel_origin[0], channel_origin[0] + channel_dim[0])
        lr_bounds_y = (channel_origin[1], channel_origin[1] + channel_dim[1])
        lr_bounds_z = (channel_origin[2], channel_origin[2] + channel_dim[2])
        self.lr_bounds = {x: lr_bounds_x, y: lr_bounds_y, z: lr_bounds_z}
        hr_bounds_x = (flow_box_origin[0], flow_box_origin[0] + flow_box_dim[0])
        hr_bounds_y = (flow_box_origin[1], flow_box_origin[1] + flow_box_dim[1])
        hr_bounds_z = (flow_box_origin[2], flow_box_origin[2] + flow_box_dim[2])
        self.hr_bounds = {x: hr_bounds_x, y: hr_bounds_y, z: hr_bounds_z}

        # inlet and outlet
        self.inlet = Plane(
            channel_origin,
            (
                channel_origin[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            -1,
            parameterization=pr,
        )
        self.outlet = Plane(
            (channel_origin[0] + channel_dim[0], channel_origin[1], channel_origin[2]),
            (
                channel_origin[0] + channel_dim[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            1,
            parameterization=pr,
        )

        # planes for integral continuity
        self.integral_plane = Plane(
            (x_pos, channel_origin[1], channel_origin[2]),
            (
                x_pos,
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            1,
        )
