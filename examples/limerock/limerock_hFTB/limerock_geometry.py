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

# import PhysicsNeMo library
from sympy import Symbol
import numpy as np
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.geometry import Parameterization, Parameter


class LimeRock(object):
    def __init__(self):
        # scale STL
        self.scale = 5 / 0.3
        self.translate = (0, 0, -0.055)

        # set fins
        self.nr_fins = 47
        self.fin_gap = 0.0018

        # make solids
        self.copper = None

        # parse file
        print("parsing stl file...")
        self._parse_file("../stl_files/limerock.stl")
        print("finished parsing")

        # inlet area
        self.inlet_area = (self.geo_bounds_upper[1] - self.geo_bounds_lower[1]) * (
            self.geo_bounds_upper[2] - self.geo_bounds_lower[2]
        )

        # geo
        self.heat_sink_bounds = (-0.7, 0.7)
        self.geo = self.channel - self.copper
        self.geo_solid = self.copper
        self.geo_bounds = {
            Symbol("x"): (self.geo_bounds_lower[0], self.geo_bounds_upper[0]),
            Symbol("y"): (self.geo_bounds_lower[1], self.geo_bounds_upper[1]),
            Symbol("z"): (self.geo_bounds_lower[2], self.geo_bounds_upper[2]),
        }
        self.geo_hr_bounds = {
            Symbol("x"): self.heat_sink_bounds,
            Symbol("y"): (self.geo_bounds_lower[1], self.geo_bounds_upper[1]),
            Symbol("z"): (self.geo_bounds_lower[2], self.geo_bounds_upper[2]),
        }

        # integral plane
        x_pos = Parameter("x_pos")
        self.integral_plane = Plane(
            (x_pos, self.geo_bounds_lower[1], self.geo_bounds_lower[2]),
            (x_pos, self.geo_bounds_upper[1], self.geo_bounds_upper[2]),
            1,
            parameterization=Parameterization({x_pos: self.heat_sink_bounds}),
        )

    def solid_names(self):
        return list(self.solids.keys())

    def _parse_file(self, filename):
        # Read file
        reader = open(filename)
        while True:
            line = reader.readline()
            if "solid" == line.split(" ")[0]:
                solid_name = line.split(" ")[-1].rstrip()
                bounds_lower, bounds_upper = self.read_solid(reader)
                if solid_name == "opening.1":
                    self.inlet = Plane(bounds_lower, bounds_upper, -1)
                    self.geo_bounds_lower = bounds_lower
                elif solid_name == "fan.1":
                    self.outlet = Plane(bounds_lower, bounds_upper, 1)
                    self.geo_bounds_upper = bounds_upper
                elif solid_name == "FIN":
                    fin = Box(bounds_lower, bounds_upper)
                    fin = fin.repeat(
                        self.scale * self.fin_gap,
                        repeat_lower=(0, 0, 0),
                        repeat_higher=(0, 0, self.nr_fins - 1),
                        center=_center(bounds_lower, bounds_upper),
                    )
                    if self.copper is not None:
                        self.copper = self.copper + fin
                    else:
                        self.copper = fin
                else:
                    solid = Box(bounds_lower, bounds_upper)
                    if self.copper is not None:
                        self.copper = self.copper + solid
                    else:
                        self.copper = solid
            else:
                break
        self.channel = Channel(self.geo_bounds_lower, self.geo_bounds_upper)

    def read_solid(self, reader):
        # solid pieces
        faces = []
        while True:
            line = reader.readline()
            split_line = line.split(" ")
            if len(split_line) == 0:
                break
            elif "endsolid" == split_line[0]:
                break
            elif "facet" == split_line[0]:
                # read outer loop line
                _ = reader.readline()
                # read 3 vertices
                a_0 = [float(x) for x in reader.readline().split(" ")[-3:]]
                a_1 = [float(x) for x in reader.readline().split(" ")[-3:]]
                a_2 = [float(x) for x in reader.readline().split(" ")[-3:]]
                faces.append([a_0, a_1, a_2])
                # read end loop/end facet
                _ = reader.readline()
                _ = reader.readline()
        faces = np.array(faces)
        bounds_lower = (
            np.min(faces[..., 2]),
            np.min(faces[..., 1]),
            np.min(faces[..., 0]),
        )  # flip axis
        bounds_upper = (
            np.max(faces[..., 2]),
            np.max(faces[..., 1]),
            np.max(faces[..., 0]),
        )
        bounds_lower = tuple(
            [self.scale * (x + t) for x, t in zip(bounds_lower, self.translate)]
        )
        bounds_upper = tuple(
            [self.scale * (x + t) for x, t in zip(bounds_upper, self.translate)]
        )
        return bounds_lower, bounds_upper


def _center(bounds_lower, bounds_upper):
    center_x = bounds_lower[0] + (bounds_upper[0] - bounds_lower[0]) / 2
    center_y = bounds_lower[1] + (bounds_upper[1] - bounds_lower[1]) / 2
    center_z = bounds_lower[2] + (bounds_upper[2] - bounds_lower[2]) / 2
    return center_x, center_y, center_z
