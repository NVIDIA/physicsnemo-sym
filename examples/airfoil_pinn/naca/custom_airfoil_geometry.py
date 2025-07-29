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

from physicsnemo.sym.geometry.primitives_2d import Channel2D
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter

import warp as wp
import numpy as np

wp.init()


@wp.func
def local_camber(
    xi: float, max_camber: float, loc_camber: float, chord: float
) -> float:
    """
    Computes the mean camber line for a NACA 4-digit airfoil.

    NACA implementation modified from https://stackoverflow.com/questions/31815041/plotting-a-naca-4-series-airfoil
    https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil

    Parameters
    ----------
    xi: float
        x-coordinate of an individual point along the chord
    max_camber: float
        Amount of maximum camber, expressed as a fraction of chord (y/c)
    loc_camber: float
        x-coordinate of the location of maximum camber, expressed as a fraction of chord (x/c)
    chord: float
        Chord length

    Returns
    -------
    yi: float
        y-coordinate of the mean camber line at the point xi

    """
    x_nondim = xi / chord  # Nondimensionalizes the x-coordinate

    if x_nondim < 0.0 or x_nondim > 1.0:  # Out of bounds
        return 0.0

    elif x_nondim < loc_camber:  # Front part of the camber line
        return (
            (max_camber / loc_camber**2.0)
            * (2.0 * loc_camber * x_nondim - x_nondim**2.0)
            * chord
        )

    else:  # Rear part of the camber line
        return (
            (max_camber / (1.0 - loc_camber) ** 2.0)
            * (1.0 - 2.0 * loc_camber + 2.0 * loc_camber * x_nondim - x_nondim**2.0)
            * chord
        )


@wp.func
def local_thickness(xi: float, max_thickness: float, chord: float) -> float:
    """
    Computes the thickness distribution for a NACA 4-digit airfoil.

    Parameters
    ----------
    xi: float
        x-coordinate of an individual point along the chord
    max_thickness: float
        Maximum thickness of the airfoil, expressed as a fraction of chord (y/c)
    chord: float
        The chord length of the airfoil

    Returns
    -------
    ti: float
        Thickness of the airfoil at the point xi
    """
    x_nondim = xi / chord
    # fmt: off
    return (
        5.0
        * max_thickness
        * chord
        * (
              0.2969 * x_nondim**0.5
            - 0.1260 * x_nondim
            - 0.3516 * x_nondim**2.0
            + 0.2843 * x_nondim**3.0
            - 0.1036 * x_nondim**4.0  # Modified from -0.1015 to close the trailing edge
        )
    )
    # fmt: on


@wp.func
def op_subtract(sdf_1: float, sdf_2: float) -> float:
    """
    Returns a signed distance function for two objects that are subtracted

    Parameters
    ----------
    sdf_1: float
        Signed distance function computed with respect to object 1
    sdf_2: float
        Signed distance function computed with respect to object 2

    Returns
    -------
    d: float
        Signed distance function computed with respect to the subtracted objects
    """
    return wp.max(-sdf_1, sdf_2)


@wp.func
def channel_sdf(x_query: float, y_query: float, y_top: float, y_bot: float) -> float:
    """
    Computes the signed distance function at (x_query, y_query) w.r.t. a channel with a top and bottom boundary.

    Parameters
    ----------
    x_query: float
        x-coordinate of the query point where the signed distance function is computed
    y_query: float
        y-coordinate of the query point where the signed distance function is computed
    y_top: float
        y-coordinate of the top boundary of the channel
    y_bot: float
        y-coordinate of the bottom boundary of the channel

    Returns
    -------
    d: float
        Signed distance function at (x_query, y_query) w.r.t. the channel. Positive values indicate that the point is
        outside the channel, while negative values indicate that the point is inside the channel.

    """
    if wp.abs(y_top - y_query) < wp.abs(y_bot - y_query):
        # Query point is closer to the top boundary
        return y_query - 5.0  # TODO fix this to use the top boundary y_top
    else:
        return -5.0 - y_query  # TODO fix this to use the bottom boundary y_bot


@wp.func
def naca_sdf(
    airfoil_resolution: int,
    max_camber: float,
    loc_camber: float,
    max_thickness: float,
    chord: float,
    x_query: float,
    y_query: float,
) -> float:
    """
    Computes the signed distance function at (x_query, y_query) w.r.t. a NACA 4-digit airfoil.

    Parameters
    ----------
    airfoil_resolution: int
        The resolution, measured as the number of discrete sampling points used to compute the SDF.
    max_camber: float
        The maximum camber of the airfoil, expressed as a fraction of chord (y/c)
    loc_camber
        The location of the maximum camber, expressed as a fraction of chord (x/c)
    max_thickness
        The maximum thickness of the airfoil, expressed as a fraction of chord (y/c)
    chord
        The chord length of the airfoil
    x_query
        The x-coordinate of the query point where the signed distance function is computed
    y_query
        The y-coordinate of the query point where the signed distance function is computed

    Returns
    -------
    d: float
        Signed distance function at (x_query, y_query) w.r.t. the NACA 4-digit airfoil. Positive values indicate that
        the point is outside the airfoil, while negative values indicate that the point is inside the airfoil.
    """

    dx = chord / float(airfoil_resolution)

    sdf = float(wp.inf)

    for i in range(airfoil_resolution):
        x_i = float(i) * dx
        camber_i = local_camber(x_i, max_camber, loc_camber, chord)
        thickness_i = local_thickness(x_i, max_thickness, chord)
        sdf_i = (
            wp.sqrt((x_i - x_query) ** 2.0 + (camber_i - y_query) ** 2.0) - thickness_i
        )
        sdf = wp.min(sdf_i, sdf)

    return sdf


@wp.func
def naca_boundary(
    airfoil_resolution: int,
    max_camber: float,
    loc_camber: float,
    max_thickness: float,
    chord: float,
    x_query: float,
    y_query: float,
):
    """
    Computes various quantities at the point on the airfoil boundary closest to the query point.

    Can be thought of as an extension of the `naca_sdf` function.

    Parameters
    ----------
    airfoil_resolution: int
        The resolution, measured as the number of discrete sampling points used to compute the quantities.
    max_camber: float
        The maximum camber of the airfoil, expressed as a fraction of chord (y/c)
    loc_camber
        The location of the maximum camber, expressed as a fraction of chord (x/c)
    max_thickness
        The maximum thickness of the airfoil, expressed as a fraction of chord (y/c)
    chord
        The chord length of the airfoil
    x_query
        The x-coordinate of the query point where the signed distance function is computed
    y_query
        The y-coordinate of the query point where the signed distance function is computed

    Returns
    -------
    out: tuple[float, float, float, float]
        A tuple containing the following values at the point on the airfoil boundary closest to the query point:
        - x-coordinate
        - y-coordinate
        - x-component of the normal vector
        - y-component of the normal vector
    """

    dx = chord / float(airfoil_resolution)

    sdf = float(wp.inf)

    # The x-coordinate, camber, and thickness at the point on the airfoil mean camber line closest to the query point
    x_closest_camberline = float(wp.inf)
    y_closest_camberline = float(wp.inf)
    thickness_closest = float(wp.inf)

    for i in range(airfoil_resolution):
        x_i = float(i) * dx
        camber_i = local_camber(x_i, max_camber, loc_camber, chord)
        thickness_i = local_thickness(x_i, max_thickness, chord)
        sdf_i = (
            wp.sqrt((x_i - x_query) ** 2.0 + (camber_i - y_query) ** 2.0) - thickness_i
        )

        if sdf_i < sdf:
            sdf = sdf_i
            x_closest_camberline = x_i
            y_closest_camberline = camber_i
            thickness_closest = thickness_i

    # x- and y-components of the normal vector at the point on the airfoil closest to the query point
    normal_x = (x_query - x_closest_camberline) / (sdf + thickness_closest)
    normal_y = (y_query - y_closest_camberline) / (sdf + thickness_closest)
    normal_magnitude = wp.sqrt(normal_x**2.0 + normal_y**2.0)
    normal_x /= normal_magnitude
    normal_y /= normal_magnitude

    # The x- and y-coordinates of the point on the airfoil boundary closest to the query point
    x_closest_boundary = thickness_closest * normal_x + x_closest_camberline
    y_closest_boundary = thickness_closest * normal_y + y_closest_camberline

    return (x_closest_boundary, y_closest_boundary, normal_x, normal_y)


@wp.kernel
def sample_interior(
    rand_seed: int,
    airfoil_resolution: int,
    camber_min: float,
    camber_max: float,
    loc_camber: float,
    thickness_min: float,
    thickness_max: float,
    chord: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    alpha_min: float,
    alpha_max: float,
    xs: wp.array(dtype=float),
    ys: wp.array(dtype=float),
    sdfs: wp.array(dtype=float),
    sdfs_x: wp.array(dtype=float),
    sdfs_y: wp.array(dtype=float),
    alphas: wp.array(dtype=float),
    thicknesses: wp.array(dtype=float),
    cambers: wp.array(dtype=float),
) -> None:
    """
    Samples PINN nodes from the interior of the domain (airfoil + channel).

    Note: Also samples from a random point in parameter space (e.g., random camber, thickness, alpha, etc.).

    Parameters
    ----------
    rand_seed: int
        Random seed used to initialize Warp's random number generator
    airfoil_resolution: int
        The resolution of the airfoil, measured as the number of discrete sampling points used to compute the SDF.
    camber_min: float
        Together with camber_max, defines the range of camber values to sample from
    camber_max: float
        Together with camber_min, defines the range of camber values to sample from
    loc_camber: float
        The location of the maximum camber, expressed as a fraction of chord (x/c)
    thickness_min: float
        Together with thickness_max, defines the range of thickness values to sample from
    thickness_max: float
        Together with thickness_min, defines the range of thickness values to sample from
    chord: float
        The chord length of the airfoil
    x_min: float
        The minimum x-coordinate of the domain
    y_min: float
        The minimum y-coordinate of the domain
    x_max: float
        The maximum x-coordinate of the domain
    y_max: float
        The maximum y-coordinate of the domain
    alpha_min: float
        Together with alpha_max, defines the range of angles of attack to sample from
    alpha_max: float
        Together with alpha_min, defines the range of angles of attack to sample from

    Returns
    -------
    None (in-place kernel). Writes per-thread to the following arrays:
        - xs: x-coordinates of the sampled points
        - ys: y-coordinates of the sampled points
        - sdfs: signed distance functions at the sampled points
        - sdfs_x: derivative of the signed distance functions w.r.t. changing x of the sampled points
        - sdfs_y: derivative of the signed distance functions w.r.t. changing y of the sampled points2
        - alphas: angles of attack of the airfoil at the sampled points
        - thicknesses: thicknesses of the airfoil at the sampled points
        - cambers: cambers of the airfoil at the sampled points
    """
    tid = wp.tid()
    random_state = wp.rand_init(rand_seed, tid)

    # Sample a point in the domain (the "query point")
    x = wp.randf(random_state, x_min, x_max)
    y = wp.randf(random_state, y_min, y_max)

    # Sample parameter values
    alpha = -wp.randf(
        random_state, alpha_min, alpha_max
    )  # rotate the volume the other way before we subtract non-rotated boundary
    thickness = wp.randf(random_state, thickness_min, thickness_max)
    camber = wp.randf(random_state, camber_min, camber_max)

    # Precompute some trig values, to reduce duplicate computations
    sa = wp.sin(alpha)
    ca = wp.cos(alpha)

    # Compute the signed distance function at the query point
    x_query = ca * x - sa * y
    y_query = sa * x + ca * y
    sdf = op_subtract(
        naca_sdf(
            airfoil_resolution, camber, loc_camber, thickness, chord, x_query, y_query
        ),
        channel_sdf(x_query=x, y_query=y, y_top=y_max, y_bot=y_min),
    )

    ### Use finite-differencing to compute the sdf's gradients at the query point
    eps = 1e-5

    # x-derivative
    x_query = ca * (x + eps) - sa * y
    y_query = sa * (x + eps) + ca * y
    sdf_dx = op_subtract(
        naca_sdf(
            airfoil_resolution, camber, loc_camber, thickness, chord, x_query, y_query
        ),
        channel_sdf(x_query=x + eps, y_query=y, y_top=y_max, y_bot=y_min),
    )

    # y-derivative
    x_query = ca * x - sa * (y + eps)
    y_query = sa * x + ca * (y + eps)
    sdf_dy = op_subtract(
        naca_sdf(
            airfoil_resolution, camber, loc_camber, thickness, chord, x_query, y_query
        ),
        channel_sdf(x_query=x, y_query=y + eps, y_top=y_max, y_bot=y_min),
    )

    # Performs the SDF derivative finite-differencing
    normal_x = sdf_dx - sdf
    normal_y = sdf_dy - sdf
    normal_magnitude = wp.sqrt(normal_x**2.0 + normal_y**2.0)
    normal_x /= normal_magnitude
    normal_y /= normal_magnitude

    # Write kernel outputs
    xs[tid] = x
    ys[tid] = y
    sdfs[tid] = sdf
    sdfs_x[tid] = -normal_x
    sdfs_y[tid] = -normal_y
    alphas[tid] = -alpha  # need minus here to feed correct sign to NN
    thicknesses[tid] = thickness
    cambers[tid] = camber


@wp.kernel
def sample_boundary(
    rand_seed: int,
    airfoil_resolution: int,
    camber_min: float,
    camber_max: float,
    loc_camber: float,
    thickness_min: float,
    thickness_max: float,
    chord: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    alpha_min: float,
    alpha_max: float,
    xs: wp.array(dtype=float),
    ys: wp.array(dtype=float),
    normals_x: wp.array(dtype=float),
    normals_y: wp.array(dtype=float),
    alphas: wp.array(dtype=float),
    thicknesses: wp.array(dtype=float),
    cambers: wp.array(dtype=float),
) -> None:
    """
    Samples PINN nodes from the boundary of the domain (airfoil).

    Note: Also samples from a random point in parameter space (e.g., random camber, thickness, alpha, etc.).

    Parameters
    ----------
    rand_seed: int
        Random seed used to initialize Warp's random number generator
    airfoil_resolution: int
        The resolution of the airfoil, measured as the number of discrete sampling points used to compute the SDF.
    camber_min: float
        Together with camber_max, defines the range of camber values to sample from
    camber_max: float
        Together with camber_min, defines the range of camber values to sample from
    loc_camber: float
        The location of the maximum camber, expressed as a fraction of chord (x/c)
    thickness_min: float
        Together with thickness_max, defines the range of thickness values to sample from
    thickness_max: float
        Together with thickness_min, defines the range of thickness values to sample from
    chord: float
        The chord length of the airfoil
    x_min: float
        The minimum x-coordinate of the domain
    y_min: float
        The minimum y-coordinate of the domain
    x_max: float
        The maximum x-coordinate of the domain
    y_max: float
        The maximum y-coordinate of the domain
    alpha_min: float
        Together with alpha_max, defines the range of angles of attack to sample from
    alpha_max: float
        Together with alpha_min, defines the range of angles of attack to sample from

    Returns
    -------
    None (in-place kernel). Writes per-thread to the following arrays:
        - xs: x-coordinates of the sampled points
        - ys: y-coordinates of the sampled points
        - normals_x: x-components of the normals at the sampled points
        - normals_y: y-components of the normals at the sampled points
        - alphas: angles of attack of the airfoil at the sampled points
        - thicknesses: thicknesses of the airfoil at the sampled points
        - cambers: cambers of the airfoil at the sampled points

    """
    tid = wp.tid()
    rand_state = wp.rand_init(rand_seed, tid)

    # Sample a point in the domain (the "query point")
    x_query = wp.randf(rand_state, x_min, x_max)
    y_query = wp.randf(rand_state, y_min, y_max)

    # Sample parameter values
    alpha = wp.randf(rand_state, alpha_min, alpha_max)
    thickness = wp.randf(rand_state, thickness_min, thickness_max)
    camber = wp.randf(rand_state, camber_min, camber_max)

    # Precompute some trig values, to reduce duplicate computations
    sa = wp.sin(alpha)
    ca = wp.cos(alpha)

    (
        x_closest_rotated,
        y_closest_rotated,
        normal_x_rotated,
        normal_y_rotated,
    ) = naca_boundary(
        airfoil_resolution, camber, loc_camber, thickness, chord, x_query, y_query
    )

    # Computes the query point, after rotating the airfoil
    x_closest = ca * x_closest_rotated - sa * y_closest_rotated
    y_closest = sa * x_closest_rotated + ca * y_closest_rotated

    # rotate the normals as well
    xs[tid] = x_closest
    ys[tid] = y_closest

    normal_x = ca * normal_x_rotated - sa * normal_y_rotated
    normal_y = sa * normal_x_rotated + ca * normal_y_rotated

    # Write kernel outputs
    xs[tid] = x_closest
    ys[tid] = y_closest
    normals_x[tid] = normal_x
    normals_y[tid] = normal_y
    alphas[tid] = alpha
    thicknesses[tid] = thickness
    cambers[tid] = camber


class Example:
    def __init__(self, n_points: int = 1000000):
        self.n_points = n_points
        self.xs = wp.zeros(n_points, dtype=float)
        self.ys = wp.zeros(n_points, dtype=float)
        self.normals_x = wp.zeros(n_points, dtype=float)
        self.normals_y = wp.zeros(n_points, dtype=float)
        self.sdfs = wp.zeros(n_points, dtype=float)
        self.sdfs_x = wp.zeros(n_points, dtype=float)
        self.sdfs_y = wp.zeros(n_points, dtype=float)

        self.alphas = wp.zeros(n_points, dtype=float)
        self.thicknesses = wp.zeros(n_points, dtype=float)
        self.cambers = wp.zeros(n_points, dtype=float)

    def sample_interior(
        self,
        rand_seed: int,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        camber_min: float,
        camber_max: float,
        loc_camber: float,
        thickness_min: float,
        thickness_max: float,
        chord: float,
        airfoil_resolution: int,
        alpha_min: float,
        alpha_max: float,
    ):
        with wp.ScopedTimer(f"Sample Interior [{self.n_points:,} pts]"):
            wp.launch(
                kernel=sample_interior,
                dim=self.n_points,
                inputs=[
                    rand_seed,
                    airfoil_resolution,
                    camber_min,
                    camber_max,
                    loc_camber,
                    thickness_min,
                    thickness_max,
                    chord,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    alpha_min,
                    alpha_max,
                    self.xs,
                    self.ys,
                    self.sdfs,
                    self.sdfs_x,
                    self.sdfs_y,
                    self.alphas,
                    self.thicknesses,
                    self.cambers,
                ],
            )

        sdf = -self.sdfs.numpy()
        mask = sdf > 0
        alphas = self.alphas.numpy()[mask]
        thicknesses = self.thicknesses.numpy()[mask]
        cambers = self.cambers.numpy()[mask]
        return (
            self.xs.numpy()[mask],
            self.ys.numpy()[mask],
            sdf[mask],
            self.sdfs_x.numpy()[mask],
            self.sdfs_y.numpy()[mask],
            alphas,
            thicknesses,
            cambers,
        )

    def sample_boundary(
        self,
        rand_seed: int,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        camber_min: float,
        camber_max: float,
        loc_camber: float,
        thickness_min: float,
        thickness_max: float,
        chord: float,
        airfoil_resolution: int,
        alpha_min: float,
        alpha_max: float,
    ) -> dict[str, np.ndarray]:
        with wp.ScopedTimer(f"Sample Boundary [{self.n_points:,} pts]"):
            wp.launch(
                kernel=sample_boundary,
                dim=self.n_points,
                inputs=[
                    rand_seed,
                    airfoil_resolution,
                    camber_min,
                    camber_max,
                    loc_camber,
                    thickness_min,
                    thickness_max,
                    chord,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    alpha_min,
                    alpha_max,
                    self.xs,
                    self.ys,
                    self.normals_x,
                    self.normals_y,
                    self.alphas,
                    self.thicknesses,
                    self.cambers,
                ],
            )

        normals_x = self.normals_x.numpy()
        normals_y = self.normals_y.numpy()
        mask = (normals_x != 0) & (normals_y != 0)

        alphas = self.alphas.numpy()[mask]
        thicknesses = self.thicknesses.numpy()[mask]
        cambers = self.cambers.numpy()[mask]
        return (
            self.xs.numpy()[mask],
            self.ys.numpy()[mask],
            normals_x[mask],
            normals_y[mask],
            alphas,
            thicknesses,
            cambers,
        )


class AirfoilInChannel:
    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        params: Parameterization = None,
        include_channel_boundary: bool = False,
        loc_camber: float = 0.4,
        chord: float = 1.0,
        airfoil_resolution: int = 501,
    ):
        """ """
        if params is None:
            params = {}

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.params = params

        self.channel = Channel2D(point_1=(x_min, y_min), point_2=(x_max, y_max))
        self.include_channel_boundary = include_channel_boundary

        self.loc_camber = loc_camber
        self.chord = chord
        self.airfoil_resolution = airfoil_resolution

    def _get_param_min_max(
        self, param: Parameter, parameterization: Parameterization = None
    ) -> tuple[float, float]:
        if parameterization is None:
            param_ranges = {}
        else:
            param_ranges = parameterization.param_ranges

        vals = param_ranges.get(param, self.params.param_ranges[param])
        if isinstance(vals, float):
            val_min, val_max = vals, vals
        else:
            val_min, val_max = min(vals), max(vals)
        return val_min, val_max

    def sample_boundary(
        self,
        n_points: int,
        rand_seed: int = 0,
        criteria=None,
        parameterization=None,
        quasirandom: bool = False,
    ) -> dict[str, np.ndarray]:
        channel_boundary = self.channel.sample_boundary(
            n_points, criteria=criteria, parameterization=parameterization
        )
        alpha_min, alpha_max = self._get_param_min_max(
            Parameter("alpha"), parameterization
        )
        camber_min, camber_max = self._get_param_min_max(
            Parameter("camber"), parameterization
        )
        thickness_min, thickness_max = self._get_param_min_max(
            Parameter("thickness"), parameterization
        )
        xs = np.zeros((n_points, 1))
        ys = np.zeros((n_points, 1))
        normals_x = np.zeros((n_points, 1))
        normals_y = np.zeros((n_points, 1))
        alphas = np.zeros((n_points, 1))
        thicknesses = np.zeros((n_points, 1))
        cambers = np.zeros((n_points, 1))

        n_points_collected: int = 0

        while n_points_collected < n_points:
            example = Example(n_points=n_points)
            (
                x,
                y,
                normal_x,
                normal_y,
                alpha,
                thickness,
                camber,
            ) = example.sample_boundary(
                rand_seed=rand_seed + n_points_collected,
                x_min=self.x_min,
                y_min=self.y_min,
                x_max=self.x_max,
                y_max=self.y_max,
                camber_min=camber_min,
                camber_max=camber_max,
                loc_camber=self.loc_camber,
                thickness_min=thickness_min,
                thickness_max=thickness_max,
                chord=self.chord,
                airfoil_resolution=self.airfoil_resolution,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )

            n_new_points = min(len(x), n_points - n_points_collected)

            indices = slice(n_points_collected, n_points_collected + n_new_points)

            xs[indices] = x[:n_new_points].reshape(-1, 1)
            ys[indices] = y[:n_new_points].reshape(-1, 1)
            normals_x[indices] = normal_x[:n_new_points].reshape(-1, 1)
            normals_y[indices] = normal_y[:n_new_points].reshape(-1, 1)
            alphas[indices] = alpha[:n_new_points].reshape(-1, 1)
            thicknesses[indices] = thickness[:n_new_points].reshape(-1, 1)
            cambers[indices] = camber[:n_new_points].reshape(-1, 1)

            n_points_collected += n_new_points

        if self.include_channel_boundary:
            idx = np.random.choice(np.arange(2 * n_points), n_points)
            xs = np.concatenate([xs, channel_boundary["x"]])[idx]
            ys = np.concatenate([ys, channel_boundary["y"]])[idx]
            normals_x = np.concatenate([-normals_x, channel_boundary["normal_x"]])[idx]
            normals_y = np.concatenate([-normals_y, channel_boundary["normal_y"]])[idx]
            return {
                "x": xs,
                "y": ys,
                "normal_x": -normals_x,
                "normal_y": -normals_y,
                "alpha": alphas,
                "thickness": thicknesses,
                "camber": cambers,
                "area": channel_boundary["area"],
            }
        else:
            return {
                "x": xs,
                "y": ys,
                "normal_x": -normals_x,
                "normal_y": -normals_y,
                "alpha": alphas,
                "thickness": thicknesses,
                "camber": cambers,
                "area": channel_boundary["area"],
            }

    def sample_interior(
        self,
        n_points: int,
        rand_seed: int = 0,
        bounds=None,
        criteria=None,
        parameterization=None,
        compute_sdf_derivatives: bool = False,
        quasirandom: bool = False,
    ) -> dict[str, np.ndarray]:
        channel_interior = self.channel.sample_interior(
            n_points, criteria=criteria, parameterization=parameterization
        )
        alpha_min, alpha_max = self._get_param_min_max(
            Parameter("alpha"), parameterization
        )
        camber_min, camber_max = self._get_param_min_max(
            Parameter("camber"), parameterization
        )
        thickness_min, thickness_max = self._get_param_min_max(
            Parameter("thickness"), parameterization
        )

        xs = np.zeros((n_points, 1))
        ys = np.zeros((n_points, 1))
        sdfs = np.zeros((n_points, 1))
        sdfs_x = np.zeros((n_points, 1))
        sdfs_y = np.zeros((n_points, 1))
        alphas = np.zeros((n_points, 1))
        thicknesses = np.zeros((n_points, 1))
        cambers = np.zeros((n_points, 1))

        n_points_collected = 0

        while n_points_collected < n_points:
            example = Example(n_points=n_points)
            x, y, sdf, sdf_x, sdf_y, alpha, thickness, camber = example.sample_interior(
                rand_seed=rand_seed + n_points_collected,
                x_min=self.x_min,
                y_min=self.y_min,
                x_max=self.x_max,
                y_max=self.y_max,
                camber_min=camber_min,
                camber_max=camber_max,
                loc_camber=self.loc_camber,
                thickness_min=thickness_min,
                thickness_max=thickness_max,
                chord=self.chord,
                airfoil_resolution=self.airfoil_resolution,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )

            n_new_points = min(len(x), n_points - n_points_collected)

            indices = slice(n_points_collected, n_points_collected + n_new_points)

            xs[indices] = x[:n_new_points].reshape(-1, 1)
            ys[indices] = y[:n_new_points].reshape(-1, 1)
            sdfs[indices] = sdf[:n_new_points].reshape(-1, 1)
            sdfs_x[indices] = sdf_x[:n_new_points].reshape(-1, 1)
            sdfs_y[indices] = sdf_y[:n_new_points].reshape(-1, 1)
            alphas[indices] = alpha[:n_new_points].reshape(-1, 1)
            thicknesses[indices] = thickness[:n_new_points].reshape(-1, 1)
            cambers[indices] = camber[:n_new_points].reshape(-1, 1)

            n_points_collected += n_new_points

        return {
            "x": xs,
            "y": ys,
            "sdf": sdfs,
            "alpha": alphas,
            "thickness": thicknesses,
            "camber": cambers,
            "sdf__x": sdfs_x,
            "sdf__y": sdfs_y,
            "area": channel_interior["area"],
        }

    @property
    def dims(self):
        return ["x", "y"]
