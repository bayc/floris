# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import math

import numpy as np
from scipy.stats import norm
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d

from .utilities import cosd, sind, tand
from .logging_manager import LoggerBase


class Turbine(LoggerBase):
    """
    Turbine is a class containing objects pertaining to the individual
    turbines.

    Turbine is a model class representing a particular wind turbine. It
    is largely a container of data and parameters, but also contains
    methods to probe properties for output.
    """
    def __init__(self, input_dictionary):
        """
        Args:
            input_dictionary: A dictionary containing the initialization data for
                the turbine model; it should have the following key-value pairs:

                -   **rotor_diameter** (*float*): The rotor diameter (m).
                -   **hub_height** (*float*): The hub height (m).
                -   **blade_count** (*int*): The number of blades.
                -   **pP** (*float*): The cosine exponent relating the yaw
                    misalignment angle to power.
                -   **pT** (*float*): The cosine exponent relating the rotor
                    tilt angle to power.
                -   **generator_efficiency** (*float*): The generator
                    efficiency factor used to scale the power production.
                -   **power_thrust_table** (*dict*): A dictionary containing the
                    following key-value pairs:

                    -   **power** (*list(float)*): The coefficient of power at
                        different wind speeds.
                    -   **thrust** (*list(float)*): The coefficient of thrust
                        at different wind speeds.
                    -   **wind_speed** (*list(float)*): The wind speeds for
                        which the power and thrust values are provided (m/s).

                -   **yaw_angle** (*float*): The yaw angle of the turbine
                    relative to the wind direction (deg). A positive value
                    represents a counter-clockwise rotation relative to the
                    wind direction.
                -   **tilt_angle** (*float*): The tilt angle of the turbine
                    (deg). Positive values correspond to a downward rotation of
                    the rotor for an upstream turbine.
                -   **TSR** (*float*): The tip-speed ratio of the turbine. This
                    parameter is used in the "curl" wake model.
                -   **ngrid** (*int*, optional): The square root of the number
                    of points to use on the turbine grid. This number will be
                    squared so that the points can be evenly distributed.
                    Defaults to 5.
                -   **rloc** (*float, optional): A value, from 0 to 1, that determines
                    the width/height of the grid of points on the rotor as a ratio of
                    the rotor radius.
                    Defaults to 0.5.

        Returns:
            Turbine: An instantiated Turbine object.
        """

        self.rotor_diameter: float = input_dictionary["rotor_diameter"]
        self.hub_height: float = input_dictionary["hub_height"]
        self.blade_count: int = input_dictionary["blade_count"]
        self.pP: float = input_dictionary["pP"]
        self.pT: float = input_dictionary["pT"]
        self.generator_efficiency: float = input_dictionary["generator_efficiency"]
        self.power_thrust_table: list = input_dictionary["power_thrust_table"]
        self.yaw_angle: float = input_dictionary["yaw_angle"]
        self.tilt_angle: float = input_dictionary["tilt_angle"]
        self.tsr: float = input_dictionary["TSR"]

        # For the following parameters, use default values if not user-specified
        self.ngrid = int(input_dictionary["ngrid"]) if "ngrid" in input_dictionary else 5
        self.rloc = float(input_dictionary["rloc"]) if "rloc" in input_dictionary else 0.5
        if "use_points_on_perimeter" in input_dictionary:
            self.use_points_on_perimeter = bool(input_dictionary["use_points_on_perimeter"])
        else:
            self.use_points_on_perimeter = False

        # initialize to an invalid value until calculated
        self.air_density = -1
        self.use_turbulence_correction = False

        self.initialize_turbine()

    # Private methods

    def initialize_turbine(self):
        # Initialize the turbine given saved parameter settings

        # Precompute interps
        wind_speed = self.power_thrust_table["wind_speed"]

        cp = self.power_thrust_table["power"]
        self.fCpInterp = interp1d(wind_speed, cp, fill_value="extrapolate")

        ct = self.power_thrust_table["thrust"]
        self.fCtInterp = interp1d(wind_speed, ct, fill_value="extrapolate")

        # constants
        self.grid_point_count = self.ngrid * self.ngrid
        if np.sqrt(self.grid_point_count) % 1 != 0.0:
            raise ValueError("Turbine.grid_point_count must be the square of a number")

        self.reset_velocities()

        # initialize derived attributes
        self.grid = self._create_swept_area_grid()

        # Compute list of inner powers
        inner_power = np.array([self._power_inner_function(ws) for ws in wind_speed])
        self.powInterp = interp1d(wind_speed, inner_power, fill_value="extrapolate")

        # The indices for this Turbine instance's points from the FlowField
        # are set in `FlowField._discretize_turbine_domain` and stored
        # in this variable.
        self.flow_field_point_indices = None

    def _create_swept_area_grid(self):
        # TODO: add validity check:
        # rotor points has a minimum in order to always include points inside
        # the disk ... 2?
        #
        # the grid consists of the y,z coordinates of the discrete points which
        # lie within the rotor area: [(y1,z1), (y2,z2), ... , (yN, zN)]

        # update:
        # using all the grid point because that how roald did it.
        # are the points outside of the rotor disk used later?

        # determine the dimensions of the square grid
        num_points = int(np.round(np.sqrt(self.grid_point_count)))

        pt = self.rloc * self.rotor_radius
        # syntax: np.linspace(min, max, n points)
        horizontal = np.linspace(-pt, pt, num_points)

        vertical = np.linspace(-pt, pt, num_points)

        # build the grid with all of the points
        grid = [(h, vertical[i]) for i in range(num_points) for h in horizontal]

        # keep only the points in the swept area
        if self.use_points_on_perimeter:
            grid = [
                point
                for point in grid
                if np.hypot(point[0], point[1]) <= self.rotor_radius
            ]
        else:
            grid = [
                point
                for point in grid
                if np.hypot(point[0], point[1]) < self.rotor_radius
            ]

        return grid

    def _power_inner_function(self, yaw_effective_velocity):
        """
        This method calculates the power for an array of yaw effective wind
        speeds without the air density and turbulence correction parameters.
        This is used to initialize the power interpolation method used to
        compute turbine power.
        """

        # Now compute the power
        cptmp = self._fCp(
            yaw_effective_velocity
        )  # Note Cp is also now based on yaw effective velocity
        return (
            0.5
            * (np.pi * self.rotor_radius ** 2)
            * cptmp
            * self.generator_efficiency
            * yaw_effective_velocity ** 3
        )

    def _fCp(self, at_wind_speed):
        wind_speed = self.power_thrust_table["wind_speed"]
        if at_wind_speed < min(wind_speed):
            return 0.0
        else:
            _cp = self.fCpInterp(at_wind_speed)
            if _cp.size > 1:
                _cp = _cp[0]
            if _cp > 1.0:
                _cp = 1.0
            if _cp < 0.0:
                _cp = 0.0
            return float(_cp)

    def _fCt(self, at_wind_speed):
        wind_speed = self.power_thrust_table["wind_speed"]
        if at_wind_speed < min(wind_speed):
            return 0.99
        else:
            _ct = self.fCtInterp(at_wind_speed)
            if _ct.size > 1:
                _ct = _ct[0]
            if _ct > 1.0:
                _ct = 0.9999
            if _ct <= 0.0:
                _ct = 0.0001
            return float(_ct)

    # Public methods

    def change_turbine_parameters(self, turbine_change_dict: dict):
        """
        Change a turbine parameter and reinitialize the Turbine class.

        Args:
            turbine_change_dict (dict): A dictionary of parameters to change.
        """
        for param in turbine_change_dict:
            self.logger.info(
                "Turbine: setting {} to {}".format(param, turbine_change_dict[param])
            )
            setattr(self, param, turbine_change_dict[param])
        self.initialize_turbine()

    def calculate_swept_area_velocities(self, local_wind_speed, coord, x, y, z, additional_wind_speed=None):
        """
        This method calculates and returns the wind speeds at each
        rotor swept area grid point for the turbine, interpolated from
        the flow field grid.

        Args:
            wind_direction (float): The wind farm wind direction (deg).
            local_wind_speed (np.array): The wind speed at each grid point in
                the flow field (m/s).
            coord (:py:obj:`~.utilities.Vec3`): The coordinate of the turbine.
            x (np.array): The x-coordinates of the flow field grid.
            y (np.array): The y-coordinates of the flow field grid.
            z (np.array): The z-coordinates of the flow field grid.

        Returns:
            np.array: The wind speed at each rotor grid point
            for the turbine (m/s).
        """
        u_at_turbine = local_wind_speed

        # TODO:
        # # PREVIOUS METHOD========================
        # # UNCOMMENT IF ANY ISSUE UNCOVERED WITH NEW METHOD
        # x_grid = x
        # y_grid = y
        # z_grid = z

        # yPts = np.array([point[0] for point in self.grid])
        # zPts = np.array([point[1] for point in self.grid])

        # # interpolate from the flow field to get the flow field at the grid
        # # points
        # dist = [np.sqrt((coord.x1 - x_grid)**2 \
        #      + (coord.x2 + yPts[i] - y_grid) **2 \
        #      + (self.hub_height + zPts[i] - z_grid)**2) \
        #      for i in range(len(yPts))]
        # idx = [np.where(dist[i] == np.min(dist[i])) for i in range(len(yPts))]
        # data = [np.mean(u_at_turbine[idx[i]]) for i in range(len(yPts))]
        # # PREVIOUS METHOD========================

        # Use this if no saved points (curl)
        if self.flow_field_point_indices is None:
            # # NEW METHOD========================
            # Sort by distance
            flow_grid_points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

            # Set up a grid array
            y_array = np.array(self.grid)[:, 0] + coord.x2
            z_array = np.array(self.grid)[:, 1] + self.hub_height
            x_array = np.ones_like(y_array) * coord.x1
            grid_array = np.column_stack([x_array, y_array, z_array])

            ii = np.array(
                [
                    np.argmin(
                        np.sum((flow_grid_points - grid_array[i, :]) ** 2, axis=1)
                    )
                    for i in range(len(grid_array))
                ]
            )
            self.flow_field_point_indices = ii
        else:
            ii = self.flow_field_point_indices

        if additional_wind_speed is not None:
            return (
                np.array(u_at_turbine.flatten()[ii]),
                np.array(additional_wind_speed.flatten()[ii]),
            )
        else:
            return np.array(u_at_turbine.flatten()[ii])

    def update_velocities(self, u_wake, coord, flow_field, rotated_x, rotated_y, rotated_z):
        """
        This method updates the velocities at the rotor swept area grid
        points based on the flow field freestream velocities and wake
        velocities.

        Args:
            u_wake (np.array): The wake deficit velocities at all grid points
                in the flow field (m/s).
            coord (:py:obj:`~.utilities.Vec3`): The coordinate of the turbine.
            flow_field (:py:class:`~.flow_field.FlowField`): The flow field.
            rotated_x (np.array): The x-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
            rotated_y (np.array): The y-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
            rotated_z (np.array): The z-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
        """
        # reset the waked velocities
        local_wind_speed = flow_field.u_initial - u_wake
        self.velocities = self.calculate_swept_area_velocities(
            local_wind_speed, coord, rotated_x, rotated_y, rotated_z
        )

    def reset_velocities(self) -> None:
        """
        This method sets the velocities at the turbine's rotor swept
        area grid points to zero.
        """
        self.velocities = [0.0] * self.grid_point_count

    def TKE_to_TI(self, turbulence_kinetic_energy: float) -> float:
        """
        Converts a list of turbulence kinetic energy values to
        turbulence intensity.
        Args:
            turbulence_kinetic_energy (list): Values of turbulence kinetic
                energy in units of meters squared per second squared.
            wind_speed (list): Measurements of wind speed in meters per second.
        Returns:
            list: converted turbulence intensity values expressed as a decimal
            (e.g. 10%TI -> 0.10).
        """
        total_turbulence_intensity = (
            np.sqrt((2 / 3) * turbulence_kinetic_energy)
        ) / self.average_velocity
        return total_turbulence_intensity

    def TI_to_TKE(self) -> float:
        """
        Converts TI to TKE.
        Args:
            wind_speed (list): Measurements of wind speed in meters per second.
        Returns:
            list: converted TKE values
        """
        return ((self.average_velocity * self.turbulence_intensity) ** 2) / (
            2 / 3
        )

    def u_prime(self):
        """
        Converts a TKE to horizontal deviation component.
        Args:
            wind_speed (list): Measurements of wind speed in meters per second.
        Returns:
            list: converted u_prime values in meters per second
        """
        tke = self.TI_to_TKE()
        return np.sqrt(2 * tke)

    def calculate_turbulence_correction(self):
        """
        This property calculates and returns the turbulence correction
        parameter for the turbine, a value used to account for the
        change in power output due to the effects of turbulence.

        Returns:
            float: The value of the turbulence parameter.
        """
        if not self.use_turbulence_correction:
            return 1.0
        else:
            # define wind speed, ti, and power curve components
            ws = np.array(self.power_thrust_table["wind_speed"])
            cp = np.array(self.power_thrust_table["power"])
            ws = ws[np.where(cp != 0)]
            ciws = ws[0]  # cut in wind speed
            cows = ws[len(ws) - 1]  # cut out wind speed
            speed = self.average_velocity
            ti = self.turbulence_intensity

            if ciws >= speed or cows <= speed or ti == 0.0 or math.isnan(speed):
                return 1.0
            else:
                # define mean and standard deviation to create normalized pdf with sum = 1
                mu = speed
                sigma = ti * mu
                if mu + sigma >= cows:
                    xp = np.linspace((mu - sigma), cows, 100)
                else:
                    xp = np.linspace((mu - sigma), (mu + sigma), 100)
                pdf = norm.pdf(xp, mu, sigma)
                npdf = np.array(pdf) * (1 / np.sum(pdf))

                # calculate turbulence parameter (ratio of corrected power to original power)
                return np.sum([npdf[k] * self.powInterp(xp[k]) for k in range(100)]) / (
                    self.powInterp(mu)
                )

    # Getters & Setters

    @property
    def rotor_radius(self) -> float:
        """
        Rotor radius of the turbine in meters.

        Returns:
            float: The rotor radius of the turbine.
        """
        return self.rotor_diameter / 2.0
    
    @rotor_radius.setter
    def rotor_radius(self, value: float) -> None:
        self.rotor_diameter = value * 2.0

    @property
    def average_velocity(self) -> float:
        """
        This property calculates and returns the cube root of the
        mean cubed velocity in the turbine's rotor swept area (m/s).

        **Note:** The velocity is scalled to an effective velocity by the yaw.

        Returns:
            float: The average velocity across a rotor.

        Examples:
            To get the average velocity for a turbine:

            >>> avg_vel = floris.farm.turbines[0].average_velocity()
        """
        # Remove all invalid numbers from interpolation
        data = np.array(self.velocities)[~np.isnan(self.velocities)]
        avg_vel = np.cbrt(np.mean(data ** 3))
        if np.isnan(avg_vel):
            avg_vel = 0
        elif np.isinf(avg_vel):
            avg_vel = 0

        return avg_vel

    # NOTE: Temporarily comment out because not used anywhere
    # When placing back, need to infer Cp from power since that
    # is where interpolation happens. TODO: Open research question
    # whether Cp should include yaw correction or not; or is it just
    # necessary to label it as such?
    # @property
    # def Cp(self):
    #     """
    #     Power coefficient of a turbine incorporating the yaw and tilt
    #     angles and using the rotor swept area average velocity,
    #     interpolated from the coefficient of power table (Cp vs wind speed).
    #     """
    #     # Compute the yaw effective velocity
    #     pW = self.pP / 3.0  # Convert from pP to w
    #     pV = self.pT / 3.0  # convert from pT to w
    #     yaw_effective_velocity = (
    #         self.average_velocity
    #         * (cosd(self.yaw_angle) ** pW)
    #         * (cosd(self.tilt_angle) ** pV)
    #     )

    #     P_avail = (
    #         0.5
    #         * self.air_density
    #         * np.pi
    #         * (self.rotor_diameter / 2) ** 2
    #         * yaw_effective_velocity ** 3
    #     )

    #     return self.power / P_avail

    @property
    def Ct(self) -> float:
        """
        Thrust coefficient of a turbine incorporating the yaw angle.
        The value is interpolated from the coefficient of thrust vs
        wind speed table using the rotor swept area average velocity.
        """
        return self._fCt(self.average_velocity) * cosd(self.yaw_angle)  # **self.pP

    @property
    def power(self) -> float:
        """
        Power produced by a turbine adjusted for yaw and tilt. Value
        given in Watts.
        """
        # Update to power calculation which replaces the fixed pP exponent with
        # an exponent pW, that changes the effective wind speed input to the power
        # calculation, rather than scaling the power.  This better handles power
        # loss to yaw in above rated conditions
        #
        # based on the paper "Optimising yaw control at wind farm level" by
        # Ervin Bossanyi

        # Compute the yaw effective velocity
        pW = self.pP / 3.0  # Convert from pP to w
        yaw_effective_velocity = self.average_velocity * cosd(self.yaw_angle) ** pW

        # Now compute the power
        turbulence_correction = self.calculate_turbulence_correction()
        return (self.air_density * self.powInterp(yaw_effective_velocity) * turbulence_correction)

    @property
    def axial_induction(self) -> float:
        """
        Axial induction factor of the turbine incorporating
        the thrust coefficient and yaw angle.
        """
        return ( 0.5 / cosd(self.yaw_angle) * (1 - np.sqrt(1 - self.Ct * cosd(self.yaw_angle) ) ) )