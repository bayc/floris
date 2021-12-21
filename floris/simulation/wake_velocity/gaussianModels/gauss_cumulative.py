# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import copy

import numpy as np
from scipy.special import gamma

from ....utilities import cosd, sind, tand
from .gaussian_model_base import GaussianModel
from ..base_velocity_deficit import VelocityDeficit


class GaussCumulative(GaussianModel):
    """
    The LegacyGauss model ports the previous Gauss model to the new FLORIS
    framework of inheritance of the GaussianModel. It is based on the gaussian
    wake models described in :cite:`glvm-bastankhah2014new`,
    :cite:`glvm-abkar2015influence`, :cite:`glvm-bastankhah2016experimental`,
    :cite:`glvm-niayifar2016analytical`, and :cite:`glvm-dilip2017wind`.

    References:
        .. bibliography:: /source/zrefs.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: glvm-
    """

    default_parameters = {
        "ka": 0.38,
        "kb": 0.000,
        "a_s": 0.179367259,
        "b_s": 0.0118889215,
        "c_s1": 0.0563691592,
        "c_s2": 0.13290157,
        "a_f": 3.11,
        "b_f": -0.68,
        "c_f": 2.41,
        "calculate_VW_velocities": True,
        "use_yaw_added_recovery": True,
        "eps_gain": 0.2,
        "alpha_mod": 1.0,
        "sigma_gch": True,
    }

    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                    -   **ka**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **kb**: Parameter used to determine the linear
                        relationship between the turbulence intensity and the
                        width of the Gaussian wake shape.
                    -   **alpha**: Parameter that determines the dependence of
                        the downstream boundary between the near wake and far
                        wake region on the turbulence intensity.
                    -   **beta**: Parameter that determines the dependence of
                        the downstream boundary between the near wake and far
                        wake region on the turbine's induction factor.

        """

        super().__init__(parameter_dictionary)

        self.model_string = "gauss_cumulative"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # wake expansion parameters
        self.ka = model_dictionary["ka"]
        self.kb = model_dictionary["kb"]

        # GCH Parameters
        self.calculate_VW_velocities = model_dictionary["calculate_VW_velocities"]
        self.use_yaw_added_recovery = model_dictionary["use_yaw_added_recovery"]
        self.eps_gain = model_dictionary["eps_gain"]
        self.gch_gain = 2.0

        self.Ctmp = []

        self.alpha_mod = model_dictionary["alpha_mod"]
        self.sigma_gch = model_dictionary["sigma_gch"]

        # wake expansion parameters
        # Table 4 of reference [3] in docstring
        self.a_s = model_dictionary["a_s"]
        self.b_s = model_dictionary["b_s"]
        self.c_s1 = model_dictionary["c_s1"]
        self.c_s2 = model_dictionary["c_s2"]

        # fitted parameters for super-Gaussian order n
        # Table 4 of reference [3] in docstring
        # self.b_f1 = model_dictionary["b_f1"]
        # self.b_f2 = model_dictionary["b_f2"]
        # self.b_f3 = model_dictionary["b_f3"]
        self.a_f = model_dictionary["a_f"]
        self.b_f = model_dictionary["b_f"]
        self.c_f = model_dictionary["c_f"]


    def function(
        self,
        x_locations,
        y_locations,
        z_locations,
        turbine,
        turbine_coord,
        deflection_field,
        flow_field,
        **kwargs,
    ):
        """
        Using the Gaussian wake model, this method calculates and
        returns the wake velocity deficits, caused by the specified turbine,
        relative to the freestream velocities at the grid of points
        comprising the wind farm flow field.

        Args:
            x_locations (np.array): An array of floats that contains the
                streamwise direction grid coordinates of the flow field
                domain (m).
            y_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the direction normal to
                x and parallel to the ground (m).
            z_locations (np.array): An array of floats that contains the grid
                coordinates of the flow field domain in the vertical
                direction (m).
            turbine (:py:obj:`floris.simulation.turbine`): Object that
                represents the turbine creating the wake.
            turbine_coord (:py:obj:`floris.utilities.Vec3`): Object containing
                the coordinate of the turbine creating the wake (m).
            deflection_field (np.array): An array of floats that contains the
                amount of wake deflection in meters in the y direction at each
                grid point of the flow field.
            flow_field (:py:class:`floris.simulation.flow_field`): Object
                containing the flow field information for the wind farm.

        Returns:
            np.array, np.array, np.array:
                Three arrays of floats that contain the wake velocity
                deficit in m/s created by the turbine relative to the freestream
                velocities for the U, V, and W components, aligned with the x, y,
                and z directions, respectively. The three arrays contain the
                velocity deficits at each grid point in the flow field.
        """
        n = kwargs["n"]
        sorted_map = kwargs["sorted_map"]
        u_wake = kwargs["u_wake"]
        Ctmp = kwargs["Ctmp"]
        TI = copy.deepcopy(turbine.current_turbulence_intensity)
        # self.sigma_gch = True
        if self.sigma_gch is True:
            sigma_n = self.wake_expansion(
                flow_field,
                turbine,
                turbine_coord,
                x_locations,
                y_locations,
                z_locations,
            )
        else:
            # yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
            # xR, _ = GaussianModel.mask_upstream_wake(y_locations, turbine_coord, yaw)
            Beta = (1 + np.sqrt(1 - turbine.Ct)) / (2 * np.sqrt(1 - turbine.Ct))
            epsilon = 0.2 * np.sqrt(Beta)
            sigma_n = (self.ka * TI) * (
                x_locations - turbine_coord.x1
            ) / turbine.rotor_diameter + epsilon 

        sum_lbda = 0.0
        # for m, (coord_i, turbine_i) in enumerate(sorted_map):
        # TODO: make sure n-1 and not n-2
        for m in range(0, n - 1):
            coord_i = sorted_map[m][0]
            turbine_i = sorted_map[m][1]
            if self.sigma_gch is True:
                sigma_i = self.wake_expansion(
                    flow_field,
                    turbine_i,
                    coord_i,
                    x_locations,
                    y_locations,
                    z_locations,
                )
#                 sigma_i = sigma_i / turbine_i.rotor_diameter
            else:
                # yaw = -1 * turbine_i.yaw_angle  # opposite sign convention in this model
                # xR, _ = GaussianModel.mask_upstream_wake(y_locations, coord_i, yaw)
                Beta = (1 + np.sqrt(1 - turbine_i.Ct)) / (2 * np.sqrt(1 - turbine_i.Ct))
                epsilon = 0.2 * np.sqrt(Beta)
                TI = copy.deepcopy(turbine_i.current_turbulence_intensity)
                sigma_i = (self.ka * TI) * (
                    x_locations - coord_i.x1
                ) / turbine_i.rotor_diameter + epsilon 
            S = sigma_n ** 2 + sigma_i ** 2
            # TODO: check deflection_field being a field instead of a scalar
            Y = (turbine_coord.x2 - coord_i.x2 - deflection_field) ** 2 / (2 * S)
            # Y = (turbine_coord.x2 - coord_i.x2) ** 2 / (2 * S)
            Z = (turbine_coord.x3 - coord_i.x3) ** 2 / (2 * S)
            # TODO: add alpha to show difference between derived and modified version
            lbda = self.alpha_mod * sigma_i ** 2 / S * np.exp(-Y) * np.exp(-Z)
            sum_lbda = sum_lbda + lbda * (Ctmp[m] / flow_field.u_initial)

        # Centerline velocity
        Uavg = turbine.average_velocity
        # print(np.min(sigma_n),np.max(sigma_n))
        num = turbine.Ct * (Uavg / flow_field.u_initial) ** 2
        den = (8 * (sigma_n / turbine.rotor_diameter) ** 2) * (1 - sum_lbda) ** 2
        C = flow_field.u_initial * (1 - sum_lbda) * (1 - np.sqrt(1 - num / den))

        # blondel
        # super gaussian
        # b_f = self.b_f1 * np.exp(self.b_f2 * TI) + self.b_f3
        x_tilde = np.abs(x_locations - turbine_coord.x1) / turbine.rotor_diameter
        r_tilde = (
                np.sqrt(
                    (y_locations - turbine_coord.x2 - deflection_field) ** 2 + (z_locations - turbine_coord.x3) ** 2,
                    dtype=np.float128,
                )
                / turbine.rotor_diameter
        )
        # a_f = 3.11
        # b_f = -0.68
        # c_f = 2.41
        n = self.a_f * np.exp(self.b_f * x_tilde) + self.c_f

        a1 = 2 ** (2 / n - 1)
        a2 = 2 ** (4 / n - 2)

        # # Calculate Beta (Eq 10, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
        # beta = 0.5 * (1.0 + np.sqrt(1.0 - turbine.Ct)) / np.sqrt(1.0 - turbine.Ct)
        # k = self.a_s * TI + self.b_s
        # eps = (self.c_s1 * turbine.Ct + self.c_s2) * np.sqrt(beta)
        #
        # # Calculate sigma_tilde (Eq 9, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
        # sigma_n = k * x_tilde + eps

        C = a1 - np.sqrt(a2
            - (
                    (n * turbine.Ct)
                    * cosd(turbine.yaw_angle)
                    / (
                            16.0
                            * gamma(2 / n)
                            * np.sign(sigma_n)
                            * (np.abs(sigma_n) ** (4 / n))
                            * (1 - sum_lbda) ** 2
                    )
            )
        ) 
        
        C = C * (1 - sum_lbda)

        # Max theoretical velocity deficit based on Betz theory
#         C_max = 2 * turbine.aI * Uavg
#         # TODO: determine whether C_max should vary in z (height)
#         # C_max = 2 * turbine.aI * np.reshape(turbine.velocities, (5,5)) * np.ones_like(flow_field.u_initial)
#         mask = C > C_max
#         C[mask] = C_max
#         nan_mask = np.isnan(C)
#         C[nan_mask] = 0.0

        f = np.exp(
            -((y_locations - turbine_coord.x2 - deflection_field) ** 2)
            / (2 * sigma_n ** 2)
        ) * np.exp(-((z_locations - turbine_coord.x3) ** 2) / (2 * sigma_n ** 2))

        # C[x_locations <= turbine_coord.x1] = 0.0
        # # C[y_locations < turbine_coord.x2 - 1.5 * turbine.rotor_diameter] = 0.0
        # # C[y_locations > turbine_coord.x2 + 1.5 * turbine.rotor_diameter] = 0.0
        Ctmp.append(C)
#         print(C)

        # Calculate mask values to mask upstream wake
        yR = y_locations - turbine_coord.x2
        xR = yR * tand(turbine.yaw_angle) + turbine_coord.x1
        
        # add turbines together
        velDef = C * np.exp((-1 * r_tilde ** n) / (2 * sigma_n ** 2))
        velDef[x_locations < xR] = 0
        u_wake = u_wake + Uavg * velDef

        # # mirrored wake
        # r_tilde2 = (
        #         np.sqrt(
        #             (y_locations - turbine_coord.x2 - deflection_field) ** 2 + (z_locations + turbine_coord.x3) ** 2,
        #             dtype=np.float128,
        #         )
        #         / turbine.rotor_diameter
        # )
        # velDef2 = C * np.exp((-1 * r_tilde2 ** n) / (2 * sigma_n ** 2))
        # velDef2[x_locations < xR] = 0
        
        # u_wake = u_wake + Uavg * np.sqrt(velDef ** 2 + velDef2 ** 2)




#         u_wake = u_wake + flow_field.u_initial * C * f

        # TODO integrate back in the v and w components
        # turb_u_wake = copy.deepcopy(sum_Cf)
        # turb_v_wake, turb_w_wake = self.wake.velocity_model.calculate_VW(
        #     np.zeros(np.shape(self.u_initial)),
        #     np.zeros(np.shape(self.u_initial)),
        #     coord,
        #     turbine,
        #     self,
        #     rotated_x,
        #     rotated_y,
        #     rotated_z,
        # )

        return u_wake, np.zeros(np.shape(u_wake)), np.zeros(np.shape(u_wake)), Ctmp

    def wake_expansion(
        self, flow_field, turbine, turbine_coord, x_locations, y_locations, z_locations
    ):

        # veer (degrees)
        veer = flow_field.wind_veer

        # turbulent mixing
        TI_mixing = self.yaw_added_turbulence_mixing(
            turbine_coord, turbine, flow_field, x_locations, y_locations, z_locations
        )
        turbine.current_turbulence_intensity = (
            turbine.current_turbulence_intensity + self.gch_gain * TI_mixing
        )
        TI = copy.deepcopy(turbine.current_turbulence_intensity)  # + TI_mixing

        # turbine parameters
        D = turbine.rotor_diameter
        yaw = -1 * turbine.yaw_angle  # opposite sign convention in this model
        Ct = turbine.Ct
        U_local = flow_field.u_initial

        # Calculate Beta (Eq 10, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
        beta = 0.5 * (1.0 + np.sqrt(1.0 - Ct)) / np.sqrt(1.0 - Ct)
        k = self.a_s * TI + self.b_s
        eps = (self.c_s1 * Ct + self.c_s2) * np.sqrt(beta)

        # Calculate sigma_tilde (Eq 9, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
        x_tilde = np.abs(x_locations - turbine_coord.x1) / turbine.rotor_diameter
        sigma_y = k * x_tilde + eps

        # xR, _ = GaussianModel.mask_upstream_wake(y_locations, turbine_coord, yaw)
        # uR, u0 = GaussianModel.initial_velocity_deficits(U_local, Ct)
        #
        # sigma_y0, sigma_z0 = GaussianModel.initial_wake_expansion(
        #     turbine, U_local, veer, uR, u0
        # )

        # quantity that determines when the far wake starts
        # x0 = (
        #     D
        #     * (cosd(yaw) * (1 + np.sqrt(1 - Ct)))
        #     / (
        #         np.sqrt(2)
        #         * (4 * self.alpha * TI + 2 * self.beta * (1 - np.sqrt(1 - Ct)))
        #     )
        #     + turbine_coord.x1
        # )

        # velocity deficit in the near wake
        # sigma_y = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(
        #     Ct / 2.0
        # ) + ((x_locations - xR) / (x0 - xR)) * sigma_y0
        # sigma_z = (((x0 - xR) - (x_locations - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(
        #     Ct / 2.0
        # ) + ((x_locations - xR) / (x0 - xR)) * sigma_z0
        # sigma_y[x_locations < xR] = 0.5 * D
        # sigma_z[x_locations < xR] = 0.5 * D

        # sigma_y[x_locations > x0] = 0.0 * D
        # sigma_z[x_locations > x0] = 0.0 * D

        # wake expansion in the lateral (y) and the vertical (z)
        # ky = self.ka * TI + self.kb  # wake expansion parameters
        # kz = self.ka * TI + self.kb  # wake expansion parameters
        # sigma_y1 = ky * (x_locations - x0) + sigma_y0
        # sigma_z1 = kz * (x_locations - x0) + sigma_z0
        # # sigma_y1[x_locations < x0] = sigma_y0[x_locations < x0]
        # # sigma_z1[x_locations < x0] = sigma_z0[x_locations < x0]
        # sigma_y1[x_locations < x0] = 0.0
        # sigma_z1[x_locations < x0] = 0.0
        #
        # sigma_y = sigma_y + sigma_y1
        # sigma_z = sigma_z + sigma_z1

        return sigma_y

    def calc_Un(self, U_0, Cn, sigma_n, y, y_n, z, z_n):
        """
        Calculates U_n.

        Args:
            U_0 ([type]): [description]
            Cn ([type]): [description]
            sigma_n ([type]): [description]
            y ([type]): [description]
            y_n ([type]): [description]
            z ([type]): [description]
            z_n ([type]): [description]
        """

    def calc_Cn(self, lambdas, Ci, sigma_n, U_inf, U_n_1, Ct):
        """
        Calculates C_n.

        Args:
            lambdas ([type]): [description]
            Ci ([type]): [description]
            sigma_n ([type]): [description]
            U_inf ([type]): [description]
            U_n_1 ([type]): [description]
            Ct ([type]): [description]

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """

    def calc_lambda_ni(self, sigma_i, sigma_n, y_i, y_n, z_i, z_n, delta):
        """
        Calculates lambda_ni.

        Args:
            sigma_i ([type]): [description]
            sigma_n ([type]): [description]
            y_i ([type]): [description]
            y_n ([type]): [description]
            z_i ([type]): [description]
            z_n ([type]): [description]
            delta ([type]): [description]

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """

    @property
    def ka(self):
        """
        Parameter used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._ka

    @ka.setter
    def ka(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for ka: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._ka = value
        if value != __class__.default_parameters["ka"]:
            self.logger.info(
                (
                    "Current value of ka, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["ka"])
            )

    @property
    def kb(self):
        """
        Parameter used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._kb

    @kb.setter
    def kb(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for kb: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._kb = value
        if value != __class__.default_parameters["kb"]:
            self.logger.info(
                (
                    "Current value of kb, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["kb"])
            )

    @property
    def alpha_mod(self):
        """
        Parameter used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._alpha_mod

    @alpha_mod.setter
    def alpha_mod(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for alpha_mod: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._alpha_mod = value
        if value != __class__.default_parameters["alpha_mod"]:
            self.logger.info(
                (
                    "Current value of alpha_mod, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["alpha_mod"])
            )

    @property
    def sigma_gch(self):
        """
        Parameter used to determine the linear relationship between the
        turbulence intensity and the width of the Gaussian wake shape.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._sigma_gch

    @sigma_gch.setter
    def sigma_gch(self, value):
        if type(value) is not bool:
            err_msg = (
                "Invalid value type given for sigma_gch: {}, " + "expected boolean."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._sigma_gch = value
        if value != __class__.default_parameters["sigma_gch"]:
            self.logger.info(
                (
                    "Current value of sigma_gch, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["sigma_gch"])
            )
    
    @property
    def a_s(self):
        """
        Constant coefficient used in calculation of wake expansion. See
        Eqn. 9 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._a_s

    @a_s.setter
    def a_s(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for a_s: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._a_s = value
        if value != __class__.default_parameters["a_s"]:
            self.logger.info(
                (
                    "Current value of a_s, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["a_s"])
            )

    @property
    def b_s(self):
        """
        Constant coefficient used in calculation of wake expansion. See
        Eqn. 9 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._b_s

    @b_s.setter
    def b_s(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for b_s: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._b_s = value
        if value != __class__.default_parameters["b_s"]:
            self.logger.info(
                (
                    "Current value of b_s, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["b_s"])
            )

    @property
    def c_s1(self):
        """
        Linear constant used in calculation of wake expansion. See
        Eqn. 9 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._c_s1

    @c_s1.setter
    def c_s1(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for c_s1: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c_s1 = value
        if value != __class__.default_parameters["c_s1"]:
            self.logger.info(
                (
                    "Current value of c_s1, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["c_s1"])
            )

    @property
    def c_s2(self):
        """
        Linear constant used in calculation of wake expansion. See
        Eqn. 9 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._c_s2

    @c_s2.setter
    def c_s2(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for c_s2: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c_s2 = value
        if value != __class__.default_parameters["c_s2"]:
            self.logger.info(
                (
                    "Current value of c_s2, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["c_s2"])
            )

    @property
    def a_f(self):
        """
        """
        return self._a_f

    @a_f.setter
    def a_f(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for a_f: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._a_f = value
        if value != __class__.default_parameters["a_f"]:
            self.logger.info(
                (
                    "Current value of a_f, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["a_f"])
            )

    @property
    def b_f(self):
        """
        """
        return self._b_f

    @b_f.setter
    def b_f(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for b_f: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._b_f = value
        if value != __class__.default_parameters["b_f"]:
            self.logger.info(
                (
                    "Current value of b_f, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["b_f"])
            )



    @property
    def c_f(self):
        """
        Linear constant used in calculation of the super-Gaussian order. See
        Eqn. 13 in [1] and Table 4 in [3].

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Raises:
            ValueError: Invalid value.
        """
        return self._c_f

    @c_f.setter
    def c_f(self, value):
        if type(value) is not float and type(value) is not int:
            err_msg = (
                "Invalid value type given for c_f: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._c_f = value
        if value != __class__.default_parameters["c_f"]:
            self.logger.info(
                (
                    "Current value of c_f, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["c_f"])
            )
