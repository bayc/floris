# Copyright 2023 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np

from floris.tools import FlorisInterface
from tests.conftest import (
    N_TURBINES,
    N_WIND_SPEEDS,
    N_WIND_DIRECTIONS,
    print_test_values,
    assert_results_arrays,
)
from floris.simulation import Ct, power, axial_induction, average_velocity


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9803783, 0.7634300, 1695368.7987130, 0.2568077],
            [7.9803783, 0.7634300, 1695368.7987130, 0.2568077],
            [7.9803783, 0.7634300, 1695368.7987130, 0.2568077],
        ],
        # 9 m/s
        [
            [8.9779256, 0.7625731, 2413658.0981405, 0.2563676],
            [8.9779256, 0.7625731, 2413658.0981405, 0.2563676],
            [8.9779256, 0.7625731, 2413658.0981405, 0.2563676],
        ],
        # 10 m/s
        [
            [9.9754729, 0.7527803, 3306006.2306084, 0.2513940],
            [9.9754729, 0.7527803, 3306006.2306084, 0.2513940],
            [9.9754729, 0.7527803, 3306006.2306084, 0.2513940],
        ],
        # 11 m/s
        [
            [10.9730201, 0.7304328, 4373596.1594956, 0.2404007],
            [10.9730201, 0.7304328, 4373596.1594956, 0.2404007],
            [10.9730201, 0.7304328, 4373596.1594956, 0.2404007],
        ],
    ]
)


def test_calculate_no_wake(sample_inputs_fixture):
    """
    The calculate_no_wake function calculates the power production of a wind farm
    assuming no wake losses. It does this by initializing and finalizing the
    floris simulation while skipping the wake calculation. The power for all wind
    turbines should be the same for a uniform wind condition. The chosen wake model
    is not important since it will not actually be used. However, it is left enabled
    instead of using "None" so that additional tests can be constructed here such
    as one with yaw activated.
    """
    sample_inputs_fixture.floris["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.floris["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fi = FlorisInterface(sample_inputs_fixture.floris)
    fi.calculate_no_wake()

    n_turbines = fi.floris.farm.n_turbines
    n_wind_speeds = fi.floris.flow_field.n_wind_speeds
    n_wind_directions = fi.floris.flow_field.n_wind_directions

    velocities = fi.floris.flow_field.u
    yaw_angles = fi.floris.farm.yaw_angles
    test_results = np.zeros((n_wind_directions, n_wind_speeds, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = Ct(
        velocities,
        yaw_angles,
        fi.floris.farm.turbine_fCts,
        fi.floris.farm.turbine_type_map,
    )
    farm_powers = power(
        fi.floris.flow_field.air_density,
        fi.floris.farm.ref_density_cp_cts,
        velocities,
        yaw_angles,
        fi.floris.farm.pPs,
        fi.floris.farm.turbine_power_interps,
        fi.floris.farm.turbine_type_map,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        yaw_angles,
        fi.floris.farm.turbine_fCts,
        fi.floris.farm.turbine_type_map,
    )
    for i in range(n_wind_directions):
        for j in range(n_wind_speeds):
            for k in range(n_turbines):
                test_results[i, j, k, 0] = farm_avg_velocities[i, j, k]
                test_results[i, j, k, 1] = farm_cts[i, j, k]
                test_results[i, j, k, 2] = farm_powers[i, j, k]
                test_results[i, j, k, 3] = farm_axial_inductions[i, j, k]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
        )

    assert_results_arrays(test_results[0], baseline)
