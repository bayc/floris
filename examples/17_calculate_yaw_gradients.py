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


import copy

import numpy as np

from floris.tools import FlorisInterface


"""
This example creates a FLORIS instance
1) Makes a two-turbine layout
2) Demonstrates single ws/wd simulations
3) Demonstrates mulitple ws/wd simulations

Main concept is introduce FLORIS and illustrate essential structure of most-used FLORIS calls
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("inputs/gch.yaml")

x = [0, 500.]
y = [0., 0.]

# Convert to a simple two turbine layout
fi.reinitialize( layout=( x, y), wind_directions=[270., 315.] )

# yaw_angles = np.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
yaw_angles = np.zeros((1, 1, 3))

# print(np.shape(yaw_angles))

fi.calculate_wake(yaw_angles=yaw_angles)
init_power = fi.get_farm_power()

def yaw_finite_diff(fi, yaw_angles, eps):
    fi_grad = copy.deepcopy(fi)
    wd = fi_grad.floris.flow_field.wind_directions
    # wd = np.reshape(wd, (len(wd), 1))
    # ws = fi_grad.floris.flow_field.wind_speeds
    n_turbs = np.shape(yaw_angles)[2]
    yaw_angles_tiled = np.tile(yaw_angles, (n_turbs, 1, 1))
    wd_tiled = np.tile(wd, (n_turbs, 1, 1)).reshape((n_turbs * len(wd), 1, 1))

    fi_grad.reinitialize(wind_directions=wd_tiled)

    yaw_angles_forward = copy.deepcopy(yaw_angles_tiled) \
        + np.tile(np.eye(n_turbs), len(wd)).reshape(np.shape(yaw_angles_tiled)) * eps
    yaw_angles_backward = copy.deepcopy(yaw_angles_tiled) \
        - np.tile(np.eye(n_turbs), len(wd)).reshape(np.shape(yaw_angles_tiled)) * eps

    fi_grad.calculate_wake(yaw_angles=yaw_angles_forward)
    forward_power = fi_grad.get_farm_power()

    fi_grad.calculate_wake(yaw_angles=yaw_angles_backward)
    backward_power = fi_grad.get_farm_power()

    yaw_gradient = (forward_power - backward_power) / (2 * eps)

    return yaw_gradient

def layout_finite_diff(fi, x, y, yaw_angles, eps):
    fi_grad = copy.deepcopy(fi)
    # wd = fi_grad.floris.flow_field.wind_directions
    # n_turbs = np.shape(x)[0]

    # x_forward = copy.deepcopy(x) + eps
    # x_backward = copy.deepcopy(x) - eps
    # y_forward = copy.deepcopy(y) + eps
    # y_backward = copy.deepcopy(y) - eps

    fi_grad.reinitialize
    fi_grad.calculate_wake(yaw_angles=yaw_angles)
    forward_power = fi_grad.get_farm_power()

    fi_grad.calculate_wake(yaw_angles=yaw_angles)
    backward_power = fi_grad.get_farm_power()

    yaw_gradient = (forward_power - backward_power) / (2 * eps)

    return yaw_gradient

print(layout_finite_diff(fi, x, y, 1e-4))
