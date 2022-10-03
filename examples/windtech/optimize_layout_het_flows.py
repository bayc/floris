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


import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from floris.tools import FlorisInterface
import floris.tools.optimization.pyoptsparse as opt
from floris.tools.floris_interface import generate_heterogeneous_wind_map
from floris.tools.visualization import visualize_cut_plane

"""
This example shows a simple layout optimization using the python module pyOptSparse.

A 4 turbine array is optimized such that the layout of the turbine produces the
highest annual energy production (AEP) based on the given wind resource. The turbines
are constrained to a square boundary and a randomw wind resource is supplied. The results
of the optimization show that the turbines are pushed to the outer corners of the boundary,
which makes sense in order to maximize the energy production by minimizing wake interactions.
"""
def _norm(val, x1, x2):
    return (val - x1) / (x2 - x1)

def _unnorm(val, x1, x2):
    return np.array(val) * (x2 - x1) + x1

n_turbs = 7
n_cases = 5

# np.random.seed(1)
rng = np.random.default_rng(1)

xmin = -1260.0 / 2
xmax = 1260.0 / 2
ymin = -1260.0 / 2
ymax = 1260.0 / 2

# for i in range(2):
    # while 1:
    #     coord = rng.random((n_turbs, 2))
    #     x = _unnorm(coord[:, 0], xmin, xmax)
    #     y = _unnorm(coord[:, 1], ymin, ymax)    

    #     locs = np.vstack((x, y)).T
    #     distances = cdist(locs, locs)
    #     arange = np.arange(distances.shape[0])
    #     distances[arange, arange] = 1e10
    #     dist = np.min(distances, axis=0)

    #     g = 1 - np.array(dist) / (2 * 126.0)

    #     # Following code copied from OpenMDAO KSComp().
    #     # Constraint is satisfied when KS_constraint <= 0
    #     rho = 500
    #     g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
    #     g_diff = g - g_max
    #     exponents = np.exp(rho * g_diff)
    #     summation = np.sum(exponents, axis=-1)[:, np.newaxis]
    #     KS_constraint = g_max + 1.0 / rho * np.log(summation)

    #     # print('KS_constraint:', KS_constraint)
    #     if KS_constraint <= 0.0:
    #         print(x, y)
    #         break
# print(x, y)
# lkj

# print(coord)
# lkj

for i in range(n_cases):
    print('Optimizing case: ' + str(i))
    file_name = 'layout_opt_' + str(i)

    while 1:
        coord = rng.random((n_turbs, 2))
        x = _unnorm(coord[:, 0], xmin, xmax)
        y = _unnorm(coord[:, 1], ymin, ymax)    

        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        arange = np.arange(distances.shape[0])
        distances[arange, arange] = 1e10
        dist = np.min(distances, axis=0)

        g = 1 - np.array(dist) / (2 * 126.0)

        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        rho = 500
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        KS_constraint = g_max + 1.0 / rho * np.log(summation)

        # print('KS_constraint:', KS_constraint)
        if KS_constraint <= 0.0:
            print(x, y)
            break

    if i == 2:
        data = pd.read_csv('windse_flows/hh_90_vel.csv')

        # print(data)
        speed_ups = data['velocity:0'] / 8.0
        x_locs = data['Points:0']
        y_locs = data['Points:1']

        # Generate the linear interpolation to be used for the heterogeneous inflow.
        het_map_2d = generate_heterogeneous_wind_map([speed_ups], x_locs, y_locs)

        # Initialize FLORIS with the given input file via FlorisInterface.
        # Also, pass the heterogeneous map into the FlorisInterface.
        fi = FlorisInterface("../inputs/gch.yaml", het_map=het_map_2d)

        # Set shear to 0.0 to highlight the heterogeneous inflow
        fi.reinitialize(wind_shear=0.0)

        # Setup 72 wind directions with a random wind speed and frequency distribution
        # wind_directions = np.arange(0, 360.0, 5.0)
        # np.random.seed(1)
        # wind_speeds = 8.0 + np.random.randn(1) * 0.5
        # freq = np.abs(np.sort(np.random.randn(len(wind_directions))))
        # freq = freq / freq.sum()
        # fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)

        freq = np.array([1.0])

        scale = 2

        # The boundaries for the turbines, specified as vertices
        boundaries = [(-1260.0/scale, -1260.0/scale), (-1260.0/scale, 1260.0/scale), (1260.0/scale, 1260.0/scale), (1260.0/scale, -1260.0/scale), (-1260.0/scale, -1260.0/scale)]

        # Set turbine locations to 4 turbines in a rectangle
        D = 126.0 # rotor diameter for the NREL 5MW
        # layout_x = [-5 * D, -5 * D, 0 * D, 0 * D]
        # layout_y = [-5 * D, 5 * D, -1.0 * D, 1.0 * D]

        layout_x = x
        layout_y = y

        fi.reinitialize(layout=(layout_x, layout_y))
        fi.calculate_wake()
        base_power = fi.get_farm_power()

        # Setup the optimization problem
        model = opt.layout.Layout(fi, boundaries, freq)
        tmp = opt.optimization.Optimization(model=model, solver='SNOPT')

        # Run the optimization
        sol = tmp.optimize()

        # Print and plot the results
        print(sol)
        locsx, locsy, power = model.plot_layout_opt_results_with_flow(sol, file_name=file_name)

        print(layout_x, layout_y, locsx, locsy, power, base_power)
