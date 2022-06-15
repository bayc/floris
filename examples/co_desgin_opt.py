# Copyright 2022 NREL

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

from floris.tools import FlorisInterface

from floris.tools.optimization.layout_optimization.co_design_optimization_pyoptsparse import CoDesignOptimizationPyOptSparse
from floris.tools.floris_interface import generate_heterogeneous_wind_map

"""
This example shows a simple layout optimization using the python module pyOptSparse.

A 4 turbine array is optimized such that the layout of the turbine produces the
highest annual energy production (AEP) based on the given wind resource. The turbines
are constrained to a square boundary and a randomw wind resource is supplied. The results
of the optimization show that the turbines are pushed to the outer corners of the boundary,
which makes sense in order to maximize the energy production by minimizing wake interactions.
"""

minx = 0.0
maxx = 1000.0
miny = 0.0
maxy = 1000.0

speed_ups = [[1.5, 1.5, 1.0, 1.0], [1.5, 1.5, 1.0, 1.0]]
x_locs = [minx, minx, maxx, maxx]
y_locs = [miny, maxy, miny, maxy]
het_map_2d = generate_heterogeneous_wind_map(speed_ups, x_locs, y_locs)

# Initialize the FLORIS interface fi
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = FlorisInterface('inputs/gch.yaml', het_map=het_map_2d)

# Setup 72 wind directions with a random wind speed and frequency distribution
wind_directions = np.array([270.0, 315.0])
wind_speeds = np.array([8.0])
freq = np.array([1.0])
# wind_directions = np.arange(0, 360.0, 5.0)
# np.random.seed(1)
# wind_speeds = 8.0 + np.random.randn(1) * 0.5
# freq = np.abs(np.sort(np.random.randn(len(wind_directions))))
# freq = freq / freq.sum()
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)

# The boundaries for the turbines, specified as vertices
boundaries = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)]

# Set turbine locations to 4 turbines in a rectangle
D = 126.0 # rotor diameter for the NREL 5MW
layout_x = [0, 0, 6 * D, 6 * D]
layout_y = [0, 4 * D, 0, 4 * D]
fi.reinitialize(layout=(layout_x, layout_y))
fi.calculate_wake()

# Setup the optimization problem

# layout_opt = LayoutOptimizationScipy(fi, boundaries, freq=freq)
layout_opt = CoDesignOptimizationPyOptSparse(fi, boundaries, solver='SNOPT')
# model = opt.layout.Layout(fi, boundaries, freq)
# tmp = opt.optimization.Optimization(model=model, solver='SLSQP')

# Run the optimization
sol = layout_opt.optimize()

# Print and plot the results
print(sol)

layout_opt.parse_sol_vars(sol)
print('x: ', layout_opt.x)
print('y: ', layout_opt.y)
print('yaw: ', layout_opt.yaw)
# layout_opt.plot_layout_opt_results(sol)
# layout_opt.plot_layout_opt_results()

file_name = 'results.png'
wd = [270.0]
ws = [8.0]
layout_opt.plot_layout_opt_results_with_flow(sol, file_name=file_name, wd=wd, ws=ws)
