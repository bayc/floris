# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

# This example optimization takes a 1x5 array of turbines at an initial spacing
# of 9 rotor diameters and works to compress that spacing in the streamwise (x)
# direction, while working to maintain the original energy output by leveraging
# wake steering. It is meant to be an illustrative example of some of the
# benefits of wake steering.

import os

import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct
import floris.tools.visualization as vis
from floris.tools.optimization.scipy.power_density_1D import PowerDensityOptimization1D


# Instantiate the FLORIS object
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, "../../../example_input.json")
)

# Set turbine locations in a line
nturbs = 2
D = fi.floris.farm.turbines[0].rotor_diameter
spacing = 7 * D
min_dist = 3 * D
layout_x = [i * spacing for i in range(nturbs)]
layout_y = [0 for _ in range(nturbs)]
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Wind inputs
wd = [270]
ws = [8]
freq = [1]

# Set optimization options
opt_options = {"maxiter": 250, "disp": True, "iprint": 2, "ftol": 1e-7}

# Compute initial AEP for optimization normalization
AEP_initial = fi.get_farm_AEP(wd, ws, freq)

# Set initial conditions for optimization (scaled between 0 and 1)
x0 = []
inc = 1 / (nturbs - 1)

for i in range(nturbs):
    x1 = i * inc
    x0.append(x1)

for i in range(nturbs):
    if i == 0:
        x0.append(20.0)
    else:
        x0.append(0.0)

# Instantiate the layout otpimization object
powdens_opt = PowerDensityOptimization1D(
    fi=fi,
    wd=wd,
    ws=ws,
    freq=freq,
    AEP_initial=AEP_initial,
    x0=x0,
    opt_options=opt_options,
    min_dist=min_dist,
)

# Perform layout optimization
powdens_results = powdens_opt.optimize()

print("=====================================================")
print("Layout coordinates: ")
for i in range(len(powdens_results[0])):
    print(
        "Turbine",
        i,
        ": \tx = ",
        "{:.1f}".format(powdens_results[0][i]),
        "\ty = ",
        "{:.1f}".format(layout_y[i]),
    )

print("=====================================================")
print("Yaw angles: ")
for i in range(len(powdens_results[0])):
    print(
        "Turbine", i, ": \tyaw = ", "{:.1f}".format(powdens_results[1][i]),
    )

# Calculate new AEP results
fi.reinitialize_flow_field(layout_array=(powdens_results[0], layout_y))
fi.floris.farm.set_yaw_angles(powdens_results[1])
AEP_optimized = fi.get_farm_AEP(wd, ws, freq)

print("=====================================================")
print("AEP Ratio = %.1f%%" % (100.0 * AEP_optimized / AEP_initial))
print(
    "Space Reduction = %.1f%%" % (100.0 * np.max(powdens_results[0]) / np.max(layout_x))
)
print("=====================================================")

np.set_printoptions(precision=1)
print("x_layout: ", np.array(powdens_results[0]))
print("yaw: ", np.array(powdens_results[1]))

# Plot the new layout vs the old layout
powdens_opt.plot_layout_opt_results()
plt.show()
