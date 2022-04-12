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


import matplotlib.pyplot as plt

from floris.tools import FlorisInterface
from floris.tools.floris_interface import generate_heterogeneous_wind_map
from floris.tools.visualization import visualize_cut_plane

"""
This example showcases the heterogeneous inflow capabilities of FLORIS.
Heterogeneous flow can be defined in either 2- or 3-dimensions.

For the 2-dimensional case, it can be seen that the freestream velocity
only varies in the x direction. For the 3-dimensional case, it can be
seen that the freestream velocity only varies in the z direction. This
is because of how the speed ups for each case were defined. More complex
inflow conditions can be defined.

For each case, we are plotting three slices of the resulting flow field:
1. Horizontal slice parallel to the ground and located at the hub height
2. Vertical slice parallel with the direction of the wind
3. Veritical slice parallel to to the turbine disc plane
"""


# Define the speed ups of the heterogeneous inflow, and their locations.
# For the 2-dimensional case, this requires x and y locations.
# The speed ups are multipliers of the ambient wind speed.
speed_ups_x = [[1.2, 1.2, 1.0, 1.0]]
speed_ups_y = [[3.0, 3.0, 0.0, 0.0]] # note the y speed ups are in absolute units and not used as scalars
x_locs = [-300.0, -300.0, 2600.0, 2600.0]
y_locs = [ -300.0, 300.0, -300.0, 300.0]

# Generate the linear interpolation to be used for the heterogeneous inflow.
het_map_2d_x = generate_heterogeneous_wind_map(speed_ups_x, x_locs, y_locs)
het_map_2d_y = generate_heterogeneous_wind_map(speed_ups_y, x_locs, y_locs)

# Initialize FLORIS with the given input file via FlorisInterface.
# Also, pass the heterogeneous map into the FlorisInterface.
fi_2d = FlorisInterface("inputs/gch.yaml", het_map_x=het_map_2d_x, het_map_y=het_map_2d_y)

fi_2d.reinitialize(wind_directions=[270.], wind_speeds=[8.0])

fi_2d.calculate_wake()

print(fi_2d.get_turbine_powers())

# Set shear to 0.0 to highlight the heterogeneous inflow
fi_2d.reinitialize(wind_shear=0.0)

# Using the FlorisInterface functions for generating plots, run FLORIS
# and extract 2D planes of data.
horizontal_plane_2d = fi_2d.calculate_horizontal_plane(x_resolution=200, y_resolution=100, height=90.0)
y_plane_2d = fi_2d.calculate_y_plane(x_resolution=200, z_resolution=100, crossstream_dist=0.0)
cross_plane_2d = fi_2d.calculate_cross_plane(y_resolution=100, z_resolution=100, downstream_dist=500.0)

# Create the plots
fig, ax_list = plt.subplots(3, 1, figsize=(10, 8))
ax_list = ax_list.flatten()
visualize_cut_plane(horizontal_plane_2d, ax=ax_list[0], title="Horizontal", color_bar=True)
ax_list[0].set_xlabel('x'); ax_list[0].set_ylabel('y')
visualize_cut_plane(y_plane_2d, ax=ax_list[1], title="Streamwise profile", color_bar=True)
ax_list[1].set_xlabel('x'); ax_list[1].set_ylabel('z')
visualize_cut_plane(cross_plane_2d, ax=ax_list[2], title="Spanwise profile at 500m downstream", color_bar=True)
ax_list[2].set_xlabel('y'); ax_list[2].set_ylabel('z')

plt.show()
