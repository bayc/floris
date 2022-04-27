import meshio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from floris.tools import FlorisInterface
from floris.tools.floris_interface import generate_heterogeneous_wind_map
from floris.tools.visualization import visualize_cut_plane

data = pd.read_csv('windse_flows/hh_90_vel.csv')

# print(data)
speed_ups = data['velocity:0'] / 8.0
x_locs = data['Points:0']
y_locs = data['Points:1']

print('xmin: ', np.min(x_locs))
print('xmax: ', np.max(x_locs))

print('ymin: ', np.min(y_locs))
print('ymax: ', np.max(y_locs))

# Generate the linear interpolation to be used for the heterogeneous inflow.
het_map_2d = generate_heterogeneous_wind_map([speed_ups], x_locs, y_locs)

# Initialize FLORIS with the given input file via FlorisInterface.
# Also, pass the heterogeneous map into the FlorisInterface.
fi_2d = FlorisInterface("../inputs/gch.yaml", het_map=het_map_2d)

# Set shear to 0.0 to highlight the heterogeneous inflow
fi_2d.reinitialize(layout=[[1300.0], [0.0]], wind_shear=0.0)

# Using the FlorisInterface functions for generating plots, run FLORIS
# and extract 2D planes of data.
horizontal_plane_2d = fi_2d.calculate_horizontal_plane(x_resolution=200, y_resolution=100, height=90.0, x_bounds=(-1260.0, 1260.0), y_bounds=(-1260.0, 1260.0))
y_plane_2d = fi_2d.calculate_y_plane(x_resolution=200, z_resolution=100, crossstream_dist=0.0, x_bounds=(-1260.0, 1260.0), z_bounds=(0.0, 3*126.0))
cross_plane_2d = fi_2d.calculate_cross_plane(y_resolution=100, z_resolution=100, downstream_dist=0.0, y_bounds=(-1260.0, 1260.0), z_bounds=(0.0, 3*126.0))

fig, ax_list = plt.subplots(3, 1, figsize=(10, 8))
ax_list = ax_list.flatten()
# visualize_cut_plane(horizontal_plane_2d, ax=ax_list[0], title="Horizontal at z=90m", color_bar=True)
# ax_list[0].set_xlabel('x'); ax_list[0].set_ylabel('y')
visualize_cut_plane(y_plane_2d, ax=ax_list[1], title="Streamwise profile at y=0m", color_bar=True)
ax_list[1].set_xlabel('x'); ax_list[1].set_ylabel('z')
visualize_cut_plane(cross_plane_2d, ax=ax_list[2], title="Spanwise profile at x=0m", color_bar=True)
ax_list[2].set_xlabel('y'); ax_list[2].set_ylabel('z')

plt.savefig('floris_2D_flow_all.png')
plt.show()

# mesh = meshio.read("windse_flows/velocity000000.vtu")
# # mesh = meshio.read("windse_flows/velocity_p1_000000.vtu")

# x = mesh.points[:, 0]
# y = mesh.points[:, 1]
# z = mesh.points[:, 2]

# mask_u = np.array(z < 150)
# mask_l = np.array(z > 140)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(
#     x * mask_u * mask_l,
#     y * mask_u * mask_l,
#     z * mask_u * mask_l,
#     marker='o'
# )

# plt.show()

# print(dir(mesh))

# print(np.sum(mesh.points[:,2] > 59))
# print(mesh.point_data)