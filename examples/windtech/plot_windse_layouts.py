import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from floris.tools import FlorisInterface
# import floris.tools.optimization.pyoptsparse as opt
from floris.tools.floris_interface import generate_heterogeneous_wind_map
from floris.tools.visualization import visualize_cut_plane

windse_data = pd.read_csv('windse_results/alt_initial_layout_3_opt_prog.csv')
file_name = 'windse_floris_1_layout.png'
# windse_data = pd.read_csv('windse_results/alt_random_layout_1_opt_prog.csv')
# file_name = 'windse_random_layout_1.png'

xlocs = []
ylocs = []
for i in range(len(windse_data)):
    xlocs.append([
        windse_data[' T_1_x (m)'][i],
        windse_data[' T_2_x (m)'][i],
        windse_data[' T_3_x (m)'][i],
        windse_data[' T_4_x (m)'][i],
        windse_data[' T_5_x (m)'][i],
        windse_data[' T_6_x (m)'][i],
        windse_data[' T_7_x (m)'][i]
    ])

    ylocs.append([
        windse_data[' T_1_y (m)'][i],
        windse_data[' T_2_y (m)'][i],
        windse_data[' T_3_y (m)'][i],
        windse_data[' T_4_y (m)'][i],
        windse_data[' T_5_y (m)'][i],
        windse_data[' T_6_y (m)'][i],
        windse_data[' T_7_y (m)'][i]
    ])

data = pd.read_csv('windse_flows/hh_90_vel.csv')

speed_ups = data['velocity:0'] / 8.0
x_locs = data['Points:0']
y_locs = data['Points:1']

# Generate the linear interpolation to be used for the heterogeneous inflow.
het_map_2d = generate_heterogeneous_wind_map([speed_ups], x_locs, y_locs)

# Initialize FLORIS with the given input file via FlorisInterface.
# Also, pass the heterogeneous map into the FlorisInterface.
fi = FlorisInterface("../inputs/gch.yaml", het_map=het_map_2d)

locsx = xlocs[-1]
locsy = ylocs[-1]
x0 = xlocs[0]
y0 = ylocs[0]

fi.reinitialize(layout=[np.array(x0), np.array(y0)])
fi.calculate_wake()
power_initial = fi.get_farm_power()

fi.reinitialize(layout=[np.array(locsx), np.array(locsy)])
fi.calculate_wake()
power_final = fi.get_farm_power()

print('inital: ', power_initial)
print('final: ', power_final)
lkj

horizontal_plane_2d = fi.calculate_horizontal_plane(x_resolution=200, y_resolution=200, height=90.0, x_bounds=(-1260.0, 1260.0), y_bounds=(-1260.0, 1260.0))

# plt.figure(figsize=(9, 6))
visualize_cut_plane(horizontal_plane_2d, color_bar=True)

fontsize = 16
plt.plot(x0, y0, "ob")
plt.plot(locsx, locsy, "or")
# plt.title('Layout Optimization Results', fontsize=fontsize)
plt.xlabel("x (m)", fontsize=fontsize)
plt.ylabel("y (m)", fontsize=fontsize)
plt.axis("equal")
# plt.grid()
plt.tick_params(which="both", labelsize=fontsize)
plt.legend(
    ["Old locations", "New locations"],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    fontsize=fontsize,
)

scale = 2
boundaries = [(-1260.0/scale, -1260.0/scale), (-1260.0/scale, 1260.0/scale), (1260.0/scale, 1260.0/scale), (1260.0/scale, -1260.0/scale), (-1260.0/scale, -1260.0/scale)]

verts = boundaries
for i in range(len(verts)):
    if i == len(verts) - 1:
        plt.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "b")
    else:
        plt.plot(
            [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "b"
        )

plt.savefig(file_name, dpi=300)
plt.show()
