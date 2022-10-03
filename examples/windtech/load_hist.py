from pyoptsparse import History
from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane
from floris.tools.floris_interface import generate_heterogeneous_wind_map
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _norm(val, x1, x2):
        return (val - x1) / (x2 - x1)

def _unnorm(val, x1, x2):
    return np.array(val) * (x2 - x1) + x1

hist = History("hist.hist", flag="r")

x_total = hist.getValues()['x']
y_total = hist.getValues()['y']

xmin = -1260.0 / 2
xmax = 1260.0 / 2
ymin = -1260.0 / 2
ymax = 1260.0 / 2

x_total = _unnorm(x_total, xmin, xmax)
y_total = _unnorm(y_total, ymin, ymax)

data = pd.read_csv('windse_flows/hh_90_vel.csv')

speed_ups = data['velocity:0'] / 8.0
x_locs = data['Points:0']
y_locs = data['Points:1']

# Generate the linear interpolation to be used for the heterogeneous inflow.
het_map_2d = generate_heterogeneous_wind_map([speed_ups], x_locs, y_locs)

for i in range(len(x_total)):
    print('Plotting ' + str(i))
    file_name = 'plots/floris_1_layout/layout_' + str(i) +'.png'
    fi = FlorisInterface("../inputs/gch.yaml", het_map=het_map_2d)

    x = x_total[i]
    y = y_total[i]
    fi.reinitialize(layout=(x, y))

    horizontal_plane_2d = fi.calculate_horizontal_plane(x_resolution=200, y_resolution=200, height=90.0, x_bounds=(-1260.0, 1260.0), y_bounds=(-1260.0, 1260.0))

    # plt.figure(figsize=(9, 6))
    visualize_cut_plane(horizontal_plane_2d, color_bar=True)

    fontsize = 16
    plt.plot(x, y, "ob")
    # plt.plot(locsx, locsy, "or")
    # plt.title('Layout Optimization Results', fontsize=fontsize)
    plt.xlabel("x (m)", fontsize=fontsize)
    plt.ylabel("y (m)", fontsize=fontsize)
    plt.axis("equal")
    # plt.grid()
    plt.tick_params(which="both", labelsize=fontsize)
    # plt.legend(
    #     ["Old locations", "New locations"],
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 1.01),
    #     ncol=2,
    #     fontsize=fontsize,
    # )

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

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    # plt.show()
    plt.close()
