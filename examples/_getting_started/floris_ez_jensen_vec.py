import time

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numpy import newaxis as na

import riptable as rt


def cosd(angle):
    return np.cos(np.radians(angle))


def sind(angle):
    return np.sin(np.radians(angle))


def rotate_fields(limits, mesh_x, mesh_y, wd):
    # Find center of rotation
    x_center_of_rotation = (limits[0] + limits[1]) / 2
    y_center_of_rotation = (limits[2] + limits[3]) / 2

    # Convert from compass rose angle to cartesian angle
    angle = ((wd - 270) % 360 + 360) % 360
    # np.ones_like()

    # print(np.shape(angle))
    # print(np.shape(mesh_x))
    # ((np.shape(angle)[0],) + np.shape(mesh_x)[1:])
    # mesh_x_new = np.ones((72, 1, 200, 100, 7))
    # np.multiply(mesh_x_new, mesh_x, out=mesh_x_new)
    # mesh_y_new = np.ones((72, 1, 200, 100, 7))
    # np.multiply(mesh_y_new, mesh_y, out=mesh_y_new)
    # print(np.shape(mesh_x * tmp))

    # Rotate grid points
    x_offset = mesh_x - x_center_of_rotation
    y_offset = mesh_y - y_center_of_rotation
    mesh_x_rotated = (
        x_offset * cosd(angle) - y_offset * sind(angle) + x_center_of_rotation
    )
    mesh_y_rotated = (
        x_offset * sind(angle) + y_offset * cosd(angle) + y_center_of_rotation
    )

    # print(np.shape(mesh_x_rotated))
    # lkj

    # Rotate turbine coordinates
    x_coord_offset = x_coord - x_center_of_rotation
    y_coord_offset = y_coord - y_center_of_rotation
    x_coord_rotated = (
        x_coord_offset * cosd(angle)
        - y_coord_offset * sind(angle)
        + x_center_of_rotation
    )
    y_coord_rotated = (
        x_coord_offset * sind(angle)
        + y_coord_offset * cosd(angle)
        + y_center_of_rotation
    )

    # Return rotated values
    return mesh_x_rotated, mesh_y_rotated, x_coord_rotated, y_coord_rotated


# @njit(fastmath=True)
def jensen_model(
    flow_field_u_initial,
    u_wake,
    turbine_ai,
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
    turbine_diameter,
):
    # Wake expansion parameter
    we = 0.05

    x = mesh_x_rotated - x_coord_rotated
    # Calculate the wake velocity deficit ratios
    # c = (turbine_diameter / (2 * we * (mesh_x_rotated - x_coord_rotated) + turbine_diameter)) ** 2
    c = (turbine_diameter / (2 * we * (x) + turbine_diameter)) ** 2

    # Calculate and apply wake mask
    m = we
    x = mesh_x_rotated - x_coord_rotated
    b = turbine_diameter / 2.0

    boundary_line = m * x + b

    y_upper = boundary_line + y_coord_rotated  # + deflection_field
    y_lower = -1 * boundary_line + y_coord_rotated  # + deflection_field
    z_upper = boundary_line + turbine_hub_height
    z_lower = -1 * boundary_line + turbine_hub_height

    c[mesh_x_rotated - x_coord_rotated < 0] = 0
    c[mesh_y_rotated > y_upper] = 0
    c[mesh_y_rotated < y_lower] = 0
    c[mesh_z > z_upper] = 0
    c[mesh_z < z_lower] = 0

    # Calculate the wake velocity deficits
    # u_wake = 2 * turbine_ai * c * flow_field_u_initial

    # print(np.shape(c))
    # print(np.shape(flow_field_u_initial))
    # lkj

    # Apply the velocity deficit field to the freestream
    # return flow_field_u_initial - u_wake
    return flow_field_u_initial - 2 * turbine_ai * c * flow_field_u_initial


# ///// #
# SETUP #
# ///// #

# Turbine parameters
turbine_diameter = 126.0
turbine_radius = turbine_diameter / 2.0
turbine_hub_height = 90.0
turbine_ai = 1 / 3

x_coord = np.array([0.0])
y_coord = np.array([0.0])

dtype = np.float64
# Wind parameters
ws = np.array([6.0] * 25, dtype=dtype)  # jklm
wd = np.array([270.0] * 72, dtype=dtype)  # ijklm
# wd = np.array([15 for i in range(100)])[:, na, na, na, na] # ijklm
# i  j  k   l   m
# 72 25 200 100 7
# wd ws x   y   z

# 1 25 200 100 7
# 72 1 200 100 7

specified_wind_height = 90.0
wind_shear = 0.12


# ////////////////////// #
# FULL FLOW FIELD POINTS #
# ////////////////////// #
# # Flow field bounds
# xmin = np.min(x_coord) - 9.99 * turbine_diameter
# xmax = np.max(x_coord) + 10 * turbine_diameter
# ymin = np.min(y_coord) - 10 * turbine_diameter
# ymax = np.max(y_coord) + 10 * turbine_diameter
# zmin = 0.1
# zmax = 6 * specified_wind_height
# limits = [xmin, xmax, ymin, ymax, zmin, zmax]

# # Flow field resolutions
# resolution_x1 = 200
# resolution_x2 = 100
# resolution_x3 = 7


# ///////////////// #
# ONLY ROTOR POINTS #
# ///////////////// #
# Flow field bounds
rotor_point_width = 0.25
xmin = x_coord[0]
xmax = x_coord[0]
ymin = y_coord[0] - rotor_point_width * turbine_diameter
ymax = y_coord[0] + rotor_point_width * turbine_diameter
zmin = turbine_hub_height - rotor_point_width * turbine_diameter
zmax = turbine_hub_height + rotor_point_width * turbine_diameter
limits = [xmin, xmax, ymin, ymax, zmin, zmax]

# Flow field resolutions
resolution_x1 = 1
resolution_x2 = 5
resolution_x3 = 5

x = np.linspace(xmin, xmax, int(resolution_x1), dtype=dtype)
y = np.linspace(ymin, ymax, int(resolution_x2), dtype=dtype)
z = np.linspace(zmin, zmax, int(resolution_x3), dtype=dtype)

# Flow field grid points
mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z, indexing="ij")

mesh_x = mesh_x[na, na, na, :, :, :]  # * np.ones((72, 1, 200, 100, 7))
mesh_y = mesh_y[na, na, na, :, :, :]  # * np.ones((72, 1, 200, 100, 7))
mesh_z = mesh_z[na, na, na, :, :, :]  # * np.ones((72, 1, 200, 100, 7))

# ///////////////// #
# JENSEN WAKE MODEL #
# ///////////////// #

# VECTORIZED CALLS
# Initialize field values
flow_field_u_initial = (
    ws[na, na, :, na, na, na] * (mesh_z / specified_wind_height) ** wind_shear
)
u_wake = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)
deflection_field = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)

tottic = time.perf_counter()
tic = time.perf_counter()
mesh_x_rotated, mesh_y_rotated, x_coord_rotated, y_coord_rotated = rotate_fields(
    limits, mesh_x, mesh_y, wd[na, :, na, na, na, na]
)
toc = time.perf_counter()
print(f"Computed rotation of vectorized grids in {toc - tic:0.4f} seconds")

# print(np.shape(mesh_x_rotated))
# lkj

tic = time.perf_counter()
flow_field_u = jensen_model(
    flow_field_u_initial,
    u_wake,
    turbine_ai,
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
    turbine_diameter,
)
toc = time.perf_counter()
print(f"Computed vectorized Jensen model in {toc - tic:0.4f} seconds")
tottoc = time.perf_counter()
print(f"Total time for vectorized: {tottoc - tottic:0.4f} seconds")

# SERIAL CALLS
tottic = time.perf_counter()
for i in range(len(wd)):

    # tic = time.perf_counter()
    mesh_x_rotated, mesh_y_rotated, x_coord_rotated, y_coord_rotated = rotate_fields(
        limits, mesh_x, mesh_y, wd[i]
    )
    # toc = time.perf_counter()
    # print(f"Computed rotation of serial grid in {toc - tic:0.4f} seconds")

    # tic = time.perf_counter()
    for j in range(len(ws)):
        # Initialize field values
        flow_field_u_initial = ws[j] * (mesh_z / specified_wind_height) ** wind_shear
        u_wake = np.zeros(np.shape(flow_field_u_initial))

        flow_field_u = jensen_model(
            flow_field_u_initial,
            u_wake,
            turbine_ai,
            mesh_x_rotated,
            mesh_y_rotated,
            mesh_z,
            x_coord_rotated,
            y_coord_rotated,
            turbine_diameter,
        )

    # toc = time.perf_counter()
    # print(f"Computed serial Jensen model in {toc - tic:0.4f} seconds")

tottoc = time.perf_counter()
print(f"Total time for serial: {tottoc - tottic:0.4f} seconds")

# ///////////// #
# PLOT THE WAKE #
# ///////////// #

plot_wakes = False
if plot_wakes is True:

    minSpeed = 2.0
    maxSpeed = 10.0
    cmap = "coolwarm"

    fontsize = 6
    plt.rcParams["font.size"] = fontsize - 2
    fig, axarr = plt.subplots(np.shape(wd)[0], np.shape(ws)[1])

    for i in range(np.shape(wd)[0]):
        for j in range(np.shape(ws)[1]):
            # Determine mask for z-slice/horizontal slice
            mask = (mesh_z > 90.0) & (mesh_z < 91.0)

            x_grid_length = len(np.unique(mesh_x[mask]))
            y_grid_length = len(np.unique(mesh_y[mask]))
            x_plot = np.reshape(mesh_x[mask], (x_grid_length, y_grid_length))
            y_plot = np.reshape(mesh_y[mask], (x_grid_length, y_grid_length))
            z_plot = np.reshape(
                flow_field_u[i][j][mask], (x_grid_length, y_grid_length)
            )

            im = axarr[i][j].pcolormesh(
                x_plot,
                y_plot,
                z_plot,
                cmap=cmap,
                vmin=minSpeed,
                vmax=maxSpeed,
                shading="nearest",
            )

            axarr[i][j].contour(x_plot, y_plot, z_plot)

            # Make equal axis
            axarr[i][j].set_aspect("equal")
            axarr[i][j].set_title(
                "wd: " + str(wd[i][0][0][0][0]) + ", ws: " + str(ws[0][j][0][0][0]),
                fontsize=fontsize,
            )

    plt.show()
