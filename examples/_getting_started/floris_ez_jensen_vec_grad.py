import time

import matplotlib.pyplot as plt
from numpy import newaxis as na

# import numpy as np
import jax.numpy as np
from jax import grad


def cosd(angle):
    return np.cos(np.radians(angle))


def sind(angle):
    return np.sin(np.radians(angle))


def rotate_fields(limits, mesh_x, mesh_y, wd, x_coord, y_coord):
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
    deflection_field,
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
    # print('c: ', c)
    c[mesh_y_rotated > y_upper] = 0
    c[mesh_y_rotated < y_lower] = 0
    c[mesh_z > z_upper] = 0
    c[mesh_z < z_lower] = 0

    # Calculate the wake velocity deficits and apply to the freestream
    return flow_field_u_initial - 2 * turbine_ai * c * flow_field_u_initial


def jensen_model_masked(
    flow_field_u_initial,
    u_wake,
    turbine_ai,
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
    turbine_diameter,
    deflection_field,
):
    # Wake expansion parameter
    we = 0.05

    # Calculate and apply wake mask
    m = we
    x = mesh_x_rotated - x_coord_rotated
    b = turbine_diameter / 2.0

    boundary_line = m * x + b

    y_upper = boundary_line + y_coord_rotated  # + deflection_field
    y_lower = -1 * boundary_line + y_coord_rotated  # + deflection_field
    z_upper = boundary_line + turbine_hub_height
    z_lower = -1 * boundary_line + turbine_hub_height

    x = mesh_x_rotated - x_coord_rotated
    # Calculate the wake velocity deficit ratios
    # c = (turbine_diameter / (2 * we * (mesh_x_rotated - x_coord_rotated) + turbine_diameter)) ** 2
    # c = (turbine_diameter / (2 * we * (x) + turbine_diameter)) ** 2 \
    #     * np.logical_not(np.array(mesh_x_rotated - x_coord_rotated < 0)) \
    #     * np.logical_not(np.array(mesh_y_rotated > y_upper)) \
    #     * np.logical_not(np.array(mesh_y_rotated < y_lower)) \
    #     * np.logical_not(np.array(mesh_z > z_upper)) \
    #     * np.logical_not(np.array(mesh_z < z_lower))
    c = (
        (turbine_diameter / (2 * we * (x) + turbine_diameter)) ** 2
        * ~(np.array(mesh_x_rotated - x_coord_rotated < 0))
        * ~(np.array(mesh_y_rotated > y_upper))
        * ~(np.array(mesh_y_rotated < y_lower))
        * ~(np.array(mesh_z > z_upper))
        * ~(np.array(mesh_z < z_lower))
    )
    # print('masked c: ', c)

    # mask = np.array(mesh_z > z_upper)
    # print(c * mask)
    # lkj

    # c[mesh_x_rotated - x_coord_rotated < 0] = 0
    # c[mesh_y_rotated > y_upper] = 0
    # c[mesh_y_rotated < y_lower] = 0
    # c[mesh_z > z_upper] = 0
    # c[mesh_z < z_lower] = 0

    # Calculate the wake velocity deficits and apply to the freestream
    return flow_field_u_initial - 2 * turbine_ai * c * flow_field_u_initial


# ///// #
# SETUP #
# ///// #

# Turbine parameters
turbine_diameter = 126.0
turbine_radius = turbine_diameter / 2.0
turbine_hub_height = 90.0
turbine_ai = 1 / 3

x_coord = np.array([0.0, 5 * 126.0])
y_coord = np.array([0.0, 0.0])

dtype = np.float64
# Wind parameters
# ws = np.array([6.0] * 25, dtype=dtype)  # jklm
# wd = np.array([270.0] * 72, dtype=dtype)  # ijklm
ws = np.arange(3.0, 26.0, 1.0, dtype=dtype)
wd = np.arange(0.0, 360.0, 5.0, dtype=dtype)
# wd = np.array([15 for i in range(100)])[:, na, na, na, na] # ijklm
# i  j  k  l  m
# wd ws x  y  z

specified_wind_height = 90.0
wind_shear = 0.12


# ////////////////////// #
# FULL FLOW FIELD POINTS #
# ////////////////////// #
# Flow field bounds
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
def initialize_flow_field(
    x_coord, y_coord, turbine_diameter, ws, specified_wind_height, wind_shear
):
    # Flow field bounds
    rotor_point_width = 0.25
    xmin = np.min(x_coord)
    xmax = np.max(x_coord)
    ymin = np.min(y_coord) - rotor_point_width * turbine_diameter
    ymax = np.max(y_coord) + rotor_point_width * turbine_diameter
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

    flow_field_u_initial = (
        ws[na, na, :, na, na, na] * (mesh_z / specified_wind_height) ** wind_shear
    )

    mesh_x_rotated, mesh_y_rotated, x_coord_rotated, y_coord_rotated = rotate_fields(
        limits, mesh_x, mesh_y, wd[na, :, na, na, na, na], x_coord, y_coord
    )
    # print(np.shape(mesh_x_rotated))
    # print(np.shape(mesh_y_rotated))
    # lkj

    return (
        flow_field_u_initial,
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z,
        x_coord_rotated,
        y_coord_rotated,
    )


def calculate_wake(
    turbine_diameter,
    turbine_ai,
    ws,
    wd,
    specified_wind_height,
    wind_shear,
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
    u_wake,
    deflection_field,
    flow_field_u,
):

    flow_field_u_masked = jensen_model_masked(
        flow_field_u,
        u_wake,
        turbine_ai,
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z,
        x_coord_rotated,
        y_coord_rotated,
        turbine_diameter,
        deflection_field,
    )

    return flow_field_u_masked


def calculate_power(
    locs, turbine_diameter, turbine_ai, ws, wd, specified_wind_height, wind_shear
):
    # flow_field_u = calculate_wake(
    #     locs, turbine_diameter, turbine_ai, ws, wd, specified_wind_height, wind_shear
    # )

    power = 0.0
    return power


# ///////////////// #
# JENSEN WAKE MODEL #
# ///////////////// #

# VECTORIZED CALLS
# Initialize field values
(
    flow_field_u_initial,
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
) = initialize_flow_field(
    x_coord, y_coord, turbine_diameter, ws, specified_wind_height, wind_shear
)

# print(np.shape(flow_field_u_initial))
# lkj

u_wake = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)
deflection_field = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)

# tottic = time.perf_counter()
# tic = time.perf_counter()
# flow_field_u = jensen_model(
#     flow_field_u_initial,
#     u_wake,
#     turbine_ai,
#     mesh_x_rotated,
#     mesh_y_rotated,
#     mesh_z,
#     x_coord_rotated,
#     y_coord_rotated,
#     turbine_diameter,
#     deflection_field,
# )
# toc = time.perf_counter()
# print(f"Computed vectorized Jensen model in {toc - tic:0.4f} seconds")
# tottoc = time.perf_counter()
# print(f"Total time for vectorized: {tottoc - tottic:0.4f} seconds")

tic = time.perf_counter()
# flow_field_u_masked = jensen_model_masked(
#     flow_field_u_initial,
#     u_wake,
#     turbine_ai,
#     mesh_x_rotated,
#     mesh_y_rotated,
#     mesh_z,
#     x_coord_rotated,
#     y_coord_rotated,
#     turbine_diameter,
#     deflection_field,
# )


for i in range(len(x_coord)):
    flow_field_u_masked = jensen_model_masked(
        flow_field_u_initial,
        u_wake,
        turbine_ai,
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z,
        x_coord_rotated,
        y_coord_rotated,
        turbine_diameter,
        deflection_field,
    )
toc = time.perf_counter()
print(f"Computed vectorized Jensen masked model in {toc - tic:0.4f} seconds")

print(np.unique(np.mean(flow_field_u_masked, axis=(4, 5))))
# /////////////// #
# COMPARE METHODS #
# /////////////// #

# print('max difference between methods: ', np.max((flow_field_u_masked - flow_field_u)))
# print('min difference between methods: ', np.min((flow_field_u_masked - flow_field_u)))

# //////////////// #
# COMPUTE GRADIENT #
# //////////////// #

# tic = time.perf_counter()
# calc_wake_grad = grad(calculate_wake)(locs, turbine_diameter, turbine_ai, ws, wd, specified_wind_height, wind_shear)
# print('gradient: ', calc_wake_grad)
# toc = time.perf_counter()
# print(f"Computed gradient in {toc - tic:0.4f} seconds")

# ///////////// #
# PLOT THE WAKE #
# ///////////// #
# plot_wakes = False
# if plot_wakes is True:

#     minSpeed = 2.0
#     maxSpeed = 10.0
#     cmap = "coolwarm"

#     fontsize = 6
#     plt.rcParams["font.size"] = fontsize - 2
#     # fig, axarr = plt.subplots(np.shape(wd)[0], np.shape(ws)[1])
#     fig, axarr = plt.subplots(2, 2)

#     # for i in range(np.shape(wd)[0]):
#     #     for j in range(np.shape(ws)[1]):
#     for i in range(1):
#         for j in range(2):
#             # Determine mask for z-slice/horizontal slice
#             mask = (mesh_z > 89.0) & (mesh_z < 91.0)

#             x_grid_length = len(np.unique(mesh_x[mask]))
#             y_grid_length = len(np.unique(mesh_y[mask]))
#             # print(x_grid_length)
#             # print(y_grid_length)
#             x_plot = np.reshape(mesh_x[mask], (x_grid_length, y_grid_length))
#             y_plot = np.reshape(mesh_y[mask], (x_grid_length, y_grid_length))
#             # print(x_plot)
#             # print(y_plot)
#             z_plot = np.reshape(
#                 flow_field_u[0][i][j][np.squeeze(mask, axis=(0, 1, 2))],
#                 (x_grid_length, y_grid_length),
#             )
#             # z_plot = np.reshape(
#             #     flow_field_u[i][j][mask], (x_grid_length, y_grid_length)
#             # )
#             # print(z_plot)

#             im = axarr[i][j].pcolormesh(
#                 x_plot,
#                 y_plot,
#                 z_plot,
#                 cmap=cmap,
#                 vmin=minSpeed,
#                 vmax=maxSpeed,
#                 shading="nearest",
#             )

#             axarr[i][j].contour(x_plot, y_plot, z_plot)

#             # Make equal axis
#             axarr[i][j].set_aspect("equal")
#             axarr[i][j].set_title(
#                 "wd: " + str(wd[i]) + ", ws: " + str(ws[j]), fontsize=fontsize,
#             )

#     plt.show()
