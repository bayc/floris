import time

import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis as na


# import jax.numpy as np
# from jax import grad


def cosd(angle):
    return np.cos(np.radians(angle))


def sind(angle):
    return np.sin(np.radians(angle))


def rotate_fields(limits, mesh_x, mesh_y, wd, x_coord, y_coord, sign=False):
    # Find center of rotation
    x_center_of_rotation = (limits[0] + limits[1]) / 2
    y_center_of_rotation = (limits[2] + limits[3]) / 2
    # print(x_center_of_rotation)
    # print(y_center_of_rotation)
    # print(mesh_x)
    # print(mesh_y)

    # print(x_coord)
    # lkj
    # Convert from compass rose angle to cartesian angle
    angle = ((wd - 270) % 360 + 360) % 360 * np.ones(np.shape(mesh_x))

    # Rotate grid points
    x_offset = mesh_x - x_center_of_rotation
    y_offset = mesh_y - y_center_of_rotation
    # print(np.shape(x_offset))
    # print(np.shape(y_offset))
    if sign:
        angle = -angle

    mesh_x_rotated = (
        x_offset * cosd(angle) - y_offset * sind(angle) + x_center_of_rotation
    )
    mesh_y_rotated = (
        x_offset * sind(angle) + y_offset * cosd(angle) + y_center_of_rotation
    )
    # print(np.shape(mesh_x_rotated))
    # print(np.shape(mesh_y_rotated))
    # print(mesh_x_rotated)
    # print(mesh_y_rotated)
    # print(angle)
    # lkj

    # Rotate turbine coordinates
    x_coord_offset = (x_coord - x_center_of_rotation)[na, na, na, :, na, na]
    y_coord_offset = (y_coord - y_center_of_rotation)[na, na, na, :, na, na]
    # x_coord_offset = (x_coord - x_center_of_rotation)
    # y_coord_offset = (y_coord - y_center_of_rotation)
    # print(x_center_of_rotation)
    # print(y_center_of_rotation)
    # print(np.shape(x_coord_offset))
    # print(np.shape(y_coord_offset))
    # print(np.shape(x_coord_offset * cosd(angle)))
    # print(np.shape(y_coord_offset * sind(angle)))
    # print(x_coord_offset)
    # print(y_coord_offset)
    # lkj

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
    # print(np.shape(x_coord_rotated))
    # print(np.shape(y_coord_rotated))
    # lkj

    # Return rotated values
    return mesh_x_rotated, mesh_y_rotated, x_coord_rotated, y_coord_rotated


def rotate_fields2(limits, mesh_x, mesh_y, wd, x_coord, y_coord, sign=False):
    # Find center of rotation
    x_center_of_rotation = (limits[0] + limits[1]) / 2
    y_center_of_rotation = (limits[2] + limits[3]) / 2
    # print(x_center_of_rotation)
    # print(y_center_of_rotation)
    # print(mesh_x)
    # print(mesh_y)

    # print(x_coord)
    # lkj
    # Convert from compass rose angle to cartesian angle
    angle = ((wd - 270) % 360 + 360) % 360 * np.ones(np.shape(mesh_x))

    # Rotate grid points
    x_offset = mesh_x - x_center_of_rotation
    y_offset = mesh_y - y_center_of_rotation
    # print(np.shape(x_offset))
    # print(np.shape(y_offset))
    if sign:
        angle = -angle

    mesh_x_rotated = (
        x_offset * cosd(angle) - y_offset * sind(angle) + x_center_of_rotation
    )
    mesh_y_rotated = (
        x_offset * sind(angle) + y_offset * cosd(angle) + y_center_of_rotation
    )
    # print(np.shape(mesh_x_rotated))
    # print(np.shape(mesh_y_rotated))
    # print(mesh_x_rotated)
    # print(mesh_y_rotated)
    # print(angle)
    # lkj

    # Rotate turbine coordinates
    x_coord_offset = x_coord - x_center_of_rotation
    y_coord_offset = y_coord - y_center_of_rotation
    # x_coord_offset = (x_coord - x_center_of_rotation)
    # y_coord_offset = (y_coord - y_center_of_rotation)
    # print(x_center_of_rotation)
    # print(y_center_of_rotation)
    # print(np.shape(x_coord_offset))
    # print(np.shape(y_coord_offset))
    # print(np.shape(x_coord_offset * cosd(angle)))
    # print(np.shape(y_coord_offset * sind(angle)))
    # print(x_coord_offset)
    # print(y_coord_offset)
    # lkj

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
    # print(np.shape(x_coord_rotated))
    # print(np.shape(y_coord_rotated))
    # lkj

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
    # print("0a: ", np.shape(mesh_x_rotated))
    print("0b: ", np.shape(x_coord_rotated))
    print("0c: ", np.shape(y_coord_rotated))
    boundary_line = we * (mesh_x_rotated - x_coord_rotated) + turbine_diameter / 2.0
    print("1: ", boundary_line)

    y_upper = boundary_line + y_coord_rotated  # + deflection_field
    y_lower = -1 * boundary_line + y_coord_rotated  # + deflection_field
    z_upper = boundary_line + turbine_hub_height
    z_lower = -1 * boundary_line + turbine_hub_height
    print("1: ", np.shape(boundary_line))
    print("2: ", np.shape(y_upper))
    print("3: ", np.shape(y_lower))
    print("4: ", np.shape(z_upper))
    print("5: ", np.shape(z_lower))

    # Calculate the wake velocity deficit ratios
    c = (
        (
            turbine_diameter
            / (2 * we * (mesh_x_rotated - x_coord_rotated) + turbine_diameter)
        )
        ** 2
        * ~(np.array(mesh_x_rotated - x_coord_rotated < 0.0))
        * ~(np.array(mesh_y_rotated > y_upper))
        * ~(np.array(mesh_y_rotated < y_lower))
        * ~(np.array(mesh_z > z_upper))
        * ~(np.array(mesh_z < z_lower))
    )
    # print(c)
    # print(mesh_x_rotated)
    # print(x_coord_rotated)
    # print(mesh_x_rotated - x_coord_rotated)
    # lkj

    # print('min of c: ', np.min(flow_field_u_initial))
    # print('max of c: ', np.max(flow_field_u_initial))

    # Calculate the wake velocity deficits and apply to the freestream
    # print(np.shape(flow_field_u_initial))
    # print(np.shape(c))
    print("here")
    flow = flow_field_u_initial - 2 * turbine_ai * c * flow_field_u_initial
    print(np.shape(flow))
    print("here2")
    return flow


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
# x_coord = np.array([0.0])
# y_coord = np.array([0.0])

dtype = np.float32
# Wind parameters
# ws = np.array([6.0] * 25, dtype=dtype)  # jklm
# wd = np.array([270.0] * 72, dtype=dtype)  # ijklm
# ws = np.arange(3.0, 26.0, 1.0, dtype=dtype)
# wd = np.arange(0.0, 360.0, 5.0, dtype=dtype)
ws = np.array([8.0])
wd = np.array([270.0])
# wd = np.array([15 for i in range(100)])[:, na, na, na, na] # ijklm
# i  j  k  l  m
# wd ws x  y  z

specified_wind_height = 90.0
wind_shear = 0.12

# [[2.5323186 2.6058128 2.6666667 2.718775  2.7644503]
#  [2.5323186 2.6058128 2.6666667 2.718775  2.7644503]
#  [2.5323186 2.6058128 2.6666667 2.718775  2.7644503]
#  [2.5323186 2.6058128 2.6666667 2.718775  2.7644503]
#  [2.5323186 2.6058128 2.6666667 2.718775  2.7644503]]

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
    x_coord,
    y_coord,
    turbine_diameter,
    limits,
    ws,
    wd,
    specified_wind_height,
    wind_shear,
):
    # Flow field bounds
    # rotor_point_width = 0.25
    # xmin = np.min(x_coord)
    # xmax = np.max(x_coord)
    # # ymin = np.min(y_coord) - rotor_point_width * turbine_diameter
    # # ymax = np.max(y_coord) + rotor_point_width * turbine_diameter
    # ymin = -rotor_point_width * turbine_diameter
    # ymax = rotor_point_width * turbine_diameter
    # zmin = turbine_hub_height - rotor_point_width * turbine_diameter
    # zmax = turbine_hub_height + rotor_point_width * turbine_diameter
    # limits = [xmin, xmax, ymin, ymax, zmin, zmax]

    # Flow field resolutions
    # resolution_x1 = 1
    resolution_x2 = 5
    resolution_x3 = 5

    # x = np.linspace(xmin, xmax, int(resolution_x1), dtype=dtype)
    x = x_coord
    # x = [[coord] for coord in x_coord]
    # y = np.linspace(ymin, ymax, int(resolution_x2), dtype=dtype)
    y = np.array(
        [
            np.linspace(
                y_coord[i] - rotor_point_width * turbine_diameter,
                y_coord[i] + rotor_point_width * turbine_diameter,
                int(resolution_x2),
                dtype=dtype,
            )
            for i in range(len(y_coord))
        ]
    )  # .flatten()
    z = np.linspace(limits[4], limits[5], int(resolution_x3), dtype=dtype)

    # print("x: ", x)
    # print("y: ", y)
    # print("z: ", z)

    # Flow field grid points
    mesh_x = []
    mesh_y = []
    mesh_z = []
    for i in range(len(x)):
        mesh_x_tmp, mesh_y_tmp, mesh_z_tmp = np.meshgrid(x[i], y[i], z, indexing="ij")
        mesh_x.append(mesh_x_tmp)
        mesh_y.append(mesh_y_tmp)
        mesh_z.append(mesh_z_tmp)
    #     print("mesh_x: ", mesh_x)
    #     print("mesh_y: ", mesh_y)
    #     print("mesh_z: ", mesh_z)
    # print('shape of mesh_x: ', np.shape(mesh_x))
    # print('shape of mesh_x: ', np.shape(mesh_y))
    # print('shape of mesh_x: ', np.shape(mesh_z))
    mesh_x = np.squeeze(mesh_x, axis=1)
    mesh_y = np.squeeze(mesh_y, axis=1)
    mesh_z = np.squeeze(mesh_z, axis=1)
    # print('shape of mesh_x: ', np.shape(mesh_x))
    # print('shape of mesh_x: ', np.shape(mesh_y))
    # print('shape of mesh_x: ', np.shape(mesh_z))
    # lkj

    mesh_x = mesh_x[na, na, na, :, :, :]  # * np.ones((72, 1, 200, 100, 7))
    mesh_y = mesh_y[na, na, na, :, :, :]  # * np.ones((72, 1, 200, 100, 7))
    mesh_z = mesh_z[na, na, na, :, :, :]  # * np.ones((72, 1, 200, 100, 7))

    # print("mesh_x: ", mesh_x)
    # print("mesh_y: ", mesh_y)
    # print("mesh_z: ", mesh_z)
    # lkj
    # print("mesh_y: ", mesh_y + y_coord[:, na, na] * np.ones((1, 5, 5)))

    flow_field_u_initial = (
        ws[na, na, :, na, na, na] * (mesh_z / specified_wind_height) ** wind_shear
    )
    flow_field_u_initial = flow_field_u_initial * np.ones((1, len(wd), 1, 1, 1, 1))
    # print(np.shape(flow_field_u_initial))
    # lkj

    mesh_x_rotated, mesh_y_rotated, x_coord_rotated, y_coord_rotated = rotate_fields(
        limits, mesh_x, mesh_y, wd[na, :, na, na, na, na], x_coord, y_coord, sign=True
    )
    # print(np.shape(mesh_x_rotated))
    # print(np.shape(x_coord_rotated))
    # lkj

    return (
        flow_field_u_initial,
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z,
        x_coord_rotated,
        y_coord_rotated,
    )


# ///////////////// #
# JENSEN WAKE MODEL #
# ///////////////// #

# VECTORIZED CALLS

# Flow field bounds
rotor_point_width = 0.25
xmin = np.min(x_coord)
xmax = np.max(x_coord)
# ymin = np.min(y_coord) - rotor_point_width * turbine_diameter
# ymax = np.max(y_coord) + rotor_point_width * turbine_diameter
ymin = -rotor_point_width * turbine_diameter
ymax = rotor_point_width * turbine_diameter
zmin = turbine_hub_height - rotor_point_width * turbine_diameter
zmax = turbine_hub_height + rotor_point_width * turbine_diameter
limits = [xmin, xmax, ymin, ymax, zmin, zmax]

# Initialize field values
(
    flow_field_u_initial,
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
) = initialize_flow_field(
    x_coord,
    y_coord,
    turbine_diameter,
    limits,
    ws,
    wd,
    specified_wind_height,
    wind_shear,
)

mesh_x_rotated, mesh_y_rotated, x_coord_rotated, y_coord_rotated = rotate_fields2(
    limits,
    mesh_x_rotated,
    mesh_y_rotated,
    wd[na, :, na, na, na, na],
    x_coord_rotated,
    y_coord_rotated,
)
# lkj
# print(np.shape(mesh_x_rotated))
# print(np.shape(x_coord_rotated))
# print(np.shape(flow_field_u_initial))
print(mesh_x_rotated)
print(mesh_y_rotated)
# print(mesh_x_rotated2)
# print(mesh_y_rotated2)
# lkj

u_wake = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)
turb_u_wake = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)
deflection_field = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)

# import copy
# flow_field_u = copy.deepcopy(flow_field_u_initial)

tic = time.perf_counter()
for i in range(len(x_coord)):
    # print('1: ', flow_field_u[:, :, :, i, :, :])
    # if i == 0:
    #     turb_u_wake[:, :, :, i, :, :] = jensen_model_masked(
    #         flow_field_u[:, :, :, i, :, :],
    #         turbine_ai,
    #         mesh_x_rotated[:, :, :, i, :, :],
    #         mesh_y_rotated[:, :, :, i, :, :],
    #         mesh_z[:, :, :, i, :, :],
    #         x_coord_rotated[:, :, :, i, :, :],
    #         y_coord_rotated[:, :, :, i, :, :],
    #         turbine_diameter,
    #         deflection_field,
    #     )
    # else:
    #     turb_u_wake[:, :, :, i, :, :] = jensen_model_masked(
    #         flow_field_u[:, :, :, i-1, :, :],
    #         turbine_ai,
    #         mesh_x_rotated[:, :, :, i, :, :],
    #         mesh_y_rotated[:, :, :, i, :, :],
    #         mesh_z[:, :, :, i, :, :],
    #         x_coord_rotated[:, :, :, i, :, :],
    #         y_coord_rotated[:, :, :, i, :, :],
    #         turbine_diameter,
    #         deflection_field,
    #     )
    if i == 0:
        turb_u_wake = jensen_model_masked(
            flow_field_u_initial,
            turbine_ai,
            mesh_x_rotated,
            mesh_y_rotated,
            mesh_z,
            x_coord_rotated,
            y_coord_rotated,
            turbine_diameter,
            deflection_field,
        )
    else:
        print("diff: ", flow_field_u_initial - u_wake)
        turb_u_wake = jensen_model_masked(
            flow_field_u_initial - u_wake,
            turbine_ai,
            mesh_x_rotated,
            mesh_y_rotated,
            mesh_z,
            x_coord_rotated,
            y_coord_rotated,
            turbine_diameter,
            deflection_field,
        )
    # print('now here')
    # print('2: ', turb_u_wake[:, :, :, i, :, :])

    u_wake = np.hypot(u_wake, turb_u_wake)
    print("uwake: ", u_wake)
    # flow_field_u[:, :, :, i, :, :] = flow_field_u_initial[:, :, :, i, :, :] - turb_u_wake[:, :, :, i, :, :]
    # print('3: ', flow_field_u[:, :, :, i, :, :])

    # print(np.shape(flow_field_u))
    # print(np.shape(turb_deficit_u_masked))
toc = time.perf_counter()
print(f"Computed vectorized Jensen masked model in {toc - tic:0.4f} seconds")

flow_field_u = flow_field_u_initial - u_wake

ind = 0
print(flow_field_u[0, ind, 0, 0, :, :])
print(turb_u_wake[0, ind, 0, 0, :, :])
print(flow_field_u[0, ind, 0, 1, :, :])
print(turb_u_wake[0, ind, 0, 1, :, :])

# print(np.shape(mesh_x_rotated))


# print(mesh_x_rotated[0, ind, 0, 0, :, :])
# print(mesh_y_rotated[0, ind, 0, 0, :, :])
# print(mesh_z)
# print(wd[ind])
# print(turb_u_wake[0, 0, 22, 1, :, :])
