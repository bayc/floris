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


def rotate_fields(limits, mesh_x, mesh_y, wd, x_coord, y_coord):
    # Find center of rotation
    x_center_of_rotation = (limits[0] + limits[1]) / 2
    y_center_of_rotation = (limits[2] + limits[3]) / 2

    # Convert from compass rose angle to cartesian angle
    angle = ((wd - 270) % 360 + 360) % 360

    # Rotate grid points
    x_offset = mesh_x - x_center_of_rotation
    y_offset = mesh_y - y_center_of_rotation
    print(np.shape(x_offset))
    print(np.shape(y_offset))

    mesh_x_rotated = (
        x_offset * cosd(angle) - y_offset * sind(angle) + x_center_of_rotation
    )
    mesh_y_rotated = (
        x_offset * sind(angle) + y_offset * cosd(angle) + y_center_of_rotation
    )
    print(np.shape(mesh_x_rotated))
    print(np.shape(mesh_y_rotated))

    # Rotate turbine coordinates
    x_coord_offset = (x_coord - x_center_of_rotation)[na, na, na, :, na, na]
    y_coord_offset = (y_coord - y_center_of_rotation)[na, na, na, :, na, na]
    # x_coord_offset = (x_coord - x_center_of_rotation)
    # y_coord_offset = (y_coord - y_center_of_rotation)
    print(x_center_of_rotation)
    print(y_center_of_rotation)
    print(np.shape(x_coord_offset))
    print(np.shape(y_coord_offset))
    print(x_coord_offset)
    print(y_coord_offset)

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
    print(np.shape(x_coord_rotated))
    print(np.shape(y_coord_rotated))
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
    print("0a: ", np.shape(mesh_x_rotated))
    print("0b: ", np.shape(x_coord_rotated))
    boundary_line = we * (mesh_x_rotated - x_coord_rotated) + turbine_diameter / 2.0
    print("1: ", np.shape(boundary_line))

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
        * ~(np.array(mesh_x_rotated - x_coord_rotated < 0))
        * ~(np.array(mesh_y_rotated > y_upper))
        * ~(np.array(mesh_y_rotated < y_lower))
        * ~(np.array(mesh_z > z_upper))
        * ~(np.array(mesh_z < z_lower))
    )

    # Calculate the wake velocity deficits and apply to the freestream
    print(np.shape(flow_field_u_initial))
    print(np.shape(c))
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
y_coord = np.array([0.0, 200.0])

dtype = np.float32
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
    # ymin = np.min(y_coord) - rotor_point_width * turbine_diameter
    # ymax = np.max(y_coord) + rotor_point_width * turbine_diameter
    ymin = -rotor_point_width * turbine_diameter
    ymax = rotor_point_width * turbine_diameter
    zmin = turbine_hub_height - rotor_point_width * turbine_diameter
    zmax = turbine_hub_height + rotor_point_width * turbine_diameter
    limits = [xmin, xmax, ymin, ymax, zmin, zmax]

    # Flow field resolutions
    # resolution_x1 = 1
    resolution_x2 = 5
    resolution_x3 = 5

    # x = np.linspace(xmin, xmax, int(resolution_x1), dtype=dtype)
    x = x_coord
    y = np.linspace(ymin, ymax, int(resolution_x2), dtype=dtype)
    # y = np.array([np.linspace(y_coord[i] - rotor_point_width * turbine_diameter, y_coord[i] + rotor_point_width * turbine_diameter, int(resolution_x2), dtype=dtype) for i in range(len(y_coord))]).flatten()
    z = np.linspace(zmin, zmax, int(resolution_x3), dtype=dtype)

    print("x: ", x)
    print("y: ", y)
    print("z: ", z)

    # Flow field grid points
    mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z, indexing="ij")

    mesh_x = mesh_x[na, na, na, :, :, :]  # * np.ones((72, 1, 200, 100, 7))
    mesh_y = mesh_y[na, na, na, :, :, :]  # * np.ones((72, 1, 200, 100, 7))
    mesh_z = mesh_z[na, na, na, :, :, :]  # * np.ones((72, 1, 200, 100, 7))

    print("mesh_x: ", mesh_x)
    print("mesh_y: ", mesh_y + y_coord[:, na, na] * np.ones((1, 5, 5)))

    flow_field_u_initial = (
        ws[na, na, :, na, na, na] * (mesh_z / specified_wind_height) ** wind_shear
    )

    mesh_x_rotated, mesh_y_rotated, x_coord_rotated, y_coord_rotated = rotate_fields(
        limits, mesh_x, mesh_y, wd[na, :, na, na, na, na], x_coord, y_coord
    )
    print(np.shape(mesh_x_rotated))
    print(np.shape(x_coord_rotated))
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
# lkj
print(np.shape(mesh_x_rotated))
print(np.shape(x_coord_rotated))

u_wake = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)
deflection_field = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)

flow_field_u = flow_field_u_initial

tic = time.perf_counter()
for i in range(len(x_coord)):
    if i == 0:
        turb_deficit_u_masked = jensen_model_masked(
            flow_field_u_initial,
            turbine_ai,
            mesh_x_rotated,
            mesh_y_rotated,
            mesh_z,
            x_coord_rotated[:, :, :, i, :, :],
            y_coord_rotated[:, :, :, i, :, :],
            turbine_diameter,
            deflection_field,
        )
    else:
        turb_deficit_u_masked = jensen_model_masked(
            flow_field_u,
            turbine_ai,
            mesh_x_rotated,
            mesh_y_rotated,
            mesh_z,
            x_coord_rotated[:, :, :, i, :, :],
            y_coord_rotated[:, :, :, i, :, :],
            turbine_diameter,
            deflection_field,
        )

    flow_field_u = np.hypot(turb_deficit_u_masked, flow_field_u)

    print(np.shape(flow_field_u))
    print(np.shape(turb_deficit_u_masked))
toc = time.perf_counter()
print(f"Computed vectorized Jensen masked model in {toc - tic:0.4f} seconds")
