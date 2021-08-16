import time

import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis as na
from scipy.interpolate import interp1d

import floris.tools as wfct


# import jax.numpy as np
# from jax import grad
thrust = [
    1.19187945,
    1.17284634,
    1.09860817,
    1.02889592,
    0.97373036,
    0.92826162,
    0.89210543,
    0.86100905,
    0.835423,
    0.81237673,
    0.79225789,
    0.77584769,
    0.7629228,
    0.76156073,
    0.76261984,
    0.76169723,
    0.75232027,
    0.74026851,
    0.72987175,
    0.70701647,
    0.54054532,
    0.45509459,
    0.39343381,
    0.34250785,
    0.30487242,
    0.27164979,
    0.24361964,
    0.21973831,
    0.19918151,
    0.18131868,
    0.16537679,
    0.15103727,
    0.13998636,
    0.1289037,
    0.11970413,
    0.11087113,
    0.10339901,
    0.09617888,
    0.09009926,
    0.08395078,
    0.0791188,
    0.07448356,
    0.07050731,
    0.06684119,
    0.06345518,
    0.06032267,
    0.05741999,
    0.05472609,
]
wind_speed = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
    9.5,
    10.0,
    10.5,
    11.0,
    11.5,
    12.0,
    12.5,
    13.0,
    13.5,
    14.0,
    14.5,
    15.0,
    15.5,
    16.0,
    16.5,
    17.0,
    17.5,
    18.0,
    18.5,
    19.0,
    19.5,
    20.0,
    20.5,
    21.0,
    21.5,
    22.0,
    22.5,
    23.0,
    23.5,
    24.0,
    24.5,
    25.0,
    25.5,
]


def cosd(angle):
    return np.cos(np.radians(angle))


def sind(angle):
    return np.sin(np.radians(angle))


def rotate_fields(mesh_x, mesh_y, mesh_z, wd, x_coord, y_coord, z_coord):
    # Find center of rotation
    x_center_of_rotation = np.mean([np.min(mesh_x), np.max(mesh_x)])
    y_center_of_rotation = np.mean([np.min(mesh_y), np.max(mesh_y)])

    # Convert from compass rose angle to cartesian angle
    angle = ((wd - 270) % 360 + 360) % 360

    # Rotate grid points
    x_offset = mesh_x - x_center_of_rotation
    y_offset = mesh_y - y_center_of_rotation
    mesh_x_rotated = (
        x_offset * cosd(angle) - y_offset * sind(angle) + x_center_of_rotation
    )
    mesh_y_rotated = (
        x_offset * sind(angle) + y_offset * cosd(angle) + y_center_of_rotation
    )

    x_coord_offset = (x_coord - x_center_of_rotation)[:, na, na]
    y_coord_offset = (y_coord - y_center_of_rotation)[:, na, na]

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

    inds_sorted = x_coord_rotated.argsort(axis=3)

    x_coord_rotated_sorted = np.take_along_axis(x_coord_rotated, inds_sorted, axis=3)
    y_coord_rotated_sorted = np.take_along_axis(y_coord_rotated, inds_sorted, axis=3)
    z_coord_rotated_sorted = np.take_along_axis(
        z_coord * np.ones((np.shape(x_coord_rotated))), inds_sorted, axis=3
    )

    mesh_x_rotated_sorted = np.take_along_axis(mesh_x_rotated, inds_sorted, axis=3)
    mesh_y_rotated_sorted = np.take_along_axis(mesh_y_rotated, inds_sorted, axis=3)
    mesh_z_rotated_sorted = np.take_along_axis(
        mesh_z * np.ones((np.shape(mesh_x_rotated))), inds_sorted, axis=3
    )

    inds_unsorted = x_coord_rotated_sorted.argsort(axis=3)

    return (
        mesh_x_rotated_sorted,
        mesh_y_rotated_sorted,
        mesh_z_rotated_sorted,
        x_coord_rotated_sorted,
        y_coord_rotated_sorted,
        z_coord_rotated_sorted,
        inds_sorted,
        inds_unsorted,
    )


def jimenez_model(yaw_angle, Ct, x_coord, mesh_x, rotor_diameter):
    kd = 0.05
    ad = 0.0
    bd = 0.0

    # angle of deflection
    xi_init = cosd(yaw_angle) * sind(yaw_angle) * Ct / 2.0

    x_locations = mesh_x - x_coord

    # yaw displacement
    yYaw_init = (
        xi_init
        * (15 * (2 * kd * x_locations / rotor_diameter + 1) ** 4.0 + xi_init ** 2.0)
        / (
            (30 * kd / rotor_diameter)
            * (2 * kd * x_locations / rotor_diameter + 1) ** 5.0
        )
    ) - (xi_init * rotor_diameter * (15 + xi_init ** 2.0) / (30 * kd))

    # corrected yaw displacement with lateral offset
    deflection = yYaw_init + ad + bd * x_locations

    x = np.unique(x_locations)
    for i in range(len(x)):
        tmp = np.max(deflection[x_locations == x[i]])
        deflection[x_locations == x[i]] = tmp

    return deflection


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
    turbine_hub_height,
    deflection_field,
):
    # Wake expansion parameter
    we = 0.05

    m = we
    x = mesh_x_rotated - x_coord_rotated
    b = turbine_diameter / 2.0

    boundary_line = m * x + b

    y_center = np.zeros_like(boundary_line) + y_coord_rotated + deflection_field
    # print(y_center)
    # lkj
    z_center = np.zeros_like(boundary_line) + turbine_hub_height

    # Calculate the wake velocity deficit ratios
    c = (
        (turbine_diameter / (2 * we * (x) + turbine_diameter)) ** 2
        * ~(np.array(mesh_x_rotated - x_coord_rotated < 0.0))
        * ~(
            ((mesh_y_rotated - y_center) ** 2 + (mesh_z - z_center) ** 2)
            > (boundary_line ** 2)
        )
    )

    return 2 * turbine_ai * c * (flow_field_u_initial)


def crespo_hernandez(
    ambient_TI, x_coord_upstream, x_coord_downstream, rotor_diameter, aI
):
    ti_initial = 0.1
    ti_constant = 0.5
    ti_ai = 0.8
    ti_downstream = -0.32

    # turbulence intensity calculation based on Crespo et. al.
    ti_calculation = (
        ti_constant
        * aI ** ti_ai
        * ambient_TI ** ti_initial
        * ((x_coord_downstream - x_coord_upstream) / rotor_diameter) ** ti_downstream
    )

    # Update turbulence intensity of downstream turbines
    return ti_calculation


def turbine_avg_velocity(turb_inflow_vel):
    return np.cbrt(np.mean(turb_inflow_vel ** 3, axis=(4, 5)))


fCtInterp = interp1d(wind_speed, thrust, fill_value="extrapolate")


def Ct(turb_avg_vels):
    Ct_vals = fCtInterp(turb_avg_vels)
    Ct_vals[Ct_vals > 1.0] = 0.9999
    return Ct_vals


def aI(turb_Ct):
    return 0.5 * (1 - np.sqrt(1 - turb_Ct))


# ///// #
# SETUP #
# ///// #

# Turbine parameters
turbine_diameter = 126.0
turbine_radius = turbine_diameter / 2.0
turbine_hub_height = 90.0

x_coord = np.array([0.0, 5 * 126.0])  # , 0*126.0])
y_coord = np.array([0.0, 0 * 126.0])  # , 5*126.0])
z_coord = np.array([90.0, 90.0])  # , 90.0])

y_ngrid = 5
z_ngrid = 5
rloc = 0.5

dtype = np.float64
# Wind parameters
ws = np.array([8.0])
wd = np.array([270.0])
# i  j  k  l  m
# wd ws x  y  z

specified_wind_height = 90.0
wind_shear = 0.12

# ///////////////// #
# ONLY ROTOR POINTS #
# ///////////////// #


def update_grid(x_grid_i, y_grid_i, wind_direction_i, x1, x2):
    xoffset = x_grid_i - x1[:, na, na]
    yoffset = y_grid_i - x2[:, na, na]

    wind_cos = cosd(-wind_direction_i)
    wind_sin = sind(-wind_direction_i)

    x_grid_i = xoffset * wind_cos - yoffset * wind_sin + x1[:, na, na]
    y_grid_i = yoffset * wind_cos + xoffset * wind_sin + x2[:, na, na]
    return x_grid_i, y_grid_i


def initialize_flow_field(
    x_coord,
    y_coord,
    z_coord,
    y_ngrid,
    z_ngrid,
    wd,
    ws,
    specified_wind_height,
    wind_shear,
):
    # Flow field bounds
    x_grid = np.zeros((len(x_coord), y_ngrid, z_ngrid))
    y_grid = np.zeros((len(x_coord), y_ngrid, z_ngrid))
    z_grid = np.zeros((len(x_coord), y_ngrid, z_ngrid))

    angle = ((wd - 270) % 360 + 360) % 360

    x1, x2, x3 = x_coord, y_coord, z_coord

    pt = rloc * turbine_radius

    # TODO: would it be simpler to create rotor points inherently rotated to be
    # perpendicular to the wind
    yt = np.linspace(x2 - pt, x2 + pt, y_ngrid,)
    zt = np.linspace(x3 - pt, x3 + pt, z_ngrid,)

    x_grid = np.ones((len(x_coord), y_ngrid, z_ngrid)) * x_coord[:, na, na]
    y_grid = np.ones((len(x_coord), y_ngrid, z_ngrid)) * yt.T[:, :, na]
    z_grid = np.ones((len(x_coord), y_ngrid, z_ngrid)) * zt.T[:, na, :]

    # yaw turbines to be perpendicular to the wind direction
    # TODO: update update_grid to be called something better
    x_grid, y_grid = update_grid(x_grid, y_grid, angle[na, :, na, na, na, na], x1, x2)

    mesh_x = x_grid
    mesh_y = y_grid
    mesh_z = z_grid

    flow_field_u_initial = (
        ws[na, na, :, na, na, na] * (mesh_z / specified_wind_height) ** wind_shear
    ) * np.ones((1, len(wd), 1, 1, 1, 1))

    # rotate turbine locations/fields to be perpendicular to wind direction
    (
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z_rotated,
        x_coord_rotated,
        y_coord_rotated,
        z_coord_rotated,
        inds_sorted,
        inds_unsorted,
    ) = rotate_fields(
        mesh_x, mesh_y, mesh_z, wd[na, :, na, na, na, na], x_coord, y_coord, z_coord
    )

    return (
        flow_field_u_initial,
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z_rotated,
        x_coord_rotated,
        y_coord_rotated,
        z_coord_rotated,
        inds_sorted,
        inds_unsorted,
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
    z_coord_rotated,
    inds_sorted,
    inds_unsorted,
) = initialize_flow_field(
    x_coord,
    y_coord,
    z_coord,
    y_ngrid,
    z_ngrid,
    wd,
    ws,
    specified_wind_height,
    wind_shear,
)

u_wake = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)
# deflection_field = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)
turb_inflow_field = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)

turb_TIs = np.ones_like(x_coord_rotated) * 0.06
yaw_angle = np.ones_like(x_coord_rotated) * 00
# print(yaw_angle)
# print(np.shape(yaw_angle))
# lkj

tic = time.perf_counter()

for i in range(len(x_coord)):
    turb_inflow_field[:, :, :, i, :, :] = (flow_field_u_initial - u_wake)[
        :, :, :, i, :, :
    ]

    turb_avg_vels = turbine_avg_velocity(turb_inflow_field)
    turb_Cts = Ct(turb_avg_vels)
    turb_aIs = aI(turb_Cts)

    deflection_field = jimenez_model(
        yaw_angle,
        turb_Cts[:, :, :, i],
        x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        mesh_x_rotated,
        turbine_diameter,
    )
    # print(np.shape(deflection_field))
    # print(deflection_field)
    # lkj

    turb_u_wake = jensen_model_masked(
        flow_field_u_initial,
        u_wake,
        turb_aIs[:, :, :, i][:, :, :, na, na, na],
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z,
        x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        turbine_diameter,
        turbine_hub_height,
        deflection_field,
    )

    print("##################### i: ", i)
    # print(np.shape(turb_u_wake))
    # print("vec turb_u_wake: ", turb_u_wake)

    u_wake = np.sqrt((u_wake ** 2) + (turb_u_wake ** 2))

toc = time.perf_counter()

# /////////////// #
# COMPARE METHODS #
# /////////////// #

flow_field_u = flow_field_u_initial - u_wake
print(
    "vec u: ",
    np.take_along_axis(
        np.take_along_axis(flow_field_u, inds_sorted, axis=3), inds_unsorted, axis=3
    ),
)
print("vec shape of u: ", np.shape(flow_field_u_initial - u_wake))
print("Turbine avg vels: ", turbine_avg_velocity(turb_inflow_field))

# //////////////// #
# COMPUTE GRADIENT #
# //////////////// #

# tic = time.perf_counter()
# calc_wake_grad = grad(calculate_wake)(locs, turbine_diameter, turbine_ai, ws, wd, specified_wind_height, wind_shear)
# print('gradient: ', calc_wake_grad)
# toc = time.perf_counter()
# print(f"Computed gradient in {toc - tic:0.4f} seconds")
