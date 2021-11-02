import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis as na
from scipy.interpolate import interp1d
from numpy.lib.function_base import meshgrid

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


def tand(angle):
    return np.tan(np.radians(angle))


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


def TKE_to_TI(turbulence_kinetic_energy, turb_avg_vels):
    total_turbulence_intensity = (
        np.sqrt((2 / 3) * turbulence_kinetic_energy)
    ) / turb_avg_vels
    return total_turbulence_intensity


def yaw_added_turbulence_mixing(
    x_coord_rotated,
    y_coord_rotated,
    turb_avg_vels,
    turbine_ti,
    turbine_diameter,
    flow_field_v,
    flow_field_w,
    turb_v,
    turb_w,
    mesh_x_rotated,
    mesh_y_rotated,
):
    # calculate fluctuations
    # print('mean of turb_v: ', np.mean(turb_v))
    # print('mean of turb_w: ', np.mean(turb_w))
    v_prime = flow_field_v + turb_v
    w_prime = flow_field_w + turb_w

    # get u_prime from current turbulence intensity
    # u_prime = turbine.u_prime()
    TKE = ((turb_avg_vels * turbine_ti) ** 2) / (2 / 3)
    u_prime = np.sqrt(2 * TKE)

    # compute the new TKE
    idx = np.where(
        (np.abs(mesh_x_rotated - x_coord_rotated) <= turbine_diameter / 4)
        & (np.abs(mesh_y_rotated - y_coord_rotated) < turbine_diameter)
    )
    TKE = (1 / 2) * (
        u_prime ** 2 + np.mean(v_prime[idx]) ** 2 + np.mean(w_prime[idx]) ** 2
    )

    # convert TKE back to TI
    TI_total = TKE_to_TI(TKE, turb_avg_vels)

    # convert to turbulence due to mixing
    TI_mixing = np.array(TI_total) - turbine_ti

    return TI_mixing


def calc_VW(
    x_coord_rotated,
    y_coord_rotated,
    wind_shear,
    specified_wind_height,
    turb_avg_vels,
    turbine_ti,
    turbine_Ct,
    turbine_aI,
    turbine_TSR,
    turbine_yaw,
    turbine_hub_height,
    turbine_diameter,
    flow_field_u_initial,
    flow_field,
    x_locations,
    y_locations,
    z_locations,
):
    eps_gain = 0.2
    # turbine parameters
    D = turbine_diameter
    HH = turbine_hub_height
    yaw = turbine_yaw
    Ct = turbine_Ct
    TSR = turbine_TSR
    aI = turbine_aI

    # flow parameters
    # Uinf = np.mean(flow_field_u_initial)
    Uinf = np.mean(flow_field_u_initial, axis=(4, 5))[:, :, :, :, na, na]

    scale = 1.0
    vel_top = (Uinf * ((HH + D / 2) / specified_wind_height) ** wind_shear) / Uinf
    vel_bottom = (Uinf * ((HH - D / 2) / specified_wind_height) ** wind_shear) / Uinf
    Gamma_top = scale * (np.pi / 8) * D * vel_top * Uinf * Ct * sind(yaw) * cosd(yaw)
    Gamma_bottom = (
        -scale * (np.pi / 8) * D * vel_bottom * Uinf * Ct * sind(yaw) * cosd(yaw)
    )
    Gamma_wake_rotation = 0.25 * 2 * np.pi * D * (aI - aI ** 2) * turb_avg_vels / TSR
    # print('vel_top', np.mean(vel_top))
    # print('vel_bottom', np.mean(vel_bottom))
    # print('Gamma_top', np.mean(Gamma_top))
    # print('Gamma_bottom', np.mean(Gamma_bottom))
    # print('Gamma_wake_rotation', np.mean(Gamma_wake_rotation))

    # compute the spanwise and vertical velocities induced by yaw
    eps = eps_gain * D  # Use set value
    # print('eps: ', eps)

    # decay the vortices as they move downstream - using mixing length
    lmda = D / 8
    kappa = 0.41
    lm = kappa * z_locations / (1 + kappa * z_locations / lmda)
    z = np.linspace(z_locations.min(), z_locations.max(), flow_field_u_initial.shape[5])
    # print(np.shape(flow_field_u_initial))
    # print(np.shape(z))
    dudz_initial = np.gradient(flow_field_u_initial, z, axis=5)
    # print(np.shape(dudz_initial))
    nu = lm ** 2 * np.abs(dudz_initial[:, :, :, 0, :, :][:, :, :, na, :, :])
    # print(np.shape(nu))
    # print(nu)
    # lkj
    # print('dudz_initial: ', np.mean(dudz_initial))
    # print('nu: ', np.mean(nu))
    # print('lm: ', np.mean(lm))
    # print('z: ', np.mean(z))

    # top vortex
    yLocs = y_locations + 0.01 - (y_coord_rotated)
    zT = z_locations + 0.01 - (HH + D / 2)
    rT = yLocs ** 2 + zT ** 2
    V1 = (
        (zT * Gamma_top)
        / (2 * np.pi * rT)
        * (1 - np.exp(-rT / (eps ** 2)))
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    W1 = (
        (-yLocs * Gamma_top)
        / (2 * np.pi * rT)
        * (1 - np.exp(-rT / (eps ** 2)))
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    # bottom vortex
    zB = z_locations + 0.01 - (HH - D / 2)
    rB = yLocs ** 2 + zB ** 2
    V2 = (
        (zB * Gamma_bottom)
        / (2 * np.pi * rB)
        * (1 - np.exp(-rB / (eps ** 2)))
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    W2 = (
        ((-yLocs * Gamma_bottom) / (2 * np.pi * rB))
        * (1 - np.exp(-rB / (eps ** 2)))
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    # top vortex - ground
    yLocs = y_locations + 0.01 - (y_coord_rotated)
    zLocs = z_locations + 0.01 + (HH + D / 2)
    V3 = (
        (
            ((zLocs * -Gamma_top) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
            + 0.0
        )
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    W3 = (
        ((-yLocs * -Gamma_top) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
        * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    # bottom vortex - ground
    yLocs = y_locations + 0.01 - (y_coord_rotated)
    zLocs = z_locations + 0.01 + (HH - D / 2)
    V4 = (
        (
            ((zLocs * -Gamma_bottom) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
            + 0.0
        )
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    W4 = (
        ((-yLocs * -Gamma_bottom) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
        * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    # wake rotation vortex
    zC = z_locations + 0.01 - (HH)
    rC = yLocs ** 2 + zC ** 2
    V5 = (
        (zC * Gamma_wake_rotation)
        / (2 * np.pi * rC)
        * (1 - np.exp(-rC / (eps ** 2)))
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )
    # print('zC: ', np.mean(zC))
    # print('rC: ', np.mean(rC))
    # print('Gamma_wake_rotation: ', np.mean(Gamma_wake_rotation))
    # print('eps: ', np.mean(eps))
    # print('nu: ', np.mean(nu))
    # print('x_locations: ', np.mean(x_locations))
    # print('x_coord_rotated: ', np.mean(x_coord_rotated))
    # print('Uinf: ', np.mean(Uinf))

    W5 = (
        (-yLocs * Gamma_wake_rotation)
        / (2 * np.pi * rC)
        * (1 - np.exp(-rC / (eps ** 2)))
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    # wake rotation vortex - ground effect
    yLocs = y_locations + 0.01 - y_coord_rotated
    zLocs = z_locations + 0.01 + HH
    V6 = (
        (
            ((zLocs * -Gamma_wake_rotation) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
            + 0.0
        )
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    W6 = (
        ((-yLocs * -Gamma_wake_rotation) / (2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
        * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
        * eps ** 2
        / (4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2)
    )

    # total spanwise velocity
    V = V1 + V2 + V3 + V4 + V5 + V6
    W = W1 + W2 + W3 + W4 + W5 + W6

    # print('V1 in: ', np.mean(V1))
    # print('V2 in: ', np.mean(V2))
    # print('V3 in: ', np.mean(V3))
    # print('V4 in: ', np.mean(V4))
    # print('V5 in: ', np.mean(V5))
    # print('V6 in: ', np.mean(V6))

    # print('W1 in: ', np.mean(W1))
    # print('W2 in: ', np.mean(W2))
    # print('W3 in: ', np.mean(W3))
    # print('W4 in: ', np.mean(W4))
    # print('W5 in: ', np.mean(W5))
    # print('W6 in: ', np.mean(W6))

    # print('V in: ', np.mean(V))
    # print('W in: ', np.mean(W))

    V = V * np.array(x_locations >= x_coord_rotated - 1)
    W = W * np.array(x_locations >= x_coord_rotated - 1)

    # no spanwise and vertical velocity upstream of the turbine
    # V[
    #     x_locations < x_coord_rotated - 1
    # ] = 0.0  # Subtract by 1 to avoid numerical issues on rotation
    # W[
    #     x_locations < x_coord_rotated - 1
    # ] = 0.0  # Subtract by 1 to avoid numerical issues on rotation

    W = W * np.array(W >= 0)

    # W[W < 0] = 0

    # print('!!!!!!!!!!!!')
    # print(np.min(V - V2))
    # print(np.max(V - V2))
    # print(np.min(W - W2))
    # print(np.max(W - W2))
    # print(V2)
    # lkj

    return V, W


def mask_upstream_wake(mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw):
    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(turbine_yaw) + x_coord_rotated
    return xR, yR


def initial_velocity_deficits(U_local, turbine_Ct):
    uR = U_local * turbine_Ct / (2.0 * (1 - np.sqrt(1 - turbine_Ct)))
    u0 = U_local * np.sqrt(1 - turbine_Ct)
    return uR, u0


def initial_wake_expansion(turbine_yaw, turbine_diameter, U_local, veer, uR, u0):
    yaw = -1 * turbine_yaw
    sigma_z0 = turbine_diameter * 0.5 * np.sqrt(uR / (U_local + u0))
    sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)
    return sigma_y0, sigma_z0


def calculate_effective_yaw_angle(
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
    turb_avg_vels,
    turbine_Ct,
    turbine_aI,
    turbine_TSR,
    turbine_yaw,
    turbine_hub_height,
    turbine_diameter,
    specified_wind_height,
    wind_shear,
    flow_field_v,
    flow_field_u_initial,
    use_secondary_steering,
):
    eps_gain = 0.2

    if use_secondary_steering:
        # if not flow_field.wake.velocity_model.calculate_VW_velocities:
        #     err_msg = (
        #         "It appears that 'use_secondary_steering' is set "
        #         + "to True and 'calculate_VW_velocities' is set to False. "
        #         + "This configuration is not valid. Please set "
        #         + "'use_secondary_steering' to True if you wish to use "
        #         + "yaw-added recovery."
        #     )
        #     self.logger.error(err_msg, stack_info=True)
        #     raise ValueError(err_msg)
        # turbine parameters
        Ct = turbine_Ct
        D = turbine_diameter
        HH = turbine_hub_height
        aI = turbine_aI
        TSR = turbine_TSR
        V = flow_field_v
        Uinf = np.mean(flow_field_u_initial, axis=(3, 4, 5))[:, :, :, na, na, na]
        # print(np.shape(Ct))
        # lkj

        eps = eps_gain * D  # Use set value
        # idx = np.where(
        #     (np.abs(mesh_x_rotated - x_coord_rotated) < D / 4)
        #     & (np.abs(mesh_y_rotated - y_coord_rotated) < D / 2)
        # )

        # print(np.shape(mesh_y_rotated))
        # print(np.shape(y_coord_rotated))
        # yLocs = mesh_y_rotated[idx] + 0.01 - y_coord_rotated #[na,na,na,na,na,:]
        shape = np.shape(mesh_z)
        yLocs = np.reshape(
            mesh_y_rotated + 0.01 - y_coord_rotated,
            (shape[0], shape[1], shape[2], shape[3], 1, shape[4] * shape[5]),
        )
        # print(np.shape(yLocs))
        # lkj

        # location of top vortex
        zT = np.reshape(
            mesh_z + 0.01 - (HH + D / 2),
            (shape[0], shape[1], shape[2], shape[3], 1, shape[4] * shape[5]),
        )
        # print(np.shape(zT))
        # lkj
        rT = yLocs ** 2 + zT ** 2
        # print('rt: ', np.shape(rT))

        # print(rT)
        # print(np.shape(yLocs))
        # lkj

        # location of bottom vortex
        zB = np.reshape(
            mesh_z + 0.01 - (HH - D / 2),
            (shape[0], shape[1], shape[2], shape[3], 1, shape[4] * shape[5]),
        )
        rB = yLocs ** 2 + zB ** 2

        # wake rotation vortex
        zC = np.reshape(
            mesh_z + 0.01 - (HH),
            (shape[0], shape[1], shape[2], shape[3], 1, shape[4] * shape[5]),
        )
        rC = yLocs ** 2 + zC ** 2

        # find wake deflection from CRV
        min_yaw = -45.0
        max_yaw = 45.0
        test_yaw = np.linspace(min_yaw, max_yaw, 91)
        # avg_V = np.mean(V[idx])
        avg_V = np.mean(V, axis=(4, 5))[:, :, :, :, na, na]
        # print(np.shape(idx))
        # print(np.shape(V[idx]))
        # print('2: ', np.shape(V))
        # print('6: ', avg_V)
        # print('7: ', Uinf)
        # print('turb vels: ', np.shape(turb_avg_vels))
        # print('TSR: ', np.shape(TSR))
        # print('ai: ', np.shape(aI))
        # print('1: ', np.shape((aI - aI ** 2) * turb_avg_vels))
        # print(D)

        # what yaw angle would have produced that same average spanwise velocity
        vel_top = ((HH + D / 2) / specified_wind_height) ** wind_shear
        vel_bottom = ((HH - D / 2) / specified_wind_height) ** wind_shear
        Gamma_top = (
            (np.pi / 8) * D * vel_top * Uinf * Ct * sind(test_yaw) * cosd(test_yaw)
        )
        Gamma_bottom = (
            -(np.pi / 8) * D * vel_bottom * Uinf * Ct * sind(test_yaw) * cosd(test_yaw)
        )
        Gamma_wake_rotation = (
            0.25 * 2 * np.pi * D * (aI - aI ** 2) * turb_avg_vels / TSR
        )
        print("0: ", TSR)
        print("1: ", vel_top)
        print("2: ", vel_bottom)
        print("3: ", np.mean(Gamma_top, axis=5)[0, 1, 0, 0, 0])
        print("4: ", np.mean(Gamma_bottom, axis=5)[0, 1, 0, 0, 0])
        print("5: ", Gamma_wake_rotation)
        print("8: ", Uinf)
        # print('0: ', TSR)
        # print(np.shape(Gamma_top))
        # print(np.shape(Gamma_bottom))
        # print(np.shape(Gamma_wake_rotation))
        # print(np.shape(zT))
        # print(Gamma_wake_rotation)
        # print(Gamma_top[:,:,:,:,:,na,:] * zT)
        # print(np.einsum("...i,...j", Gamma_top, zT))
        # print(np.shape(zT))
        # print(np.shape(Gamma_top))
        # print(np.shape(np.einsum("...i,...j->...ij", Gamma_top, zT)))
        # lkj
        # print(np.shape(np.divide(np.einsum("...i,...j", Gamma_top, zT), (2 * np.pi * rT))
        #     * (1 - np.exp(-rT / (eps ** 2)))))
        # print(np.shape(np.einsum("...i,...j", Gamma_bottom, zB)
        #     / (2 * np.pi * rB)
        #     * (1 - np.exp(-rB / (eps ** 2)))))
        # print(np.shape(np.einsum("...i,...j", Gamma_wake_rotation, zC)
        #     / (2 * np.pi * rC)
        #     * (1 - np.exp(-rC / (eps ** 2)))))

        # print('!!!!!!!!!!!!!!!!')
        # print(np.einsum("...i,j", Gamma_top, zT))
        # print(np.matmul(zT, Gamma_top))
        # print(Gamma_top * zT)
        # lkj
        # var = np.divide(
        #     np.einsum("...i,...j->...ij", Gamma_top, zT),
        #     (2 * np.pi * rT[:, :, :, :, :, na, :]),
        # )
        Veff = (
            np.divide(
                np.einsum("...i,...j->...ij", Gamma_top, zT),
                (2 * np.pi * rT[:, :, :, :, :, na, :]),
            )
            * (1 - np.exp(-rT[:, :, :, :, :, na, :] / (eps ** 2)))
            + np.einsum("...i,...j->...ij", Gamma_bottom, zB)
            / (2 * np.pi * rB[:, :, :, :, :, na, :])
            * (1 - np.exp(-rB[:, :, :, :, :, na, :] / (eps ** 2)))
            # + (zC * Gamma_wake_rotation)
            + np.einsum("...i,...j->...ij", Gamma_wake_rotation, zC)
            / (2 * np.pi * rC[:, :, :, :, :, na, :])
            * (1 - np.exp(-rC[:, :, :, :, :, na, :] / (eps ** 2)))
        )
        # print('3: ', np.shape(vel_top))
        # print('4: ', np.shape(vel_bottom))
        # print('3: ', np.shape(Gamma_top) )
        # print('4: ', np.shape(Gamma_bottom))
        # print('5: ', np.shape(Gamma_wake_rotation))
        # print('6a: ', np.shape(np.einsum("...i,...j->...ij", Gamma_top, zT)))
        # print('6b: ', np.shape(2 * np.pi * rT))
        # print('6c: ', np.shape((1 - np.exp(-rT / (eps ** 2)))))
        # print('6d: ', np.shape(np.divide(np.einsum("...i,...j->...ij", Gamma_top, zT), (2 * np.pi * rT))))
        # print('7: ', np.shape(np.einsum("...i,...j->...ij", Gamma_bottom, zB)
        #     / (2 * np.pi * rB)
        #     * (1 - np.exp(-rB / (eps ** 2)))))
        # print('8: ', np.shape(np.einsum("...i,...j->...ij", Gamma_wake_rotation, zC)
        #     / (2 * np.pi * rC)
        #     * (1 - np.exp(-rC / (eps ** 2)))))
        # print('9: ', np.shape(Veff))
        # print('8: ', Uinf)
        # print('9: ', Ct)

        # print(np.shape(turb_avg_vels))
        # print(np.shape(aI))
        # print('1: ', np.shape(np.divide(np.einsum("...i,j", Gamma_top, zT), (2 * np.pi * rT))
        #     * (1 - np.exp(-rT / (eps ** 2)))))
        # print('2: ', np.shape(np.einsum("...i,j", Gamma_bottom, zB)
        #     / (2 * np.pi * rB)
        #     * (1 - np.exp(-rB / (eps ** 2)))))
        # print('3: ', np.shape(np.einsum("...i,j", Gamma_wake_rotation, zC)
        #     / (2 * np.pi * rC)
        #     * (1 - np.exp(-rC / (eps ** 2)))))

        tmp = avg_V - np.mean(Veff, axis=6)
        # print('veff: ', np.shape(Veff))
        # print(np.shape(avg_V))
        # print(np.mean(Veff, axis=6))
        # print(np.shape(tmp))
        # print('tmp: ', np.mean(tmp, axis=5)[0,1,0,0,0])
        # lkj

        # return indices of sorted residuals to find effective yaw angle
        order = np.argsort(np.abs(tmp), axis=5)
        # print(order)
        # print(np.shape(tmp))
        # lkj
        # lkj
        # idx_1 = order[0][0][0][0][0][0]
        # idx_2 = order[0][0][:][0][0][1]
        idx_1 = np.take_along_axis(order, np.array([[[[[[0]]]]]]), axis=5)
        idx_2 = np.take_along_axis(order, np.array([[[[[[1]]]]]]), axis=5)

        # check edge case, if true, assign max yaw value
        if 0:
            pass
        # if idx_1 == 90 or idx_2 == 90:
        #     yaw_effective = max_yaw
        # check edge case, if true, assign min yaw value
        # elif idx_1 == 0 or idx_2 == 0:
        #     yaw_effective = -min_yaw
        # for each identified minimum residual, use adjacent points to determine
        # two equations of line and find the intersection of the two lines to
        # determine the effective yaw angle to add; the if/else structure is based
        # on which residual index is larger
        else:
            # if idx_1 > idx_2:
            #     idx_right = idx_1 + 1  # adjacent point
            #     idx_left = idx_2 - 1  # adjacent point
            #     mR = abs(tmp[0][0][0][0][0][idx_right]) - abs(tmp[0][0][0][0][0][idx_1])  # slope
            #     mL = abs(tmp[0][0][0][0][0][idx_2]) - abs(tmp[0][0][0][0][0][idx_left])  # slope
            #     bR = abs(tmp[0][0][0][0][0][idx_1]) - mR * float(idx_1)  # intercept
            #     bL = abs(tmp[0][0][0][0][0][idx_2]) - mL * float(idx_2)  # intercept
            # else:
            #     idx_right = idx_2 + 1  # adjacent point
            #     idx_left = idx_1 - 1  # adjacent point
            #     mR = abs(tmp[0][0][0][0][0][idx_right]) - abs(tmp[0][0][0][0][0][idx_2])  # slope
            #     mL = abs(tmp[0][0][0][0][0][idx_1]) - abs(tmp[0][0][0][0][0][idx_left])  # slope
            #     bR = abs(tmp[0][0][0][0][0][idx_2]) - mR * float(idx_2)  # intercept
            #     bL = abs(tmp[0][0][0][0][0][idx_1]) - mL * float(idx_1)  # intercept

            # if idx_1.flatten() > idx_2.flatten():
            #     idx_right = idx_1 + 1  # adjacent point
            #     idx_left = idx_2 - 1  # adjacent point
            # #     mR = abs(tmp[idx_right]) - abs(tmp[idx_1])  # slope
            # #     mL = abs(tmp[idx_2]) - abs(tmp[idx_left])  # slope
            # #     bR = abs(tmp[idx_1]) - mR * float(idx_1)  # intercept
            # #     bL = abs(tmp[idx_2]) - mL * float(idx_2)  # intercept
            #     mR = abs(np.take_along_axis(tmp, idx_right, axis=5) - abs(np.take_along_axis(tmp, idx_1, axis=5)))  # slope
            #     mL = abs(np.take_along_axis(tmp, idx_2, axis=5)) - abs(np.take_along_axis(tmp, idx_left, axis=5))  # slope
            #     bR = abs(np.take_along_axis(tmp, idx_1, axis=5)) - mR * idx_1  # intercept
            #     bL = abs(np.take_along_axis(tmp, idx_2, axis=5)) - mL * idx_2  # intercept

            # else:
            #     idx_right = idx_2 + 1  # adjacent point
            #     idx_left = idx_1 - 1  # adjacent point
            # # mR = abs(tmp[idx_right]) - abs(tmp[idx_2])  # slope
            # # mL = abs(tmp[idx_1]) - abs(tmp[idx_left])  # slope
            # # bR = abs(tmp[idx_2]) - mR * float(idx_2)  # intercept
            # # bL = abs(tmp[idx_1]) - mL * float(idx_1)  # intercept
            #     mR = abs(np.take_along_axis(tmp, idx_right, axis=5) - abs(np.take_along_axis(tmp, idx_2, axis=5)))  # slope
            #     mL = abs(np.take_along_axis(tmp, idx_1, axis=5)) - abs(np.take_along_axis(tmp, idx_left, axis=5))  # slope
            #     bR = abs(np.take_along_axis(tmp, idx_2, axis=5)) - mR * idx_2  # intercept
            #     bL = abs(np.take_along_axis(tmp, idx_1, axis=5)) - mL * idx_1  # intercept

            mask1 = np.array(idx_1 > idx_2)
            mask2 = np.array(idx_1 <= idx_2)

            idx_right_1 = idx_1 + 1  # adjacent point
            idx_left_1 = idx_2 - 1  # adjacent point
            mR_1 = abs(
                np.take_along_axis(tmp, idx_right_1, axis=5)
                - abs(np.take_along_axis(tmp, idx_1, axis=5))
            )  # slope
            mL_1 = abs(np.take_along_axis(tmp, idx_2, axis=5)) - abs(
                np.take_along_axis(tmp, idx_left_1, axis=5)
            )  # slope
            bR_1 = (
                abs(np.take_along_axis(tmp, idx_1, axis=5)) - mR_1 * idx_1
            )  # intercept
            bL_1 = (
                abs(np.take_along_axis(tmp, idx_2, axis=5)) - mL_1 * idx_2
            )  # intercept

            idx_right_2 = idx_2 + 1  # adjacent point
            idx_left_2 = idx_1 - 1  # adjacent point
            mR_2 = abs(
                np.take_along_axis(tmp, idx_right_2, axis=5)
                - abs(np.take_along_axis(tmp, idx_2, axis=5))
            )  # slope
            mL_2 = abs(np.take_along_axis(tmp, idx_1, axis=5)) - abs(
                np.take_along_axis(tmp, idx_left_2, axis=5)
            )  # slope
            bR_2 = (
                abs(np.take_along_axis(tmp, idx_2, axis=5)) - mR_2 * idx_2
            )  # intercept
            bL_2 = (
                abs(np.take_along_axis(tmp, idx_1, axis=5)) - mL_2 * idx_1
            )  # intercept

            mR = mR_1 * mask1 + mR_2 * mask2
            mL = mL_1 * mask1 + mL_2 * mask2
            bR = bR_1 * mask1 + bR_2 * mask2
            bL = bL_1 * mask1 + bL_2 * mask2

            # print('idx_1: ', idx_1)
            # print('idx_2: ', idx_2)
            # print('idx_left: ', idx_left)
            # print('idx_right: ', idx_right)
            # print('mR: ', mR)
            # print('mL: ', mL)
            # print('bR: ', bR)
            # print('bL: ', bL)
            # find the value at the intersection of the two lines
            # ival = np.divide((bR - bL), (mL - mR))
            ival = (bR - bL) / (mL - mR)
            print("ival: ", ival)
            # convert the indice into degrees
            yaw_effective = ival - max_yaw
            # print(np.shape((bR - bL) / (mL - mR)))
            # print((bR - bL) / (mL - mR))
            print("yaw_effective: ", yaw_effective)
            # print(Uinf)
            # print(mR)
            # print(mL)
            # print(bR)
            # print(bL)
            # [print('ival: ', val) for val in ival[0][0][0][0][0]]
            # [print('ival: ', val) for val in ival[0][0][1][0][0]]
            # print(ival[0][0][0][0][0])
            # print(ival[0][0][1][0][0])
            # print(np.shape(ival))
            # print((bR - bL))
            # print((mL - mR))
            # print('!!!!!!!!!!!!!!!!!!!!!!!')
            # print(yaw_effective + turbine_yaw)
            # lkj

        return yaw_effective + turbine_yaw
    else:
        return turbine_yaw


def gaussian_function(U, C, r, n, sigma):
    return U * C * np.exp(-1 * r ** n / (2 * sigma ** 2))


def gauss_vel_model(
    veer,
    wind_shear,
    specified_wind_height,
    flow_field_u_initial,
    flow_field_u,
    flow_field_v,
    flow_field_w,
    turb_avg_vels,
    turbine_ti,
    turbine_Ct,
    turbine_aI,
    turbine_TSR,
    turbine_yaw,
    turbine_hub_height,
    turbine_diameter,
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
    deflection_field,
):
    # gch_gain = 2.0
    alpha = 0.58
    beta = 0.077
    ka = 0.38
    kb = 0.004

    # turb_v, turb_w = calc_VW(
    #     x_coord_rotated, y_coord_rotated, wind_shear, specified_wind_height, turb_avg_vels, turbine_ti, turbine_Ct, turbine_aI, turbine_TSR, turbine_yaw, turbine_hub_height, turbine_diameter, flow_field_u_initial, flow_field_u, mesh_x_rotated, mesh_y_rotated, mesh_z
    # )

    # TI_mixing = yaw_added_turbulence_mixing(
    #     x_coord_rotated,
    #     y_coord_rotated,
    #     turb_avg_vels,
    #     turbine_ti,
    #     turbine_diameter,
    #     flow_field_v,
    #     flow_field_w,
    #     turb_v,
    #     turb_w,
    #     mesh_x_rotated,
    #     mesh_y_rotated,
    # )
    # print('TI_mixing: ', TI_mixing)
    # turbine_ti = (
    #     turbine_ti + gch_gain * TI_mixing
    # )
    # TI = copy.deepcopy(turbine_ti)  # + TI_mixing
    # print('TI: ', TI)

    # turbine parameters
    D = turbine_diameter
    HH = turbine_hub_height
    yaw = -1 * turbine_yaw  # opposite sign convention in this model
    Ct = turbine_Ct
    U_local = flow_field_u_initial

    # wake deflection
    delta = deflection_field

    xR, _ = mask_upstream_wake(
        mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw
    )
    uR, u0 = initial_velocity_deficits(U_local, Ct)
    sigma_y0, sigma_z0 = initial_wake_expansion(
        turbine_yaw, turbine_diameter, U_local, veer, uR, u0
    )

    # quantity that determines when the far wake starts
    x0 = (
        D
        * (cosd(yaw) * (1 + np.sqrt(1 - Ct)))
        / (np.sqrt(2) * (4 * alpha * turbine_ti + 2 * beta * (1 - np.sqrt(1 - Ct))))
        + x_coord_rotated
    )

    # velocity deficit in the near wake
    sigma_y = (((x0 - xR) - (mesh_x_rotated - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(
        Ct / 2.0
    ) + ((mesh_x_rotated - xR) / (x0 - xR)) * sigma_y0
    sigma_z = (((x0 - xR) - (mesh_x_rotated - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(
        Ct / 2.0
    ) + ((mesh_x_rotated - xR) / (x0 - xR)) * sigma_z0
    # print('x0: ', np.mean(x0))
    # print('xR: ', np.mean(xR))
    sigma_y = (
        sigma_y * np.array(mesh_x_rotated >= xR)
        + np.ones_like(sigma_y) * np.array(mesh_x_rotated < xR) * 0.5 * D
    )
    sigma_z = (
        sigma_z * np.array(mesh_x_rotated >= xR)
        + np.ones_like(sigma_z) * np.array(mesh_x_rotated < xR) * 0.5 * D
    )
    # print('sigma_y: ', np.mean(sigma_y))
    # print('sigma_z: ', np.mean(sigma_z))

    # sigma_y[mesh_x_rotated < xR] = 0.5 * D
    # sigma_z[mesh_x_rotated < xR] = 0.5 * D
    # print(np.min(sigma_y2 - sigma_y))
    # print(np.max(sigma_y2 - sigma_y))

    a = cosd(veer) ** 2 / (2 * sigma_y ** 2) + sind(veer) ** 2 / (2 * sigma_z ** 2)
    b = -sind(2 * veer) / (4 * sigma_y ** 2) + sind(2 * veer) / (4 * sigma_z ** 2)
    c = sind(veer) ** 2 / (2 * sigma_y ** 2) + cosd(veer) ** 2 / (2 * sigma_z ** 2)
    r = (
        a * ((mesh_y_rotated - y_coord_rotated) - delta) ** 2
        - 2 * b * ((mesh_y_rotated - y_coord_rotated) - delta) * ((mesh_z - HH))
        + c * ((mesh_z - HH)) ** 2
    )
    C = 1 - np.sqrt(
        np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)), 0.0, 1.0)
    )

    velDef = gaussian_function(U_local, C, r, 1, np.sqrt(0.5))
    velDef = velDef * np.array(mesh_x_rotated >= xR)
    velDef = velDef * np.array(mesh_x_rotated <= x0)
    # velDef[mesh_x_rotated < xR] = 0
    # velDef[mesh_x_rotated > x0] = 0
    # print(np.min(velDef2 - velDef))
    # print(np.max(velDef2 - velDef))

    # wake expansion in the lateral (y) and the vertical (z)
    ky = ka * turbine_ti + kb  # wake expansion parameters
    kz = ka * turbine_ti + kb  # wake expansion parameters
    sigma_y = ky * (mesh_x_rotated - x0) + sigma_y0
    sigma_z = kz * (mesh_x_rotated - x0) + sigma_z0
    sigma_y = sigma_y * np.array(mesh_x_rotated >= x0) + sigma_y0 * np.array(
        mesh_x_rotated < x0
    )
    sigma_z = sigma_z * np.array(mesh_x_rotated >= x0) + sigma_z0 * np.array(
        mesh_x_rotated < x0
    )
    # sigma_y[mesh_x_rotated < x0] = sigma_y0[mesh_x_rotated < x0]
    # sigma_z[mesh_x_rotated < x0] = sigma_z0[mesh_x_rotated < x0]
    # print(np.min(sigma_y2 - sigma_y))
    # print(np.max(sigma_z2 - sigma_z))

    # velocity deficit outside the near wake
    a = cosd(veer) ** 2 / (2 * sigma_y ** 2) + sind(veer) ** 2 / (2 * sigma_z ** 2)
    b = -sind(2 * veer) / (4 * sigma_y ** 2) + sind(2 * veer) / (4 * sigma_z ** 2)
    c = sind(veer) ** 2 / (2 * sigma_y ** 2) + cosd(veer) ** 2 / (2 * sigma_z ** 2)
    r = (
        a * (mesh_y_rotated - y_coord_rotated - delta) ** 2
        - 2 * b * (mesh_y_rotated - y_coord_rotated - delta) * (mesh_z - HH)
        + c * (mesh_z - HH) ** 2
    )
    C = 1 - np.sqrt(
        np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)), 0.0, 1.0)
    )

    # compute velocities in the far wake
    velDef1 = gaussian_function(U_local, C, r, 1, np.sqrt(0.5))
    velDef1 = velDef1 * np.array(mesh_x_rotated >= x0)
    # velDef1[mesh_x_rotated < x0] = 0

    turb_u = np.sqrt(velDef ** 2 + velDef1 ** 2)

    return turb_u


def gauss_defl_model(
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
    flow_field_u_initial,
    flow_field_v,
    wind_veer,
    wind_shear,
    turb_avg_vels,
    turbine_ti,
    turbine_Ct,
    turbine_aI,
    turbine_TSR,
    turbine_yaw,
    turbine_tilt,
    turbine_hub_height,
    turbine_diameter,
    use_secondary_steering,
):
    # free-stream velocity (m/s)
    wind_speed = flow_field_u_initial
    # veer (degrees)
    veer = wind_veer

    # added turbulence model
    TI = turbine_ti

    ka = 0.38  # wake expansion parameter
    kb = 0.004  # wake expansion parameter
    alpha = 0.58  # near wake parameter
    beta = 0.077  # near wake parameter
    ad = 0.0  # natural lateral deflection parameter
    bd = 0.0  # natural lateral deflection parameter
    dm = 1.0

    # turbine parameters
    D = turbine_diameter
    # yaw = -1 * calculate_effective_yaw_angle(
    #     mesh_x_rotated,
    #     mesh_y_rotated,
    #     mesh_z,
    #     turb_avg_vels,
    #     turbine_Ct,
    #     turbine_aI,
    #     turbine_TSR,
    #     turbine_yaw,
    #     turbine_hub_height,
    #     turbine_diameter,
    #     specified_wind_height,
    #     wind_shear,
    #     x_coord_rotated,
    #     y_coord_rotated,
    #     flow_field_v,
    #     flow_field_u_initial,
    #     use_secondary_steering,
    # ) #* np.array(mesh_x_rotated > x_coord_rotated)
    # print('yaw: ', np.shape(yaw))

    # opposite sign convention in this model
    tilt = turbine_tilt
    Ct = turbine_Ct

    # U_local = flow_field.wind_map.grid_wind_speed
    # just a placeholder for now, should be initialized with the flow_field
    U_local = flow_field_u_initial

    # initial velocity deficits
    uR = (
        U_local
        * Ct
        * cosd(tilt)
        * cosd(yaw)
        / (2.0 * (1 - np.sqrt(1 - (Ct * cosd(tilt) * cosd(yaw)))))
    )
    u0 = U_local * np.sqrt(1 - Ct)

    # length of near wake
    x0 = (
        D
        * (cosd(yaw) * (1 + np.sqrt(1 - Ct * cosd(yaw))))
        / (np.sqrt(2) * (4 * alpha * TI + 2 * beta * (1 - np.sqrt(1 - Ct))))
        + x_coord_rotated
    )

    # wake expansion parameters
    ky = ka * TI + kb
    kz = ka * TI + kb

    C0 = 1 - u0 / wind_speed
    M0 = C0 * (2 - C0)
    E0 = C0 ** 2 - 3 * np.exp(1.0 / 12.0) * C0 + 3 * np.exp(1.0 / 3.0)

    # initial Gaussian wake expansion
    sigma_z0 = D * 0.5 * np.sqrt(uR / (U_local + u0))
    sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)

    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(yaw) + x_coord_rotated

    # yaw parameters (skew angle and distance from centerline)
    theta_c0 = (
        dm * (0.3 * np.radians(yaw) / cosd(yaw)) * (1 - np.sqrt(1 - Ct * cosd(yaw)))
    )  # skew angle in radians
    delta0 = np.tan(theta_c0) * (x0 - x_coord_rotated)  # initial wake deflection;
    # NOTE: use np.tan here since theta_c0 is radians

    # deflection in the near wake
    delta_near_wake = ((mesh_x_rotated - xR) / (x0 - xR)) * delta0 + (
        ad + bd * (mesh_x_rotated - x_coord_rotated)
    )
    delta_near_wake = delta_near_wake * np.array(mesh_x_rotated >= xR)
    delta_near_wake = delta_near_wake * np.array(mesh_x_rotated <= x0)
    # delta_near_wake[mesh_x_rotated < xR] = 0.0
    # delta_near_wake[mesh_x_rotated > x0] = 0.0
    # print(np.min(delta_near_wake2 - delta_near_wake))
    # print(np.max(delta_near_wake2 - delta_near_wake))

    # deflection in the far wake
    sigma_y = ky * (mesh_x_rotated - x0) + sigma_y0
    sigma_z = kz * (mesh_x_rotated - x0) + sigma_z0
    sigma_y = sigma_y * np.array(mesh_x_rotated >= x0) + sigma_y0 * np.array(
        mesh_x_rotated < x0
    )
    sigma_z = sigma_z * np.array(mesh_x_rotated >= x0) + sigma_z0 * np.array(
        mesh_x_rotated < x0
    )
    # sigma_y[mesh_x_rotated < x0] = sigma_y0[mesh_x_rotated < x0]
    # sigma_z[mesh_x_rotated < x0] = sigma_z0[mesh_x_rotated < x0]
    # print(np.min(sigma_y2 - sigma_y))
    # print(np.max(sigma_z2 - sigma_z))

    ln_deltaNum = (1.6 + np.sqrt(M0)) * (
        1.6 * np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0)) - np.sqrt(M0)
    )
    ln_deltaDen = (1.6 - np.sqrt(M0)) * (
        1.6 * np.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0)) + np.sqrt(M0)
    )

    delta_far_wake = (
        delta0
        + (theta_c0 * E0 / 5.2)
        * np.sqrt(sigma_y0 * sigma_z0 / (ky * kz * M0))
        * np.log(ln_deltaNum / ln_deltaDen)
        + (ad + bd * (mesh_x_rotated - x_coord_rotated))
    )

    delta_far_wake = delta_far_wake * np.array(mesh_x_rotated > x0)
    # delta_far_wake[mesh_x_rotated <= x0] = 0.0
    # print(np.min(delta_far_wake2 - delta_far_wake))
    # print(np.max(delta_far_wake2 - delta_far_wake))

    deflection = delta_near_wake + delta_far_wake
    # print('delta_near_wake: ', np.mean(delta_near_wake))
    # print('delta_far_wake: ', np.mean(delta_far_wake))

    return deflection


def crespo_hernandez(
    ambient_TI, x_coord_downstream, x_coord_upstream, rotor_diameter, aI
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
    print("MIN!!!!!!!!!!!!!: ", np.min(x_coord_downstream - x_coord_upstream))
    print("MAX!!!!!!!!!!!!!: ", np.max(x_coord_downstream - x_coord_upstream))
    # Update turbulence intensity of downstream turbines
    return ti_calculation


def turbine_avg_velocity(turb_inflow_vel):
    return np.cbrt(np.mean(turb_inflow_vel ** 3, axis=(4, 5)))


fCtInterp = interp1d(wind_speed, thrust, fill_value="extrapolate")


def Ct(turb_avg_vels):
    Ct_vals = fCtInterp(turb_avg_vels)
    Ct_vals[Ct_vals > 1.0] = 0.9999
    Ct_vals[Ct_vals < 0.0] = 0.0001
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

x_spc = 5 * 126.0
x_coord = np.array([0.0, x_spc, 2 * x_spc])  # , 3*x_spc, 4*x_spc])  # , 0*126.0])
y_coord = np.array([0.0, 0.0, 0.0])  # , 0.0, 0.0])  # , 5*126.0])
z_coord = np.array([90.0] * len(x_coord))  # , 90.0])

y_ngrid = 5
z_ngrid = 5
rloc = 0.5

dtype = np.float64
# Wind parameters
ws = np.array([8.0])
wd = np.array([275.0, 270.0])
# i  j  k  l  m
# wd ws x  y  z

specified_wind_height = 90.0
wind_shear = 0.12
wind_veer = 0.0

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


def calculate_area_overlap(wake_velocities, freestream_velocities, y_ngrid, z_ngrid):
    """
    compute wake overlap based on the number of points that are not freestream velocity, i.e. affected by the wake
    """
    count = np.sum(freestream_velocities - wake_velocities <= 0.05, axis=(4, 5))
    return (y_ngrid * z_ngrid - count) / (y_ngrid * z_ngrid)


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
    flow_field_v_initial = np.zeros(np.shape(flow_field_u_initial))
    flow_field_w_initial = np.zeros(np.shape(flow_field_u_initial))

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
        flow_field_v_initial,
        flow_field_w_initial,
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
    flow_field_v_initial,
    flow_field_w_initial,
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
flow_field_u = flow_field_u_initial - u_wake
flow_field_v = flow_field_v_initial
flow_field_w = flow_field_w_initial
# deflection_field = np.zeros(np.shape(flow_field_u_initial), dtype=dtype)
turb_inflow_field = (
    np.ones(np.shape(flow_field_u_initial), dtype=dtype) * flow_field_u_initial
)

TI = 0.06
turb_TIs = np.ones_like(x_coord_rotated) * TI
ambient_TIs = np.ones_like(x_coord_rotated) * TI
yaw_angle = np.ones_like(x_coord_rotated) * 0.0
# print(np.shape(yaw_angle))
# yaw_angle[:,:,:,0,:,:] = 20.0
turbine_tilt = np.ones_like(x_coord_rotated) * 0.0
turbine_TSR = np.ones_like(x_coord_rotated) * 8.0

tic = time.perf_counter()

use_secondary_steering = True

for i in range(len(x_coord)):
    # print(np.shape((flow_field_u_initial - u_wake)))
    # print(np.shape(turb_inflow_field))
    # turb_inflow_field[:, :, :, i, :, :] = (flow_field_u_initial - u_wake)[
    #     :, :, :, i, :, :
    # ]
    # print('shape 1: ', np.shape(flow_field_u_initial))
    # print('shape 2: ', np.shape(u_wake))
    turb_inflow_field = turb_inflow_field * np.array(
        mesh_x_rotated != x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
    ) + (flow_field_u_initial - u_wake) * np.array(
        mesh_x_rotated == x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
    )
    # print('!!!!!!!!!!!!!!!!!!')
    # print(u_wake * np.array(mesh_x_rotated >= x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]))
    # print('$$$$$$$$$$$$$')

    print(np.shape(turb_inflow_field))
    turb_avg_vels = turbine_avg_velocity(turb_inflow_field)
    print("222: ", turb_avg_vels)
    print(np.shape(turb_avg_vels))
    turb_Cts = Ct(turb_avg_vels)
    turb_aIs = aI(turb_Cts)
    # turb_TIs = np.ones(np.shape(turb_aIs)) * 0.06

    # deflection_field = jimenez_model(
    #     yaw_angle,
    #     turb_Cts[:, :, :, i],
    #     x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
    #     mesh_x_rotated,
    #     turbine_diameter,
    # )

    yaw = -1 * calculate_effective_yaw_angle(
        mesh_x_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        mesh_y_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        mesh_z[:, :, :, i, :, :][:, :, :, na, :, :],
        x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
        turb_Cts[:, :, :, i][:, :, :, na, na, na],
        turb_aIs[:, :, :, i][:, :, :, na, na, na],
        turbine_TSR[:, :, :, i, :, :][:, :, :, na, :, :],
        yaw_angle,
        turbine_hub_height,
        turbine_diameter,
        specified_wind_height,
        wind_shear,
        flow_field_v[:, :, :, i, :, :][:, :, :, na, :, :],
        flow_field_u_initial,
        use_secondary_steering,
    )

    print("1: ", np.shape(flow_field_v))
    deflection_field = gauss_defl_model(
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z,
        x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        flow_field_u_initial,
        flow_field_v[:, :, :, i, :, :][:, :, :, na, :, :],
        wind_veer,
        wind_shear,
        turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
        turb_TIs[:, :, :, i, :, :][:, :, :, na, :, :],
        turb_Cts[:, :, :, i][:, :, :, na, na, na],
        turb_aIs[:, :, :, i][:, :, :, na, na, na],
        turbine_TSR[:, :, :, i, :, :][:, :, :, na, :, :],
        yaw_angle,
        turbine_tilt,
        turbine_hub_height,
        turbine_diameter,
        use_secondary_steering,
    )

    turb_v_wake, turb_w_wake = calc_VW(
        x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        wind_shear,
        specified_wind_height,
        turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
        turb_TIs[:, :, :, i, :, :][:, :, :, na, :, :],
        turb_Cts[:, :, :, i][:, :, :, na, na, na],
        turb_aIs[:, :, :, i][:, :, :, na, na, na],
        turbine_TSR,
        yaw_angle,
        turbine_hub_height,
        turbine_diameter,
        flow_field_u_initial,
        flow_field_u,
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z,
    )

    TI_mixing = yaw_added_turbulence_mixing(
        x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
        turb_TIs[:, :, :, i, :, :][:, :, :, na, :, :],
        turbine_diameter,
        flow_field_v,
        flow_field_w,
        turb_v_wake,
        turb_w_wake,
        mesh_x_rotated,
        mesh_y_rotated,
    )
    # TI_mixing = 0.0
    # print('TI_mixing: ', TI_mixing)
    gch_gain = 2
    # print('20: ', turb_TIs)
    turb_TIs = turb_TIs + gch_gain * TI_mixing * (
        np.array(
            x_coord_rotated == x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
        )
    )
    # print('21: ', turb_TIs)
    # print('22: ', (
    #         np.array(
    #             x_coord_rotated == x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
    #         )
    #     ))
    # TI = copy.deepcopy(turb_TIs)  # + TI_mixing
    # print('TI: ', TI)
    # print('Ct: ', turb_aIs)

    veer = 0.0
    turb_u_wake = gauss_vel_model(
        veer,
        wind_shear,
        specified_wind_height,
        flow_field_u_initial,
        flow_field_u,
        flow_field_v,
        flow_field_w,
        turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
        turb_TIs[:, :, :, i, :, :][:, :, :, na, :, :],
        turb_Cts[:, :, :, i][:, :, :, na, na, na],
        turb_aIs[:, :, :, i][:, :, :, na, na, na],
        turbine_TSR,
        yaw_angle,
        turbine_hub_height,
        turbine_diameter,
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z,
        x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        deflection_field,
    )

    # turb_v_wake = np.zeros_like(turb_u_wake)
    # turb_w_wake = np.zeros_like(turb_u_wake)
    # print('10: ', np.mean(turb_u_wake))

    print("##################### i: ", i)
    # print(turb_u_wake)

    print("")
    u_wake = np.sqrt((u_wake ** 2) + (np.array(turb_u_wake) ** 2))
    flow_field_u = flow_field_u_initial - u_wake
    flow_field_v = flow_field_v + turb_v_wake
    flow_field_w = flow_field_w + turb_w_wake

    turb_wake_field = flow_field_u_initial - turb_u_wake

    area_overlap = calculate_area_overlap(
        turb_wake_field, flow_field_u_initial, y_ngrid, z_ngrid
    )

    # print('10: ', turb_TIs)
    WAT_TIs = crespo_hernandez(
        ambient_TIs,
        x_coord_rotated,
        x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        turbine_diameter,
        turb_aIs[:, :, :, i][:, :, :, na, na, na],
    )
    print("11: ", WAT_TIs)
    # print('111: ', turb_aIs)
    # TODO: will need to make the rotor_diameter part of this mask work for turbines of different types
    downstream_influence_length = 15 * turbine_diameter
    ti_added = (
        area_overlap[:, :, :, :, na, na]
        * np.nan_to_num(WAT_TIs, posinf=0.0)
        * (
            np.array(
                x_coord_rotated > x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
            )
        )
        * (
            np.array(
                np.abs(
                    y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
                    - y_coord_rotated
                )
                < 2 * turbine_diameter
            )
        )
        * (
            np.array(
                x_coord_rotated
                <= downstream_influence_length
                + x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
            )
        )
    )
    # # print('mask: ', (
    #     np.array(x_coord_rotated > x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :])
    # ) * (
    #     np.array(np.abs(y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :] - y_coord_rotated) < 2*turbine_diameter)
    # ) * (
    #     np.array(x_coord_rotated <= downstream_influence_length + x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :])
    # ))
    print("12: ", ti_added[0, 1, 0, 2, 0, 0])

    # print('13: ', np.sqrt(ti_added ** 2 + turb_TIs ** 2))
    print("14: ", turb_TIs)
    turb_TIs = np.maximum(np.sqrt(ti_added ** 2 + ambient_TIs ** 2), turb_TIs,)
    print("2: ", turb_TIs)


[print(val) for val in turb_TIs.flatten()]


toc = time.perf_counter()

# /////////////// #
# COMPARE METHODS #
# /////////////// #

# flow_field_u = flow_field_u_initial - u_wake
# print(
#     "vec u: ",
#     np.take_along_axis(
#         np.take_along_axis(flow_field_u, inds_sorted, axis=3), inds_unsorted, axis=3
#     ),
# )
# print("vec shape of u: ", np.shape(flow_field_u_initial - u_wake))
# print("Turbine avg vels: ", turbine_avg_velocity(turb_inflow_field))

# //////////////// #
# COMPUTE GRADIENT #
# //////////////// #

# tic = time.perf_counter()
# calc_wake_grad = grad(calculate_wake)(locs, turbine_diameter, turbine_ai, ws, wd, specified_wind_height, wind_shear)
# print('gradient: ', calc_wake_grad)
# toc = time.perf_counter()
# print(f"Computed gradient in {toc - tic:0.4f} seconds")
