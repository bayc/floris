# import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
from floris.utilities import Vec3
# from functions import get_power_at_coord, sweep_power, sweep_long

import numpy as np
import os
import copy
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid

# Parameters

data_folder = '/mnt/c/Users/cbay/Box Sync/sowfa_library/full_runs/' + \
              'three_turbine_eagle/eagle_runs/'
floris_input_gauss = '../example_input.json'
floris_input_curl = '../example_input_curl.json'
fig_folder = 'figures/three_turb'

wind_speed = 8.38
TI = 0.09

D = 126 # rotor diameter [m]
dist_downstream = 6*D
turbine_to_measure_behind = 0

minspeed = -3.0
maxspeed = 3.01

# Make the figure folder
if not os.path.exists(fig_folder):
    os.mkdir(fig_folder)

# Read in the cases
# 'C_0014_0_0_7_0_0' - veer: 0.08 at 50; 0.025 at 100;  -0.1 at 150; -0.225 at 200; -0.33 at 250
# 'C_0050_20_0_7_0_0' - veer: 0.1 at 50; -0.01 at 100; -0.12 at 150; -0.25 at 200; -0.325 at 250
for case_name in ['C_0050_20_0_7_0_0']:

    # Select case based on number
    # case_name = case_name_list[case_number]
    fig_root = os.path.join(fig_folder, case_name)

    no_turbine_case_folder = data_folder + '../../near_wake/hi_no_turbine'

    m_ind = [m.start() for m in re.finditer('_', case_name)]
    yaw_angle_T1 = case_name[m_ind[1]+1:m_ind[2]]
    yaw_angle_T2 = case_name[m_ind[2]+1:m_ind[3]]
    turb_spacing = case_name[m_ind[3]+1:m_ind[4]]
    rotor_offset_T1 = case_name[m_ind[4]+1:m_ind[5]]
    rotor_offset_T2 = case_name[m_ind[5]+1:]

    # Import SOWFA
    case_folder = os.path.join(data_folder,case_name)

    sowfa_case = wfct.sowfa_utilities.SowfaInterface(
        case_folder, flow_data_sub_path='array_mean/array.mean0D_UAvg.vtk')
    
    layout_x = sowfa_case.layout_x
    layout_y = sowfa_case.layout_y

    y_center = layout_y[0]

    # print('layout_x: ', layout_x)
    # print('layout_y: ', layout_y)

    x_loc = sowfa_case.layout_x[turbine_to_measure_behind] + dist_downstream
    print('layout_x: ', layout_x)
    print('layout_y: ', layout_y)
    print('x_loc: ', x_loc)

    x_floris_bnds = [350, 650]
    y_floris_bnds = [10, 180]

    # print('x_loc: ', x_loc)

    sowfa_flow_data = sowfa_case.flow_data
    cut_plane_sowfa = sowfa_case.get_cross_plane(
        x_loc,
        x_bounds=x_floris_bnds,
        y_bounds=y_floris_bnds
        )
    hor_plane_sowfa = sowfa_case.get_hor_plane(height=90)

    # Get the no-turbine sowfa flow
    sowfa_case_no_turbine = wfct.sowfa_utilities.SowfaInterface(
                        no_turbine_case_folder,
                        flow_data_sub_path='array_mean/array.mean0D_UAvg.vtk')

    sowfa_flow_data_no_turbine = sowfa_case_no_turbine.flow_data
    # cut_plane_sowfa_base = sowfa_case_no_turbine.get_cross_plane(x_loc)
    cut_plane_sowfa_base = sowfa_case.get_cross_plane(10,
                                                      x_bounds=x_floris_bnds,
                                                      y_bounds=y_floris_bnds)
    

    # Load the FLORIS case in
    fi_gauss = wfct.floris_interface.FlorisInterface(floris_input_gauss)
    fi_gauss.reinitialize_flow_field(wind_speed=[wind_speed],
                               turbulence_intensity=[TI],
                               layout_array=(layout_x, layout_y))
    fi_gauss.floris.farm.wake.velocity_model.use_yaw_added_recovery = False
    fi_gauss.floris.farm.wake.velocity_model.use_secondary_steering = False
    fi_gauss.calculate_wake(yaw_angles=sowfa_case.yaw_angles)

    cut_plane_floris_gauss_base = fi_gauss.get_cross_plane(10)
    cut_plane_floris_gauss = fi_gauss.get_cross_plane(x_loc)
    cut_plane_floris_gauss_vel_def = cp.subtract(cut_plane_floris_gauss,
                                                 cut_plane_floris_gauss_base)
    # hor_plane_floris_gauss = fi_gauss.get_hor_plane()

    # Project FLORIS cut plane onto grid of SOWFA cut plane
    cut_plane_floris_gauss_project = cp.project_onto(
                                        cut_plane_floris_gauss_vel_def,
                                        cut_plane_sowfa)

    # Load the FLORIS case in
    fi_curl = wfct.floris_interface.FlorisInterface(floris_input_curl)
    fi_curl.reinitialize_flow_field(wind_speed=[wind_speed],
                               turbulence_intensity=[TI],
                               layout_array=(layout_x, layout_y))
    fi_curl.floris.farm.set_wake_model('curl')
    fi_curl.floris.farm.wake.velocity_model.use_yaw_added_recovery = False
    fi_curl.floris.farm.wake.velocity_model.use_secondary_steering = False
    fi_curl.calculate_wake(yaw_angles=sowfa_case.yaw_angles)

    cut_plane_floris_curl_base = fi_curl.get_cross_plane(10)
    cut_plane_floris_curl = fi_curl.get_cross_plane(x_loc)
    cut_plane_floris_curl_vel_def = cp.subtract(cut_plane_floris_curl,
                                                 cut_plane_floris_curl_base)
    # hor_plane_floris_curl = fi_curl.get_hor_plane()

    # Project FLORIS cut plane onto grid of SOWFA cut plane
    cut_plane_floris_curl_project = cp.project_onto(
                                        cut_plane_floris_curl_vel_def,
                                        cut_plane_sowfa)

    # fig, axarr = plt.subplots(1,3,figsize=(18, 4))
    fig = plt.figure(figsize=(18, 4))
    font_size = 14

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
        nrows_ncols=(1,3),
        axes_pad=0.35,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )
    axarr = [ax for ax in grid]

    # ax = axarr[0,0]
    ax = axarr[0]
    wfct.visualization.visualize_cut_plane(
        cp.subtract(cut_plane_sowfa, cut_plane_sowfa_base),
        ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    turb_rotor = plt.Circle((y_center, 90), D/2,lw=0.75, color='k', fill=False)
    ax.add_artist(turb_rotor)
    ax.set_title('SOWFA Velocity Deficit', fontsize=font_size)
    ax.tick_params(labelsize=font_size)
    # Plot streamlines
    vis.visualize_quiver(
        cp.subtract(cut_plane_sowfa, cut_plane_sowfa_base),
        ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    ax.set_xlim([350.01,649.01])
    ax.set_ylim(10,170)
    ax.set_ylabel('Height (m)', fontsize=font_size)
    ax.set_xlabel('Spanwise Distance (m)', fontsize=font_size)

    # ax = axarr[1]
    # vis.visualize_cut_plane(
    #     cp.subtract(cut_plane_floris_gauss_project, cut_plane_sowfa_base),
    #     ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    # ax.set_title('FLORIS', fontsize=font_size)
    # ax.tick_params(labelsize=font_size)
    # ax = axarr[0,1]
    ax = axarr[1]
    vis.visualize_cut_plane(
        cut_plane_floris_gauss_project,
        ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    turb_rotor = plt.Circle((y_center, 90), D/2,lw=0.75, color='k', fill=False)
    ax.add_artist(turb_rotor)
    ax.set_title('Gauss Velocity Deficit', fontsize=font_size)
    ax.tick_params(labelsize=font_size)
    ax.set_xlim([350.01,649.01])
    ax.set_ylim(10,170)
    ax.set_xlabel('Spanwise Distance (m)', fontsize=font_size)

    # ax = axarr[2]
    # vis.visualize_cut_plane(
    #     cp.subtract(
    #         cp.subtract(
    #             cut_plane_sowfa, cut_plane_sowfa_base
    #         ),
    #         cp.subtract(
    #             cut_plane_floris_gauss_project, cut_plane_sowfa_base
    #         )
    #     ),
    #     ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    # ax.set_title('FLORIS', fontsize=font_size)
    # ax.tick_params(labelsize=font_size)
    # ax = axarr[0,2]
    ax = axarr[2]
    im = vis.visualize_cut_plane(
        cut_plane_floris_curl_project,
        ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    turb_rotor = plt.Circle((y_center, 90), D/2,lw=0.75, color='k', fill=False)
    ax.add_artist(turb_rotor)
    ax.set_title('Curl Velocity Deficit', fontsize=font_size)
    ax.tick_params(labelsize=font_size)
    ax.set_xlabel('Spanwise Distance (m)', fontsize=font_size)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.set_ticks(np.arange(minspeed, maxspeed, 1.0))
    # cbar.set_label('Wind speed (m/s)', fontsize=font_size)
    # cbar.ax.tick_params(labelsize=font_size)
    # Plot streamlines
    vis.visualize_quiver(
        cut_plane_floris_curl_project,
        ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    ax.set_xlim([350.01,649.01])
    ax.set_ylim(10,170)

    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    cbar = fig.colorbar(im, cax=ax.cax)
    cbar.set_ticks(np.arange(minspeed, maxspeed, 1.0))
    cbar.set_label('Velocity Deficit (m/s)', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    # ax = axarr[1,1]
    # vis.visualize_cut_plane(
    #     cp.subtract(
    #         cp.subtract(
    #             cut_plane_sowfa, cut_plane_sowfa_base
    #         ),
    #         cut_plane_floris_gauss_project
    #     ),
    #     ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    # turb_rotor = plt.Circle((y_center, 90), D/2,lw=0.625, color='k', fill=False)
    # ax.add_artist(turb_rotor)
    # ax.set_title('SOWFA - Gauss', fontsize=font_size)
    # ax.tick_params(labelsize=font_size)

    # ax = axarr[1,2]
    # vis.visualize_cut_plane(
    #     cp.subtract(
    #         cp.subtract(
    #             cut_plane_sowfa, cut_plane_sowfa_base
    #         ),
    #         cut_plane_floris_curl_project
    #     ),
    #     ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)
    # turb_rotor = plt.Circle((y_center, 90), D/2,lw=0.625, color='k', fill=False)
    # ax.add_artist(turb_rotor)
    # ax.set_title('SOWFA - Curl', fontsize=font_size)
    # ax.tick_params(labelsize=font_size)

    # ax = axarr[2,0]
    # vis.visualize_cut_plane(
    #     hor_plane_sowfa,
    #     ax=ax,
    #     minSpeed=3.0,
    #     maxSpeed=9.0
    # )

    # ax = axarr[2,1]
    # vis.visualize_cut_plane(
    #     hor_plane_floris_gauss,
    #     ax=ax,
    #     minSpeed=3.0,
    #     maxSpeed=9.0
    # )

    # ax = axarr[2,2]
    # vis.visualize_cut_plane(
    #     hor_plane_floris_curl,
    #     ax=ax,
    #     minSpeed=3.0,
    #     maxSpeed=9.0
    # )
    
plt.savefig('3turb_cross_6D_T0.png', bbox_inches='tight')
plt.show()