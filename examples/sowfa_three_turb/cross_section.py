# Code to tune the guass model to 1-turbine case
# import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
from floris.utilities import Vec3
# from functions import get_power_at_coord, sweep_power, sweep_long

import numpy as np
import os
import copy
import re

# ## Parameters
# data_folder = '/Users/cbay/Box Sync/sowfa_library/full_runs/three_turbine_sims/good_efficiency_peregrine_runs'
data_folder = '/mnt/c/Users/cbay/Box Sync/sowfa_library/full_runs/three_turbine_eagle/eagle_runs/'
floris_input_gauss = '../example_input.json'
floris_input_curl = '../example_input.json'
fig_folder = 'figures/three_turb'
show_horizontal = True

## Plotting parameters
# Long wise
# x_points_in_D = np.arange(-.25,10,.5)
x_points_in_D = np.arange(3,10,1.)

# cross wise
y_points_in_D = np.arange(-2,2,.05)
# y_points_in_D = np.arange(-2,2,.5)
x_location_downstream_in_D = 7

## Make the figure folder
if not os.path.exists(fig_folder):
    os.mkdir(fig_folder)

# Make a list of case name options
# case_name_list = os.listdir(data_folder)
# case_name_list = [c for c in os.listdir(data_folder) if '_08mps' in c and 'GE' not in c and 'A_' in c and 'high' in c]
# print(case_name_list)

# Read in the high turbulence one turbine case
for case_name in ['C_0019_0_10_5_-1_-1']:

    # Select case based on number
    # case_name = case_name_list[case_number]
    fig_root = os.path.join(fig_folder,case_name[9:])

    # no_turbine_case_folder = '/Users/cbay/Box Sync/sowfa_library/full_runs/three_turbine_sims/good_efficiency_peregrine_runs/C_no_turbine_nigh'
    no_turbine_case_folder = data_folder + '../../near_wake/hi_no_turbine'

    m_ind = [m.start() for m in re.finditer('_', case_name)]
    yaw_angle_T1 = case_name[m_ind[1]+1:m_ind[2]]
    yaw_angle_T2 = case_name[m_ind[2]+1:m_ind[3]]
    turb_spacing = case_name[m_ind[3]+1:m_ind[4]]
    rotor_offset_T1 = case_name[m_ind[4]+1:m_ind[5]]
    rotor_offset_T2 = case_name[m_ind[5]+1:]


    if 'y00' in case_name:
        yaw_angle = 0.0
    elif 'y10' in case_name: 
        yaw_angle = 10.0
    else:
        yaw_angle = 20.0


    # ## Import SOWFA x 3
    case_folder_c = os.path.join(data_folder,case_name)

    sowfa_case_c = wfct.sowfa_utilities.SowfaInterface(
        case_folder_c, flow_data_sub_path='array_mean/array.mean0D_UAvg.vtk')

    sowfa_flow_data_c = sowfa_case_c.flow_data

    # Work out position of hypothetical 3rd turbine
    # y_center = sowfa_case_c.layout_y[0]
    # turbine_x = sowfa_case_c.layout_x[1] + (sowfa_case_c.layout_x[1]-sowfa_case_c.layout_x[0])
    # turbine_x = sowfa_case_c.layout_x[0] + 2.*126 

    # ## Get the no-turbine sowfa flow
    sowfa_case_no_turbine = wfct.sowfa_utilities.SowfaInterface(
        no_turbine_case_folder,flow_data_sub_path='array_mean/array.mean0D_UAvg.vtk')
    sowfa_flow_data_no_turbine = sowfa_case_no_turbine.flow_data

    # Set up as A,B,C no turbine flow fields
    sowfa_flow_data_no_turbine_c = copy.deepcopy(sowfa_flow_data_no_turbine)

    sowfa_flow_data_no_turbine_a.y = sowfa_flow_data_no_turbine_a.y - 150
    sowfa_flow_data_no_turbine_b.y = sowfa_flow_data_no_turbine_b.y + 150

    # Setup FLORIS
    floris_interface_gauss = wfct.floris_utilities.FlorisInterface(floris_input_gauss)

    # Also curl
    floris_interface_curl = wfct.floris_utilities.FlorisInterface(floris_input_curl)

    # Set up turbines
    floris_interface_gauss.floris.farm.set_turbine_locations(sowfa_case_c.layout_x, sowfa_case_c.layout_y, calculate_wake=False)
    floris_interface_curl.floris.farm.set_turbine_locations(sowfa_case_c.layout_x, sowfa_case_c.layout_y, calculate_wake=False)

    # Set up yaw
    floris_interface_gauss.floris.farm.set_yaw_angles(sowfa_case_c.yaw_angles, calculate_wake=False)
    floris_interface_curl.floris.farm.set_yaw_angles(sowfa_case_c.yaw_angles, calculate_wake=False)

    # Run FLORIS
    floris_interface_gauss.run_floris()
    floris_interface_curl.run_floris()

    # Print FLORIS power
    print('FLORIS Gauss Turbine Powers: ', floris_interface_gauss.get_turbine_power())
    print('FLORIS Gauss Power: ', floris_interface_gauss.get_farm_power())
    print('FLORIS Curl Turbine Powers: ', floris_interface_curl.get_turbine_power())
    print('FLORIS Curl Power: ', floris_interface_curl.get_farm_power())

    # Set FLORIS resolution
    xmin, xmax, ymin, ymax, zmin, zmax = floris_interface_gauss.floris.farm.flow_data.domain_bounds
    resolution = Vec3(
        1 + (xmax - xmin) / 10,
        1 + (ymax - ymin) / 10,
        1 + (zmax - zmin) / 10
    )
    floris_flow_data_gauss = floris_interface_gauss.get_flow_data(resolution=resolution)
    floris_flow_data_ctx2 = floris_interface_ctx2.get_flow_data(resolution=resolution)

    # Curl needs to match
    resolution = floris_interface_curl.floris.farm.flow_data.wake.velocity_model.model_grid_resolution
    floris_flow_data_curl =  floris_interface_curl.get_flow_data(resolution=resolution)


    floris_domain_limits = [[np.min(floris_flow_data_curl.x), np.max(floris_flow_data_curl.x)],
                        [np.min(floris_flow_data_curl.y), np.max(floris_flow_data_curl.y)], 
                        [np.min(floris_flow_data_curl.z), np.max(floris_flow_data_curl.z)]]

    # COMPARE THE HORIZONTALS
    if show_horizontal:
        # fig, axarr = plt.subplots(4,1,figsize=(10,12))
        # for label,flow_data,ax in zip(['SOWFA','FLORIS GAUSS gauss','FLORIS GAUSS CTX2','FLORIS CURL'], 
        #                             [sowfa_flow_data_a,floris_flow_data_gauss,floris_flow_data_ctx2,floris_flow_data_curl],
        #                                     axarr):
        fig, axarr = plt.subplots(3,1,figsize=(10,12))
        for label,flow_data,ax in zip(['SOWFA','FLORIS GAUSS','FLORIS CURL'], 
                                    [sowfa_flow_data_a,floris_flow_data_gauss,floris_flow_data_curl],
                                            axarr):
            hor_plane = wfct.cut_plane.HorPlane(flow_data,90)
            wfct.visualization.visualize_cut_plane(hor_plane,ax=ax,maxSpeed=8.,minSpeed=5.)
            vis.plot_turbines(ax, floris_interface_gauss.floris.farm.layout_x, floris_interface_gauss.floris.farm.layout_y, floris_interface_gauss.get_yaw_angles(), floris_interface_gauss.floris.farm.turbine_map.turbines[0].rotor_diameter)
            ax.set_title(label)
            ax.grid(True)
            ax.set_xlim(floris_domain_limits[0][0], floris_domain_limits[0][1])
            ax.set_ylim(floris_domain_limits[1][0], floris_domain_limits[1][1])
        figname = os.path.join(fig_root + 'horizontal.png')
        fig.savefig(figname,dpi=300)


    # # COMPARE THE SOWFA HORIZONTALS
    # if show_horizontal:
    #     fig, axarr = plt.subplots(3,2,figsize=(10,12),sharex=True, sharey=True)
    #     for ax, flow_data in zip(axarr.flatten(),[sowfa_flow_data_a,sowfa_flow_data_no_turbine_a,sowfa_flow_data_b,sowfa_flow_data_no_turbine_b,sowfa_flow_data_c,sowfa_flow_data_no_turbine_c]):
    #         hor_plane = wfct.cut_plane.HorPlane(flow_data,90)
    #         wfct.visualization.visualize_cut_plane(hor_plane,ax=ax,maxSpeed=8.,minSpeed=5.)
    #         # ax.set_title(label)
    #         ax.grid(True)
    #         # ax.set_xlim(floris_domain_limits[0][0], floris_domain_limits[0][1])
    #         ax.set_ylim([0,1000])
    #         figname = os.path.join(fig_root + '_sowfa_horizontal.png')
    #     ax.set_ylim([0,1000])
    #     fig.savefig(figname,dpi=300)



    
    D = sowfa_case_a.D
    # Get the FLORIS free power
    floris_cross = wfct.cut_plane.CrossPlane(floris_flow_data_gauss,1)
    floris_free_power = get_power_at_coord(floris_cross,y_center)
    D = sowfa_case_a.D
    # # ## Long-wise profile comparison==============================================================================
    # D = sowfa_case_a.D
    # x_points = x_points_in_D * D + turbine_x
    # x_plot_points = (x_points - turbine_x) / D

    # # Get the longitudinal points
    # sowfa_power_a = sweep_long(sowfa_flow_data_a,x_points,y_center)
    # sowfa_power_b = sweep_long(sowfa_flow_data_b,x_points,y_center)
    # sowfa_power_c = sweep_long(sowfa_flow_data_c,x_points,y_center)
    # sowfa_power_no_turbine_a = sweep_long(sowfa_flow_data_no_turbine_a,x_points,y_center)
    # sowfa_power_no_turbine_b = sweep_long(sowfa_flow_data_no_turbine_b,x_points,y_center)
    # sowfa_power_no_turbine_c = sweep_long(sowfa_flow_data_no_turbine_c,x_points,y_center)
    # floris_power = sweep_long(floris_flow_data,x_points,y_center)
    # floris_power_curl = sweep_long(floris_flow_data_curl,x_points,y_center)
  


    # # Get the FLORIS free power
    # floris_cross = wfct.cut_plane.CrossPlane(floris_flow_data,1)
    # floris_free_power = get_power_at_coord(floris_cross,y_center)

    # fig, ax = plt.subplots(figsize=(8,5))
    # sowfa_min = np.amin([sowfa_power_a/sowfa_power_no_turbine_a,sowfa_power_b/sowfa_power_no_turbine_b,sowfa_power_c/sowfa_power_no_turbine_c],0)
    # sowfa_max = np.amax([sowfa_power_a/sowfa_power_no_turbine_a,sowfa_power_b/sowfa_power_no_turbine_b,sowfa_power_c/sowfa_power_no_turbine_c],0)
    # ax.fill_between(x_plot_points,100*sowfa_min,100*sowfa_max,color='k',alpha=0.6,label='SOWFA')
    # ax.plot(x_plot_points,100 * floris_power/floris_free_power,color='r',ls='-.',lw=3,label='FLORIS GAUSS')
    # ax.plot(x_plot_points,100 * floris_power_curl/floris_free_power,color='b',ls='--',lw=3,label='FLORIS CURL')
    # ax.legend()
    # ax.grid(True)
    # ax.set_xlabel('Distance Downstream (D)')
    # ax.set_ylabel('Percent of Freestream Power (%)')
    # figname = os.path.join(fig_root + 'longwise.png')
    # fig.savefig(figname,dpi=300)

    # # ## Cross flow comparison at 7D===============================================================================
    x_location_downstream = turbine_x


    # # Loop through and get the power
    sowfa_cross_a = wfct.cut_plane.CrossPlane(sowfa_flow_data_a,x_location_downstream)
    sowfa_cross_b = wfct.cut_plane.CrossPlane(sowfa_flow_data_b,x_location_downstream)
    sowfa_cross_c = wfct.cut_plane.CrossPlane(sowfa_flow_data_c,x_location_downstream)
    sowfa_cross_c3D = wfct.cut_plane.CrossPlane(sowfa_flow_data_c,sowfa_case_c.layout_x[0] + 3.*126)
    sowfa_cross_c4D = wfct.cut_plane.CrossPlane(sowfa_flow_data_c,sowfa_case_c.layout_x[0] + 4.*126)
    sowfa_cross_c5D = wfct.cut_plane.CrossPlane(sowfa_flow_data_c,sowfa_case_c.layout_x[0] + 5.*126)
    sowfa_cross_no_turbine_a = wfct.cut_plane.CrossPlane(sowfa_flow_data_no_turbine_a,x_location_downstream)
    sowfa_cross_no_turbine_b = wfct.cut_plane.CrossPlane(sowfa_flow_data_no_turbine_b,x_location_downstream)
    sowfa_cross_no_turbine_c = wfct.cut_plane.CrossPlane(sowfa_flow_data_no_turbine_c,x_location_downstream)
    floris_cross_gauss = wfct.cut_plane.CrossPlane(floris_flow_data_gauss,x_location_downstream)
    floris_cross_ctx2 = wfct.cut_plane.CrossPlane(floris_flow_data_ctx2,x_location_downstream)
    floris_cross_curl = wfct.cut_plane.CrossPlane(floris_flow_data_curl,x_location_downstream)
    floris_cross_curl3D = wfct.cut_plane.CrossPlane(floris_flow_data_curl,sowfa_case_c.layout_x[0] + 3.*126)
    floris_cross_curl4D = wfct.cut_plane.CrossPlane(floris_flow_data_curl,sowfa_case_c.layout_x[0] + 4.*126)
    floris_cross_curl5D = wfct.cut_plane.CrossPlane(floris_flow_data_curl,sowfa_case_c.layout_x[0] + 5.*126)
    floris_empty = wfct.cut_plane.CrossPlane(floris_flow_data_gauss,1)
    floris_empty_curl = wfct.cut_plane.CrossPlane(floris_flow_data_curl,1)
    floris_empty_curl3D = wfct.cut_plane.CrossPlane(floris_flow_data_curl,1)
    floris_empty_curl4D = wfct.cut_plane.CrossPlane(floris_flow_data_curl,1)
    floris_empty_curl5D = wfct.cut_plane.CrossPlane(floris_flow_data_curl,1)
    sowfa_empty = wfct.cut_plane.CrossPlane(sowfa_flow_data_c,1)
    sowfa_empty3D = wfct.cut_plane.CrossPlane(sowfa_flow_data_c,1)
    sowfa_empty4D = wfct.cut_plane.CrossPlane(sowfa_flow_data_c,1)
    sowfa_empty5D = wfct.cut_plane.CrossPlane(sowfa_flow_data_c,1)

    # Get the power sweeps
    y_points, sowfa_power_a = sweep_power(sowfa_cross_a,y_center)
    y_points, sowfa_power_b = sweep_power(sowfa_cross_b,y_center)
    y_points, sowfa_power_c = sweep_power(sowfa_cross_c,y_center)
    y_points, sowfa_power_no_turbine_a = sweep_power(sowfa_cross_no_turbine_a,y_center)
    y_points, sowfa_power_no_turbine_b = sweep_power(sowfa_cross_no_turbine_b,y_center)
    y_points, sowfa_power_no_turbine_c = sweep_power(sowfa_cross_no_turbine_c,y_center)
    y_points, floris_power_gauss = sweep_power(floris_cross_gauss,y_center)
    y_points, floris_power_ctx2 = sweep_power(floris_cross_ctx2,y_center)
    y_points, floris_power_curl = sweep_power(floris_cross_curl,y_center)

    # Show these cross streams

    fig, axarr = plt.subplots(1,3,figsize=(12,4))
    for label,cp,cp_base,ax in zip(['SOWFA','FLORIS GAUSS','FLORIS CURL'], 
                                [sowfa_cross_c,floris_cross_gauss,floris_cross_curl],[sowfa_empty,floris_empty,floris_empty_curl],
                                        axarr):
        print(label)
        cp_plot = copy.deepcopy(cp)
        cp_plot.u_mesh = cp_plot.u_mesh - cp_base.u_mesh
        wfct.visualization.visualize_cut_plane(cp_plot,ax=ax,maxSpeed=0.,minSpeed=-2.)
        # vis.plot_turbines(ax, floris_interface_gauss.floris.farm.layout_x, floris_interface_gauss.floris.farm.layout_y, np.degrees(floris_interface_gauss.get_yaw_angles()), floris_interface_gauss.floris.farm.turbine_map.turbines[0].rotor_diameter)
        ax.set_title(label)
        ax.grid(True)
        # ax.set_xlim(floris_domain_limits[1][0], floris_domain_limits[1][1])
        ax.set_xlim([350,650])
        ax.set_ylim(20,160)
        circle2 = plt.Circle((y_center, 90), D/2,lw=2, color='k', fill=False)
        ax.add_artist(circle2)
    figname = os.path.join(fig_root + 'cross_viz.png')
    fig.savefig(figname,dpi=300, bbox_inches = 'tight')
    # plt.show()

    fig, axarr = plt.subplots(1,8,figsize=(12,4))
    for label,cp,cp_base,ax in zip(['Curl 2D','Curl 3D','Curl 4D','Curl 5D', 'SOWFA 2D', 'SOWFA 3D', 'SOWFA 4D', 'SOWFA 5D'], 
                                [floris_cross_curl,floris_cross_curl3D,floris_cross_curl4D,floris_cross_curl5D, sowfa_cross_c, sowfa_cross_c3D, sowfa_cross_c4D, sowfa_cross_c5D],
                                [floris_empty_curl,floris_empty_curl3D,floris_empty_curl4D,floris_empty_curl5D, sowfa_empty, sowfa_empty3D, sowfa_empty4D, sowfa_empty5D], axarr):
        cp_plot = copy.deepcopy(cp)
        cp_plot.u_mesh = cp_plot.u_mesh - cp_base.u_mesh
        wfct.visualization.visualize_cut_plane(cp_plot,ax=ax,maxSpeed=0.,minSpeed=-2.)
        # vis.plot_turbines(ax, floris_interface_gauss.floris.farm.layout_x, floris_interface_gauss.floris.farm.layout_y, np.degrees(floris_interface_gauss.get_yaw_angles()), floris_interface_gauss.floris.farm.turbine_map.turbines[0].rotor_diameter)
        ax.set_title(label)
        ax.grid(True)
        # ax.set_xlim(floris_domain_limits[1][0], floris_domain_limits[1][1])
        ax.set_xlim([350,650])
        ax.set_ylim(20,160)
        circle2 = plt.Circle((y_center, 90), D/2,lw=2, color='k', fill=False)
        ax.add_artist(circle2)
    fig.savefig(figname,dpi=300, bbox_inches = 'tight')

    y_points_plot = (y_points - y_center) / D


    fig, ax = plt.subplots(figsize=(8,5))
    sowfa_min = np.amin([sowfa_power_a/sowfa_power_no_turbine_a,sowfa_power_b/sowfa_power_no_turbine_b,sowfa_power_c/sowfa_power_no_turbine_c],0)
    sowfa_max = np.amax([sowfa_power_a/sowfa_power_no_turbine_a,sowfa_power_b/sowfa_power_no_turbine_b,sowfa_power_c/sowfa_power_no_turbine_c],0)
    ax.fill_between(y_points_plot,100*sowfa_min,100*sowfa_max,color='k',alpha=0.6,label='SOWFA')
    ax.plot(y_points_plot,100 * floris_power_gauss/floris_free_power,color='r',ls='-.',lw=2,label='FLORIS GAUSS gauss')
    ax.plot(y_points_plot,100 * floris_power_ctx2/floris_free_power,color='m',ls='-.',lw=2,label='FLORIS GAUSS CTX2')
    ax.plot(y_points_plot,100 * floris_power_curl/floris_free_power,color='b',ls='--',lw=2,label='FLORIS CURL')

    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Distance Cross Stream (D)')
    ax.set_ylabel('Percent of Freestream Power (%)')
    ax.axvline(0,color='k')
    ax.set_ylim([25,100])
    figname = os.path.join(fig_root + 'crosswise.png')
    # figname = os.path.join(fig_root + 'crosswise.png')
    fig.savefig(figname,dpi=300)

    if 'base' in case_name: # this is baseline, save some stuff
        sowfa_base = sowfa_power_c
        floris_power_gauss_base = floris_power_gauss
        floris_power_ctx2_base = floris_power_ctx2
        floris_power_curl_base = floris_power_curl

    fig, ax = plt.subplots(figsize=(8,5))
    
    ax.plot(y_points_plot,100*(sowfa_power_c - sowfa_base)/sowfa_base,color='k',alpha=0.6,label='SOWFA')
    ax.plot(y_points_plot,100 * (floris_power_gauss - floris_power_gauss_base)/floris_power_gauss_base,color='r',ls='-.',lw=2,label='FLORIS GAUSS')
    # ax.plot(y_points_plot,100 * (floris_power_ctx2 - floris_power_ctx2_base)/floris_power_ctx2_base,color='m',ls='-.',lw=2,label='FLORIS GAUSS CTX2')
    ax.plot(y_points_plot,100 * (floris_power_curl - floris_power_curl_base)/floris_power_curl_base,color='b',ls='--',lw=2,label='FLORIS CURL')

    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Distance Cross Stream (D)')
    ax.set_ylabel('Relative Difference of Baseline Power (%)')
    ax.axvline(0,color='k')
    figname = os.path.join(fig_root + 'crosswise_gain.png')
    # ax.set_ylim([25,100])
    fig.savefig(figname,dpi=300)

plt.show()