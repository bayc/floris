import numpy as np


flowfield = []
flow_field_u_initial = []
turbine_RD = 0.0
no_wake = False
Vec3 = []
speed = 0.0

flowfield._grid_wind_speed = np.full(np.shape(flowfield.grid_layout[0]), speed[0])
flowfield.u_initial = (
    flowfield.wind_map.grid_wind_speed
    * (flowfield.z / flowfield.specified_wind_height) ** flowfield.wind_shear
)


# calculate_wake

# reinitialize the turbines
for i, turbine in enumerate(flowfield.turbine_map.turbines):
    turbine.current_turbulence_intensity = flowfield.wind_map.turbine_turbulence_intensity[
        i
    ]
    turbine.reset_velocities()

# define the center of rotation with reference to 270 deg as center of
# flow field
x0 = np.mean([np.min(flowfield.x), np.max(flowfield.x)])
y0 = np.mean([np.min(flowfield.y), np.max(flowfield.y)])
center_of_rotation = Vec3(x0, y0, 0)

# Rotate the turbines such that they are now in the frame of reference
# of the wind direction simplifying computing the wakes and wake overlap
rotated_map = flowfield.turbine_map.rotated(
    flowfield.wind_map.turbine_wind_direction, center_of_rotation
)


x_coord = [coord.x1 for coord in rotated_map.coords]
y_coord = [coord.x2 for coord in rotated_map.coords]
# re-setup the grid for the curl model
xmin = np.min(x_coord) - 2 * turbine_RD
xmax = np.max(x_coord) + 10 * turbine_RD
ymin = np.min(y_coord) - 2 * turbine_RD
ymax = np.max(y_coord) + 2 * turbine_RD
zmin = 0.1
zmax = 6 * flowfield.specified_wind_height

resolution = flowfield.wake.velocity_model.model_grid_resolution

x = np.linspace(xmin, xmax, int(resolution.x1))
y = np.linspace(ymin, ymax, int(resolution.x2))
z = np.linspace(zmin, zmax, int(resolution.x3))
mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z, indexing="ij")

rotated_x, rotated_y, rotated_z = flowfield._rotated_grid(0.0, center_of_rotation)

# sort the turbine map
sorted_map = rotated_map.sorted_in_x_as_list()

# calculate the velocity deficit and wake deflection on the mesh
u_wake = np.zeros(np.shape(flowfield.u))

rx = np.array([coord.x1prime for coord in flowfield.turbine_map.coords])
ry = np.array([coord.x2prime for coord in flowfield.turbine_map.coords])

for coord, turbine in sorted_map:
    # update the turbine based on the velocity at its hub
    turbine.update_velocities(u_wake, coord, flowfield, rotated_x, rotated_y, rotated_z)

    # get the velocity deficit accounting for the deflection
    (turb_u_wake) = flowfield._compute_turbine_velocity_deficit(
        rotated_x, rotated_y, rotated_z, turbine, coord, flowfield
    )

    # combine this turbine's wake into the full wake field
    u_wake = flowfield.wake.combination_function(u_wake, turb_u_wake)

# apply the velocity deficit field to the freestream
if not no_wake:
    flowfield.u = flowfield.u_initial - u_wake
