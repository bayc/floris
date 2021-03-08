import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis as na


def cosd(angle):
    return np.cos(np.radians(angle))


def sind(angle):
    return np.sin(np.radians(angle))


# ///// #
# SETUP #
# ///// #

# Turbine parameters
turbine_diameter = 126.0
turbine_radius = turbine_diameter / 2.0
turbine_hub_height = 90.0
turbine_aI = 1 / 3

x_coord = [0.0]
y_coord = [0.0]

# Wind parameters
ws = np.array([8.0, 10.0])
wd = 270.0

specified_wind_height = 90.0
wind_shear = 0.12

# Flow field bounds
xmin = np.min(x_coord) - 10 * turbine_diameter
xmax = np.max(x_coord) + 10 * turbine_diameter
ymin = np.min(y_coord) - 10 * turbine_diameter
ymax = np.max(y_coord) + 10 * turbine_diameter
zmin = 0.1
zmax = 6 * specified_wind_height

# Flow field resolutions
resolution_x1 = 100
resolution_x2 = 200
resolution_x3 = 7

x = np.linspace(xmin, xmax, int(resolution_x1))
y = np.linspace(ymin, ymax, int(resolution_x2))
z = np.linspace(zmin, zmax, int(resolution_x3))
# z = np.ones_like(x) * 90.0

# Flow field grid points
mesh_x, mesh_y, mesh_z = np.meshgrid(x, y, z, indexing="ij")

# Initialize field values
# flow_field_u_initial = (ws * (mesh_z / specified_wind_height) ** wind_shear)
print("shape of ws: ", np.shape(ws))
print("shape of mesh_z: ", np.shape((mesh_z / specified_wind_height) ** wind_shear))
flow_field_u_initial = ws * (mesh_z[na, :] / specified_wind_height) ** wind_shear
print(np.shape(flow_field_u_initial))
# lkj
u_wake = np.zeros(np.shape(flow_field_u_initial))
deflection_field = np.zeros(np.shape(flow_field_u_initial))

# ///////////////// #
# JENSEN WAKE MODEL #
# ///////////////// #

we = 0.05

m = we
x = mesh_x - x_coord
b = turbine_radius

boundary_line = m * x + b

y_upper = boundary_line + y_coord + deflection_field
y_lower = -1 * boundary_line + y_coord + deflection_field

z_upper = boundary_line + turbine_hub_height
z_lower = -1 * boundary_line + turbine_hub_height

# Rotate something
x_center_of_rotation = (xmin + xmax) / 2
y_center_of_rotation = (ymin + ymax) / 2
angle = ((wd - 270) % 360 + 360) % 360

x_offset = mesh_x - x_center_of_rotation
y_offset = mesh_y - y_center_of_rotation
x_coord_offset = x_coord - x_center_of_rotation
y_coord_offset = y_coord - y_center_of_rotation

mesh_x_rotated = x_offset * cosd(angle) - y_offset * sind(angle) + x_center_of_rotation
mesh_y_rotated = x_offset * sind(angle) + y_offset * cosd(angle) + y_center_of_rotation

x_coord_rotated = (
    x_coord_offset * cosd(angle) - y_coord_offset * sind(angle) + x_center_of_rotation
)
y_coord_rotated = (
    x_coord_offset * sind(angle) + y_coord_offset * cosd(angle) + y_center_of_rotation
)

# Calculate the wake velocity deficit ratios
# c = (turbine_diameter / (2 * we * (mesh_x - x_coord) + turbine_diameter)) ** 2
c = (
    turbine_diameter / (2 * we * (mesh_x_rotated - x_coord_rotated) + turbine_diameter)
) ** 2

# filter points upstream and beyond the upper and lower bounds of the wake
# c[mesh_x - x_coord < 0] = 0
# c[mesh_y > y_upper] = 0
# c[mesh_y < y_lower] = 0
# c[mesh_z > z_upper] = 0
# c[mesh_z < z_lower] = 0
c[mesh_x_rotated - x_coord_rotated < 0] = 0

m = we
x = mesh_x_rotated - x_coord_rotated
b = turbine_radius

boundary_line = m * x + b

y_upper = boundary_line + y_coord_rotated + deflection_field
y_lower = -1 * boundary_line + y_coord_rotated + deflection_field
c[mesh_y_rotated > y_upper] = 0
c[mesh_y_rotated < y_lower] = 0
c[mesh_z > z_upper] = 0
c[mesh_z < z_lower] = 0

# Calculate the wake velocity deficits
u_wake = 2 * turbine_aI * c * flow_field_u_initial

# Apply the velocity deficit field to the freestream
flow_field_u = flow_field_u_initial - u_wake

# ///////////// #
# PLOT THE WAKE #
# ///////////// #

minSpeed = 2.0
maxSpeed = 8.0
cmap = "coolwarm"

fig, ax = plt.subplots()

# Determine mask for z-slice/horizontal slice
mask = (mesh_z > 90.0) & (mesh_z < 91.0)

x_grid_length = len(np.unique(mesh_x[mask]))
y_grid_length = len(np.unique(mesh_y[mask]))
x_plot = np.reshape(mesh_x[mask], (x_grid_length, y_grid_length))
y_plot = np.reshape(mesh_y[mask], (x_grid_length, y_grid_length))
z_plot = np.reshape(flow_field_u[mask], (x_grid_length, y_grid_length))

im = ax.pcolormesh(
    x_plot, y_plot, z_plot, cmap=cmap, vmin=minSpeed, vmax=maxSpeed, shading="nearest"
)

ax.contour(x_plot, y_plot, z_plot)

# Make equal axis
ax.set_aspect("equal")

plt.show()
