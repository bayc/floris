# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata

from floris.tools import FlorisInterface


"""
"""

fi = FlorisInterface("inputs/gch.yaml")

fi.reinitialize(layout=([0.0], [0.0]))

x_resolution = 10
y_resolution = 50

# fi.calculate_wake()
horizontal_plane = fi.calculate_horizontal_plane(
    x_resolution=x_resolution,
    y_resolution=y_resolution,
    height=90.0
)

print(np.shape(horizontal_plane.df.x1))
print(np.shape(horizontal_plane.df.u))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# X, Y = np.meshgrid(fi.floris.grid.x.flatten(), fi.floris.grid.y.flatten())

# # X = fi.floris.grid.x
# # Y = fi.floris.grid.y
# Z = np.reshape(fi.floris.flow_field.u, (np.shape(X)[0], np.shape(X)[1]))

# print(np.shape(X))
# print(np.shape(Y))
# print(np.shape(Z))

x = horizontal_plane.df.x1.values
y = horizontal_plane.df.x2.values

mask = np.array(x > -500.0)

X, Y = np.meshgrid(x, y)

z = horizontal_plane.df.u.values / 8.0

Z = griddata((x, y), z, (X,Y), method='linear')

xx = x*mask
yy = y*mask
zz = z*mask

# X = np.reshape(x, (x_resolution, y_resolution))
# Y = np.reshape(y, (x_resolution, y_resolution))
# Z = np.reshape(z, (x_resolution, y_resolution))

wire = ax.plot_wireframe(X, Y, Z, rstride=0, cstride=4)
# surf = ax.plot_surface(X, Y, Z, alpha=0.2)
surf = ax.plot_trisurf(xx[xx!=0], yy[yy!=0], zz[zz!=0], alpha=0.2, color='k')
vel = ax.tricontourf(xx[xx!=0], yy[yy!=0], zz[zz!=0], zdir='z', offset=0.15, cmap=cm.coolwarm)

# ax.set_ylim(-200.0, 200.0)
ax.set_zlim(0.15, 1.0)

print(np.shape(X))
print(np.shape(Y))
print(np.shape(Z))

plt.show()
