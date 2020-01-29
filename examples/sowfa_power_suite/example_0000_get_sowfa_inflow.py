#
# Copyright 2019 NREL
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

# See read the https://floris.readthedocs.io for documentation

import floris.tools as wfct
import floris.tools.visualization as vis
import floris.tools.cut_plane as cp
import matplotlib.pyplot as plt
import numpy as np
import os

# # Define a minspeed and maxspeed to use across visualiztions
minspeed = 4.0
maxspeed = 8.5

# Load the SOWFA case in
sowfa_root = '/Users/pfleming/Box Sync/sowfa_library/full_runs/three_turbine_sims/peregrine_runs'
high_ti_inflow = 'C_no_turbine_nigh'

si = wfct.sowfa_utilities.SowfaInterface(os.path.join(sowfa_root,high_ti_inflow))

# Plot the SOWFA flow and turbines using the input information
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sowfa_flow_data = si.flow_data
hor_plane = si.get_hor_plane(90)
wfct.visualization.visualize_cut_plane(
    hor_plane, ax=ax, minSpeed=minspeed, maxSpeed=maxspeed)

ax.set_title('SOWFA')
ax.set_ylabel('y location [m]')


plt.show()
