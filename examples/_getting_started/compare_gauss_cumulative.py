# Copyright 2020 NREL

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

import floris.tools as wfct


# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi_gauss_cumulative = wfct.floris_interface.FlorisInterface("../example_input.json")
fi_gauss_legacy = wfct.floris_interface.FlorisInterface("../example_input.json")

fi_gauss_cumulative.floris.farm.set_wake_model("gauss_cumulative")
fi_gauss_cumulative.set_gch(enable=False)
fi_gauss_cumulative.floris.farm.flow_field.solver = "gauss_cumulative"
fi_gauss_cumulative.reinitialize_flow_field(
    layout_array=([0.0, 6 * 126.0, 12 * 126.0, 18 * 126.0], [0.0, 0.0, 0.0, 0.0])
)
# Calculate wake
fi_gauss_cumulative.calculate_wake()

fi_gauss_legacy.floris.farm.set_wake_model("gauss_legacy")
fi_gauss_legacy.floris.farm.flow_field.solver = "floris"
fi_gauss_legacy.reinitialize_flow_field(
    layout_array=([0.0, 6 * 126.0, 12 * 126.0, 18 * 126.0], [0.0, 0.0, 0.0, 0.0])
)
# Calculate wake
fi_gauss_legacy.calculate_wake()

# Get horizontal plane at default height (hub-height)
hor_plane = fi_gauss_cumulative.get_hor_plane()
hor_plane_gauss = fi_gauss_legacy.get_hor_plane()

# Plot and show
fig, axarr = plt.subplots(2, 1)
wfct.visualization.visualize_cut_plane(hor_plane, ax=axarr[0])
axarr[0].set_title("Gauss Cumulative")
wfct.visualization.visualize_cut_plane(hor_plane_gauss, ax=axarr[1])
axarr[1].set_title("Gauss Legacy")
plt.show()
