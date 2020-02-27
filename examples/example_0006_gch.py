# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import floris.tools as wfct
import numpy as np
# from floris.utilities import Vec3

# Initialize the FLORIS interface fi, use default gauss model
fi = wfct.floris_interface.FlorisInterface("example_input.json")

fi.floris.farm.wake.velocity_model.use_yaw_added_recovery = True
fi.floris.farm.wake.deflection_model.use_secondary_steering = True

# Force dm to 1.0
fi.floris.farm.wake._deflection_model.deflection_multiplier = 1.0

# Change the layout
D = fi.floris.farm.flow_field.turbine_map.turbines[0].rotor_diameter
layout_x = [0, 7*D, 14*D]
layout_y = [0, 0, 0]
yaw_angles = [25, 0, 0]

print('------------------')
print('3 Turbine case')
print('------------------')

fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
fi.calculate_wake(yaw_angles=yaw_angles)

print('------------------')
print('Larger farm case')
print('------------------')

layout_x = [2436.7995, 3082.3995, 2759.5995, 2113.9995, 1791.1995, 2113.9995,
       2759.5995, 3727.9995, 3554.9995, 3082.3995, 2436.7995, 1791.1995,
       1318.5995, 1145.5995, 1318.5995, 1791.1995, 2436.7995, 3082.3995,
       3554.9995, 4373.5995, 4268.6995, 3965.1995, 3496.0995, 2912.2995,
       2276.8995, 1658.7995, 1124.9995,  733.3995,  526.3995,  526.3995,
        733.3995, 1124.9995, 1658.7995, 2276.8995, 2912.2995, 3496.0995,
       3965.1995, 4268.6995]
layout_y = [2436.799, 2436.799, 2995.899, 2994.999, 2436.799, 1876.999,
       1877.699, 2435.999, 3082.399, 3554.999, 3727.999, 3554.999,
       3082.399, 2436.799, 1791.199, 1318.599, 1145.599, 1318.599,
       1791.199, 2436.799, 3065.699, 3626.399, 4058.199, 4314.299,
       4366.999, 4210.499, 3861.799, 3358.599, 2755.599, 2117.999,
       1514.999, 1011.799,  663.099,  506.599,  559.299,  815.399,
       1247.199, 1807.899]
yaw_angles = [21.0,
 18.099999999999994,
 6.100000000000023,
 11.399999999999977,
 23.19999999999999,
 15.400000000000006,
 8.800000000000011,
 13.899999999999977,
 10.5,
 8.600000000000023,
 0.0,
 12.199999999999989,
 0.0,
 23.599999999999994,
 14.599999999999994,
 0.0,
 1.5,
 0.0,
 10.899999999999977,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 3.6999999999999886,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 4.699999999999989,
 0.0,
 14.599999999999994,
 0.0,
 0.0,
 0.0,
 0.0]
 
yaw_angles = np.array(yaw_angles)*1.0

fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# Calculate wake
# fi.calculate_wake(yaw_angles=yaw_angles)

# Calculate wake
fi.calculate_wake(yaw_angles=yaw_angles)
