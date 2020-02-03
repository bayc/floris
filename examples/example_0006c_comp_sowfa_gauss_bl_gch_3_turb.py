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
import pandas as pd
# from floris.utilities import Vec3

# Write out SOWFA results
layout_x = (1000.0, 1756.0, 2512.0)
layout_y = (1000.0, 1000.0, 1000.0)
sowfa_results = np.array([
[1940.1,	844.2,	861.7,	0,	0,	0],
[1940.8,	756.9,	966.5,	0,	20,	0],
[1689.5,	1143.1,	988.5,	20,	0,	0],
[1690.6,	979.5,	1223.2,	20,	20,	0],
[1576.9,	1053.1,	884.5,	25,	-30,	0],
[1575.9,	1181.3,	873.9,	25,	-20,	0],
[1575.4,	1245.6,	923.7,	25,	-10,	0],
[1575.2,	1248,	1015.3,	25,	0,	0],
[1575,	1186.5,	1144.2,	25,	10,	0],
[1576.4,	1064.9,	1289.6,	25,	20,	0],
[1577.2,	893.5,	1410.9,	25,	30,	0]
])
df_sowfa = pd.DataFrame(sowfa_results, 
                        columns = ['p1','p2','p3','y1','y2','y3'] )

print(df_sowfa.head())

## SET UP FLORIS AND MATCH TO BASE CASE
wind_speed = 8.38
TI = 0.09

# Initialize the FLORIS interface fi, use default gauss model
fi = wfct.floris_interface.FlorisInterface("example_input.json")
fi.reinitialize_flow_field(wind_speed=[wind_speed],turbulence_intensity=[TI],layout_array=(layout_x, layout_y))
fi.calculate_wake(yaw_angles=[0,0,0])
print('Gauss',np.array(fi.get_turbine_power())/1000.0)


# fi_gch = wfct.floris_interface.FlorisInterface("example_input.json")

# # Force dm to 1.0
# fi.floris.farm.wake._deflection_model.deflection_multiplier = 1.0
# fi_gch.floris.farm.wake._deflection_model.deflection_multiplier = 1.0

# # Set up gch
# fi_gch.floris.farm.wake.velocity_model = "gauss_curl_hybrid"
# fi_gch.floris.farm.wake.deflection_model = "gauss_curl_hybrid"
# fi_gch.floris.farm.wake.velocity_models["gauss_curl_hybrid"].use_yar = False
# fi_gch.floris.farm.wake.deflection_models["gauss_curl_hybrid"].use_ss = False

# # Change the layout
# D = fi.floris.farm.flow_field.turbine_map.turbines[0].rotor_diameter
# layout_x = [0, 7*D, 14*D]
# layout_y = [0, 0, 0]
# yaw_angles = [0, 0, 0]
# fi.reinitialize_flow_field(layout_array=(layout_x, layout_y))
# fi_gch.reinitialize_flow_field(layout_array=(layout_x, layout_y))

# # Calculate baseline wake
# fi.calculate_wake(yaw_angles=yaw_angles)
# fi_gch.calculate_wake(yaw_angles=yaw_angles)

# # Print the turbine power
# print("Power in baseline")
# print('Gauss',np.array(fi.get_turbine_power())/1000.0)
# print('GCH',np.array(fi_gch.get_turbine_power())/1000.0)


# # # Calculate wake
# # fi.calculate_wake(yaw_angles=yaw_angles)

# # # Print the turbine power
# # print(np.array(fi.get_turbine_power())/1000.0)
