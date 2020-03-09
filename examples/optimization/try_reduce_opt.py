# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import dill as pickle

import floris.tools as wfct
import floris.tools.optimization as opt
from floris.utilities import Vec3
import floris.tools.wind_rose as rose

file_name = 'AEP_data.p'

# df_data = pickle.load(open(file_name, "rb"))
# print(df_data.head())
# print(df_data['AEP'].argmax())
# print(df_data.iloc[297])
# print(df_data.iloc[207])
# print(df_data.iloc[117])
# print(df_data.iloc[27])
# print(df_data.)
# lkj

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")
df_data = pd.DataFrame()

wind_rose = rose.WindRose()
# df = wind_rose.load('lat35_101876000000004long-121_402157wtk.p')
# df = wind_rose.load('lat40_99776long-124_67084399999999wtk_1deg.p')
df = wind_rose.load('lat35_591651long-121_80595100000001wtk.p')
# df = wind_rose.load('lat37_901969long-75_14755799999999wtk.p')
# df = wind_rose.load('lat32_944133long-78_934997wtk.p')
wind_rose.plot_wind_rose(wd_bins=np.arange(0, 360, 5.))
# plt.show()
# print(df)
# lkj
# df2 = wind_rose.resample_wind_direction(copy.deepcopy(df),
#                                         wd=np.arange(0, 360, 1.))
df = wind_rose.resample_average_ws_by_wd(df)
print(df.head())

potential = df['ws'] * df['freq_val']
# print(potential.values.argmax())
# print(potential[65])
# print(potential.values.argmin())
# print(potential[8])
# print(potential[potential > 0.3])

# lkj

# df2 = wind_rose.resample_average_ws_by_wd(df2)
# print(df2[df2['wd'] == 4.0])
# lkj
# print(df['freq_val'].max())

boundaries = [[0., 0.], [0., 1000.], [1000., 1000.], [1000., 0.]]

# wd = [270]
# wd = np.arange(0., 355., 5.)
# np.random.seed(1)
# ws = 8.0 + np.random.randn(len(wd))*0.5
# freq = np.abs(np.sort(np.random.randn(len(wd))))
# freq = np.ones(len(wd))
# freq[0] = 10
# freq = freq/freq.sum()

# wd_new = np.arange(0., 355., 15.)
# ws_new = [np.average(ws[3*i:3*i+3]) for i in range(24)]
# freq_new = np.array([np.average(freq[3*i:3*i+3]) for i in range(24)])
# freq_new = freq_new/freq_new.sum()

n_wt = 9; D = 126.; grid_spc = 7

model = opt.layout2var.Layout2Var(fi, boundaries, n_wt,
                                                  D,
                                                  grid_spc,
                                                  wdir=df.wd,
                                                  wspd=df.ws,
                                                  wfreq=df.freq_val)

# x_layout, y_layout = model.make_grid_layout(n_wt, D, grid_spc)
# print(x_layout, y_layout)

# rotate = np.arange(0, 355., 5.)
AEP_init = model.fi.get_farm_AEP(model.wdir, model.wspd, model.wfreq)
Pow_min = 1.0e20
wd_min = 0.0

# wake_loss_wd = np.arange(0.0, 360.0, 1.0)
# wake_loss_ws = [8.0]

# for i in range(len(model.wdir)):
#     # model.fi.reinitialize_flow_field(wind_direction=[model.wdir[i]],
#     #                                   wind_speed=[model.wspd[i]])
#     model.fi.reinitialize_flow_field(wind_direction=[wake_loss_wd[i]],
#                                       wind_speed=[model.wspd[i]])
#     model.fi.calculate_wake()
#     Pow_tmp = model.fi.get_farm_power()
#     # print('Pow_tmp: ', Pow_tmp/177781.7557436897)
#     if Pow_tmp < Pow_min:
#         Pow_min = Pow_tmp
#         wd_min = model.wdir[i]
#         # print('min power update: ', Pow_min)

# print('Greatest wake loss direction: ', wd_min)
# print('Greatest wake loss power: ', Pow_min)


# for i in range(len(rotate)):

# rotate = np.arange(0, 360., 1.)
# AEP_list = []
# coords = copy.deepcopy(model.fi.floris.farm.flow_field.turbine_map.coords)
# for i in range(len(rotate)):
#     x_layout, y_layout = model.rotate_farm(coords, rotate[i])
#     model.fi.reinitialize_flow_field(layout_array=[x_layout, y_layout])
#     AEP = model.fi.get_farm_AEP(model.wdir, model.wspd, model.wfreq)
#     AEP_list.append(AEP)
#     print('Rotation ', str(rotate[i]), ' done!')

# df_data['AEP'] = AEP_list
# df_data['farm_rotation'] = rotate

# df_data.to_pickle('AEP_data.p')

# print('Initial AEP: ', AEP_init)
# print('New AEP: ', AEP)
# print('AEP Increase: ', (AEP - AEP_init)/AEP*100)

# plt.figure()
# plt.plot(model.x0, model.y0, '.k')
# plt.plot(x_layout, y_layout, '.r')
# plt.axis('equal')
# plt.grid()
# plt.show()

tmp = opt.optimization.Optimization(model=model, solver='SLSQP')

sol = tmp.optimize()

# print(dir(sol))
# print(sol.objFun())
# print(sol.objectiveIdx)
# print(dir(sol.objectives))
# print(sol.objectives.items())
# print(sol.objectives.keys())
# print(sol.objectives.values())
# print(dir(sol.objectives['obj']))
# print(sol.objectives['obj'].value)

AEP = sol.objectives['obj'].value*-1e9

# print()
# print('theta opt: ', sol.getDVs()['theta'])

print('Initial AEP: ', AEP_init)
print('New AEP: ', AEP)
print('AEP Increase: ', (AEP - AEP_init)/AEP*100)

model.plot_layout_opt_results(sol)
plt.show()