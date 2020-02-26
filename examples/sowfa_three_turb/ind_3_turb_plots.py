import numpy as np
import matplotlib.pyplot as plt

# 0, 0, 0 yaw
SOWFA_base = [1941.1,	918.7,	947.9]
gauss_base = [1942.591748,	1010.112048,	1009.259718]
curl_base =	[1941.925636,	1022.32436,	988.0414309]

# 20, 0, 0 yaw
SOWFA_20_0_0 = [1690.1,	1216.9,	1086.1]
gauss_20_0_0 = [1728.3664,	1193.056031,	1027.832666]
curl_20_0_0 = [1727.773727,	1156.040391,	1033.833333]

# 0, 20, 0 yaw
SOWFA_0_20_0 = [1942.8,	814.2,	1055.1]
gauss_0_20_0 = [1942.591748,	895.7760013,	1138.249194]
curl_0_20_0 = [1941.925636,	906.8830232,	1111.959693]

# 20, 20, 0 yaw
SOWFA_20_20_0 = [1692.2,	1036.4,	1330.7]
gauss_20_20_0 = [1728.3664,	1059.130578,	1171.306639]
curl_20_20_0 = [1727.773727,	1025.700302,	1227.111409]

plt.figure()
# plt.plot(np.array([SOWFA_base[1], SOWFA_20_0_0[1],
#          SOWFA_0_20_0[1], SOWFA_20_20_0[1]])/SOWFA_base[1], '-s')

# plt.plot(np.array([gauss_base[1], gauss_20_0_0[1],
#          gauss_0_20_0[1], gauss_20_20_0[1]])/gauss_base[1], '-o')

# plt.plot(np.array([curl_base[1], curl_20_0_0[1],
#          curl_0_20_0[1], curl_20_20_0[1]])/curl_base[1], '-*')

turb_ind = 2
# plt.plot(np.array([SOWFA_base[turb_ind], SOWFA_20_0_0[turb_ind],
#          SOWFA_0_20_0[turb_ind], SOWFA_20_20_0[turb_ind]])/SOWFA_base[turb_ind], '-s')

# plt.plot(np.array([gauss_base[turb_ind], gauss_20_0_0[turb_ind],
#          gauss_0_20_0[turb_ind], gauss_20_20_0[turb_ind]])/gauss_base[turb_ind], '-o')

# plt.plot(np.array([curl_base[turb_ind], curl_20_0_0[turb_ind],
#          curl_0_20_0[turb_ind], curl_20_20_0[turb_ind]])/curl_base[turb_ind], '-*')

plt.plot(np.array([SOWFA_base[turb_ind], SOWFA_20_0_0[turb_ind],
         SOWFA_0_20_0[turb_ind], SOWFA_20_20_0[turb_ind]]), '-s')

plt.plot(np.array([gauss_base[turb_ind], gauss_20_0_0[turb_ind],
         gauss_0_20_0[turb_ind], gauss_20_20_0[turb_ind]]), '-o')

plt.plot(np.array([curl_base[turb_ind], curl_20_0_0[turb_ind],
         curl_0_20_0[turb_ind], curl_20_20_0[turb_ind]]), '-*')         

plt.show()