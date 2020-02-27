import numpy as np
import matplotlib.pyplot as plt

# 0, 0, 0, 0, 0 yaw
SOWFA_base = [1940,	843.9,	856.9,	893.1,	926.2]
gauss_base = [1942.591748,	883.2172842,	871.1370133,	942.9237698,	1023.576209]
curl_base = [1941.925636,	951.9408138,	913.7743983,	910.0919239,	908.3445362]

# 25, 0, 0, 0, 0 yaw
SOWFA_25_0_0_0_0 = [1575.3,	1247.3,	1008.4,	955.4,	887.1]
gauss_25_0_0_0_0 = [1615.042379,	1172.50937,	903.9127636,	935.3838198,	1007.223905]
curl_25_0_0_0_0 = [1614.48992,	1146.232113,	981.5624361,	941.7905274,	924.6832865]

# 25, 25, 0, 0, 0 yaw
SOWFA_25_25_0_0_0 = [1577,	986.9,	1338.7,	1089.4,	999.8]
gauss_25_25_0_0_0 = [1615.042379,	970.5601478,	1137.815245,	959.8743055,	998.672105]
curl_25_25_0_0_0 = [1614.48992,	948.4422098,	1254.015189,	1041.912109,	967.9919165]


# plt.figure()
# plt.plot(np.array([SOWFA_base[1], SOWFA_20_0_0[1],
#          SOWFA_0_20_0[1], SOWFA_20_20_0[1]])/SOWFA_base[1], '-s')

# plt.plot(np.array([gauss_base[1], gauss_20_0_0[1],
#          gauss_0_20_0[1], gauss_20_20_0[1]])/gauss_base[1], '-o')

# plt.plot(np.array([curl_base[1], curl_20_0_0[1],
#          curl_0_20_0[1], curl_20_20_0[1]])/curl_base[1], '-*')

yaw_combinations = [
    (0,0,0,0,0), (25,0,0,0,0), (25,25,0,0,0)
]

fig, axarr = plt.subplots(1,5,sharex=False,sharey=True,figsize=(21,3))

titles = ['Turbine 0', 'Turbine 1', 'Turbine 2', 'Turbine 3', 'Turbine 4']

# turb_ind = 4
n_turbs = 5
for turb_ind in range(n_turbs):
    ax = axarr[turb_ind]
    ax.plot(np.array([SOWFA_base[turb_ind], SOWFA_25_0_0_0_0[turb_ind],
            SOWFA_25_25_0_0_0[turb_ind]])/SOWFA_base[turb_ind], '-s', label='SOWFA')

    ax.plot(np.array([gauss_base[turb_ind], gauss_25_0_0_0_0[turb_ind],
            gauss_25_25_0_0_0[turb_ind]])/gauss_base[turb_ind], '-o', label='Gauss')

    ax.plot(np.array([curl_base[turb_ind], curl_25_0_0_0_0[turb_ind],
            curl_25_25_0_0_0[turb_ind]])/curl_base[turb_ind], '-*', label='Curl')
    
    ax.set_title(titles[turb_ind])
    ax.set_xlabel('Yaw (deg.)')
    if turb_ind == 0:
        ax.set_ylabel('Relative Turbine Power')
    ax.set_xticks(np.arange(len(yaw_combinations)))
    ax.set_xticklabels([str(yaw[turb_ind]) for yaw in yaw_combinations])
    # plt.xticks(np.arange(3), (yaw_combinations[0][turb_ind], yaw_combinations[1][turb_ind], yaw_combinations[2][turb_ind]))

axarr[-1].legend()

# plt.plot(np.array([SOWFA_base[turb_ind], SOWFA_25_0_0_0_0[turb_ind],
#          SOWFA_25_25_0_0_0[turb_ind]]), '-s')

# plt.plot(np.array([gauss_base[turb_ind], gauss_25_0_0_0_0[turb_ind],
#          gauss_25_25_0_0_0[turb_ind]]), '-o')

# plt.plot(np.array([curl_base[turb_ind], curl_25_0_0_0_0[turb_ind],
#          curl_25_25_0_0_0[turb_ind]]), '-*')         

plt.savefig('5turb_hiTI_relative_power.png', bbox_inches='tight')
plt.show()