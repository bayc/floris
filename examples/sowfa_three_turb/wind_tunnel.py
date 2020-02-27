from scipy import io
import numpy as np
# from mat4py import loadmat

# file_name = '/mnt/c/Users/cbay/Desktop/torque_paper/Yaw0_active.mat'
file_name = '/mnt/c/Users/cbay/Desktop/torque_paper/Yaw30_active.mat'
# file_name = '/mnt/c/Users/cbay/Desktop/torque_paper/Effective_velocity.mat'

data = io.loadmat(file_name, struct_as_record=True)
# data = loadmat(file_name)

# print(dir(data))

# print(data.keys())

# print(dir(data['Lidar2']))
# print(data['Lidar2'].dtype)

# for values in data['Lidar2']:
#     for val in values:
#         for tmp in val:
#             print(tmp)
#             print(data['Lidar2'].dtype)
#             print(tmp.dtype)
#             input('Press a key to continue')

print(data['Lidar2'][0][0][3][0][0][19])
print(data['Lidar2'][0][0][3][0][0].dtype)

# print(data.keys())
# print(data['v_eff_yaw30_7D'])
# print(data['yy'])

# print(data)