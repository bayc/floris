from scipy import io

file_name = '/mnt/c/Users/cbay/Downloads/Yaw0_active.mat'

data = io.loadmat(file_name)

print(dir(data))

print(dir(data['Lidar2']))