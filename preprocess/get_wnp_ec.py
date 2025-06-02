from netCDF4 import Dataset
import os
import datetime
from natsort import natsorted
import numpy as np
from scipy.interpolate import interp2d
import h5py

ec_ori_path = '/media/data4/EC/uv'
ec_wnp_path = '/media/data3/wyp/InitalField/data/ec_data'


def era5_time_to_datetime(time_nc):
    inittime = datetime.datetime(1900, 1, 1, 00, 00, 00)
    time_temp = []
    for elem in time_nc:
        time_temp.append(inittime + datetime.timedelta(hours=int(elem)))
    return time_temp


def interpolate_data(data):
    # data shape is (2, 5, 600, 600)
    # 原始网格大小
    y = np.arange(0, 60, 0.1, dtype=np.float32) # 起点，终点，步长
    x = np.arange(100, 160, 0.1, dtype=np.float32)
    # 目标网格大小
    y_new = np.arange(0, 60, 0.25)
    x_new = np.arange(100, 160, 0.25)
    
    new_data = np.zeros((2, 5, 240, 240), dtype=np.float32)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f = interp2d(x, y, data[i, j], kind='linear')
            new_data[i, j] = f(x_new, y_new)
    
    return new_data


def read_ec(ec_file, ec_wnp_path):
    print(ec_file)
    ec_data = Dataset(ec_file)
    ec_time = ec_data.variables['time'][:]
    ec_time = era5_time_to_datetime(ec_time)
    # print(ec_time[0])
    ec_lat = ec_data.variables['latitude'][:].data
    ec_lon = ec_data.variables['longitude'][:].data
    
    # extract the area of 0°-60°N, 100°E-160°E
    lat_index = np.where((ec_lat >= 0) & (ec_lat <= 60))[0]
    lon_index = np.where((ec_lon >= 100) & (ec_lon <= 160))[0]
    
    ec_u = []
    ec_v = []
    
    hours = [0, 6, 12, 18, 24]
    hours = [datetime.timedelta(hours=hour) for hour in hours]
    
    for temp_time in ec_time:
        if temp_time - ec_time[0] in hours:
            ec_u.append(ec_data.variables['u10'][ec_time.index(temp_time), 0, lat_index[0]:lat_index[-1], lon_index[0]:lon_index[-1]].data)
            ec_v.append(ec_data.variables['v10'][ec_time.index(temp_time), 0, lat_index[0]:lat_index[-1], lon_index[0]:lon_index[-1]].data)
        elif ec_time.index(temp_time) in [0, 2, 4, 6, 8]:
            # 若该预报时间不存在，则使用nan填充
            print('Forecast time is not exist! {} is filled with nan.'.format(temp_time))
            ec_u.append(np.zeros((600, 600), dtype=np.float32))
            ec_v.append(np.zeros((600, 600), dtype=np.float32))
            
    ec_data.close()
    ec_uv = np.array([ec_u, ec_v], dtype=np.float32)
    # print(ec_uv.shape)
    
    # interpolate the data to 0.25°
    ec_uv = interpolate_data(ec_uv)
    
    # save the data
    ec_name = datetime.datetime.strftime(ec_time[0], '%Y-%m-%d-%H') + '.h5'
    ec_path = os.path.join(ec_wnp_path, ec_name)
    with h5py.File(ec_path, 'w') as f:
        f.create_dataset('data', data=ec_uv, dtype=np.float32)
        f.close()
    print(ec_name, 'is saved!')


if __name__ == '__main__':
    ec_files = natsorted(os.listdir(ec_ori_path))
    for ec_file in ec_files:
        if ec_file.split('_')[4][:4] == '2021':
            read_ec(os.path.join(ec_ori_path, ec_file), ec_wnp_path)
    # read_ec(os.path.join(ec_ori_path, ec_files[0]), ec_wnp_path)