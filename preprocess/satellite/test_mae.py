import numpy as np
import os
import h5py

from natsort import natsorted
from scipy.interpolate import interp2d

def interpolate_data(data):
    # data shape is (32, 3000, 3000)
    # 原始网格大小
    y = np.arange(0, 60, 0.25, dtype=np.float32) # 起点，终点，步长
    x = np.arange(100, 160, 0.25, dtype=np.float32)
    # 目标网格大小
    y_new = np.arange(0, 60, 0.5)
    x_new = np.arange(100, 160, 0.5)
    
    new_data = np.zeros((2, 120, 120), dtype=np.float32)
    for i in range(data.shape[0]):
        f = interp2d(x, y, data[i], kind='linear')
        new_data[i] = f(x_new, y_new)
    
    return new_data
def get_mae(sat_path,era5_path):
    folders = natsorted(os.listdir(sat_path))
    mae_total = 0
    rmse_total = 0
    num = 0
    for m in folders:
        files = natsorted(os.listdir(os.path.join(sat_path, m)))
        for file in files:
            sat_data = np.load(os.path.join(sat_path, m, file), allow_pickle=True)[1:, :, :]
            era5_data = h5py.File(os.path.join(era5_path, file.replace('npy', 'h5')), 'r')['data'][:]
            
        
            # valid_indices = np.where(~np.isnan(sat_data) & ~np.isnan(sat_data))
            sat_data[sat_data == np.nan] = 0
            
            # 硬拷贝
            mask = np.copy(sat_data)
            mask[mask != 0] = 1
            era5_data  = era5_data * mask
            
            point_num = np.sum(mask)
            mae = np.mean(np.abs(sat_data - era5_data))
            rmse = np.sqrt(np.mean((sat_data - era5_data) ** 2))
            mae_total += mae
            rmse_total += rmse
            num += 1
            # print(f"{file} MAE为：{mae}")

            
    print(f"MAE为：{mae_total/num}",f"RMSE为: {rmse_total/num}")


if __name__ == "__main__":
    sat_path = '/media/data3/wyp/InitalField/data/satelite/grid'
    era5_path = '/media/data3/wyp/InitalField/data/era5/swh/test'
    get_mae(sat_path, era5_path)
    print("MAE done !")