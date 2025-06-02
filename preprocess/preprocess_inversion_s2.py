import sys
sys.path.append('/media/data5/wyp/lq/SWH_Retrieval')
import os
import numpy as np
import h5py
import datetime
from natsort import natsorted
from scipy.interpolate import interp2d
import random
import time
from multiprocessing import Pool
from tqdm import tqdm

from utils.read_config import configs
from utils.utils import *


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def interpolate_data(data):
    # data shape is (32, 3000, 3000)
    # 原始网格大小
    y = np.arange(0, 60, 0.02, dtype=np.float32) # 起点，终点，步长
    x = np.arange(100, 160, 0.02, dtype=np.float32)
    # 目标网格大小
    y_new = np.arange(0, 60, 0.25)
    x_new = np.arange(100, 160, 0.25)
    
    new_data = np.zeros((2, 240, 240), dtype=np.float32)
    for i in range(data.shape[0]):
        f = interp2d(x, y, data[i], kind='linear')
        new_data[i] = f(x_new, y_new)
    
    return new_data


def get_h8_data_one_file(h8_file, h8_ori_path, norm_info, saved_path):
    """
    create sample from one h8 file
    """
    start_time = time.time()
    # extract the time of hb_file
    h8_time = h8_file.split('_')[2] + h8_file.split('_')[3]
    h8_time = datetime.datetime.strptime(h8_time, '%Y%m%d%H%M')

    # read h8 data
    try:
        h8_nc = h5py.File(os.path.join(h8_ori_path, h8_file), 'r')
        h8_data = h8_nc['data'][:][:, :-1, :-1]
        h8_nc.close()
    except:
        print(f'Can\'t Read file: {h8_file}')
        return None, None

    if h8_data.shape[0] == 16:
        # 取 band 7 and band 13, 即下标为6和12的通道
        h8_data = h8_data[[6, 12]]
    elif h8_data.shape[0] == 3:
        # 为3，则表示只有band 3,7,13，取 band 7 and band 13, 即下标为1和2的通道
        h8_data = h8_data[[1, 2]]

    h8_data = filter_data(h8_data)
    
    if h8_data is None:
        return None, None
    # now the shape of h8_data is (2, 3000, 3000)
    
    # interpolate data
    new_h8_data = interpolate_data(h8_data)
    # now the shape of new_h8_data is (2, 240, 240)
    
    # normalize data
    new_h8_data = (new_h8_data - norm_info[0]) / norm_info[1]
    
    # save data
    file_name = h8_time.strftime('%Y-%m-%d-%H')

    saved_file = os.path.join(saved_path, file_name+'.h5')
    
    with h5py.File(saved_file, 'w') as f:
        f.create_dataset('data', data=new_h8_data)
        f.close()
    
    print(f'Save file: {saved_file}, cost time: {time.time() - start_time:.2f}s')


def get_norm_h8_data(h8_ori_path, time_range, saved_path):
    years = time_range    #natsorted(os.listdir(h8_ori_path))
    # sub_pathes = ['train', 'valid', 'test']
    sub_pathes = ['train']
    
    norm_info = read_norm_info(info_type='h8', variable=True)
    
    # for year in years:
    for sub_path in sub_pathes:
        for year in years[sub_path]:
            if int(year) in time_range[sub_path]:
                input_h8_path = os.path.join(saved_path, 'h8', sub_path)
                create_dir(input_h8_path)
                # break
        
            temp_year_path = os.path.join(h8_ori_path, str(year))
            months = natsorted(os.listdir(temp_year_path))
            
            for month in months:
                temp_month_path = os.path.join(temp_year_path, month)
                h8_files = natsorted(os.listdir(temp_month_path))
                
                pool = Pool(4)
            
                for h8_file in h8_files:
                    # get_h8_data_one_file(h8_file, temp_month_path, norm_info, input_h8_path)
                    pool.apply_async(get_h8_data_one_file, args=(h8_file, temp_month_path, norm_info, input_h8_path))
                pool.close()
                pool.join()


def norm_swint_output(swint_output_path, saved_path):
    """
    normalize swint output
    """
    norm_info = read_norm_info(info_type='era5')
    
    sub_dirs = natsorted(os.listdir(swint_output_path))
    for sub_dir in sub_dirs:
        swint_output_sub_dir = os.path.join(swint_output_path, sub_dir, 'data')
        saved_sub_dir = os.path.join(saved_path, 'swint', sub_dir)
        
        create_dir(saved_sub_dir)
        
        swint_files = natsorted(os.listdir(swint_output_sub_dir))
        for swint_file in tqdm(swint_files):
            swint_nc = h5py.File(os.path.join(swint_output_sub_dir, swint_file), 'r')
            swint_data = swint_nc['data'][:].astype(np.float32)
            swint_nc.close()
            
            swint_data = (swint_data - norm_info[0]) / norm_info[1]
            
            saved_file = os.path.join(saved_sub_dir, swint_file)
            
            with h5py.File(saved_file, 'w') as f:
                f.create_dataset('data', data=swint_data)
                f.close()


def norm_era5_uv(era5_uv_path, saved_path):
    """
    normalize era5 uv
    """
    print('Start normalize era5 uv data (inversion stage 2 target).')
    create_dir(saved_path)
    
    norm_info = read_norm_info(info_type='era5')
    
    era5_files = natsorted(os.listdir(era5_uv_path))
    for era5_file in tqdm(era5_files):
        era5_nc = h5py.File(os.path.join(era5_uv_path, era5_file), 'r')
        era5_data = era5_nc['data'][:].astype(np.float32)
        era5_nc.close()
        
        era5_data = (era5_data - norm_info[0]) / norm_info[1]

        saved_file = os.path.join(saved_path, era5_file)
        with h5py.File(saved_file, 'w') as f:
            f.create_dataset('data', data=era5_data)
            f.close()


if __name__ == '__main__':
    config = configs
    
    seed_everything()
    
    h8_ori_path = config['data_path']['himawari8_wnp']
    time_range = config['inversion_stage2']['time_range']
    input_path = config['inversion_stage2']['input_path']
    
    get_norm_h8_data(h8_ori_path, time_range, input_path)
    
    # swint_output_path = config['inversion_stage1']['output_path']
    # # norm_swint_output(swint_output_path, input_path)

    # era5_uv_path = config['data_path']['era5_uv']
    # target_path = config['inversion_stage2']['target_path']
    # norm_era5_uv(era5_uv_path, target_path)