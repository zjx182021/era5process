"""
预处理接收的是向日葵8卫星下载并切出WNP区域的数据，shape为(3, 3001 , 3001)；
ERA5数据为原始下载数据，为多个NC文件
the preprocess mainly include:
1. 将ERA5拆分为单个时间的uv，保存为.h5文件
2. 开始做样本。先取出向日葵8卫星的band7和13通道(即后3个通道中的后两个通道)；
   去除向日葵卫星和ERA5数据的最后一列和一行，使得向日葵8卫星的shape为(2, 3000, 3000)，ERA5的shape为(2, 240, 240)；
   采用滑动窗口采样对向日葵8卫星数据和ERA5数据进行采样；
   若采样的样本中有无效值(-32767.0)，则舍弃该样本；
3. 从h5文件中读取数据，计算每个通道的mean和std，保存为csv文件
4. 对样本进行归一化，保存为h5文件
"""
from netCDF4 import Dataset
import numpy as np
from natsort import natsorted
import datetime
import os
import sys
import h5py
sys.path.append('/media/data5/wyp/lq/SWH_Retrieval')
import tqdm
import pandas as pd
from multiprocessing import Pool
import time
import random
import multiprocessing
from itertools import product

from utils.read_config import configs
from utils.utils import *


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def filter_data(data):
    """
    filter data have invalid data or have invalid dtype
    """
    if -32767.0 in data or data.dtype != np.float32:
        return None
    else:
        return data


def reform_era5(params):
    """
    read era5 uv from era5 file, and save one by one, each file contains one time's uv
    """
    era5_uv_path = params['data_path']['era5_uv']
    era5_ori_path = params['data_path']['era5_ori']
    
    for file in natsorted(os.listdir(era5_ori_path)):
        print('Read ' + file)
        era5_ori_file = os.path.join(era5_ori_path, file)
        era5_ori_data = Dataset(era5_ori_file, 'r')
        era5_time = era5_time_to_datetime(era5_ori_data.variables['time'][:].data)
        era5_ori_u = era5_ori_data.variables['u10'][:].data
        era5_ori_v = era5_ori_data.variables['v10'][:].data
        
        for i in range(len(era5_time)):
            era5_uv = np.array([era5_ori_u[i], era5_ori_v[i]], dtype=np.float32)[:, :-1, :-1]
            
            file_name = datetime.datetime.strftime(era5_time[i], '%Y-%m-%d-%H')
            era5_uv_file = os.path.join(era5_uv_path, file_name + '.h5')
            with h5py.File(era5_uv_file, 'w') as f:
                f.create_dataset('data', data=era5_uv, compression='gzip')
                f.close()
                # print('save ' + file_name)


def create_sample_one_file(h8_file, h8_ori_path, era5_uv_path, 
                           input_path, target_path, mean_info_path, input_size, 
                           target_size, input_stride, target_stride):
    """
    create sample from one h8 file
    """
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
        return None
    
    if h8_data.shape[0] == 16:
        # 取 band 7 and band 13, 即下标为6和12的通道
        h8_data = h8_data[[6, 12]]
    elif h8_data.shape[0] == 3:
        # 为3，则表示只有band 3,7,13，取 band 7 and band 13, 即下标为1和2的通道
        h8_data = h8_data[[1, 2]]
    # now the shape of h8_data is (2, 3000, 3000)
    
    # read era5 data
    # get era5 file name from h8_time
    # era5 file name example: 2021-01-01-00.h5
    era5_file_name = datetime.datetime.strftime(h8_time, '%Y-%m-%d-%H') + '.h5'
    era5_file = os.path.join(era5_uv_path, era5_file_name)

    try:
        era5_nc = h5py.File(era5_file, 'r')
        era5_data = era5_nc['data'][:]
        era5_nc.close()
    except:
        print(f'Can\'t Read file: {era5_file}')
        return None
    # now the shape of era5_data is (2, 240, 240)
    
    # create sample
    samples_input = []
    samples_target = []
    for i in range(0, h8_data.shape[1] - input_size + 1, input_stride):
        for j in range(0, h8_data.shape[2] - input_size + 1, input_stride):
            temp = h8_data[:, i:i+input_size, j:j+input_size]
            # filter invalid data
            if np.min(temp) < -30000.0 or np.max(temp) > 30000.0:
                continue
            samples_input.append(h8_data[:, i:i+input_size, j:j+input_size])
    
    for i in range(0, era5_data.shape[1] - target_size + 1, target_stride):
        for j in range(0, era5_data.shape[2] - target_size + 1, target_stride):
            samples_target.append(era5_data[:, i:i+target_size, j:j+target_size])
    
    samples_input = np.array(samples_input)
    samples_target = np.array(samples_target)
    
    # save sample one by one
    for i in range(len(samples_input)):
        sample_name = era5_file_name.split('.')[0] + '_' + str(i)
        sample_input_file = os.path.join(input_path, sample_name + '.h5')
        sample_target_file = os.path.join(target_path, sample_name + '.h5')
        
        with h5py.File(sample_input_file, 'w') as f:
            f.create_dataset('data', data=samples_input[i], dtype=np.float32, compression='gzip')
            f.close()
        with h5py.File(sample_target_file, 'w') as f:
            f.create_dataset('data', data=samples_target[i], dtype=np.float32, compression='gzip')
            f.close()
        
    # save norm info
    mean = np.mean(samples_input, axis=(0, 2, 3))
    max_value = np.max(samples_input, axis=(0, 2, 3))
    min_value = np.min(samples_input, axis=(0, 2, 3))
    np.save(os.path.join(mean_info_path, era5_file_name.split('.')[0] + '.npy'), np.array([mean, max_value, min_value]))

    print(f'Save sample {era5_file_name}, and the number of samples is {len(samples_input)}')


def create_sample(params):
    h8_ori_path = params['data_path']['himawari8_wnp']
    era5_uv_path = params['data_path']['era5_uv']
    input_path = params['inversion_stage1']['input_path']
    target_path = params['inversion_stage1']['target_path']
    time_range = params['inversion_stage1']['time_range']
    
    input_size, target_size = params['inversion_stage1']['input_size'], params['inversion_stage1']['target_size']
    input_stride, target_stride = params['inversion_stage1']['input_stride'], params['inversion_stage1']['target_stride']
    
    sub_dir = ['train'] #, 'valid', 'test'
    
    for dir in sub_dir:
        temp_time_range = time_range[dir]
        for year in temp_time_range:
            year = str(year)
            months = natsorted(os.listdir(os.path.join(h8_ori_path, year)))
            for month in months:
                temp_h8_ori_month_path = os.path.join(h8_ori_path, year, month)
                h8_files = natsorted(os.listdir(temp_h8_ori_month_path))
                
                temp_input_path = os.path.join(input_path, dir)
                temp_target_path = os.path.join(target_path, dir)
                mean_info_path = os.path.join(input_path, 'mean_info')
                
                create_dir([temp_input_path, temp_target_path, mean_info_path])
                
                pool = Pool(8)
                for h8_file in h8_files:
                    create_sample_one_file(h8_file, temp_h8_ori_month_path, era5_uv_path, 
                                              temp_input_path, temp_target_path, mean_info_path,
                                              input_size, target_size, input_stride, target_stride)
                    #pool.apply_async(create_sample_one_file, args=(h8_file, temp_h8_ori_month_path, era5_uv_path, 
                                                                   #temp_input_path, temp_target_path, mean_info_path,
                                                                   #input_size, target_size, input_stride, target_stride))
                #pool.close()
                #pool.join()


def calculate_mse_hdf5(args):
    file_path, mean = args
    try:
        with h5py.File(file_path, 'r') as f:
            data = f['data'][:]
            mse = np.mean((data - mean)**2, axis=(1, 2))
            f.close()
            print(f'Calculate {file_path}')
    except:
        return None
    return mse


def cal_norm_info(params):
    """
    get norm info from each channel
    """
    norm_info_files_path = os.path.join(params['inversion_stage1']['input_path'], 'mean_info')
    h8_norm_info = params['data_path']['h8_norm_info']
    time_years = params['inversion_stage1']['time_range']['train']

    # collect norm info
    dim_mean = []
    dim_std = []
    dim_max = []
    dim_min = []
    
    files = os.listdir(norm_info_files_path)
    
    # filter files by time_range
    new_files = []
    for file in files:
        file_year = int(file.split('-')[0])
        if file_year not in time_years:
            continue
        new_files.append(file)
    files = new_files
    
    # get norm info from h8 data
    for file in files:
        temp_norm_info = np.load(os.path.join(norm_info_files_path, file))
        dim_mean.append(temp_norm_info[0])
        dim_max.append(temp_norm_info[1])
        dim_min.append(temp_norm_info[2])
    
    dim_mean = np.mean(np.array(dim_mean), axis=0)
    dim_max = np.max(np.array(dim_max), axis=0)
    dim_min = np.min(np.array(dim_min), axis=0)
    print('-------------------')
    print(dim_mean, dim_max, dim_min)
    
    # get std from h8 data
    input_train_path = os.path.join(params['inversion_stage1']['input_path'], 'train')
    train_files = os.listdir(input_train_path)
    train_files = [os.path.join(input_train_path, file) for file in train_files]
    h8_files = train_files
    # h8_files = h8_files[:32]
    
    # Create Cartesian product of h8_files and temp_dim_mean
    temp_dim_mean = dim_mean.reshape(2, 1, 1)
    args_list = list(product(h8_files, [temp_dim_mean]))
    
    with multiprocessing.Pool(processes=16) as pool:
        statistics_list = pool.map(calculate_mse_hdf5, args_list)
    # filter None
    statistics_list = [mse for mse in statistics_list if mse is not None]
    statistics_list = np.array(statistics_list)
    dim_std = np.sqrt(np.mean(statistics_list, axis=0))

    # save norm info to csv
    print(dim_mean, dim_std)
    norm_info = pd.DataFrame({'mean': dim_mean, 'std': dim_std, 'max': dim_max, 'min': dim_min})
    norm_info.to_csv(h8_norm_info, index=False)
    print(f'Save norm info to {h8_norm_info}')

    # get norm info from era5 data
    era5_uv_path = params['data_path']['era5_uv']
    era5_norm_info = params['data_path']['era5_norm_info']
    era5_files = os.listdir(era5_uv_path)
    # filter files by time_range
    new_era5_files = []
    for file in era5_files:
        file_year = int(file.split('-')[0])
        if file_year not in time_years:
            continue
        new_era5_files.append(file)
    era5_files = new_era5_files
    
    # get all era5 data
    era5_data = []
    for file in era5_files:
        era5_file = os.path.join(era5_uv_path, file)
        with h5py.File(era5_file, 'r') as f:
            data = f['data'][:]
            era5_data.append(data)
            f.close()
    era5_data = np.array(era5_data)
    
    # get mean and std from era5 data
    era5_mean = np.mean(era5_data, axis=(0, 2, 3))
    era5_std = np.std(era5_data, axis=(0, 2, 3))
    era5_max = np.max(era5_data, axis=(0, 2, 3))
    era5_min = np.min(era5_data, axis=(0, 2, 3))

    # save norm info to csv
    norm_info = pd.DataFrame({'mean': era5_mean, 'std': era5_std, 'max': era5_max, 'min': era5_min})
    norm_info.to_csv(era5_norm_info, index=False)
    print(f'Save norm info to {era5_norm_info}')


def norm_one_file(file_path, file, norm_info, norm_path):
    file_name = os.path.join(file_path, file)
    with h5py.File(file_name, 'r') as f:
        input_image = f['data'][:]
        f.close()
    input_image = (input_image - norm_info[0]) / norm_info[1]
    
    # save
    saved_file_name = os.path.join(norm_path, file)
    with h5py.File(saved_file_name, 'w') as f:
        f.create_dataset('data', data=input_image, dtype=np.float32, compression='gzip')
        f.close()
        print(f'Save {saved_file_name}')


def norm_h8_era5(params):
    # load mean and std
    h8_mean, h8_std = read_norm_info(info_type='h8')
    era5_mean, era5_std = read_norm_info(info_type='era5')
    
    input_norm_info = (h8_mean, h8_std)
    target_norm_info = (era5_mean, era5_std)
    
    pool = Pool(8)
    
    sub_path = ['train'] #, 'valid', 'test'
    input_path = params['inversion_stage1']['input_path']
    target_path = params['inversion_stage1']['target_path']
    timerange = params['inversion_stage1']['time_range']
    for path in sub_path:
        # temp_timerange = timerange[path]
        for i in range(2):
            print(f'Norm {path} {i}')
            root_path = os.path.join(input_path, path) if i == 0 else os.path.join(target_path, path)
            norm_info = input_norm_info if i == 0 else target_norm_info
            norm_path = root_path
            files = natsorted(os.listdir(root_path))
            
            #files = [file for file in files if file.split('-')[0] in temp_timerange]
            
            for file in files:
                # norm_one_file(root_path, file, norm_info, norm_path)
                pool.apply_async(norm_one_file, args=(root_path, file, norm_info, norm_path))
    
    pool.close()
    pool.join()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    seed_everything()
    
    params = configs
    # reform_era5(params)
    #create_sample(params)
    # cal_norm_info(params)
    # norm_h8_era5(params)
    
