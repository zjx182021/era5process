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
import fcntl
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
    era5_swh_path = params['data_path']['era5_swh']
    era5_ori_path = params['data_path']['era5_ori']
    
    for file in natsorted(os.listdir(era5_ori_path)):
        year = file.split('_')[2]
        print('Read ' + file)
        era5_ori_file = os.path.join(era5_ori_path, file)
        era5_ori_data = Dataset(era5_ori_file, 'r')
        era5_time = era5_time_to_datetime(era5_ori_data.variables['time'][:].data)
        era5_ori_swh = era5_ori_data.variables['swh'][:].data
        
        
        for i in range(len(era5_time)):
            era5_swh = np.array(era5_ori_swh, dtype=np.float32)[:, :-1, :-1]
            
            file_name = datetime.datetime.strftime(era5_time[i], '%Y-%m-%d-%H')
            if i == 0:
                temp_era5_swh_path = os.path.join(era5_swh_path, year)                   
            era5_swh_file = os.path.join(temp_era5_swh_path, file_name + '.h5')
            if not os.path.exists(temp_era5_swh_path): 
                os.makedirs(temp_era5_swh_path, exist_ok=True)
            
            with h5py.File(era5_swh_file, 'w') as f:
                f.create_dataset('data', data=era5_swh[i], compression='gzip')
                f.close()
                print('save ' + file_name)
            


def create_sample_one_file(h8_file, h8_ori_path, temp_h8_ori_month_path, era5_swh_path, 
                           input_path, target_path, input_size, 
                           target_size, input_stride, target_stride):
    """
    create sample from one h8 file
    """
    # extract the time of hb_file
    h8_time = h8_file.split('_')[2] + h8_file.split('_')[3]
    h8_time = datetime.datetime.strptime(h8_time, '%Y%m%d%H%M')
    
    h8_data = get_seq_h8_data(h8_file, h8_ori_path, temp_h8_ori_month_path)
    
    # # read h8 data
    # try:
    #     h8_nc = h5py.File(os.path.join(h8_ori_path, h8_file), 'r')
    #     h8_data = h8_nc['data'][:, :-1, :-1]
    #     # h8_data = h8_nc['data'][:][1:2, :-1, :-1]
    #     h8_nc.close()
        
    # except:
    #     print(f'Can\'t Read file: {h8_file}')
    #     return None
    
    # if h8_data.shape[0] == 16:
    #     # 取 band 7 and band 13, 即下标为6和12的通道
    #     h8_data = h8_data[[6, 12]]
    # elif h8_data.shape[0] == 3:
    #     # 为3，则表示只有band 3,7,13，取 band 7 and band 13, 即下标为1和2的通道
    #     h8_data = h8_data[[1, 2]]
    # # now the shape of h8_data is (2, 3000, 3000)
    
    # read era5 data
    # get era5 file name from h8_time
    # era5 file name example: 2021-01-01-00.h5
    while h8_data is not None:
        
        era5_file_name = datetime.datetime.strftime(h8_time, '%Y-%m-%d-%H') + '.h5'
        temp_era5_file = os.path.join(era5_swh_path, h8_file.split('_')[2][0:4])
        era5_file = os.path.join(temp_era5_file, era5_file_name)

        try:
            era5_nc = h5py.File(era5_file, 'r')
            era5_data = era5_nc['data'][:]
            era5_data[era5_data == -32767] = 0
            era5_data = np.expand_dims(era5_data, axis=0)
            era5_nc.close()
        except:
            print(f'Can\'t Read file: {era5_file}')
            return None
        # now the shape of era5_data is (2, 240, 240)
        
        # create sample    切分为中尺度小图
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
        samples_input = np.expand_dims(samples_input, axis=1)
        samples_target = np.expand_dims(samples_target, axis=1)
        
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
            
        # # save norm info
        # mean = np.mean(samples_input, axis=(0, 2, 3))
        # max_value = np.max(samples_input, axis=(0, 2, 3))
        # min_value = np.min(samples_input, axis=(0, 2, 3))
        # np.save(os.path.join(mean_info_path, era5_file_name.split('.')[0] + '.npy'), np.array([mean, max_value, min_value]))

        print(f'Save sample {era5_file_name}, and the number of samples is {len(samples_input)}')
        h8_data = None

def get_seq_h8_data(h8_file, h8_ori_path, temp_h8_ori_month_path):
    t_ori_file = os.path.join(temp_h8_ori_month_path, h8_file)
    # t_file = h8_file
    h8_time = get_h8_time(os.path.basename(t_ori_file))
    
    # find t-6, t-12, t-18, t-24
    t_6 = h8_time - datetime.timedelta(hours=6)
    t_12 = h8_time - datetime.timedelta(hours=12)
    # t_18 = h8_time - datetime.timedelta(hours=18)
    # t_24 = h8_time - datetime.timedelta(hours=24)
    
    # transform to h8 file name
    #h8 file name example: NC_H08_20160101_0000_R21_FLDK.06001_06001.h5
    t_file = f'NC_H08_{h8_time.strftime("%Y%m%d")}_{h8_time.strftime("%H%M")}_R21_FLDK.06001_06001.h5'
    t_file = os.path.join(h8_ori_path, h8_time.strftime('%Y'), h8_time.strftime('%m').zfill(2), t_file)
    t_6_file = f'NC_H08_{t_6.strftime("%Y%m%d")}_{t_6.strftime("%H%M")}_R21_FLDK.06001_06001.h5'
    t_6_file = os.path.join(h8_ori_path, t_6.strftime('%Y'), t_6.strftime('%m').zfill(2), t_6_file)
    t_12_file = f'NC_H08_{t_12.strftime("%Y%m%d")}_{t_12.strftime("%H%M")}_R21_FLDK.06001_06001.h5'
    t_12_file = os.path.join(h8_ori_path, t_12.strftime('%Y'), t_12.strftime('%m').zfill(2), t_12_file)
    # t_18_file = f'NC_H08_{t_18.strftime("%Y%m%d")}_{t_18.strftime("%H%M")}_R21_FLDK.06001_06001.h5'
    # t_18_file = os.path.join(h8_ori_path, t_18.strftime('%Y'), t_18.strftime('%m').zfill(2), t_18_file)
    # t_24_file = f'NC_H08_{t_24.strftime("%Y%m%d")}_{t_24.strftime("%H%M")}_R21_FLDK.06001_06001.h5'
    # t_24_file = os.path.join(h8_ori_path, t_24.strftime('%Y'), t_24.strftime('%m').zfill(2), t_24_file)
    
    # read h8 file
    t_24_data, t_18_data, t_12_data, t_6_data, t_data = None, None, None, None, None
    # for file in [t_24_file, t_18_file, t_12_file, t_6_file, t_file]:#
    for file in [t_12_file, t_6_file, t_file]:#
        try:
            with h5py.File(file, 'r') as f:
                data = f['data'][:][1:, :-1, :-1]
                # if file == t_24_file:
                #     t_24_data = filter_data_np(data)
                # elif file == t_18_file:
                #     t_18_data = filter_data_np(data)
                if file == t_12_file:
                    t_12_data = filter_data_np(data)
                elif file == t_6_file:
                    t_6_data = filter_data_np(data)
                else:
                    t_data = filter_data_np(data)
        
        except Exception as e:
            # print(f'Error: {e}')
            # print(f'file: {file}')
            return None
    # if t_24_data is None or t_18_data is None or t_12_data is None or t_6_data is None or t_data is None:
    if t_12_data is None or t_6_data is None or t_data is None:
        return None
    else:
        # input_data = np.concatenate((t_24_data, t_18_data, t_12_data, t_6_data, t_data), axis=0)
        input_data = np.concatenate((t_12_data, t_6_data, t_data), axis=0)
        return input_data


def create_sample(params):
    h8_ori_path = params['data_path']['himawari8_wnp']
    # era5_uv_path = params['data_path']['era5_uv']
    era5_swh_path = params['data_path']['era5_swh']
    input_path = params['inversion_stage1']['input_path']
    target_path = params['inversion_stage1']['target_path']
    time_range = params['inversion_stage1']['time_range']
    
    input_size, target_size = params['inversion_stage1']['input_size'], params['inversion_stage1']['target_size']
    input_stride, target_stride = params['inversion_stage1']['input_stride'], params['inversion_stage1']['target_stride']
    
    sub_dir = ['train','valid'] #, 'valid', 'test'
    
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
                # mean_info_path = os.path.join(input_path, 'mean_info')
                
                # create_dir([temp_input_path, temp_target_path, mean_info_path])
                create_dir([temp_input_path, temp_target_path])
                
                pool = Pool(8)
                for h8_file in h8_files:
                    create_sample_one_file(h8_file, h8_ori_path, temp_h8_ori_month_path, era5_swh_path, 
                                              temp_input_path, temp_target_path,
                                              input_size, target_size, input_stride, target_stride)
                    # pool.apply_async(create_sample_one_file, args=(h8_file, temp_h8_ori_month_path, era5_uv_path, 
                    #                                                temp_input_path, temp_target_path, mean_info_path,
                    #                                                input_size, target_size, input_stride, target_stride))
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
    files = natsorted(files)
    
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
    # args_list = args_list[:100]
    
    with multiprocessing.Pool(processes=16) as pool: # 使用 multiprocessing.Pool 并行执行任务，并将结果收集到一个列表中。通过这种方式，你可以大大加快处理大量数据的速度
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
    era5_swh_path = params['data_path']['era5_swh']
    era5_norm_info = params['data_path']['era5_norm_info_swh']
    era5_files = os.listdir(era5_swh_path)
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
        files = os.listdir(os.path.join(era5_swh_path, file))
        temp_era5_swh_path = os.path.join(era5_swh_path, file)
        for f in files:
            era5_file = os.path.join(temp_era5_swh_path, f)
            with h5py.File(era5_file, 'r') as f:
                data = f['data'][:]
                data[data == -32767] = 0
                era5_data.append(data)
                f.close()
    era5_data = np.array(era5_data)
    
    # get mean and std from era5 data
    era5_mean = np.mean(era5_data, axis=(0, 1, 2))
    era5_std = np.std(era5_data, axis=(0, 1, 2))
    era5_max = np.max(era5_data, axis=(0, 1, 2))
    era5_min = np.min(era5_data, axis=(0, 1, 2))

    # save norm info to csv
    norm_info = pd.DataFrame({'mean': era5_mean, 'std': era5_std, 'max': era5_max, 'min': era5_min}, index=[0])
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
    era5_mean, era5_std = read_norm_info(info_type='era5_swh', variable=False)
    
    input_norm_info = (h8_mean, h8_std)
    target_norm_info = (era5_mean, era5_std)
    
    pool = Pool(8)
    
    sub_path = ['train'] #, 'valid', 'test'
    input_path = params['inversion_stage1']['input_path']
    target_path = params['inversion_stage1']['target_path']
    timerange = params['inversion_stage1']['time_range']
    for path in sub_path:
        # temp_timerange = timerange[path]
        name_path = path + '_norm'
        for i in range(2):
            print(f'Norm {path} {i}')
            root_path = os.path.join(input_path, path) if i == 0 else os.path.join(target_path, path)
            norm_path = os.path.join(input_path, name_path) if i == 0 else os.path.join(target_path, name_path)
            if not os.path.exists(norm_path):
                os.makedirs(norm_path)
            norm_info = input_norm_info if i == 0 else target_norm_info
            # norm_path = root_path
            files = natsorted(os.listdir(root_path))
            
            #files = [file for file in files if file.split('-')[0] in temp_timerange]
            
            for file in files:
                # norm_one_file(root_path, file, norm_info, norm_path)
                pool.apply_async(norm_one_file, args=(root_path, file, norm_info, norm_path))
    
    pool.close()
    pool.join()
    
def process_era5_swh(data_path):
    folders = os.listdir(data_path)
    for folder in folders:
        folder_path = os.path.join(data_path, folder)
        save_folder_path = os.path.join(data_path, folder + '_takeout_-32767')
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        # Process each folder here
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            save_file_path = os.path.join(save_folder_path, file)
            with h5py.File(file_path, 'r') as f:
                data = f['data'][:]
                data[data == -32767] = 0
                f.close()
            with h5py.File(save_file_path, 'w') as f:
                f.create_dataset('data', data=data, dtype=np.float32, compression='gzip')
                f.close()
                print(f'Save {file_path}')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    seed_everything()
    
    params = configs
    # reform_era5(params)
    create_sample(params)
    # cal_norm_info(params)
    # norm_h8_era5(params)
    
    # data_path = '/media/data7/lq/fanyan_data/data/inversion_swh_new/stage1/target'
    # process_era5_swh(data_path)