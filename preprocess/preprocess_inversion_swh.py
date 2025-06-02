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
from netCDF4 import Dataset
from multiprocessing import Pool
import multiprocessing
from itertools import product

from tqdm import tqdm

from utils.read_config import configs
from utils.utils import *


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
def reform_era5(params):
    """
    read era5 uv from era5 file, and save one by one, each file contains one time's uv
    """
    era5_swh_path = params['data_path']['era5_swh']
    era5_ori_path = params['data_path']['era5_ori']
    files = [file for file in os.listdir(era5_ori_path) if '20S' not in file]
    for file in natsorted(files):
        print('Read ' + file)
        era5_ori_file = os.path.join(era5_ori_path, file)
        era5_ori_data = Dataset(era5_ori_file, 'r')
        era5_time = era5_time_to_datetime(era5_ori_data.variables['valid_time'][:].data)
        era5_ori_swh = era5_ori_data.variables['mwp'][:].data
        
        
        for i in range(len(era5_time)):
            
            era5_swh = np.array(era5_ori_swh[i:i+1], dtype=np.float32)[:, :-1, :-1]
            # if era5_swh 中的数为-32767，则表示该点没有数据，将其置为0
            
            era5_swh[np.isnan(era5_swh)] = 0

            file_name = datetime.datetime.strftime(era5_time[i], '%Y-%m-%d-%H')
            era5_swh_file = os.path.join(era5_swh_path, file_name + '.h5')
            with h5py.File(era5_swh_file, 'w') as f:
                f.create_dataset('data', data=era5_swh, compression='gzip', dtype=np.float32)
                f.close()
                print('save ' + file_name)



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
    # new_h8_data = interpolate_data(h8_data)
    # now the shape of new_h8_data is (2, 240, 240)
    
    # normalize data
    # new_h8_data = (new_h8_data - norm_info[0]) / norm_info[1]
    h8_data = (h8_data - norm_info[0]) / norm_info[1]
    h8_data = h8_data.reshape(2, 3000, 3000)
    # save data
    file_name = h8_time.strftime('%Y-%m-%d-%H')

    saved_file = os.path.join(saved_path, file_name+'.h5')
    
    with h5py.File(saved_file, 'w') as f:
        f.create_dataset('data', data=h8_data, compression='gzip', dtype=np.float32)
        f.close()
    
    print(f'Save file: {saved_file}, cost time: {time.time() - start_time:.2f}s')


def get_norm_h8_data(h8_ori_path, time_range, saved_path):
    # years = natsorted(os.listdir(h8_ori_path))
    years = ['2016']
    sub_pathes = ['train', 'test']
    
    norm_info = read_norm_info(info_type='h8', variable=True, need_batch=True)
    
    for year in years:
        for sub_path in sub_pathes:
            if int(year) in time_range[sub_path]:
                input_h8_path = os.path.join(saved_path, 'h8_3000', sub_path)
                create_dir(input_h8_path)
                break
        
        # temp_year_path = os.path.join(h8_ori_path, year)
        temp_year_path = os.path.join(h8_ori_path)
        months = natsorted(os.listdir(temp_year_path))
        
        for month in months:
            temp_month_path = os.path.join(temp_year_path, month)
            h8_files = natsorted(os.listdir(temp_month_path))
            
            pool = Pool(4)
        
            for h8_file in h8_files:
                get_h8_data_one_file(h8_file, temp_month_path, norm_info, input_h8_path)
            #     pool.apply_async(get_h8_data_one_file, args=(h8_file, temp_month_path, norm_info, input_h8_path))
            # pool.close()
            # pool.join()


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


def cal_mean_std(era5_reform_path, mean_std_saved_file, train_time_range):
    """
    Calculate the mean and std of the era5 data (only the train time range).
    :param era5_reform_path: the path of the uv data.
    :param mean_std_saved_file: the file to save the mean and std.
    :param train_time_range: the train time range.
    :return: None
    """
    print('Calculate the mean and std of the era5 data')
    # years = natsorted(os.listdir(era5_reform_path))
    
    # train time range last 10 years
    # train_end_year = int(train_time_range[1].split('-')[0])
    # train_start_year = int(train_time_range[0].split('-')[0])
    # start_year = train_end_year - 10 if train_end_year - 10 > train_start_year else train_start_year
    
    era5_data = []
    
    # if int(year) not in range(start_year, train_end_year+1):
    #     continue
    
    # temp_era5_reform_path = os.path.join(era5_reform_path, year)
    
    era5_swh_files = natsorted(os.listdir(era5_reform_path))

    for file in tqdm(era5_swh_files):
        if int(file.split('-')[0]) == 2021:
            continue
        
        file_path = os.path.join(era5_reform_path, file)
        with h5py.File(file_path, 'r') as f:
            # era5_swh = f['data'][:] # type: ignore
            era5_swh = f['data'][()]
            # invalid_value = -32767
            # mask_data = np.where(era5_swh == invalid_value, 0, 1)
            # era5_swh = era5_swh * mask_data
            f.close()
        era5_data.append(era5_swh)
    era5_data = np.array(era5_data)
    era5_data[era5_data < 0] = 0  # 将小于零的点置零
    
    # get mean and std from era5 data
    
    # non_zero_mask = era5_data != 0
    # era5_mean = np.mean(era5_data[non_zero_mask], axis=0)
    # era5_std = np.std(era5_data[non_zero_mask], axis=0)
    # era5_max = np.max(era5_data[non_zero_mask], axis=0)
    # era5_min = np.min(era5_data[non_zero_mask], axis=0)

    # era5_mean = np.mean(era5_data, axis=(0,1,2))
    # era5_std = np.std(era5_data, axis=(0,1,2))
    # era5_max = np.max(era5_data, axis=(0,1,2))
    # era5_min = np.min(era5_data, axis=(0,1,2))
    era5_mean = np.mean(era5_data, axis=(0,2,3))
    era5_std = np.std(era5_data, axis=(0,2,3))
    era5_max = np.max(era5_data, axis=(0,2,3))
    era5_min = np.min(era5_data, axis=(0,2,3))

    
    # save the mean and std, using pandas DataFrame
    df = pd.DataFrame({'mean': era5_mean, 'std': era5_std, 'max': era5_max, 'min': era5_min})
    df.to_csv(mean_std_saved_file, index=['swh']) # type: ignore
    print(f'Save norm info to {mean_std_saved_file}')
    
    
def norm_era5_swh(era5_swh_path, saved_path):
    """
    normalize era5 uv
    """
    print('Start normalize era5 uv data (inversion stage 2 target).')
    # create_dir(saved_path)
    
    norm_info = read_norm_info(info_type='era5_swh',variable=False)
    
    era5_files = natsorted(os.listdir(era5_swh_path))
    for era5_file in tqdm(era5_files):
        
        era5_nc = h5py.File(os.path.join(era5_swh_path, era5_file), 'r')
        era5_data = era5_nc['data'][:].astype(np.float32)
        era5_nc.close()
        
        era5_data = (era5_data - norm_info[0]) / norm_info[1]

        saved_file = os.path.join(saved_path, 'valid', era5_file)
        with h5py.File(saved_file, 'w') as f:
            f.create_dataset('data', data=era5_data)
            f.close()


if __name__ == '__main__':
    config = configs
    
    seed_everything()
    
    h8_ori_path = config['data_path']['himawari8_wnp']
    time_range = config['inversion_swh_0']['time_range']
    input_path = config['inversion_swh_0']['input_path']
    
    # get_norm_h8_data(h8_ori_path, time_range, input_path)
    
    # swint_output_path = config['inversion_stage1']['output_path']
    # norm_swint_output(swint_output_path, input_path)
    # reform_era5(config)

    era5_reform_path = config['data_path']['era5_swh']
    mean_std_saved_file = config['data_path']['mean_std_path']
    cal_mean_std(era5_reform_path, mean_std_saved_file, time_range['train'])

    # era5_swh_path = config['data_path']['era5_swh']
    # target_path = config['inversion_swh_0']['target_path']
    # norm_era5_swh(era5_swh_path, target_path)