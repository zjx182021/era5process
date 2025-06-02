import gc
from io import StringIO
import sys

sys.path.append('/media/data5/wyp/lq/SWH_Retrieval')
from utils.utils import *
from utils.read_config import configs
import os
import numpy as np
import h5py
from netCDF4 import Dataset # type: ignore
import pandas as pd
import datetime
from tqdm import tqdm
from natsort import natsorted
import scipy.ndimage


def reform_data(era5_ori_path, era5_reform_path):
    """
    Reform the origin data, and save the reform data.
    :param era5_ori_path: the path of the origin data, maybe have more than one file.
    :param era5_reform_path: the path to save the reform data, each file contains one time's uv.
    :return: None
    """
    era5_ori_files = os.listdir(era5_ori_path)
    era5_ori_files.sort()
    
    for file in era5_ori_files:
        print("Reform " + file)
        file_path = os.path.join(era5_ori_path, file)
        dataset = Dataset(file_path, 'r')
        
        year = file.split('_')[3]
        if int(year) == 2021:
            temp_era5_reform_path = os.path.join(era5_reform_path, "test")
        else:
            temp_era5_reform_path = os.path.join(era5_reform_path, "train")
        if not os.path.exists(temp_era5_reform_path):
            os.mkdir(temp_era5_reform_path)
        
        era5_time = dataset.variables['valid_time'][:].data
        era5_time = era5_time_to_datetime(era5_time)
        
        # swh = dataset.variables['swh'][:].data
        # swh = dataset.variables['mwd'][:].data
        swh = dataset.variables['mwp'][:].data
        

        # reform the data
        for i in tqdm(range(len(era5_time))): # 对于每一个时刻的波高数据
            # if era5_time[i] < time_range[0] or era5_time[i] > time_range[1]:
            #     continue
            # if era5_time[i].hour % 3 != 0:
            #     continue
            
            era5_swh = np.array(swh[i], dtype=np.float32)[:-1, :-1]  # 每一个时刻的波高数据
            era5_swh[np.isnan(era5_swh)] = 0

            # mask = np.where(era5_swh == 0, 0, 1)
            # np.save('/media/data5/wyp/lq/Point_To_Area_Forecast/data_Hawaiian/mask_swh.npy', mask)
            
            
            file_name = datetime.datetime.strftime(era5_time[i], '%Y-%m-%d-%H')
            era5_swh_file = os.path.join(temp_era5_reform_path, file_name)
            with h5py.File(os.path.join(era5_swh_file + '.h5'), 'w') as f:
                f.create_dataset('data', data=era5_swh, compression='gzip', dtype=np.float32)
                f.close()
                print('save ' + file_name)


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
        if int(file.split('-')[0]) < 2000:
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


def create_sample_index(time_range, sample_index, samples_path, need_flip=False, input_time_steps=1, output_time_steps=1, time_stride=6, move_stride=3):
    """
    Create the index of the samples.
    :param time_range: the time range of the samples.
    :param sample_index_file: the file to save the index.
    :param samples_path: the path of the samples.
    :return: None
    """
    sub_dir = ['train', 'test']

    for dir in sub_dir:
        temp_time_range = time_range[dir]
        temp_sample_index_file = sample_index[dir]
        
        start_time = datetime.datetime.strptime(temp_time_range[0], '%Y-%m-%d-%H')
        end_time = datetime.datetime.strptime(temp_time_range[1], '%Y-%m-%d-%H')
        
        sample_index_list = []
        while start_time + datetime.timedelta(hours=(input_time_steps+output_time_steps-1)*time_stride) <= end_time:
            temp_input_index, temp_target_index = [], []
            for i in range(input_time_steps):
                temp_input_time = start_time + datetime.timedelta(hours=time_stride * i)
                temp_input_time = datetime.datetime.strftime(temp_input_time, '%Y-%m-%d-%H')
                temp_input = os.path.join(temp_input_time.split('-')[0], temp_input_time+'.h5')
                
                # whether the sample exists
                if not os.path.exists(os.path.join(samples_path, temp_input)):
                    break
                else:
                    temp_input_index.append(temp_input)
                    
            for i in range(output_time_steps):
                temp_target_time = start_time + datetime.timedelta(hours=time_stride * (input_time_steps + i))
                temp_target_time = datetime.datetime.strftime(temp_target_time, '%Y-%m-%d-%H')
                temp_target = os.path.join(temp_target_time.split('-')[0], temp_target_time+'.h5')
                
                # whether the sample exists
                # if not os.path.exists(os.path.join(samples_path, temp_target)):
                #     break
                # else:
                temp_target_index.append(temp_target)
            
            if len(temp_input_index) != input_time_steps or len(temp_target_index) != output_time_steps:
                pass
            else:
                sample_index_list.append([','.join(temp_input_index), ','.join(temp_target_index)])
            
            start_time += datetime.timedelta(hours=move_stride)
        
        df = pd.DataFrame(sample_index_list, columns=['input_time', 'target_time'])
        df.to_csv(temp_sample_index_file, index=False)
    
        print(f'Save {dir} sample index to {temp_sample_index_file}')


def create_sample(era5_reform_path, samples_path, time_range=None, need_flip=False):
    """
    Create the samples from the uv data.
    Need to normalize and filp the data.
    :param era5_reform_path: the path of the uv data.
    :param samples_path: the path to save the samples.
    :param time_range: the time range of the samples.
    :param need_flip: whether to flip the train data to increase the samples.
    :return: None
    """

    train_save_path = os.path.join(samples_path, 'train')
    test_save_path = os.path.join(samples_path, 'test')
    if not os.path.exists(train_save_path):
        os.mkdir(train_save_path)
    if not os.path.exists(test_save_path):
        os.mkdir(test_save_path)

    years = natsorted(os.listdir(era5_reform_path))
    for year in years:
        if year == '2021':
            temp_save_path = test_save_path
        else:
            temp_save_path = train_save_path

        temp_era5_reform_path = os.path.join(era5_reform_path, year)
        era5_swh_files = natsorted(os.listdir(temp_era5_reform_path))
        for file in tqdm(era5_swh_files):
            file_path = os.path.join(temp_era5_reform_path, file)
            with h5py.File(file_path, 'r') as f:
                era5_swh = f['data'][:]
                f.close()
            era5_swh = era5_swh.reshape(1, 24, 24)
            with h5py.File(os.path.join(temp_save_path, file), 'w') as f:
                f.create_dataset('data', data=era5_swh, compression='gzip', dtype=np.float32)
                f.close()
            print(f'Save {file} to {temp_save_path}/{file}')
            
            

        
        


def pro_inversion_data(inversion_model_output_path, inversion_sample_path):
    """
    Process the inversion model output data. reform and normalize.
    :param inversion_model_output_path: the path of the inversion model output data.
    :param inversion_sample_path: the path to save the inversion model output data.
    :return: None
    """
    sub_dirs = os.listdir(inversion_model_output_path)
    
    all_inversion_files = []
    for sub_dir in sub_dirs:
        temp_sub_dir = os.path.join(inversion_model_output_path, sub_dir)
        inversion_files = os.listdir(temp_sub_dir)
        
        for file in inversion_files:
            all_inversion_files.append(os.path.join(temp_sub_dir, file))
    
    # read_norm_info
    norm_info = read_norm_info()
    
    for file in tqdm(all_inversion_files):
        with h5py.File(file, 'r') as f:
            inversion_data = f['data'][:] # type: ignore
            f.close()
        
        inversion_data = (inversion_data - norm_info[0]) / norm_info[1]
        
        if file[-5] == '_':
            new_file_name = file.split('/')[-1].split('_')[0] + '.h5'
        else:
            new_file_name = file.split('/')[-1].split('_')[0]
            
        temp_year = new_file_name.split('-')[0]
        
        temp_inversion_sample_path = os.path.join(inversion_sample_path, temp_year)
        if not os.path.exists(temp_inversion_sample_path):
            os.mkdir(temp_inversion_sample_path)
        
        with h5py.File(os.path.join(temp_inversion_sample_path, new_file_name), 'w') as f:
            f.create_dataset('data', data=inversion_data, dtype=np.float32)
            f.close()
        

def chazhi_data(ori_path, source_path):
    """
    Interpolate the original data to 240*240.
    :param ori_path: the path of the original data.
    :param source_path: the path to save the interpolated data.
    :return: None
    """
    ori_files = os.listdir(ori_path)
    ori_files.sort()
    for file in ori_files:
        print("Interpolate " + file)
        file_path = os.path.join(ori_path, file)
        new_file_name = file[0:4] + '-' + file[4:6] + '-' + file[6:8] + '-' + file[8:10]
        # file_date = datetime.datetime.strptime(file.split('_')[3], '%Y%m%d%H')
        # new_file_name = file.strftime('%Y-%m-%d-%H') + '.h5'
        temp_source_path = os.path.join(source_path, new_file_name)
        # if not os.path.exists(temp_source_path):
        #     os.mkdir(temp_source_path)
        # year = file.split('_')[3]
        # temp_source_path = os.path.join(source_path, year)
        # if not os.path.exists(temp_source_path):
        #     os.mkdir(temp_source_path)
        
        # era5_time = dataset.variables['time'][:].data
        # era5_time = era5_time_to_datetime(era5_time)
        
        # swh = dataset.variables['swh'][:].data
        with h5py.File(file_path, 'r') as f:
            ec_ori_data = f['data'][:]
            f.close()
        
        ec_ori_data = ec_ori_data[:-1, :-1]
        
        ec_data = scipy.ndimage.zoom(ec_ori_data, (150 / ec_ori_data.shape[0], 180 / ec_ori_data.shape[1]), order=1)
        # ec_data = torch.nn.functional.interpolate(ec_ori_data, size=(150, 180), mode='bilinear', align_corners=False)
        # ec_data = torch.tensor(ec_data, dtype=torch.float32)
        ec_data = np.array(ec_data, dtype=np.float32)
        
        with h5py.File(os.path.join(temp_source_path + '.h5'), 'w') as f:
                f.create_dataset('data', data=ec_data, compression='gzip', dtype=np.float32)
                f.close()
                print('save ' + new_file_name + '.h5')


def clip(data):
    # 网格参数
    lat_start, lat_end = 0, 60
    lon_start, lon_end = 100, 160
    grid_size = 120

    # 计算分辨率
    lat_resolution = (lat_end - lat_start) / grid_size
    lon_resolution = (lon_end - lon_start) / grid_size

    # 浮标点的经纬度
    buoy_lat = 13.5
    buoy_lon = 145

    # 计算浮标点的网格索引
    lat_index = int((buoy_lat - lat_start) / lat_resolution)
    lon_index = int((buoy_lon - lon_start) / lon_resolution)

    # 确定8x8网格的索引范围
    lat_start_index = max(0, lat_index - 2)  # 防止索引超出边界
    lat_end_index = min(grid_size - 1, lat_index + 1)
    lon_start_index = max(0, lon_index - 2)
    lon_end_index = min(grid_size - 1, lon_index + 1)

    # 由于纬度是从上到下，所以纬度索引要反转
    # 裁剪数据时，纵坐标（纬度）需要反向处理，因为纬度值越小索引越大
    clipped_data = data[grid_size - lat_end_index - 1: grid_size - lat_start_index, lon_start_index: lon_end_index + 1]

    return clipped_data



def clip_data(era5_reform_path, era5_clip_path):
    """
    Clip the data from the era5 reform path and save it to the era5 clip path.
    :param era5_reform_path: the path of the reform data.
    :param era5_clip_path: the path to save the clipped data.
    :return: None
    """
    train_save_path = os.path.join(era5_clip_path, 'train')
    test_save_path = os.path.join(era5_clip_path, 'test')
    if not os.path.exists(train_save_path):
        os.mkdir(train_save_path)
    if not os.path.exists(test_save_path):
        os.mkdir(test_save_path)
    years = natsorted(os.listdir(era5_reform_path))
    for year in years:
        print('Clip ' + year)
        if int(year) == 2023:
            save_path = test_save_path
        else:
            save_path = train_save_path
        temp_era5_reform_path = os.path.join(era5_reform_path, year)
        era5_swh_files = natsorted(os.listdir(temp_era5_reform_path))
        for file in tqdm(era5_swh_files):
            with h5py.File(os.path.join(temp_era5_reform_path, file), 'r') as f:
                era5_data = f['data'][:]
                f.close()

            era5_data = clip(era5_data)
            with h5py.File(os.path.join(save_path, file), 'w') as f:
                f.create_dataset('data', data=era5_data, compression='gzip', dtype=np.float32)
                f.close()
            print('save ' + file)



def reform_buoy_data(origin_buoy_path, reform_buoy_path):
    """
    Reform the origin buoy data, and save the reform data.
    :param origin_buoy_path: the path of the origin buoy data, maybe have more than one file.
    :param reform_buoy_path: the path to save the reform buoy data, each file contains one time's uv.
    :return: None
    """
    buoy_files_dir = natsorted(os.listdir(origin_buoy_path))
    for file_dir in tqdm(buoy_files_dir):
        print("Reform " + file_dir)
        temp_file_path = os.path.join(origin_buoy_path, file_dir)
        # save_file_path = os.path.join(reform_buoy_path, file_dir[6:])
        # dataset = Dataset(file_path, 'r')
        year = file_dir[6:]
        # if int(year) < 2018:
        #     continue
        temp_reform_buoy_path = os.path.join(reform_buoy_path, year)
        if not os.path.exists(temp_reform_buoy_path):
            os.mkdir(temp_reform_buoy_path)
        txt_files = natsorted(os.listdir(temp_file_path))
        for file in txt_files:
            txt_file_path = os.path.join(temp_file_path, file)
            with open(txt_file_path, 'r') as f:
                lines = f.readlines()
                # data = []
                for line in lines[2:]:
                    temp_data = line.split()
                    year_, month_, day_, hour_, minute_, swh = temp_data[0], temp_data[1], temp_data[2], temp_data[3], temp_data[4], temp_data[8]
                    if int(hour_) % 1 == 0:
                        save_file_name = year_ + '-' + month_ + '-' + day_ + '-' + hour_ + '-' + minute_
                        save_file_path = os.path.join(temp_reform_buoy_path, save_file_name + '.h5')
                        # data.append([float(temp_data[1]), float(temp_data[2])])
                        with h5py.File(save_file_path, 'w') as f:
                            f.create_dataset('data', data=float(swh), dtype=np.float32)
                            f.close()
                            print('save ' + save_file_name)
                
                
def create_buoy_sample(reform_buoy51101_path, reform_buoy51001_path, reform_buoy51000_path,\
                       reform_buoy51208_path, reform_buoy51201_path, reform_buoy51206_path,\
                       reform_buoy51003_path, reform_buoy51002_path, reform_buoy51004_path,\
                        buoy_samples_path):
    """
    Create the samples from the buoy data.
    :param reform_buoy_path: the path of the reform buoy data.
    :param buoy_samples_path: the path to save the samples.
    :return: None
    """
    train_save_path = os.path.join(buoy_samples_path, 'train')
    test_save_path = os.path.join(buoy_samples_path, 'test')
    if not os.path.exists(train_save_path):
        os.mkdir(train_save_path)
    if not os.path.exists(test_save_path):
        os.mkdir(test_save_path)
    
    yesrs = natsorted(os.listdir(reform_buoy51201_path))
    for year in yesrs:
        print('Create samples for ' + year)
        if int(year) == 2021:
            save_path = test_save_path
        else:
            save_path = train_save_path
        # save_path = os.path.join(buoy_samples_path, year)
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)


        temp_reform_buoy51101_path = os.path.join(reform_buoy51101_path, year)
        temp_reform_buoy51001_path = os.path.join(reform_buoy51001_path, year)
        temp_reform_buoy51000_path = os.path.join(reform_buoy51000_path, year)
        temp_reform_buoy51208_path = os.path.join(reform_buoy51208_path, year)
        temp_reform_buoy51201_path = os.path.join(reform_buoy51201_path, year)
        temp_reform_buoy51206_path = os.path.join(reform_buoy51206_path, year)
        temp_reform_buoy51003_path = os.path.join(reform_buoy51003_path, year)
        temp_reform_buoy51002_path = os.path.join(reform_buoy51002_path, year)
        temp_reform_buoy51004_path = os.path.join(reform_buoy51004_path, year)
        buoy_files = natsorted(os.listdir(temp_reform_buoy51000_path))
        for file in tqdm(buoy_files):
            temp_buoy51101_file_path = os.path.join(temp_reform_buoy51101_path, file)
            temp_buoy51001_file_path = os.path.join(temp_reform_buoy51001_path, file)
            temp_buoy51000_file_path = os.path.join(temp_reform_buoy51000_path, file)
            temp_buoy51208_file_path = os.path.join(temp_reform_buoy51208_path, file)
            temp_buoy51201_file_path = os.path.join(temp_reform_buoy51201_path, file)
            temp_buoy51206_file_path = os.path.join(temp_reform_buoy51206_path, file)
            temp_buoy51003_file_path = os.path.join(temp_reform_buoy51003_path, file)
            temp_buoy51002_file_path = os.path.join(temp_reform_buoy51002_path, file)
            temp_buoy51004_file_path = os.path.join(temp_reform_buoy51004_path, file)
            if os.path.exists(temp_buoy51101_file_path) and os.path.exists(temp_buoy51001_file_path) and os.path.exists(temp_buoy51000_file_path) \
                and os.path.exists(temp_buoy51208_file_path) and os.path.exists(temp_buoy51201_file_path) and os.path.exists(temp_buoy51206_file_path) \
                    and os.path.exists(temp_buoy51003_file_path) and os.path.exists(temp_buoy51002_file_path) and os.path.exists(temp_buoy51004_file_path):
                with h5py.File(temp_buoy51101_file_path, 'r') as f:
                    buoy51101_data = f['data'][()]
                    f.close()
                with h5py.File(temp_buoy51001_file_path, 'r') as f:
                    buoy51001_data = f['data'][()]
                    f.close()
                with h5py.File(temp_buoy51000_file_path, 'r') as f:
                    buoy51000_data = f['data'][()]
                    f.close()
                with h5py.File(temp_buoy51208_file_path, 'r') as f:
                    buoy51208_data = f['data'][()]
                    f.close()
                with h5py.File(temp_buoy51201_file_path, 'r') as f:
                    buoy51201_data = f['data'][()]
                    f.close()
                with h5py.File(temp_buoy51206_file_path, 'r') as f:
                    buoy51206_data = f['data'][()]
                    f.close()
                with h5py.File(temp_buoy51003_file_path, 'r') as f:
                    buoy51003_data = f['data'][()]
                    f.close()
                with h5py.File(temp_buoy51002_file_path, 'r') as f:
                    buoy51002_data = f['data'][()]
                    f.close()
                with h5py.File(temp_buoy51004_file_path, 'r') as f:
                    buoy51004_data = f['data'][()]
                    f.close()
                
                buoy_data = np.array([buoy51101_data, buoy51001_data, buoy51000_data, buoy51208_data, buoy51201_data, buoy51206_data, buoy51003_data, buoy51002_data, buoy51004_data], dtype=np.float32)
                buoy_data = buoy_data.reshape(1, 3, 3)
                with h5py.File(os.path.join(save_path, file), 'w') as f:
                    f.create_dataset('data', data=buoy_data, dtype=np.float32)
                    f.close()
                    print('save ' + file)
                



def create_sample_index_optimized(origin_path, sample_index, batch_size=None):
    """
    Create the sample index for the origin data with batch saving into a single CSV file.
    :param origin_path: the path of the origin data.
    :param sample_index: the path to save the sample index (single file).
    :param batch_size: the number of samples to save in each batch.
    :return: None
    """
    sub_dirs = ['train', 'test']
    for sub_dir in sub_dirs:
        temp_origin_path = os.path.join(origin_path, sub_dir)
        temp_sample_index_file = sample_index[sub_dir]
        origin_files = natsorted(os.listdir(temp_origin_path))

        sample_index_list = []
        origin_files_set = set(origin_files)  # 转换为集合，加快查找速度

        # Clear the CSV file if it exists to start fresh
        open(temp_sample_index_file, 'w').close()
        write_header = True

        for idx, file in enumerate(tqdm(origin_files)):
            time_str = file.split('.')[0]
            try:
                # Convert time string to datetime object
                time_dt = datetime.datetime.strptime(time_str, '%Y-%m-%d-%H')
            except ValueError:
                continue

            # Calculate previous and future time strings
            # time_offsets = [-12,-9,-6,-3,0]
            time_offsets = [-18, -12, -6, 0, 6, 12, 18, 24]  
            # time_offsets = [0]
            time_files = [
                (time_dt + datetime.timedelta(hours=offset)).strftime('%Y-%m-%d-%H') + '.h5'
                for offset in time_offsets
            ]

            # Check if all required files exist
            if all(file_name in origin_files_set for file_name in time_files):
                input_files = time_files[:4]
                # target_files = [time_str + '.h5']
                target_files = time_files[4:]
                sample_index_list.append([','.join(input_files), ','.join(target_files)])

            # Save to CSV in batches
            if len(sample_index_list) >= batch_size or idx == len(origin_files) - 1:
                batch_df = pd.DataFrame(sample_index_list, columns=['input_time', 'target_time'])
                
                # 使用 StringIO 提高写入效率
                buffer = StringIO()
                batch_df.to_csv(buffer, index=False, header=write_header)
                with open(temp_sample_index_file, 'a') as f:
                    f.write(buffer.getvalue())

                print(f'Save {sub_dir} sample index to {temp_sample_index_file} ({idx + 1}/{len(origin_files)})')
                buffer.close()

                write_header = False  # 表头只写入一次
                sample_index_list.clear()  # Clear the list for the next batch
                # 调整垃圾回收频率
            if idx % (batch_size * 10) == 0:
                gc.collect()

        gc.collect()




if __name__=='__main__':
    data_path = configs['data_path']
    # origin_buoy_path = data_path['origin_buoy']
    # reform_buoy_path = data_path['reform_buoy']
    # reform_buoy51101_path = data_path['reform_buoy51101']
    # reform_buoy51001_path = data_path['reform_buoy51001']
    # reform_buoy51000_path = data_path['reform_buoy51000']
    # reform_buoy51208_path = data_path['reform_buoy51208']
    # reform_buoy51201_path = data_path['reform_buoy51201']
    # reform_buoy51206_path = data_path['reform_buoy51206']
    # reform_buoy51003_path = data_path['reform_buoy51003']
    # reform_buoy51002_path = data_path['reform_buoy51002']
    # reform_buoy51004_path = data_path['reform_buoy51004']
    # buoy_samples_path = data_path['buoy_samples']
    era5_ori_path = data_path['era5_ori']
    # era5_clip_path = data_path['era5_clip']
    era5_reform_path = data_path['era5_swh']
    # mean_std_saved_file = data_path['mean_std']
    # time_range = configs['time_range']
    # ori_path = '/media/data5/wyp/lq/SUPER_Resolution_Wave/data/ec_data_600x720'
    # source_path = '/media/data5/wyp/lq/SUPER_Resolution_Wave/data/ec_data_150x180'
    
    # chazhi
    # chazhi_data(ori_path, source_path)
    
    # reform the data
    # reform_data(era5_ori_path, era5_reform_path)
    
    # calculate the mean and std
    # cal_mean_std(era5_reform_path, mean_std_saved_file, time_range['train'])
    
    # create the inversion model output samples
    # inversion_model_output_path = data_path['inversion_model_output_path']
    # inversion_sample_path = data_path['inversion_samples']
    # pro_inversion_data(inversion_model_output_path, inversion_sample_path)
    

    # create the samples
    # samples_path = data_path['samples']
    # create_sample(era5_reform_path, samples_path)


    
    # clip the swh data
    # clip_data(era5_reform_path, era5_clip_path)


    # preprocess the buoy data
    # reform_buoy_data(origin_buoy_path, reform_buoy_path)


    # create the buoy samples
    # create_buoy_sample(reform_buoy51101_path, reform_buoy51001_path, reform_buoy51000_path,\
    #                    reform_buoy51208_path, reform_buoy51201_path, reform_buoy51206_path,\
    #                    reform_buoy51003_path, reform_buoy51002_path, reform_buoy51004_path,\
    #                     buoy_samples_path)



    # create the sample index
    origin_path = '/media/data5/wyp/lq/SWH_Retrieval/data_era5_80year/era5_swh'
    sample_index = data_path['sample_index']
    create_sample_index_optimized(origin_path, sample_index, batch_size=5000)








