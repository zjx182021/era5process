"""
从原始的卫星数据中选出相应的时间、区域的数据
卫星: HY-2B、CFOSAT、Metop-A、Metop-B、Metop-C
时间：每个文件保存当前时刻后若干小时(temporal_res)的数据
区域: 15°S-15°N, 155°E-85°W
保存的数据：[[time, lat, lon, wind_speed], ...]
"""

import sys
sys.path.append('./preprocess')

import config as CFG
import os
from natsort import natsorted
from tqdm import tqdm
from tqdm import trange
from netCDF4 import Dataset
import time
import numpy as np
import datetime
import warnings
from read_sat_files import *
import sys

cfg = CFG.DefaultConfigure
warnings.filterwarnings('ignore')


def filter_files_by_id(sat_files, sat_name):
    """
    接收卫星的文件名，过滤掉重复轨道号的文件。
    :rtype: list
    """
    sat_new = []

    if sat_name == 'HY-2B':
        # HY-2B命名格式：H2B_OPER_GDR_2PC_0057_0306_20210101T213603_20210101T222817.h5
        nums = []
        for elem in sat_files:
            num_temp = elem.split('_')[-5]  # 轨道号，类似文件唯一标识，不应重复
            if num_temp not in nums:
                sat_new.append(elem)
                nums.append(num_temp)
            # 若重复，看后面这轨的时间是不是更长，若是则替换。
            elif num_temp == sat_new[-1].split('_')[-5]:
                if elem.split('_')[-6] > sat_new[-1].split('_')[-6]:
                    sat_new[-1] = elem

    elif sat_name == 'CFOSAT':
        # CFOSAT命名格式：CFO_EXPR_SCA_C_L2B_OR_20210601T003612_14333_125_32_owv.nc
        nums = []
        for elem in sat_files:
            num_temp = elem.split('_')[-5]
            if '_250_' in elem:
                sat_new.append(elem)
                nums.append(num_temp)

    return sat_new

def filter_files_by_time(sat, time, hours):
    """
    筛选出time时刻前两小时和后hours小时以内的文件。
    :rtype: list
    """
    sat_new = []
    # sat_ftime 和 sat_files 下标是对应的
    # sat[0]为sat_files， sat[1]为sat_ftime
    for i in range(len(sat[0])):
        # 判断两时间点的差值是否在-2~hours之间
        if (sat[1][i] - time).total_seconds() >= -2 * 3600 and (sat[1][i] - time).total_seconds() <= hours * 3600:
            sat_new.append(sat[0][i])

    return sat_new

def lon_tran(lon):
    """将HY-2B和CFOSAT卫星的经度值从(-180, 180)表达方式变为(0, 360)"""
    lon = np.where(lon < 0, lon + 360, lon)
    return lon

def filter_data_by_timearea(data, time, area, hours=3, h2b=False):
    """
    从data中筛选出距time时刻后hours小时内的、且经纬在area内的数据。
    :type time: datetime
    :param area: [下纬度，上纬度，左经度，右纬度]
    :type area: list
    :param h2b: 由于HY-2B的风速值要分别 /10 /100
    :type h2b: bool
    """
    temp_data = []

    for t in range(len(data[0])):
        # 每个时间都是一排数据的扫描时间，先过滤时间
        if 0 <= (data[0][t] - time).total_seconds() <= (3600 * hours):
            # 依次取出这排数据中的纬度值
        
            # 判断经纬度的范围
            if area[0] <= data[1][t] <= area[1] and area[2] <= data[2][t] <= area[3]:
                # 经纬度对应的波高是否是有效值
                # 首先把data[3]和data[4]转换成numpy array
                data[3] = np.ma.filled(data[3], -32767)
                data[4] = np.ma.filled(data[3], -32767)
                if float(data[3][t]) != -32767 and float(data[4][t]) != -32767:
                    if h2b:
                        temp_data.append([data[0][t], data[1][t], data[2][t],
                                            (data[3][t] ), (data[4][t])])
                    else:
                        temp_data.append([data[0][t], data[1][t], data[2][t],
                                            data[3][t], data[4][t]])

    return temp_data

def read_files(sat_files, sat_name, time, hours, area):
    """
    依次读取三个列表中的文件, 选出其中在time后hours时间内的、并且区域在area的数据。
    :return: 满足需求的数据，格式:[times, lats, lons, wind_speed, wind_dir]
    :rtype: list
    """
    data = []

    if sat_name == 'HY-2B':
        # HY-2B
        for file in sat_files:
            temp_data = read_h2b_file(file)  # [times, lats, lons, wind_speed, wind_dir]
            if not temp_data:
                continue

            # 将时间变为datetime格式
            temp_times = []
            for i in range(len(temp_data[0])):
                
                temp_time = datetime.datetime(2000, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=int(temp_data[0][i]))
                formatted_string = temp_time.strftime('%Y%m%dT%H:%M:%S')
                temp_times.append(datetime.datetime.strptime(formatted_string, '%Y%m%dT%H:%M:%S'))
            temp_data[0] = temp_times
            temp_data[2] = lon_tran(temp_data[2])

            # 筛选出time时刻前两小时和后hours小时以内的数据
            h2b_data = filter_data_by_timearea(temp_data, time, area, hours, h2b=True)
            data += h2b_data

    elif sat_name == 'CFOSAT':
        # CFOSAT
        for file in sat_files:
            temp_data = read_cfo_file(file)  # [times, lats, lons, wind_speed, wind_dir]
            if not temp_data:
                continue

            # 将时间变为datetime格式
            temp_data[0] = cfotime_to_datetime(temp_data[0])
            temp_data[2] = lon_tran(temp_data[2])

            cfo_data = filter_data_by_timearea(temp_data, time, area, hours)
            data += cfo_data

    data.sort(key=lambda x: (x[1], x[2]), reverse=False)
    data = np.array(data)
    return data


def spdir2uv(data):
    """
    将风速和风向转换为u和v分量
    """
    # data shape is (n, 5), where 5 is [times, lats, lons, wind_speed, wind_dir]
    spd, dir = data[:, -2].astype('float'), data[:, -1].astype('float')
    u = spd * np.sin(dir * np.pi / 180)
    v = spd * np.cos(dir * np.pi / 180)
    
    data[:, -2] = u
    data[:, -1] = v
    
    return data


def filter_sat_data(sat_name, sat_origin_dir, sat_filter_dir, months, area, filter_temporal_res):
    """
    过滤卫星数据，将原始卫星数据中的时间和空间信息过滤出来，存入对应的文件中。
    """
    for mon in months:
        print('正在处理{}月份的数据...'.format(mon))
        start = datetime.datetime.now()
        # 创建保存文件夹
        result_dir = os.path.join(sat_filter_dir, mon)
        os.makedirs(result_dir, exist_ok=True)

        # 卫星原始文件夹路径
        sat_month_dir = os.path.join(sat_origin_dir, mon)
        # if not os.path.exists(sat_month_dir):
        #     continue

        # 获取卫星原始文件
        sat_files = natsorted(os.listdir(sat_month_dir))
        print(f"total files num: {len(sat_files)}")
        # 过滤卫星原始文件
        # sat_files = filter_files_by_id(sat_files, sat_name)
        # print(f"after filter files num: {len(sat_files)}")
        # 将文件名中的时间转换为datatime格式
        sat_ftime = filetime_to_datatime(sat_files, sat_name)
        # 将文件路径加入到文件名中
        sat_files = [os.path.join(sat_month_dir, file) for file in sat_files]

        # 起始时刻为最后一个已保存的文件的时刻，最后这个文件要重新保存，因为可能上一次异常终止了
        # 或原始卫星数据的第一个时刻
        saved_files = natsorted(os.listdir(result_dir))
        if len(saved_files):
            startime = datetime.datetime.strptime(saved_files[-1].split('.')[0], "%Y-%m-%d-%H")
        else:
            # startime = sat_ftime[0]
            startime = datetime.datetime.strptime('2021-'+mon+'-01-00', "%Y-%m-%d-%H")

        temp_time = startime

        while temp_time.month == startime.month:  #处理每个月份的原始卫星数据
            # 找出temp_time后hours小时的文件
            sat_temp_files = filter_files_by_time([sat_files, sat_ftime], temp_time, filter_temporal_res)

            # 读取每个文件中的数据，过滤出相应的时间和空间的数据
            data = read_files(sat_temp_files, sat_name, temp_time, filter_temporal_res, area)

            # 保存数据
            if len(data):
                # transform spd and dir to u and v
                # data = spdir2uv(data)
                # now data is (n, 5), where 5 is [times, lats, lons, swh_ku, swh_c]
                
                new_file = str(temp_time.date()) + "-{:0>2d}".format(temp_time.hour)
                np.save(os.path.join(result_dir, new_file), data)
                print("%s is saved" % new_file)
            # 至此，已经处理完了temp_time时刻的数据，接下来处理下一个temp_time时刻的数据
            temp_time = temp_time + datetime.timedelta(hours=filter_temporal_res)  # 向后推filter_temporal_res小时
        
        end = datetime.datetime.now()
        print('处理{}月份的数据完成，耗时{}秒'.format(mon, (end - start).seconds))


if __name__ == "__main__":
    
    for sat_name in cfg.sat_names:
        print('正在处理{}卫星的数据...'.format(sat_name))
        if sat_name == 'HY-2B':
            sat_origin_dir = cfg.sat_ori_path_h2b 
            sat_filter_dir = cfg.sat_filter_path_h2b
        elif sat_name == 'CFOSAT':
            sat_origin_dir = cfg.cfo_origin_dir
            sat_filter_dir = cfg.cfo_filter_dir
        filter_sat_data(sat_name, sat_origin_dir, sat_filter_dir, cfg.months, cfg.area, cfg.filter_temporal_res)
