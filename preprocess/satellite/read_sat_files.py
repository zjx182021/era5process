"""读取卫星数据"""
from netCDF4 import Dataset
import h5py
import os
import datetime
import numpy as np


def read_h2b_file(file_name):
    try:
        h5data = Dataset(file_name)
    except OSError:
        print(file_name, "can't read !")
        return None
    # keys = list(h5data.keys())

    times_sec = h5data['time'][:]  # (2352,)
    lats = h5data['lat'][:]  # (2352, 76)
    lons = h5data['lon'][:]  # (1624, 76)
    swh_ku = h5data['swh_ku'][:]  # (1624, 76)
    swh_c = h5data['swh_c'][:]  # (1624, 76)
    swh_numval_ku = h5data['swh_numval_ku'][:]  # (1624, 76)
    
    # 关闭文件
    h5data.close()
    # times_day = np.ma.filled(times_day)
    
    return [times_sec, lats, lons, swh_ku, swh_c]

def read_cfo_file(file_name):
    try:
        nc_obj = Dataset(file_name)
    except OSError:
        print(file_name, "can't read !")
        return None
    lats = nc_obj.variables['wvc_lat'][:]  # (3440, 84)
    lons = nc_obj.variables['wvc_lon'][:]  # (3440, 84)
    wind_dir = nc_obj.variables['wind_dir_selection'][:]  # (3440, 84)
    wind_speed = nc_obj.variables['wind_speed_selection'][:]  # (3440, 84)
    times = nc_obj.variables['row_time'][:]  # (3440, 20)
    
    # 关闭文件
    nc_obj.close()
    return [times, lats, lons, wind_speed, wind_dir]

def read_met_file(file_name):
    try:
        nc_obj = Dataset(file_name)
    except OSError:
        print(file_name, "can't read !")
        return None
    lats = nc_obj.variables['lat'][:]
    lons = nc_obj.variables['lon'][:]
    # wind_dir = nc_obj.variables['wind_dir'][:]
    wind_speed = nc_obj.variables['wind_speed'][:]
    times = nc_obj.variables['time'][:]
    # 关闭文件
    nc_obj.close()
    return [times, lats, lons, wind_speed]

def cfotime_to_datetime(time_nc):
    """将CFOSAT的时间转换为datetime格式"""
    times = []
    temp = []
    for elems in time_nc:
        for elem in elems:
            temp.append(bytes.decode(elem))
        strtemp = ''.join(temp)

        if strtemp[:4] != '0000':
            if int(strtemp[11:13]) >= 24:
                bal = int(strtemp[11:13]) % 24
                new_strtemp = strtemp[:11] + str(bal) + strtemp[13:]
                times.append(
                    datetime.datetime.strptime(new_strtemp, '%Y-%m-%dT%H:%M:%SZ') + datetime.timedelta(hours=24))
            else:
                ddd = datetime.datetime.strptime(strtemp, '%Y-%m-%dT%H:%M:%SZ')
                times.append(ddd)
        else:
            times.append(datetime.datetime(2100, 1, 1, 00, 00, 00))
        temp = []
    return times

def filetime_to_datatime(sat_files, sat_name):
    """
    将文件名中的日期时间转换为datatime格式, 并保存进list中返回, 时间list下标与文件List下标一致。
    :rtype: list
    """
    sat_ftime = []

    if sat_name == 'HY-2B':
        # HY-2B    命名格式：H2B_OPER_GDR_2PC_0057_0306_20210101T213603_20210101T222817.nc
        for file in sat_files:
            
            temp = file.split('_')[6]
            temp_time = datetime.datetime.strptime(temp, '%Y%m%dT%H%M%S')
            sat_ftime.append(temp_time)

    elif sat_name == 'CFOSAT':
        # CFOSAT   命名格式：CFO_EXPR_SCA_C_L2B_OR_20210601T003612_14333_125_32_owv.nc
        for file in sat_files:
            temp = file.split('_')[6]
            temp_time = datetime.datetime.strptime(temp, '%Y%m%dT%H%M%S')
            sat_ftime.append(temp_time)

    else:
        # Metop-A/B/C  命名格式：ascat_20210601_005700_metopb_45155_eps_o_coa_3202_ovw.l2.nc
        for file in sat_files:
            temp = file.split('_')[1] + file.split('_')[2]
            temp_time = datetime.datetime.strptime(temp, '%Y%m%d%H%M%S')
            sat_ftime.append(temp_time)

    return sat_ftime

def mettime_to_datetime(time_nc):
    """将metop的时间转换为datetime格式"""
    inittime = datetime.datetime(1990, 1, 1, 00, 00, 00)
    time_temp = []
    for elems in time_nc:
        time_temp.append(inittime + datetime.timedelta(seconds=int(elems[0])))
    return time_temp
