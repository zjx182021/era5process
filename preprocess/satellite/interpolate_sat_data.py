"""
将挑选出来的卫星数据使用进行插值，使其由原本的空间散乱点变为经纬方向上的网格状数据。
保存为整图，格式：[时刻索引][纬度][经度]，保存的特征为风速。
插值方法采用IDW，插值窗口为50km，时间采用最近邻的1h.
纬度随index增大而增大，经度随index增大而增大。
"""

import sys
sys.path.append('./preprocess')
import datetime
import sys
import numpy as np
import time
import os
from natsort import natsorted
from haversine import haversine
import math
import config as CFG
cfg = CFG.DefaultConfigure


def STW(target_dot, near_dots, spatiol_window, temporal_window, tar_time):
    """
    使用near_dots中最近的五个点，用IDW算法插值dot
    :param target_dot: [lat, lon]
    :param near_dots: [[time, lat, lon, u, v], ...]
    """
    u, v = 0, 0

    W = []
    # 算权重
    for i in range(len(near_dots)):
        # 以经纬度之间的平方和作为两点的距离
        di = (near_dots[i][1] - target_dot[0]) ** 2 + (near_dots[i][2] - target_dot[1]) ** 2
        if di == 0 and near_dots[i][0] == tar_time:
            return float(near_dots[i][-2]), float(near_dots[i][-1])
        # 时间差值的平方
        te = ((max(near_dots[i][0], tar_time) - min(near_dots[i][0], tar_time)).seconds / 3600) ** 2
        Wr = di / (spatiol_window ** 2)
        Wt = te / (temporal_window ** 2)
        W.append((2 - (Wr + Wt)) / (2 + (Wr + Wt)))

    W_sum = sum(W)
    for w in range(len(W)):
        u += W[w] * near_dots[w][-2] / W_sum
        v += W[w] * near_dots[w][-1] / W_sum

    return u, v     # 风速


def IDW(near_dots):
    """
    使用near_dots中的点, 用IDW算法插值dot
    :param target_dot: [lat, lon]
    :param near_dots: [[[time, lat, lon, spd], di], ...]
    """
    u, v = 0, 0

    W = []
    # 算权重
    for i in range(len(near_dots)):
        # 以经纬度之间的平方和作为两点的距离
        di = near_dots[i][1]
        if di == 0:
            return float(near_dots[i][-2]), float(near_dots[i][-1])
        W.append(1 / math.pow(di, 2))

    # W = sorted(W, reverse=True)[:5]

    W_sum = sum(W)
    for w in range(len(W)):
        u += W[w] * near_dots[w][-2] / W_sum
        v += W[w] * near_dots[w][-1] / W_sum

    return u, v


def intplt_onedot(dot, sat_data, grid_window, target_res, temporal_window, 
                  temp_time, type='STW', spatial_window=None):
    """
    找sata_data中距dot点distance以内的点，如果这些点多于五个，就用它们插值dot点
    :return: None
    """
    datas_bef = []  # 符合要求的卫星数据
    datas_aft = []
    number_bef = 0
    number_aft = 0

    # 取出
    temp_data = []
    lat = dot[0] - grid_window
    lon = dot[1] - grid_window
    while lat <= dot[0] + (grid_window - target_res):
        while lon <= dot[1] + (grid_window - target_res):
            try:
                temp_data += sat_data[(lat, lon)]
            except KeyError:
                lon += target_res
                continue
            lon += target_res
        lon = dot[1] - grid_window
        lat += target_res

    # 若space_window不为None，则需要进一步筛选
    if spatial_window is not None:
        new_data = []
        for p in temp_data:
            # 将lon 由 0~360 转换为 -180~180
            p[2] = p[2] - 360 if p[2] > 180 else p[2]
            dot[1] = dot[1] - 360 if dot[1] > 180 else dot[1]
            di = haversine([dot[0], dot[1]], [p[1], p[2]])
            if di <= spatial_window:
                new_data.append([p, di])
        temp_data = new_data

    if type == 'STW':
        # for elem in temp_data:
        #     # elem : [time, lat, lon, u, v]
        #     # dot : [lat, lon]
        #     # 分类前后时刻
        #     if elem[0] < temp_time:
        #         number_bef += 1
        #         datas_bef.append(elem)
        #     else:
        #         number_aft += 1
        #         datas_aft.append(elem)

        # # 若目标点前后点过少，不进行插值
        # if number_aft < 1 or number_bef < 1:
        #     return -32767.0, -32767.0

        if len(temp_data) < 5:
            return -32767.0, -32767.0
        
        datas = temp_data   # [*datas_bef, *datas_aft]
        sw = spatial_window if spatial_window is not None else grid_window
        u, v = STW(dot, datas, sw, temporal_window, temp_time)

    elif type == 'IDW':
        if len(temp_data) < 5:
            return -32767.0, -32767.0
    # datas = sorted(datas, key=lambda x: x[1], reverse=False)  # 按di进行排序，由近到远

        u, v = IDW(temp_data)
    return u, v


def create_init_image(area, gap):
    """
    构建整图状的初试数据结构。
    :return: 维度又外向内分别表示：[纬度][经度]
    :rtype:
    """
    init_image = np.zeros((2, int((area[1] - area[0])/gap + 1), int((area[3] - area[2])/gap + 1)))
    init_image += -32767.
    return init_image


def split_dots(ori_data, area, inter_grid_spatio):# 切分area，构建0.25*0.25的网格点
    """分类ori_data, 方便选点"""
    split_data = {}
    lat, lon = area[0], area[2]
    while lat < area[1]:
        while lon < area[3]:
            split_data[(lat, lon)] = []
            lon += inter_grid_spatio
        lon = area[2]
        lat += inter_grid_spatio

    for dot in ori_data: # 将数据点的位置调整为规则的0.25*0.25的网格点
        lat = dot[1] - dot[1] % 0.25
        lon = dot[2] - dot[2] % 0.25
        try:
            split_data[(lat, lon)].append(dot)
        except KeyError:
            continue

    # 去掉无值的键
    lat, lon = area[0], area[2]
    while lat < area[1]:
        while lon < area[3]:
            if not len(split_data[(lat, lon)]):
                split_data.pop((lat, lon))
            lon += inter_grid_spatio
        lon = area[2]
        lat += inter_grid_spatio

    return split_data


def interpolation(ori_data, init_image, spatiol_window, temporal_window, temp_time, 
                  inter_type, area, target_spatial_res, inter_grid_spatio):
    """
    用 file 文件中的数据 使用IDW方法，对init_grid进行插值。
    :param init_image: [[lat, lon, spd], ...]
    :return: None
    """
    # sat_data = np.load(os.path_ori.join(sys.argv[1], file))     # [[time, lat, lon, spd, dir], ...]

    split_data = split_dots(ori_data, area, target_spatial_res)

    # 遍历init_image，并进行插值
    for lat in range(init_image.shape[1]):
        for lon in range(init_image.shape[2]):
            u, v = intplt_onedot(
                [lat * target_spatial_res + area[0],
                 lon * target_spatial_res + area[2]],
                split_data, inter_grid_spatio, target_spatial_res, temporal_window, temp_time, inter_type, spatiol_window)
            init_image[0][lat][lon], init_image[1][lat][lon] = u, v

    return init_image


def select_files_by_time(ori_files, ori_datetime, temp_time, inter_windows_temporal, type='STW'):
    """
    temp_time为待插值时刻，从ori_files过滤出前后temporal_res时间内的文件，
    :param ori_files: 原始文件列表
    :param ori_datetime: 原始文件列表对应的datetime格式的列表
    :param temp_time: 目标插值时刻
    :param inter_windows_temporal: STW的时间窗口大小
    :return: 在时间窗口内的文件
    :rtype: list
    """
    bef_files, aft_files = [], []
    for t in range(len(ori_datetime)):
        # 目标时刻的时间窗口前
        if ori_datetime[t] + datetime.timedelta(hours=inter_windows_temporal) < temp_time:
            continue
        # 目标时刻的时间窗口后
        elif ori_datetime[t] > temp_time + datetime.timedelta(hours=(inter_windows_temporal-1)):
            break
        # 时间窗口内且早于目标时刻
        elif ori_datetime[t] < temp_time:
            bef_files.append(ori_files[t])
        # 时间窗口内且晚于目标时刻
        elif ori_datetime[t] >= temp_time:
            aft_files.append(ori_files[t])

    # 前或后时刻无数据
    # if type == 'STW' and (len(bef_files) == 0 or len(aft_files) == 0):
    #     print(temp_time, "can't interpolating ! No before(or after) file.")
    #     return None

    select_files = [*bef_files, *aft_files]
    return select_files


def filename_to_datetime(files):
    """
    建立一个同files对应的datetime list，方便时间上的比较
    :param files: 格式 '%Y-%m-%d-%H.npy'
    """
    files_time = []
    for file in files:
        temp_time = datetime.datetime.strptime(file.split('.')[0], '%Y-%m-%d-%H')
        files_time.append(temp_time)
    return files_time


def read_files(select_files, opened_files, readed_data, path_dir):
    """读选中的文件"""
    # 读取未读的文件
    for f in range(len(select_files)):
        # 跳过已读的
        if select_files[f] in opened_files:
            continue
        # 读取未读的
        else:
            readed_data.append(np.load(os.path.join(path_dir, select_files[f]), allow_pickle=True))
            opened_files.append(select_files[f])

    # 去除不需要的文件
    opened_files_new = []
    readed_data_new = []
    for f in range(len(opened_files)):
        if opened_files[f] in select_files:
            opened_files_new.append(opened_files[f])
            readed_data_new.append(readed_data[f])
    return opened_files_new, readed_data_new


def interpolate_sat(sat_name, months, sat_filter_path, sat_grid_path, target_spatial_res, 
                    target_temporal_res, inter_grid_spatial, inter_window_spatial, 
                    inter_window_temporal, inter_type, area):
    '''
    对卫星数据进行插值
    '''
    for m in months:
        print("Interpolating", sat_name)
        sat_filter_dir = os.path.join(sat_filter_path, m)
        sat_grid_dir = os.path.join(sat_grid_path, m)
        # ori_files = os.listdir(sat_filter_dir)
        ori_files = natsorted(os.listdir(sat_filter_dir))
        os.makedirs(sat_grid_dir, exist_ok=True)
        saved_files = natsorted(os.listdir(sat_grid_dir))

        # 将原始文件的文件时间存入一个list
        ori_files_datetime = filename_to_datetime(ori_files)

        # 确定待插值文件的起始时刻
        if len(saved_files):
            startime = datetime.datetime.strptime(saved_files[0].split('.')[0], "%Y-%m-%d-%H")
        else:
            startime = datetime.datetime.combine(ori_files_datetime[0].date(), datetime.time(0, 0, 0))
        temp_time = startime

        last_ori_file_time = datetime.datetime.strptime(ori_files[-1].split('.')[0], "%Y-%m-%d-%H")

        # 把已经读取出来的文件保存一个列表，避免重复读
        opened_files = []
        readed_data = []
        cost_times = []

        while temp_time <= last_ori_file_time:
            # 该时刻已保存
            # if datetime.datetime.strftime(temp_time, "%Y-%m-%d-%H")+'.npy' in saved_files:
            #     temp_time += datetime.timedelta(hours=CFG.target_temporal_res)
            #     continue
            select_files = select_files_by_time(ori_files, ori_files_datetime, temp_time, inter_window_temporal, inter_type)

            # 时间上无法满足插值条件
            while not select_files and temp_time <= last_ori_file_time:
                temp_time += datetime.timedelta(hours=target_temporal_res)
                select_files = select_files_by_time(ori_files, ori_files_datetime, temp_time, inter_window_temporal, inter_type)
            if temp_time > last_ori_file_time:
                break

            opened_files, readed_data = read_files(select_files, opened_files, readed_data, sat_filter_dir)

            # 将readed_data中的散点拿出来, [点个数, 每个点的五个属性]
            ori_data = readed_data[0]
            for data in readed_data[1:]:
                ori_data = np.append(ori_data, data, axis=0)
            ori_data = sorted(ori_data, key=lambda x: (x[1], x[2]))
            # 将ori_data中的spd, dir转换为u, v
            # ori_data = np.array(ori_data)
            # spd, dir = ori_data[:, 3].astype('float'), ori_data[:, 4].astype('float')
            # ori_data[:, 3] = np.array([-1])*spd * np.sin(dir * np.pi / 180)
            # ori_data[:, 4] = np.array([-1])*spd * np.cos(dir * np.pi / 180)

            # 初始化一个待插值整图
            init_image = create_init_image(area, target_spatial_res)

            start = time.time()
            
            # 插值
            itplted_image = interpolation(ori_data, init_image, inter_window_spatial, inter_window_temporal,
                                        temp_time, inter_type, area, target_spatial_res, inter_grid_spatial)
            # 该时刻没有点满足插值条件
            if itplted_image.max() == -32767.0:
                print(temp_time, "can't interpolating ! No match point.")
                temp_time += datetime.timedelta(hours=target_temporal_res)
                continue

            # 保存，保存的整图 纬度从大到小（与ERA5相同），经度从小到大
            # 这段代码的主要目的是将插值后的数据保存为文件，以及记录保存每个文件所需的时间。
            new_file_name = datetime.datetime.strftime(temp_time, "%Y-%m-%d-%H")
            np.save(os.path.join(sat_grid_dir, new_file_name), itplted_image[:, ::-1])
            end = time.time()
            print(new_file_name, "is saved. Cost : ", (end - start))
            cost_times.append(end - start)
            temp_time += datetime.timedelta(hours=target_temporal_res)

        print("Ave time: ", sum(cost_times) / len(cost_times))


if __name__ == '__main__' :
    # 读取卫星数据
    sat_name = 'HY-2B'
    months = cfg.months
    sat_filter_path = cfg.sat_filter_path_h2b
    sat_grid_path = cfg.sat_grid_path_h2b
    target_spatial_res = cfg.target_spatial_res
    target_temporal_res = cfg.target_temporal_res
    inter_grid_spatial = cfg.inter_grid_spatial
    inter_window_spatial = cfg.inter_window_spatial
    inter_window_temporal = cfg.inter_window_temporal
    inter_type = 'STW'
    area = cfg.area

    interpolate_sat(sat_name, months, sat_filter_path, sat_grid_path, target_spatial_res, 
                    target_temporal_res, inter_grid_spatial, inter_window_spatial, 
                    inter_window_temporal, inter_type, area)