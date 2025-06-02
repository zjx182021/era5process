# 根据卫星原始文件名，把对应月份的数据分别放到不同的文件夹（01，02，....，11，12）中

from netCDF4 import Dataset
from natsort import natsorted

import os

def select_file(file_path, save_path):
    """
    根据卫星原始文件名，把对应月份的数据分别放到不同的文件夹（01，02，....，11，12）中
    :param file_path: 原始文件路径
    :param save_path: 保存文件路径
    :return: None
    """
    # 获取文件名
    folder_list = natsorted(os.listdir(file_path))
    # print(file_list)
    # 遍历文件
    for folder in folder_list:
        # 获取文件路径
        folder_path = file_path + '/' + folder
        # 获取文件名
        file_list = natsorted(os.listdir(folder_path))
        # # 调用函数
        # move_file(folder_path, file_list, save_path)
        for file in file_list:
            # 获取文件名
            file_name = file.split('_')[-2]
            # 获取文件月份
            file_month = file_name[4:6]
            # 获取文件年份
            file_year = file_name[0:4]
            # 获取文件路径
            file_name = file_path + '/' + folder + '/' + file
            # 获取保存路径
            save_file_path = save_path + '/' + file_month
            # 判断文件夹是否存在
            if not os.path.exists(save_file_path):
                os.makedirs(save_file_path)
            # 移动文件
            os.rename(file_name, save_file_path + '/' + file)
        

if __name__ == '__main__':
    
    # 原始文件路径
    file_path = '/media/data3/wyp/InitalField/data/satelite/HY-2B'
    # 保存文件路径
    save_path = '/media/data3/wyp/InitalField/data/satelite/origin'
    # 调用函数
    select_file(file_path, save_path)