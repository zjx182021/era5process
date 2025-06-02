import numpy as np
from scipy.interpolate import griddata
from datetime import datetime
from scipy.spatial.qhull import QhullError
from natsort import natsorted
import os

def interpolate(filer_path, grid_path,spatial_res = 0.5):
    folders = natsorted(os.listdir(filer_path))
    for folder in folders:
        files = natsorted(os.listdir(os.path.join(filer_path, folder)))
        if not os.path.exists(os.path.join(grid_path, folder)):
            os.makedirs(os.path.join(grid_path, folder))
        for file in files:
           
            # 读取数据
            data = np.load(os.path.join(filer_path, folder, file), allow_pickle=True)
           
            # 提取经纬度数据
            lats = np.array([lat for _, lat, _, _, _ in data])
            lons = np.array([lon for _, _, lon, _, _ in data])

            # 提取swh_ku和swh_c数据
            swh_ku = np.array([swh_ku for _, _, _, swh_ku, _ in data])
            swh_c = np.array([swh_c for _, _, _, _, swh_c in data])

            # 定义标准网格的经纬度范围和空间分辨率
            lat_range = np.arange(0, 60.0, spatial_res)
            lon_range = np.arange(100, 160.0, spatial_res)

            # 生成标准网格的经纬度坐标
            grid_lats, grid_lons = np.meshgrid(lat_range, lon_range)

            # 进行插值
            # STW 插值
            
            try:
                interp_swh_ku = griddata((lats, lons), swh_ku, (grid_lats, grid_lons), method='linear')
                interp_swh_c = griddata((lats, lons), swh_c, (grid_lats, grid_lons), method='linear')
                interp_swh_ku[np.isnan(interp_swh_ku)] = 0
                interp_swh_c[np.isnan(interp_swh_c)] = 0
                # 合并swh_ku和swh_c
                interp_swh_ku = np.expand_dims(interp_swh_ku, axis=0)
                interp_swh_c = np.expand_dims(interp_swh_c, axis=0)
                interp_swh = np.concatenate((interp_swh_ku, interp_swh_c), axis=0)
                # 保存插值后的数据到文件或者继续后续处理
            # 例如，如果需要将插值后的数据保存到文件：
                # print(interp_swh_c.mean())
                # print(interp_swh_ku.mean())
                np.save(os.path.join(grid_path, folder, file), interp_swh)
            except QhullError:
                # 当出现 QhullError 异常时，处理异常情况
                print(file, "数据点不足以进行插值，无法构建初始单纯形。")
          
        print(folder, "月 interpolated !")

if __name__ == "__main__":
    file_path = '/media/data3/wyp/InitalField/data/satelite/filter1'
    grid_path = '/media/data3/wyp/InitalField/data/satelite/grid1'
    interpolate(file_path, grid_path)
    print("interpolation done !")