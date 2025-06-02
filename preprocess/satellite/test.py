from netCDF4 import Dataset

import matplotlib.pyplot as plt
import numpy as np


# 读取nc文件
nc_file = Dataset('/media/data3/wyp/InitalField/data/satelite/origin/02/H2B_OPER_GDR_2PC_0059_0364_20210201T001232_20210201T005738.nc', 'r')

# 提取lon、lat和swh变量数据
lon = np.ma.filled(nc_file.variables['lon'][:])
lat = np.ma.filled(nc_file.variables['lat'][:])
swh = np.ma.filled(nc_file.variables['wind_speed_model_u'][:])

# 
# 关闭nc文件
nc_file.close()

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(lon, lat, c=swh, cmap='jet', marker='o', edgecolors='none')
plt.colorbar(label='swh')  # 添加颜色条
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of SWH')
plt.grid(True)
plt.savefig('/media/data3/wyp/InitalField/data/1.png')
