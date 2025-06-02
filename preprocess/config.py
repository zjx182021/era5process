class DefaultConfigure(object):
    sat_ori_path_h2b = '/media/data3/wyp/InitalField/data/satelite/origin'
    sat_filter_path_h2b = '/media/data3/wyp/InitalField/data/satelite/filter1'
    sat_grid_path_h2b = '/media/data3/wyp/InitalField/data/satelite/grid1'
    
    sat_ori_path_cfo = '/media/data1/wyp/ocean_fusion_2/fusion_wind_spd/preprocess/satellite/origin/CFOSAT/2021'
    sat_filter_path_cfo = '/media/data1/wyp/ocean_fusion_2/fusion_wind_spd/preprocess/satellite/filter_uv/CFOSAT'
    sat_grid_path_cfo = '/media/data1/wyp/ocean_fusion_2/fusion_wind_spd/preprocess/satellite/grid_uv/CFOSAT'
    
    sat_names = ['HY-2B', 'CFOSAT']
    year = ['2021']
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']  # 
    area = [0, 60, 100, 160]  # [下纬度，上纬度，左经度，右经度]，经度为0-360
    filter_temporal_res = 6     # h

    target_spatial_res = 0.25   # 度
    target_temporal_res = 6     # h
    inter_grid_spatial = 0.5    # 度
    inter_window_spatial = None  # 度
    inter_window_temporal = 6  # h

    inter_type = 'STW'       # 插值方法
