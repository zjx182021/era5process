import cdsapi


years = [str(x) for x in range(2003,2025)]
for year in years:
    # temp_file_name = f'predict/patch_corrformer/era5_data_ori-1h/ERA5_uvtp_{year}_60N-0_100-160E.nc'
    # temp_file_name = f'/media/data7/lq/Indian Ocean/era5_uv/era5_uv_origin/ERA5_uv_{year}_40N-80S_20-140E.nc'
    print('---------------------')
    print(f'Get {year} data')
    print('---------------------')
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": ["mean_wave_period"],
        "year": [year],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "06:00",
            "12:00", "18:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [60, 100, 0, 160]
    }
    temp_file_name = f'/media/data5/wyp/lq/SWH_Retrieval/data_era5_80year/era5_wave_cycle_origin/ERA5_wave_cycle_{year}_60N-0_100-160E.nc'
    client = cdsapi.Client()
    client.retrieve(dataset, request, temp_file_name)
