import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["significant_height_of_combined_wind_waves_and_swell"],
    "year": ["2022"],
    "month": ["01"],
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
        "00:00", "03:00", "06:00",
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

temp_file_name = f'/media/data5/wyp/lq/SWH_Retrieval/ZZZZZZ/ERA5_swh_2022_60N-0_100-160E.nc'
client = cdsapi.Client()
client.retrieve(dataset, request, temp_file_name)
