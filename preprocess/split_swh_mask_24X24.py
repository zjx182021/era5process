import os
import numpy as np
import h5py

era5_swh_mask_path = '/media/data5/wyp/lq/SWH_Retrieval/data/mask_swh.npy'
save_path = '/media/data7/lq/fanyan_data/data/inversion_swh_new/stage1/era5_swh_mask_24x24'

def creat_sample():
    # 读取数据
    era5_swh_mask = np.load(era5_swh_mask_path)
    # era5_swh_mask = era5_swh_mask['era5_swh_mask']
    print(era5_swh_mask.shape)

    # 生成样本, 切24X24的小图
    sample = []
    for i in range(0, era5_swh_mask.shape[1] - 24 + 1, 12):
        for j in range(0, era5_swh_mask.shape[2] - 24 + 1, 12):
            sample.append(era5_swh_mask[:, i:i+24, j:j+24])
    samples = np.array(sample)
    print(samples.shape)

    # 保存样本
    for i in range(len(samples)):
        sample_name = save_path.split('/')[-1] + '_' + str(i)
        sample_file = os.path.join(save_path, sample_name + '.h5')
        with h5py.File(sample_file, 'w') as f:
            f.create_dataset('data', data=samples[i], dtype=np.float32, compression='gzip')
            f.close()
    
    
    
    
    
if __name__ == '__main__':    
    
    creat_sample()