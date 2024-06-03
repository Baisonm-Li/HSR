

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import numpy as np
import torchvision

def down_sample(hsi,scale=1/4):
    """下采样"""
    hsi = hsi.unsqueeze(0)
    down_hsi = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=0.5)(hsi)
    down_hsi = F.interpolate(down_hsi,scale_factor=scale)
    return down_hsi.squeeze(0)

def get_RGB(hsi,bands=[1,2,3]):
    """给定RGB index 抽取RGB图像"""
    return hsi[bands]


data_root_path = '/wxw/lbs/SR/datasets/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat'
# 生成chikushi dataset 
# 生成chikushi给的是mat数据格式
with h5py.File(data_root_path,'r') as f:
    data_mat = np.array(f['chikusei'])
    print(data_mat)
image_data = data_mat
sample_stride = 32
patche_size = 64
c,h,w = image_data.shape
# 这里提取中心区域作为测试数据集
window_h,window_w = h // 3, w // 3
window_start_y = h//2 - window_h//2
window_start_x = w//2 - window_w//2
window_end_y   = h//2 + window_h//2
window_end_x   = w//2 + window_w//2
test_region = image_data[:,window_start_y:window_end_y, window_start_x:window_end_x]
c, test_region_h, test_region_w = test_region.shape

test_data_list = []
for i in range(0,test_region_h-patche_size+1,sample_stride):
    for j in range(0,test_region_w-patche_size+1,sample_stride):
        tar_GT = test_region[0,i:i+patche_size,j:j+patche_size]
        tar_LRHSI = down_sample(tar_GT)
        tar_RGB = get_RGB(tar_GT)
        test_data_list.appne((tar_GT,tar_LRHSI,tar_RGB))
