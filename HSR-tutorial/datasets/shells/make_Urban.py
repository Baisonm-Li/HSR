import h5py
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
"""
 size: 307 x 307 x 162
 中心切割 128 x 128 作为测试数据
    测试数据数量：64
    训练数据数量：705
"""

def down_sample(hsi,scale=1/4):
    """下采样"""
    hsi = hsi.unsqueeze(0)
    down_hsi = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=0.5)(hsi)
    down_hsi = F.interpolate(down_hsi,scale_factor=scale)
    return down_hsi.squeeze(0)

def get_RGB(hsi,bands=[1,2,3]):
    """给定RGB index 抽取RGB图像"""
    return hsi[bands]


def is_overlap(x,y,h,w,window_size=128):
    """
        判断是否进入测试区域
    """
    mid_x = h // 2
    mid_y = w // 2
    x_0 = mid_x - window_size // 2
    y_0 = mid_y - window_size // 2
    x_1 = mid_x + window_size // 2
    y_1 = mid_y + window_size // 2
    
    if x >= x_0 and x <= x_1 and y <= y_1 and y >= y_0:
        return True
    else:
        return False

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loaded_data = scio.loadmat('/wxw/lbs/SR/datasets/Urban_R162.mat')
    stride = 8
    window_size = 128
    tar_size = 64
    mat_data = np.array(loaded_data['Y']).reshape(162,307,307)
    mat_data = (mat_data - mat_data.min()) / mat_data.max()
    mat_data = torch.from_numpy(mat_data).to(device)
    c,h,w = mat_data.shape
    test_region = mat_data[:,h // 2 - window_size // 2:h // 2 + window_size // 2, 
                             w // 2 - window_size // 2:w // 2 + window_size // 2]
    test_list = []
    train_list = []
    RGB_indexs = [10,60,100]
    print('==切割测试数据==')
    # 切割测试数据
    for i in range(0,window_size-tar_size,stride):
        for j in range(0,window_size-tar_size,stride):
            GT = test_region[:,i:i+tar_size,j:j+tar_size]
            LRHSI = down_sample(GT)
            RGB = get_RGB(GT,RGB_indexs)
            test_list.append((GT.detach().cpu().numpy(),LRHSI.detach().cpu().numpy(),RGB.detach().cpu().numpy()))
    
    
    print('==切割训练数据==')
    train_region = mat_data
    train_region[:,h // 2 - window_size // 2:h // 2 + window_size // 2, 
                   w // 2 - window_size // 2:w // 2 + window_size // 2]
    # 切割训练数据
    for i in range(0,h-tar_size,stride):
        for j in range(0,w-tar_size,stride):
            if is_overlap(i+tar_size,j+tar_size,h,w,window_size=128):
                continue
            else:
                GT = train_region[:,i:i+tar_size,j:j+tar_size]
                LRHSI = down_sample(GT)
                RGB = get_RGB(GT,RGB_indexs)
                train_list.append((GT.detach().cpu().numpy(),LRHSI.detach().cpu().numpy(),RGB.detach().cpu().numpy()))   
            
    train_dict = {
        "GT": [GT for GT,LRHSI,RGB in train_list],
        "LRHSI":[LRHSI for GT,LRHSI,RGB in train_list],
        "RGB":[RGB for GT,LRHSI,RGB in train_list]
    }
    test_dict = {
        "GT": [GT for GT,LRHSI,RGB in test_list],
        "LRHSI":[LRHSI for GT,LRHSI,RGB in test_list],
        "RGB":[RGB for GT,LRHSI,RGB in test_list]
    }
    print(f"测试数据数量：{len(test_list)}")
    print(f"训练数据数量：{len(train_list)}")
    np.savez('/wxw/lbs/SR/HSR/datasets/Urban_train.npz',**train_dict)
    np.savez('/wxw/lbs/SR/HSR/datasets/Urban_test.npz',**test_dict)
    print('==数据保存完毕==')
    # plt.imshow(mat_data[:3,:,:].transpose(2,1,0))
    # plt.savefig('./urban.png')

    