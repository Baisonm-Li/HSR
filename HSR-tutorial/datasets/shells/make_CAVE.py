
""" 
    制作CAVE超分辨数据集, 切割的stride为32, 不做填充和切割边缘操作
    size: 512 x 512 x 31
    训练数据数量:3920
    测试数据数量:2156
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
from PIL import Image
import os 
import torchvision
import numpy as np

def get_hsi_image(image_path):
    """拼凑高光谱图像，由于下载到的CAVE图像是分波段的灰度图，这里把每个高光谱图像对应的31张灰度图叠加在一起
    对应CAVE高光谱图像的31个波段"""
    hsi = []
    for dir in sorted(os.listdir(image_path)):
        if dir.endswith('.png'):
            image_dir = os.path.join(image_path,dir)
            image_data = Image.open(image_dir)
            image_data = torch.from_numpy(np.array(image_data))
            hsi.append(image_data )

    hsi = torch.stack(hsi,dim=0) / 65535 # 这里对数据进行归一化，注意CAVE图像存储是16位无符号int 这里除以（2**16-1) i.e. 65535
    hsi = hsi.float()
    return hsi

def plot_tensor_image(image_data): # 这里将高光谱图像转换位plot可以展示的RGB格式，主要是转换图像的维度（axis）
    """plot 格式"""
    if len(image_data.shape)==2:
        image_data = image_data.unsqueeze(0)
    return image_data.permute(1,2,0).detach().cpu().numpy()

def down_sample(hsi,scale=1/4): # 这里是图像的降质
    """下采样"""
    hsi = hsi.unsqueeze(0)
    down_hsi = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=0.5)(hsi) # 高斯模糊
    down_hsi = F.interpolate(down_hsi,scale_factor=scale) # 下采样
    return down_hsi.squeeze(0)

def get_RGB(hsi,bands=[1,2,3]): 
    """给定RGB index 抽取RGB图像"""
    return hsi[bands]



if __name__ == "__main__":
    root_dir = r'/wxw/lbs/SR/datasets/CAVE_IMG'
    
    # 测试图像 11 张
    test_image_themes = ['balloons_ms','cd_ms','chart_and_stuffed_toy_ms','clay_ms','fake_and_real_beers_ms',
                        'fake_and_real_lemon_slices_ms','fake_and_real_tomatoes_ms','feathers_ms','flowers_ms','hairs_ms','jelly_beans_ms']
    # 丢弃一张图像
    discarded_image = ['watercolors_ms']
    # 测试的图像
    train_image_themes = ["beads_ms","cloth_ms","egyptian_statue_ms","face_ms","fake_and_real_food_ms","fake_and_real_lemons_ms",
                        "fake_and_real_peppers_ms","fake_and_real_strawberries_ms","fake_and_real_sushi_ms","glass_tiles_ms",
                        "oil_painting_ms","paints_ms","photo_and_face_ms","pompoms_ms","real_and_fake_apples_ms","real_and_fake_peppers_ms",
                        "sponges_ms","stuffed_toys_ms","superballs_ms","thread_spools_ms"]

    train_data_list = []
    test_data_list = []
    test_dirs = [f'{root_dir}/{theme}/{theme}' for theme in test_image_themes]
    train_dirs =  [f'{root_dir}/{theme}/{theme}' for theme in train_image_themes]
    test_hsi_list = [get_hsi_image(dir) for dir in test_dirs]
    train_hsi_list = [get_hsi_image(dir) for dir in train_dirs]
    RGB_bands = [(700 - 400) // 10,(530 - 400) // 10,(450 - 400) // 10] # 抽取RGB图像 注意RGB对应的波段
    
    print("==开始切割数据==") 
    # 切割测试数据
    for hsi in train_hsi_list:
        for i in range(0,512-64,32):
            for j in range(0,512-64,32):
                GT = hsi[:,i:i+64,j:j+64]
                LRHSI = down_sample(hsi=GT)
                RGB = get_RGB(GT,bands=RGB_bands)
                train_data_list.append((GT,LRHSI,RGB))
    print("==训练数据加载完==")
    
    # 切割训练数据
    for hsi in test_hsi_list:
        for i in range(0,512-64,32):
            for j in range(0,512-64,32):
                GT = hsi[:,i:i+64,j:j+64]
                LRHSI = down_sample(hsi=GT)
                RGB = get_RGB(GT,bands=RGB_bands)
                test_data_list.append((GT,LRHSI,RGB))
                
    print("==测试数据加载完成==")
    print(f'训练数据数量:{len(train_data_list)}') # 3920
    print(f'测试数据数量:{len(test_data_list)}') # 
    test_data_dict = {
        'LRHSI':[LRHSI.numpy() for GT,LRHSI,RGB in test_data_list],
        'GT':[GT.numpy() for GT,LRHSI,RGB in test_data_list],
        'RGB':[RGB.numpy() for GT,LRHSI,RGB in test_data_list]
    }
    train_data_dict = {
        'LRHSI':[LRHSI.numpy() for GT,LRHSI,RGB in train_data_list],
        'GT':[GT.numpy() for GT,LRHSI,RGB in train_data_list],
        'RGB':[RGB.numpy() for GT,LRHSI,RGB in train_data_list]
    }
    # 这里将数据保存位npz格式，https://blog.csdn.net/wangkaidehao/article/details/103434442
    np.savez('CAVE_test.npz',**test_data_dict)
    np.savez('CAVE_train.npz',**train_data_dict)
    print("==数据保存完毕==")