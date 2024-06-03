import torchvision
from einops import rearrange, repeat, einsum
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import scipy.io as sio
import os
import torch.nn.functional as F

# https://liangjiandeng.github.io/Projects_Res/HSRnet_2021tnnls.html
def _phase_shift(I, r):
    return F.interpolate(I,scale_factor=r,mode='bicubic')


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        residual = x
        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # 第二层
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # 加上残差
        out += residual
        return out
    

# Define the PS (phase shift) block
class PSBlock(nn.Module):
    def __init__(self, num_channels, r):
        super(PSBlock, self).__init__()
        self.r = r

    def forward(self, X):
        Xc = torch.chunk(X, 31, dim=1)
        X = torch.cat([_phase_shift(x, self.r) for x in Xc], dim=1)
        return X

# Define the Channel Attention (CA) block
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, num_spectral):
        super(ChannelAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, num_spectral, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_spectral)

    def forward(self, x):
        gap = self.global_pool(x)
        ca = F.relu(self.bn1(self.conv1(gap)))
        ca = F.relu(self.bn2(self.conv2(ca)))
        return ca

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=6,padding=2)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=6, stride=1,padding=3)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=False)
    def forward(self, x):
        gap = torch.mean(x,keepdim=True,dim=1)
        sa = self.relu1(self.conv1(gap))
        sa = torch.sigmoid(self.conv2(sa))
        return sa


avg_pooling_layer = nn.AvgPool2d(kernel_size=2, stride=2)

class HSRNet(nn.Module):
    def __init__(self,HSI_bands=31, MSI_bands=3, res_nums=6,):
        super(HSRNet, self).__init__()
        self.psb_block = PSBlock(HSI_bands,4)
        self.channel_attention = ChannelAttention(HSI_bands,HSI_bands)
        self.spatial_attention = SpatialAttention()
        self.downsample = nn.AvgPool2d(kernel_size=4, stride=4)
        self.conv1 = nn.Conv2d(HSI_bands + MSI_bands,64,kernel_size=3,stride=1, padding=1)
        resblocks = [ResBlock(64, 64) for _ in range(res_nums)]
        self.resblocks = nn.Sequential(*resblocks)
        self.conv2 = nn.Conv2d(64, HSI_bands,kernel_size=3,stride=1, padding=1)
        self.convPS = nn.Conv2d((HSI_bands + MSI_bands), HSI_bands * 4 * 4,kernel_size=3,stride=1, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        
    def forward(self, lr_hs,hr_ms):
        lr_hs = torch.clamp(lr_hs,0,1)
        ca = self.channel_attention(lr_hs) # channle attention 
        y_u = self.psb_block(lr_hs)
        sa = self.spatial_attention(hr_ms) # spatial attention 
        z_d = self.downsample(hr_ms)
        c0 = torch.cat([z_d,lr_hs],dim=1)
        c0 = self.convPS(c0)
        y_u_ps = F.pixel_shuffle(c0, upscale_factor=4)
        c1 = torch.cat([hr_ms,y_u_ps],dim=1)
        x =self.relu1( self.conv1(c1))
        x = self.resblocks(x) 
        x = x * sa
        x =self.relu2(self.conv2(x))
        y = ca * x 
        y = y + y_u
        y.clamp_(0,1)
        return y
