

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

# https://github.com/rs-lsl/CSSNet
class CSSNet(nn.Module):
    def __init__(self,HSI_bands=31,MSI_bands=3,ratio=4,):
        super(CSSNet, self).__init__()
        self.cross_spectral_scale_attention_block = CrossSpectralScaleAttentionBlock(ratio=ratio,HSI_bands=HSI_bands,MSI_bands=MSI_bands)
        self.cross_spatial_scale_attention_block = CrossSpatialScaleAttentionBlock(ratio=ratio,HSI_bands=HSI_bands,MSI_bands=MSI_bands)
        self.aggregation_block = AggregationBlock(in_channels=5 * HSI_bands,out_channels=HSI_bands)

    def forward(self, LRHSI, HRMSI):
        up_LRHSI = F.interpolate(LRHSI,scale_factor=4,mode='bicubic')
        spatial_feature = self.cross_spatial_scale_attention_block(LRHSI,HRMSI)
        spectral_feature = self.cross_spectral_scale_attention_block(LRHSI,HRMSI)
        feature = torch.concat([spatial_feature,spectral_feature],dim=1)
        feature = self.aggregation_block(feature)
        out = feature  + up_LRHSI
        return out.clamp(0,1)


class CrossSpectralScaleAttentionBlock(nn.Module):
    def __init__(self,ratio=4,HSI_bands=31,MSI_bands=3):
        super(CrossSpectralScaleAttentionBlock, self).__init__()
        self.MSI_down_conv = nn.Sequential(
            nn.Conv2d(MSI_bands,MSI_bands,5,4 * ratio,),
            nn.LeakyReLU()
        )
        self.HSI_down_conv = nn.Sequential(
            nn.Conv2d(HSI_bands,HSI_bands,5,4,1),
            nn.LeakyReLU()
        )
        self.align_conv = nn.Sequential(
            nn.Conv2d(MSI_bands,MSI_bands,3,1,1),
            nn.LeakyReLU()
        )
    def forward(self, LRHSI, HRMSI):
        b,C,h,w = LRHSI.shape
        b,c,H,W = HRMSI.shape
        HRMSI_align = self.align_conv(HRMSI)
        HRMSI_align = rearrange(HRMSI_align,'b c h w -> b c (h w)')
        LRHSI_feature = self.HSI_down_conv(LRHSI)
        HRMSI_feature = self.MSI_down_conv(HRMSI)
        LRHSI_feature = rearrange(LRHSI_feature,'b C h w -> b (h w) C')
        HRMSI_feature = rearrange(HRMSI_feature,'b c h w -> b c (h w)')
        simi_matrix = torch.matmul(HRMSI_feature,LRHSI_feature) # c x C
        simi_matrix = F.softmax(simi_matrix,dim=-1)
        out_feature = torch.matmul(simi_matrix.transpose(-1,-2),HRMSI_align)
        out_feature = rearrange(out_feature,'b c (h w) -> b c h w',h=H)
        return out_feature

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out
    
class EncodingBlock(nn.Module):
    """high-level feature encoding block"""
    def __init__(self, in_channels,res_nums=5, c_h=64):
        super(EncodingBlock, self).__init__()
        self.c_h = c_h
        self.res_nums = res_nums
        self.in_channels = in_channels
        self.conv_ = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,1,1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels,self.c_h,3,1,1),
            nn.BatchNorm2d(self.c_h),
        )
        self.res_blocks = nn.Sequential(
            *[BasicBlock(self.c_h,self.c_h) for _ in range(res_nums)]
        )
    def forward(self, x):
        x = self.conv_(x)
        x = self.res_blocks(x)
        return x
    

class SpatialBasicBlock(nn.Module):
    def __init__(self,ratio=4,HSI_bands=31,MSI_bands=3):
        super(SpatialBasicBlock, self).__init__()
        self.HSI_encoding_block = EncodingBlock(HSI_bands)
        self.MSI_encoding_block = EncodingBlock(MSI_bands)
        self.low_level_conv = nn.Sequential(nn.Conv2d(HSI_bands,HSI_bands,3,1,1),nn.LeakyReLU())
        self.out_conv = nn.Sequential(nn.Conv2d(MSI_bands,MSI_bands,3,1,1),nn.LeakyReLU())

    def forward(self, LRHSI, HRMSI):
        b,c,H,W = LRHSI.shape
        HSI_feature = self.HSI_encoding_block(LRHSI)
        MSI_feature = self.MSI_encoding_block(HRMSI)
        HSI_feature_low = self.low_level_conv(LRHSI)

        HSI_feature_patches = rearrange(HSI_feature[:,:,:15,:15] ,'b c (n_x p1) (n_y p2) -> b (n_x n_y) c p1 p2',n_x=5,n_y=5)
        HSI_feature_low_patches = rearrange(HSI_feature_low[:,:,:15,:15] ,'b c (n_x p1) (n_y p2) -> b (n_x n_y) c p1 p2',n_x=5,n_y=5)

        # MSI_feature = F.conv2d(MSI_feature,HSI_feature_patches,bias=None)
        # MSI_feature = F.conv_transpose2d(MSI_feature,HSI_feature_low_patches,bias=None)
        simi_matrix = torch.concat([F.conv2d(MSI_feature[i].unsqueeze(0),HSI_feature_patches[i],bias=None) for i in range(b)])
        simi_matrix = F.softmax(simi_matrix,dim=-1)
        out_MSI = torch.concat([F.conv_transpose2d(simi_matrix[i].unsqueeze(0),HSI_feature_low_patches[i],bias=None) for i in range(b)],dim=0)
        return out_MSI
    



class CrossSpatialScaleAttentionBlock(nn.Module):
    def __init__(self,ratio=4,HSI_bands=31,MSI_bands=3):
        super(CrossSpatialScaleAttentionBlock, self).__init__()
        self.spatial_block_0 = SpatialBasicBlock(ratio=ratio,HSI_bands=HSI_bands,MSI_bands=MSI_bands)
        self.spatial_block_1 = SpatialBasicBlock(ratio=ratio,HSI_bands=HSI_bands,MSI_bands=MSI_bands)
        self.spatial_block_2 = SpatialBasicBlock(ratio=ratio,HSI_bands=HSI_bands,MSI_bands=MSI_bands)
        self.spatial_block_3 = SpatialBasicBlock(ratio=ratio,HSI_bands=HSI_bands,MSI_bands=MSI_bands)
    
    def shift_window(self,HSI,shift_type=0):
        b,c,h,w = HSI.shape
        shift_HSI = torch.zeros(b,c,h+2,w+2).to(HSI.device) 
        shift_HSI[:,:,1:h+1,1:w+1] = HSI
        h = h + 2
        w = w + 2
        if shift_type == 0:
            shift_HSI = shift_HSI[:,:,1:h-1,1:w-1]
        if shift_type == 1:
            shift_HSI = shift_HSI[:,:,0:h-2,1:w-1]
        if shift_type == 2:
            shift_HSI = shift_HSI[:,:,1:h-1,0:w-2]
        if shift_type == 3:
            shift_HSI = shift_HSI[:,:,0:h-2,0:w-2]
        return shift_HSI
                                                                                                                                                                           
    def forward(self, LRHSI, HRMSI):
        LRHSI_0 = self.shift_window(LRHSI,0)
        LRHSI_1 = self.shift_window(LRHSI,1)
        LRHSI_2 = self.shift_window(LRHSI,2)
        LRHSI_3 = self.shift_window(LRHSI,3)
        feature_0 = self.spatial_block_0(LRHSI_0,HRMSI)
        feature_1 = self.spatial_block_0(LRHSI_1,HRMSI)
        feature_2 = self.spatial_block_0(LRHSI_2,HRMSI)
        feature_3 = self.spatial_block_0(LRHSI_3,HRMSI)
        return torch.concat([feature_0,feature_1,feature_2,feature_3],dim=1)
    
class AggregationBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=11,padding=5):
        super(AggregationBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
    
# if __name__ == "__main__":
    
#     x = torch.randn((32,103,16,16))
#     y = torch.randn((32,3,64,64))
#     net = MainNet(HSI_bands=103)
#     o = net(x,y)
#     print(o.shape)

