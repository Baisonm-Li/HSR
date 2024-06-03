import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
#https://github.com/XieQi2015/MHF-net
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1,padding=0):
        super(DWConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size,stride=stride, padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
    
class UpSample_x2(nn.Module):
    def __init__(self,in_channels):
        super(UpSample_x2, self).__init__()
        self.conv_transpose_x2 = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels,2,2))
        
    def forward(self, x):
        x = self.conv_transpose_x2(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self,in_channels, scale=4):
        super(UpSampleBlock, self).__init__()
        self.up_convs= nn.Sequential(*[nn.Sequential(
            UpSample_x2(in_channels),
            UpSample_x2(in_channels),
            BasicBlock(in_channels,in_channels)
        ) for _ in range(scale // 4)])
    
    def forward(self, x):
        up_x = self.up_convs(x)
        return up_x


class DownSample(nn.Module):
    def __init__(self,in_channels):
        super(DownSample, self).__init__()
        
        self.DW_conv_1 = DWConv(in_channels,in_channels,6,1)
        self.DW_conv_2 = DWConv(in_channels,in_channels,6,1)
        self.DW_conv_3 = DWConv(in_channels,in_channels,3,1,1)

    def forward(self, x):
        x = nn.functional.pad(x,[3,2,3,2])
        down_x4 = self.DW_conv_1(x)[:,:,::4,::4]
        down_x16 = nn.functional.pad(down_x4,[3,2,3,2])
        down_x16 = self.DW_conv_2(down_x16)[:,:,::4,::4]
        down_x32 = self.DW_conv_3(down_x16)[:,:,::2,::2]
        return down_x4, down_x16, down_x32
     

class BasicBlock(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class MHFNet(nn.Module):
    def __init__(self,HSI_bands=31,MSI_bands=3,subnet_len=2,layer_num=20,up_rank=12):
        super(MHFNet, self).__init__()
        self.B = nn.Parameter(torch.randn([up_rank - MSI_bands,HSI_bands],dtype=torch.float32))
        self.A = nn.Parameter(torch.randn([HSI_bands,MSI_bands],dtype=torch.float32))

        self.first_layer = FirstLayer(self.A ,self.B,subnet_len,up_rank,MSI_bands,HSI_bands)
        self.k_layers = nn.ModuleList([KLayer(self.A ,self.B,subnet_len,up_rank,MSI_bands,HSI_bands) 
                                       for _ in range(layer_num)]) 
        self.last_layer = LastLayer(self.A ,self.B,subnet_len,up_rank,MSI_bands,HSI_bands)
        
    def forward(self,LRHSI,HRMSI):
        main_feature = self.first_layer(LRHSI,HRMSI)
        for layer in self.k_layers:
            main_feature = layer(LRHSI,HRMSI,main_feature)
        main_feature,main_feature_down = self.last_layer(LRHSI,HRMSI,main_feature)
        return main_feature



class FirstLayer(nn.Module):
    def __init__(self,A,B,subnet_len=2,up_rank=12,MSI_bands=3,HSI_bands=31):
        super(FirstLayer, self).__init__()
        self.B = B
        self.A = A
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(MSI_bands,HSI_bands,1),
            nn.LeakyReLU()
        )
        self.conv_A = nn.Sequential(
            nn.Conv2d(HSI_bands,HSI_bands,1),
            nn.LeakyReLU()
        )
        self.down_sample = DownSample(HSI_bands)
        self.up_sample = UpSampleBlock(HSI_bands)
        self.up_conv= nn.Sequential(
            nn.Conv2d(HSI_bands,HSI_bands,1),
            nn.LeakyReLU()
        )
        self.res_block = nn.Sequential(*[BasicBlock(up_rank-MSI_bands,up_rank-MSI_bands) 
                                          for _ in range(subnet_len)])

    def forward(self, LRHSI, HRMSI):
        y_A = F.conv2d(HRMSI,weight=self.A.unsqueeze(-1).unsqueeze(-1))
        down_x4_y_A,_,down_x32_y_A = self.down_sample(y_A)
        feature = (down_x4_y_A - LRHSI)
        up_feature = self.up_sample(feature)
        up_feature = self.up_conv(up_feature)
        up_feature = rearrange(up_feature,'b c h w -> b h w c')
        up_feature = torch.matmul(up_feature,self.B.transpose(-1,-2))
        up_feature = rearrange(up_feature,'b h w c -> b c h w')
        out_feature = self.res_block(up_feature)
        return out_feature # [b up_rank-MSI_bands H W]

class KLayer(nn.Module):
    def __init__(self, A,B,subnet_len=2,up_rank=12,MSI_bands=3,HSI_bands=31):
        super(KLayer, self).__init__()
        self.B = B
        self.A = A
        self.conv_init_1x1 = nn.Sequential(nn.Conv2d(up_rank-MSI_bands,up_rank-MSI_bands,1),nn.LeakyReLU())
        self.conv_init_msi = nn.Sequential(nn.Conv2d(MSI_bands,HSI_bands,1),nn.LeakyReLU())
        self.down_sample_block = DownSample(HSI_bands)
        self.up_sample_block = UpSampleBlock(HSI_bands)
        self.res_block = nn.Sequential(*[BasicBlock(up_rank-MSI_bands,up_rank-MSI_bands)  for _ in range(subnet_len)]) 

    def forward(self, LRHSI, HRMSI, pre_feature):
        pre_feature_ = F.conv2d(pre_feature,weight=self.B.transpose(-1,-2).unsqueeze(-1).unsqueeze(-1))
        pre_MSI = F.conv2d(HRMSI,weight=self.A.unsqueeze(-1).unsqueeze(-1))# H W HSI_bands
        main_feature = pre_feature_ + pre_MSI
        main_feature_down,_,_ = self.down_sample_block(main_feature)
        main_feature_down = main_feature_down - LRHSI
        main_feature_up = self.up_sample_block(main_feature_down)
        main_feature_up = rearrange(main_feature_up,'b c h w -> b h w c')
        main_feature_up = torch.matmul(main_feature_up,self.B.transpose(-1,-2))
        main_feature_up = rearrange(main_feature_up,'b h w c -> b c h w') # H W HSI_bands
        main_feature_up = main_feature_up + pre_feature
        main_feature_up = self.res_block(main_feature_up)
        return main_feature_up # HSI_bands-up_rank  H W 
    

class LastLayer(nn.Module):
    def __init__(self, A,B,subnet_len=2,up_rank=12,MSI_bands=3,HSI_bands=31):
        super(LastLayer, self).__init__()
        self.B = B
        self.A = A
        self.conv_init_1x1 = nn.Sequential(nn.Conv2d(up_rank-MSI_bands,up_rank-MSI_bands,1),nn.LeakyReLU())
        self.conv_init_msi = nn.Sequential(nn.Conv2d(MSI_bands,HSI_bands,1),nn.LeakyReLU())
        self.down_sample_block = DownSample(HSI_bands)
        self.up_sample_block = UpSampleBlock(HSI_bands)
        self.res_block = nn.Sequential(*[BasicBlock(HSI_bands,HSI_bands)  for _ in range(subnet_len)]) 
    
    def forward(self, LRHSI, HRMSI, pre_feature):
        pre_feature_ = F.conv2d(pre_feature,weight=self.B.transpose(-1,-2).unsqueeze(-1).unsqueeze(-1))
        pre_MSI = F.conv2d(HRMSI,weight=self.A.unsqueeze(-1).unsqueeze(-1))# H W HSI_bands
        main_feature = pre_feature_ + pre_MSI
        main_feature_down,_,_ = self.down_sample_block(main_feature)
        main_feature_down = main_feature_down - LRHSI
        main_feature = self.res_block(main_feature)
        return main_feature,main_feature_down