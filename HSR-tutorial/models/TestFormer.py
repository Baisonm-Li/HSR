from einops import rearrange, repeat, einsum
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()


    def forward(self, x):
        b, c, h,w=x.shape
        x =   x.view(b*self.groups,-1,h,w)   #bs*g,dim//g,h,w
        xn =  x * self.avg_pool(x)            #bs*g,dim//g,h,w
        xn =  xn.sum(dim=1,keepdim=True)    #bs*g,1,h,w  第二维度相加
        t =   xn.view(b*self.groups,-1)      #bs*g,h*w 
        t =   t-t.mean(dim=1,keepdim=True)   #bs*g,h*w 归一化
        std = t.std(dim=1,keepdim=True)
        t =   t/std                          #bs*g,h*w  1e-5
        t =   t.view(b,self.groups,h,w)      #bs,g,h*w
        t =   t*self.weight+self.bias        #bs,g,h*w
        t =   t.view(b*self.groups,1,h,w)    #bs*g,1,h*w
        x =   x*self.sig(t)
        x =   x.view(b,c,h,w)
        return x 

class ShuffleAttention(nn.Module):
    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()
        self.sge = SpatialGroupEnhance(G)
        
    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        x=x.view(b*self.G,-1,h,w) 
        x_0,x_1=x.chunk(2,dim=1)
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)
        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial + self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w
        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)
        # channel shuffle
        out = self.sge(out)
        out = self.channel_shuffle(out, 2)
        return out
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
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
    
class MainExtractionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(64,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return x

class DownsampleConvNet(nn.Module):
    def __init__(self,input_channels,hidden_dim):
        super(DownsampleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=input_channels, kernel_size=3, stride=1, padding=1)
      

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        return x
from fightingcv_attention.attention.SKAttention import SKAttention
import torch


class TestFormer(nn.Module):
    def __init__(self,HSI_bands=31,MSI_bands=3,hidden_dim=64,groups=8):
        super(TestFormer, self).__init__()
        self.downSampleConv = DownsampleConvNet(MSI_bands,hidden_dim) 
        self.pixel_conv = nn.Conv2d(HSI_bands + MSI_bands,(HSI_bands+MSI_bands) * 4 * 4,3,1,1)
        self.reduce_conv = nn.Conv2d((HSI_bands+MSI_bands)*2,hidden_dim,3,1,1)
        self.se = SKAttention(hidden_dim)
        self.mainblocks = MainExtractionBlock()
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim * 2,3,1,1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim * 2,HSI_bands,3,1,1)
        )
        
        
    def forward(self, hsi, msi):
        up_hsi = torch.nn.functional.interpolate(hsi,scale_factor=4)
        down_msi = self.downSampleConv(msi)
        up_hsi_msi = torch.concat([up_hsi,msi],dim=1) # (HSI_bands + MSI_bands) * 64 * 64
        down_msi_hsi = torch.concat([down_msi,hsi],dim=1) # (HSI_bands + MSI_bands) * 16 * 16
        
        # 做下采样
        main_feature = torch.nn.functional.pixel_shuffle(
            self.pixel_conv(down_msi_hsi), 4
        )
        main_feature = torch.concat([main_feature,up_hsi_msi],dim=1) # (HSI_bands + MSI_bands) * 2
        main_feature = self.reduce_conv(main_feature)
  
        main_feature = self.se(main_feature) 
        main_feature = self.mainblocks(main_feature)
        main_feature = self.refine(main_feature)
        res = main_feature + up_hsi
        return res.clamp(0,1)


if __name__ == '__main__':
    x = torch.randn((32,31,16,16))
    y = torch.randn((32,3,64,64))
    model = TestFormer()
    print(model(x,y).shape)
    