


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# https://crabwq.github.io/pdf/2020%20SSR-NET%20Spatial-Spectral%20Reconstruction%20Network%20for%20Hyperspectral%20and%20Multispectral%20Image%20Fusion.pdf
class ResTFNet(nn.Module):
    def __init__(self,HSI_bands=31,MSI_bands=3,):
        super(ResTFNet, self).__init__()
        self.feature_extraction_block = FeatureExtractionBlock(HSI_bands,MSI_bands)
        self.feature_fusion_block = FeatureFusionBlock(in_channels=128)
        self.res_block = ImageReconstructionBlock(in_channels=128,out_channels=HSI_bands)
    def forward(self, LRHSI, HRMSI):
        out_feature,HRMSI_feature_,LRHSI_feature_ = self.feature_extraction_block(LRHSI,HRMSI)
        feature = self.feature_fusion_block(out_feature)
        out_feature = self.res_block(feature,HRMSI_feature_,LRHSI_feature_)
        return out_feature



class FeatureExtractionBlock(nn.Module):
    def __init__(self,HSI_bands,MSI_bands):
        super(FeatureExtractionBlock, self).__init__()
        self.conv1_M = nn.Conv2d(MSI_bands,32,3,1,1)
        self.prelu1_M = nn.PReLU()
        self.conv2_M = nn.Conv2d(32,32,3,1,1)
        self.prelu2_M = nn.PReLU()
        self.conv3_M = nn.Conv2d(32,64,2,2,0)

        self.conv1_P = nn.Conv2d(HSI_bands,32,3,1,1)
        self.prelu1_P = nn.PReLU()
        self.conv2_P = nn.Conv2d(32,32,3,1,1)
        self.prelu2_P = nn.PReLU()
        self.conv3_P = nn.Conv2d(32,64,2,2,0)

    def forward(self, LRHSI, HRMSI):
        up_LRHSI = F.interpolate(LRHSI,scale_factor=4)
        HRMSI_feature = self.conv1_M(HRMSI)
        HRMSI_feature = self.prelu1_M (HRMSI_feature)
        HRMSI_feature = self.conv2_M(HRMSI_feature)
        HRMSI_feature_ = self.prelu2_M(HRMSI_feature)
        HRMSI_feature = self.conv3_M(HRMSI_feature_)

        LRHSI_feature = self.conv1_P(up_LRHSI)
        LRHSI_feature = self.prelu1_P(LRHSI_feature)
        LRHSI_feature = self.conv2_P(LRHSI_feature)
        LRHSI_feature_ = self.prelu2_P(LRHSI_feature)
        LRHSI_feature = self.conv3_P(LRHSI_feature_)
        out_feature = torch.concat([HRMSI_feature,LRHSI_feature],dim=1)
        return out_feature,HRMSI_feature_,LRHSI_feature_
    

class FeatureFusionBlock(nn.Module):
    def __init__(self,in_channels):
        super(FeatureFusionBlock, self).__init__()
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels,128,3,1,1),nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128,128,3,1,1),nn.PReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(128,256,2,2,0),nn.PReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.PReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.PReLU())
        self.conv9 = nn.Sequential(nn.ConvTranspose2d(256,128,2,2,0),nn.PReLU())
        self.res_conv1 = nn.Sequential(nn.Conv2d(256,128,1),nn.PReLU())

    def forward(self, x):
        x_ = self.conv4(x)
        x = self.conv5(x_)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.res_conv1(torch.concat([x,x_],dim=1))
        return x
    

class ImageReconstructionBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ImageReconstructionBlock, self).__init__()
        self.conv10 = nn.Sequential(nn.Conv2d(in_channels,128,3,1,1),nn.PReLU())
        self.conv11 = nn.Sequential(nn.Conv2d(128,128,3,1,1),nn.PReLU())
        self.up_conv12 = nn.Sequential(nn.ConvTranspose2d(in_channels,64,2,2,0),nn.PReLU())

        self.res_conv2 = nn.Sequential(nn.Conv2d(128,64,1),nn.PReLU())
        self.conv13 = nn.Sequential(nn.Conv2d(64,64,3,1,1),nn.PReLU())
        self.conv14 = nn.Sequential(nn.Conv2d(64,64,3,1,1),nn.PReLU())
        self.conv15 = nn.Sequential(nn.Conv2d(64,out_channels,3,1,1),nn.Tanh())

        
    def forward(self, x, HRMSI_feature_,LRHSI_feature_):
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.up_conv12(x)
        x = torch.concat([x,HRMSI_feature_,LRHSI_feature_],dim=1)
        x = self.res_conv2(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)

        return x
