import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#import numpy as np
from encoder_decoder import *

from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init




class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, active=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if active:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
 
    
class CCIM(nn.Module):
    def __init__(self, in_channels):
        super(CCIM,self).__init__()

        self.downchans = BasicConv(in_channels * 2 , in_channels, kernel_size=1, active=True, stride=1)
        self.se = SELayer(in_channels)      
        self.qk_conv = BasicConv(in_channels, in_channels, 3, 1)
        self.softmax = nn.Softmax(-1)
               
    def forward(self,left,right):

        fuse = self.downchans(torch.cat([left,right],dim=1))
        fuse = self.se(fuse)
        out_left = self.downchans(torch.cat([left,fuse],dim=1))
        out_right = self.downchans(torch.cat([right,fuse],dim=1))
        Q, K = self.qk_conv(left), self.qk_conv(right)
        b, c, h, w = Q.shape
        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))

        out_right =torch.bmm(self.softmax(score), out_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)
        out_left =torch.bmm(self.softmax(score.permute(0, 2, 1)), out_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w, c).permute(0, 3, 1, 2) 

        return out_left + left, out_right + right

class Dehaze(nn.Module):
    def __init__(self,in_channels=128):
        super(Dehaze,self).__init__()
        
        
        self.enc_stage1 = Encoder()
        self.enc_stage2 = Encoder()
        self.enc_stage3 = Encoder()
        self.lv3_downchans = BasicConv(in_channels * 2, in_channels, kernel_size=1, active=True, stride=1)
        self.lv2_downchans = BasicConv(in_channels * 2, in_channels, kernel_size=1, active=True, stride=1)
        self.lv1_downchans = BasicConv(in_channels * 2, in_channels, kernel_size=1, active=True, stride=1)
        
        self.fuse_lv3_1 = CCIM(in_channels)
        self.fuse_lv3_2 = CCIM(in_channels)
        self.fuse_lv3_3 = CCIM(in_channels)
        self.fuse_lv3_4 = CCIM(in_channels)
        
        self.fuse_lv2_1 = CCIM(in_channels)
        self.fuse_lv2_2 = CCIM(in_channels)
        
        self.fuse_lv1 = CCIM(in_channels)

        self.dec_stage1 = Decoder()
        self.dec_stage2 = Decoder()
        self.dec_stage3 = Decoder()
    
    
    def forward(self,img_left,img_right):

        H = img_left.size(2)          
        W = img_left.size(3)
        img_left_lv1 = img_left
        img_right_lv1 = img_right

        # split left image
        img_left_lv2_1 = img_left[:,:,0:int(H/2),:]
        img_left_lv2_2 = img_left[:,:,int(H/2):H,:] 
        img_left_lv3_1 = img_left_lv2_1[:,:,:,0:int(W/2)]
        img_left_lv3_2 = img_left_lv2_1[:,:,:,int(W/2):W]
        img_left_lv3_3 = img_left_lv2_2[:,:,:,0:int(W/2)]
        img_left_lv3_4 = img_left_lv2_2[:,:,:,int(W/2):W]

        # split right image
        img_right_lv2_1 = img_right[:,:,0:int(H/2),:]
        img_right_lv2_2 = img_right[:,:,int(H/2):H,:] 
        img_right_lv3_1 = img_right_lv2_1[:,:,:,0:int(W/2)]
        img_right_lv3_2 = img_right_lv2_1[:,:,:,int(W/2):W]
        img_right_lv3_3 = img_right_lv2_2[:,:,:,0:int(W/2)]
        img_right_lv3_4 = img_right_lv2_2[:,:,:,int(W/2):W]

        # processing
        # stage 1
        feature_left_lv3_1 = self.enc_stage1(img_left_lv3_1)
        feature_left_lv3_2 = self.enc_stage1(img_left_lv3_2)
        feature_left_lv3_3 = self.enc_stage1(img_left_lv3_3)
        feature_left_lv3_4 = self.enc_stage1(img_left_lv3_4)

        feature_right_lv3_1 = self.enc_stage1(img_right_lv3_1)
        feature_right_lv3_2 = self.enc_stage1(img_right_lv3_2)
        feature_right_lv3_3 = self.enc_stage1(img_right_lv3_3)
        feature_right_lv3_4 = self.enc_stage1(img_right_lv3_4)

        
        feature_left_lv3_1, feature_right_lv3_1 = self.fuse_lv3_1(feature_left_lv3_1,feature_right_lv3_1)
        feature_left_lv3_2, feature_right_lv3_2 = self.fuse_lv3_2(feature_left_lv3_2,feature_right_lv3_2)
        feature_left_lv3_3, feature_right_lv3_3 = self.fuse_lv3_3(feature_left_lv3_3,feature_right_lv3_3)
        feature_left_lv3_4, feature_right_lv3_4 = self.fuse_lv3_4(feature_left_lv3_4,feature_right_lv3_4)
        

        feature_left_lv3_top = torch.cat([feature_left_lv3_1,feature_left_lv3_2],dim=3)
        feature_left_lv3_bot= torch.cat([feature_left_lv3_3,feature_left_lv3_4],dim=3)

        feature_right_lv3_top = torch.cat([feature_right_lv3_1,feature_right_lv3_2],dim=3)
        feature_right_lv3_bot= torch.cat([feature_right_lv3_3,feature_right_lv3_4],dim=3)

        feature_left_lv3 = torch.cat([feature_left_lv3_top,feature_left_lv3_bot],dim=2)
        feature_right_lv3 = torch.cat([feature_right_lv3_top,feature_right_lv3_bot],dim=2)


        resiual_left_lv3_top = self.dec_stage1(feature_left_lv3_top)
        resiual_left_lv3_bot = self.dec_stage1(feature_left_lv3_bot)


        resiual_right_lv3_top = self.dec_stage1(feature_right_lv3_top)
        resiual_right_lv3_bot = self.dec_stage1(feature_right_lv3_bot)
        
        # stage 2
        feature_left_lv2_1 = self.enc_stage2(img_left_lv2_1 + resiual_left_lv3_top)
        feature_left_lv2_2 = self.enc_stage2(img_left_lv2_2 + resiual_left_lv3_bot)

        feature_right_lv2_1 = self.enc_stage2(img_right_lv2_1 + resiual_right_lv3_top)
        feature_right_lv2_2 = self.enc_stage2(img_right_lv2_2 + resiual_right_lv3_bot)

        
        feature_left_lv2_1, feature_right_lv2_1 = self.fuse_lv2_1(feature_left_lv2_1,feature_right_lv2_1)
        feature_left_lv2_2, feature_right_lv2_2 = self.fuse_lv2_2(feature_left_lv2_2,feature_right_lv2_2)


        feature_left_lv2 = torch.cat((feature_left_lv2_1, feature_left_lv2_2), 2) + feature_left_lv3
        feature_right_lv2 = torch.cat((feature_right_lv2_1, feature_right_lv2_2), 2) + feature_right_lv3


        residual_left_lv2 = self.dec_stage2(feature_left_lv2)
        residual_right_lv2 = self.dec_stage2(feature_right_lv2)

        # stage 3
        feature_left_lv1 = self.enc_stage3(img_left_lv1 + residual_left_lv2 ) + feature_left_lv2
        feature_right_lv1 = self.enc_stage3(img_right_lv1 + residual_right_lv2) + feature_right_lv2
        
        feature_left_lv1, feature_right_lv1 = self.fuse_lv1(feature_left_lv1,feature_right_lv1)

        dehaze_left = self.dec_stage3(feature_left_lv1)
        dehaze_right = self.dec_stage3(feature_right_lv1)

        return dehaze_left,dehaze_right,resiual_left_lv3_top,resiual_left_lv3_bot,resiual_right_lv3_top,resiual_right_lv3_bot,residual_left_lv2,residual_right_lv2





























