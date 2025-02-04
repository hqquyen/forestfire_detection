
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from mmdet.registry import MODELS

# from ..builder import BACKBONES

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
#modify
class wf_inception_module(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv_1 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 1, padding = 0, dilation=1, groups=in_channels//4)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.conv_1_3 = nn.Sequential(
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (1,3), padding = (0,1), dilation=1, groups=in_channels//4),
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (3,1), padding = (1,0), dilation=1, groups=in_channels//4)
        )      
        self.conv_1_5 = nn.Sequential(
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (1,5), padding = (0,2), dilation=1, groups=in_channels//4),
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (5,1), padding = (2,0), dilation=1, groups=in_channels//4)
        )  
        self.max_3_1 = nn.Sequential(
                nn.MaxPool2d(3, 1, 1),
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 1, padding = 0, dilation=1, groups=in_channels//4)
        )
    def forward(self, x):
        a,b,c,d= torch.split(x, self.in_channels//4, dim = 1)
        out_1 = self.conv_1(a)    
        out_2 = self.conv_1_3(b)
        out_3 = self.conv_1_5(c)
        out_4 = self.max_3_1(d)
        out_5 = torch.cat((out_1, out_2, out_3, out_4), 1)
        #out_5 = out_5 + self.gap(x) 
        return out_5

# #main
# class wf_inception_module(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.conv_1 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 1, padding = 0, dilation=1, groups= in_channels//4)
#         #self.gap = nn.AdaptiveAvgPool2d((1,1))
#         self.conv_1_3 = nn.Sequential(
#                 nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (1,3), padding = (0,1), dilation=1, groups=in_channels//4),
#                 nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (3,1), padding = (1,0), dilation=1, groups=in_channels//4)
#         )      
#         self.conv_1_5 = nn.Sequential(
#                 nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (1,3), padding = (0,2), dilation=2, groups=in_channels//4),
#                 nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (3,1), padding = (2,0), dilation=2, groups=in_channels//4)
#         )  
#         self.max_3_1 = nn.Sequential(
#                 nn.MaxPool2d(3, 1, 1),
#                 nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 1, padding = 0, dilation=1, groups=in_channels//4)
#         )
#     def forward(self, x):
#         a,b,c,d= torch.split(x, self.in_channels//4, dim = 1)
#         out_1 = self.conv_1(a)    
#         out_2 = self.conv_1_3(b)
#         out_3 = self.conv_1_5(c)
#         out_4 = self.max_3_1(d)
#         out_5 = torch.cat((out_1, out_2, out_3, out_4), 1)
#         #out_5 = out_5 + self.gap(x) 
#         return out_5

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        #self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dwconv = wf_inception_module(dim)
        self.norm = nn.BatchNorm2d(dim, eps=.001)
        self.conv1 = nn.Conv2d(dim, 4*dim, kernel_size=1, stride=1)
        self.conv12 = nn.Conv2d(4*dim, dim, kernel_size=1, stride=1)
        ##self.norm = LayerNorm(dim, eps=1e-6)
        #self.norm = nn.BatchNorm2d(dim, eps=.001)
        #self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        #self.act = nn.GELU()
        self.act = nn.ReLU(inplace=True)
        ###self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        #x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.conv1(x)
        #x = self.pwconv1(x)
        x = self.act(x)
        x = self.conv12(x)
        #x = self.conv1(x)
        #x = self.pwconv2(x)
        #if self.gamma is not None:
            #x = self.gamma * x
        #x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

@MODELS.register_module()
class FF_Backbone(nn.Module):
    """ ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1, 
                 depths=[1, 1, 3, 1], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        
        self.ca2 = ChannelAttention(384)
        self.ca3 = ChannelAttention(768)
        self.sa = SpatialAttention()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0], eps=.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(dims[0], eps=.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(dims[0], eps=.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)            
        )
        # stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=2),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        #     nn.GELU(),
        #     nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        #     nn.GELU(),
        #     nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        #     nn.GELU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)            
        # )
        # self.downsample_layers = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer  = nn.Sequential(
                ##LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size = 1, padding = 0, dilation=1),
                nn.BatchNorm2d(dims[i+1], eps=.001),
                nn.ReLU(inplace=True),  
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)               
            )
            # downsample_layer = nn.Sequential(
            #         LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
            #         nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            # )
            # downsample_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 2:
                x = self.ca2(x)*x
                x = self.sa(x)*x
            if i == 3:
                x = self.ca3(x)*x
                x = self.sa(x)*x
            outs.append(x)
        #return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return outs


    # def forward(self, x):
    #     outs = []
    #     for i in range(4):
    #         x = self.downsample_layers[i](x)
    #         x = self.stages[i](x)
    #         outs.append(x)
    #     #return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
    #     return outs

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     #x = self.head(x)
    #     return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


