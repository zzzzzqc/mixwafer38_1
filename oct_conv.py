#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai Li(lxtpku@pku.edu.cn)
# Pytorch Implementation of Octave Conv Operation
# This version use nn.Conv2d because alpha_in always equals alpha_out

import torch
import torch.nn as nn


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        # print(in_channels,out_channels)
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.stride = stride
        # todo 这里对齐常规模型，实际通道数/2
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(int(in_channels) - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(int(in_channels) - int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

        # self.upsample = None  # 初始化时不定义具体的上采样层

    def forward(self, x):
        X_h, X_l = x
        if self.stride == 2:
            X_h, X_l = self.downsample(X_h), self.downsample(X_l)
        # print('oct input', X_h.shape, X_l.shape)
        X_h2l = self.downsample(X_h)
        # print('octconv1',X_h2l.shape)
        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)
        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)
        # print('octconv mid process',X_h2h.shape, X_l2h.shape,X_l2l.shape, X_h2l.shape )
        if X_l2h.shape[-2:] != X_h2h.shape[-2:]:
            target_size = (X_h2h.size(-2), X_h2h.size(-1))
            X_l2h = torch.nn.functional.interpolate(X_l2h, size=target_size, mode='nearest')
        if X_h2l.shape[-1:] != X_l2l.shape[-1:]:
            target_size = (X_l2l.size(-2), X_l2l.size(-1))
            X_h2l = torch.nn.functional.interpolate(X_h2l, size=target_size, mode='nearest')
        # X_l2h = self.upsample(X_l2h)
        # print('oct_conv2', X_l2h.shape, X_h2h.shape)
        X_h = X_h2h + X_l2h 
        X_l = X_l2l + X_h2l 
        # print('octconv3',X_h.shape, X_l.shape)
        return X_h, X_l


class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)

        X_h = x
        # print(X_h.shape)
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l


class FirstOctaveConvLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConvLeakyReLU, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)

        X_h = x
        # print(X_h.shape)
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(X_l2h)

        X_h = X_h2h + X_l2h

        return X_h


class FreqOctaveCR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(FreqOctaveCR, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(x_h)
        x_l = self.relu(x_l)
        return x_h, x_l
    

class FreqOctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(FreqOctaveCB, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l


class OctaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCBR, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return x_h, x_l


class OctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCB, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,
                               groups, bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l


class FirstOctaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCBR, self).__init__()
        self.conv = FirstOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups,
                                    bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return x_h, x_l


class LastOCtaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCBR, self).__init__()
        self.conv = LastOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups,
                                   bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        return x_h


class out_OCtaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(out_OCtaveConv, self).__init__()
        self.conv = LastOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups,
                                   bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        return x_h


class FirstOctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCB, self).__init__()
        self.conv = FirstOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups,
                                    bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l


class LastOCtaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCB, self).__init__()
        self.conv = LastOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups,
                                   bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.bn_h(x_h)
        return x_h


if __name__ == '__main__':
    # nn.Conv2d
    high = torch.Tensor(1, 1, 32, 32)
    low = torch.Tensor(1, 1, 16, 16)
    # test Oc conv
    OCconv = OctaveConv(kernel_size=(3, 3), in_channels=2, out_channels=2, bias=False, stride=2, alpha=0.5)
    i = high, low
    x_out, y_out = OCconv(i)
    print(x_out.size())
    print(y_out.size())

    i = torch.Tensor(1, 3, 512, 512)
    FOCconv = FirstOctaveConv(kernel_size=(3, 3), in_channels=3, out_channels=128)
    x_out, y_out = FOCconv(i)
    print("First: ", x_out.size(), y_out.size())
    # test last Octave Cov
    LOCconv = LastOctaveConv(kernel_size=(3, 3), in_channels=128, out_channels=128, alpha=0.5)
    i = high, low
    out = LOCconv(i)
    print("Last: ", out.size())

    # test OCB
    ocb = OctaveCB(in_channels=128, out_channels=128, alpha=0.5)
    i = high, low
    x_out_h, y_out_l = ocb(i)
    print("OCB:", x_out_h.size(), y_out_l.size())

    # test last OCB
    ocb_last = LastOCtaveCBR(128, 128, alpha=0.5)
    i = high, low
    x_out_h = ocb_last(i)
    print("Last OCB", x_out_h.size())
