#https://github.com/Tshzzz/cifar10.classifer

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:23:31 2018
@author: tshzzz
"""

from pyexpat import model
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format


def conv_dw(inplane,outplane,stride=1):
    return nn.Sequential(
        nn.Conv2d(inplane,inplane,kernel_size = 3,groups = inplane,stride=stride,padding=1),
        nn.BatchNorm2d(inplane),
        nn.ReLU(),
        nn.Conv2d(inplane,outplane,kernel_size = 1,groups = 1,stride=1),
        nn.BatchNorm2d(outplane),
        nn.ReLU(
        ))

def conv_bw(inplane,outplane,kernel_size = 3,stride=1):
    return nn.Sequential(
        nn.Conv2d(inplane,outplane,kernel_size = kernel_size,groups = 1,stride=stride,padding=1),
        nn.BatchNorm2d(outplane),
        nn.ReLU()
        )


class MobileNet(nn.Module):
    def __init__(self,num_class=9):
        super(MobileNet,self).__init__()

        layers = []
        layers.append(conv_bw(3,32,3,2))
        layers.append(conv_dw(32,64,1))
        layers.append(conv_dw(64,128,2))
        layers.append(conv_dw(128,128,1))
        layers.append(conv_dw(128,256,2))
        layers.append(conv_dw(256,256,1))
        layers.append(conv_dw(256,512,2))

        for i in range(5):
            layers.append(conv_dw(512,512,1))
        layers.append(conv_dw(512,1024,2))
        layers.append(conv_dw(1024,1024,1))

        self.feature = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(7)
        self.classifer = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1024,num_class)
                )
      
    def forward(self,x):
        out = self.feature(x)
        out = self.avgpool(out) 
        out = out.view(-1,1024)
        out = self.classifer(out)
        return out

def mobilenet():
    return MobileNet(num_class=38)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input):
    flops = profile(model, inputs=(input, ))[0] * 2
    flops = clever_format(flops, "%.2f")
    return flops

if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    model = mobilenet()
    out = model(img)
    from thop import profile
    flops, params = profile(model, inputs=(img, ))    
    print(f'params is {params/1e6} M') #flops单位G，para单位M
    print(f'flops is {flops/1e9} G') #flops单位G，para单位M
    total = count_parameters(model)
    flops = count_flops(model, img)
    print("Number of parameter: %.2fM" % (total/1e6))
    print(flops)