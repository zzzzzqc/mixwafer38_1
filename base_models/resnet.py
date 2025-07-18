# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:28:39 2018
@author: tshzzz
"""

from joblib import PrintTime
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format



class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,in_planes,planes,stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.pass_by = nn.Sequential()
        if stride != 1 or in_planes != planes*self.expansion:
            self.pass_by = nn.Sequential(
                        nn.Conv2d(in_planes,planes*self.expansion,kernel_size=1,stride=stride,bias=False),
                        nn.BatchNorm2d(planes*self.expansion)
                    )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.pass_by(x)
        out = F.relu(out)
        return out


class BottlenBlock(nn.Module):
    expansion = 4
    def __init__(self,in_planes,planes,stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes,planes*self.expansion,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.pass_by = nn.Sequential()

        if stride != 1 or in_planes != planes*self.expansion:
            self.pass_by = nn.Sequential(
                        nn.Conv2d(in_planes,planes*self.expansion,kernel_size=1,stride=stride,bias=False),
                        nn.BatchNorm2d(planes*self.expansion)
                    )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.pass_by(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,block,layers,in_ch = 3, num_class=9):
        super(ResNet,self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_ch,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_class)


    def _make_layer(self,block,plane,layer,stride=1):
        layers = []
        layers.append(block(self.inplanes,plane,stride))
        self.inplanes = plane*block.expansion
        for i in range(1,layer):
            layers.append(block(self.inplanes,plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18(in_ch=1,num_class = 9):
    return ResNet(BasicBlock, [2,2,2,2],in_ch, num_class)

def ResNet50(num_class = 9):
    model = ResNet(BottlenBlock, [3, 4, 6, 3], num_class)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def count_flops(model, input):
    flops = profile(model, inputs=(input, ))[0] * 2
    flops = clever_format(flops, "%.2f")
    return flops

if __name__ == '__main__':
    img = torch.randn(1, 1, 64, 64)
    model = ResNet18(1,38)
    out = model(img)
    from thop import profile
    flops, params = profile(model, inputs=(img, ))
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M
    # time_start=time.time()
    # #在这里运行模型
    # time_end=time.time()
    # print('totally cost',time_end-time_start)
    # total = count_parameters(model)
    # flops = count_flops(model, img)
    # print("Number of parameter: %.2fM" % (total/1e6))
    # print(flops)
