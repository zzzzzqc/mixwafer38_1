"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from thop import profile
from thop import clever_format

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features,num_class=9):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x) 
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, input_channel, batch_norm=True):
    layers = []

    # input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11():
    return VGG(make_layers(cfg['A']))

def vgg13():
    return VGG(make_layers(cfg['B']))

def vgg16(in_channel = 1,num_class = 38):
    return VGG(make_layers(cfg['D'],input_channel=in_channel), num_class=num_class)

def vgg19():
    return VGG(make_layers(cfg['E']))



# batch size的大小不回影响参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input):
    flops = profile(model, inputs=(input, ))[0] * 2
    flops = clever_format(flops, "%.2f")
    return flops

if __name__ == '__main__':
    img = torch.randn(1, 1, 64, 64 )
    model = vgg16(in_channel=1, num_class=38)
    out = model(img)
    total = count_parameters(model)
    flops = count_flops(model, img)
    print("Number of parameter: %.2fM" % (total/1e6))
    print(flops)
   
