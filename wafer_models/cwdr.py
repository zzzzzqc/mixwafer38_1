from torch import nn

from torch import nn


# todo 来自github "https://github.com/Luo-wj/CWDR-Net"
# "Composite Wafer Defect Recognition Framework Based on Multiview Dynamic Feature Enhancement With Class-Specific Classifier  "

class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, relu=True, bn=True):
        super(BasicConv1D, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Class_Specific_Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Class_Specific_Classifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1 = BasicConv(input_dim, num_classes, 1, stride=1, padding=0, relu=False,
                               bn=False)  # bn must be False here
        self.conv2 = BasicConv1D(num_classes, 4, 3, stride=1, padding=1, relu=True, bn=True)
        self.conv3 = BasicConv1D(4, num_classes, 3, stride=1, padding=1, relu=True, bn=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(2)
        identity = x

        scale = self.conv2(x)
        scale = self.conv3(scale)
        scale = self.sigmoid(scale)

        out = identity * scale
        out = self.avgpool(out)
        out = out.squeeze(-1)

        return out


import torch
from torch import nn


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Pooling(nn.Module):
    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max, avg), dim=1)


class Branch(nn.Module):
    def __init__(self, kernel_size=7):
        super(Branch, self).__init__()
        self.compress = Pooling()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class MVDFE_Module(nn.Module):
    def __init__(self, kernel_size=7):
        super(MVDFE_Module, self).__init__()
        self.HW_branch = Branch(kernel_size=kernel_size)
        self.CW_branch = Branch(kernel_size=kernel_size)
        self.HC_branch = Branch(kernel_size=kernel_size)

    def forward(self, x):
        # HW branch recalibrate
        HW_out = self.HW_branch(x)

        # CW branch recalibrate
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        CW_out = self.CW_branch(x_perm1)
        CW_out = CW_out.permute(0, 2, 1, 3).contiguous()

        # HC branch recalibrate
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        HC_out = self.HC_branch(x_perm2)
        HC_out = HC_out.permute(0, 3, 2, 1).contiguous()

        final_out = 1 / 3 * (HW_out + CW_out + HC_out)

        return final_out


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, attention=False, attention_kernel_size=7):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.attention = attention

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if self.attention == True:
            self.mvdfe = MVDFE_Module(kernel_size=attention_kernel_size)

        if stride == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        if self.stride == 2:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        if self.attention == True:
            out = self.mvdfe(out)

        return out


class CWDR(nn.Module):
    def __init__(self, num_of_classes, in_channel=1):
        super(CWDR, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = BasicBlock(in_ch=64, out_ch=64, stride=1, attention=True, attention_kernel_size=7)
        self.block2 = BasicBlock(in_ch=64, out_ch=64, stride=1, attention=True, attention_kernel_size=7)
        self.block3 = BasicBlock(in_ch=64, out_ch=64, stride=1, attention=True, attention_kernel_size=7)

        self.ac = Class_Specific_Classifier(input_dim=64, num_classes=num_of_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.ac(x)

        x = self.sigmoid(x)

        return x


import torch
import torch.nn as nn
import numpy as np

class GUIDELayer(nn.Module):
    def __init__(self, defect_types, input_size=(52, 52)):
        super(GUIDELayer, self).__init__()
        self.defect_types = defect_types
        self.input_size = input_size

        # 初始化 GUIDE Layer 的权重
        # 每个缺陷类型对应一个 (H, W) 的权重矩阵，最后一个通道是原始输入
        self.weights = nn.ParameterList()
        for _ in range(defect_types):
            # 使用随机初始化（实际应替换为单缺陷数据集的均值）
            weight = nn.Parameter(torch.rand(1, *input_size))
            self.weights.append(weight)
        # 最后一个通道是原始输入
        self.weights.append(nn.Parameter(torch.ones(1, *input_size)))

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == len(self.weights), "输入通道数必须与 GUIDE Layer 通道数一致"

        # 逐元素相乘
        enhanced_features = []
        for i in range(len(self.weights)):
            weight = self.weights[i].expand(B, 1, H, W)
            enhanced = x[:, i:i+1, :, :] * weight  # (B, 1, H, W)
            enhanced_features.append(enhanced)
        
        # 拼接所有增强后的特征
        return torch.cat(enhanced_features, dim=1)


def CWDR_model(num_of_classes):
    model = CWDR(num_of_classes=num_of_classes)

    return model

def CWDR_guidance_layer(defect_types, input_size, num_of_classes):
    model = nn.Sequential(
        GUIDELayer(defect_types=defect_types, input_size=input_size),
        CWDR_model(num_of_classes=num_of_classes)
)
    return model



if __name__ == "__main__":
    model = CWDR_model(num_of_classes=38)
    # 测试前向传播
    x = torch.randn(1, 1, 64, 64)
    print("Mixed模型输出尺寸:", model(x).shape)
    from thop import profile
    flops, params = profile(model, inputs=(x, ))
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M
