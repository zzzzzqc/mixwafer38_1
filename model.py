import torch
from sympy.physics.units import planck_time
from torchsummary import summary

from torchinfo import summary
from oct_conv import *
from transformer import *
from atten import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 conv"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False, padding=0)


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,alpha,stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, First=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.first = First
        if self.first:
            self.ocb1 = FirstOctaveCBR(inplanes, width, kernel_size=(3, 3),alpha=alpha, norm_layer=norm_layer, padding=1)
        else:
            self.ocb1 = OctaveCBR(inplanes, width, kernel_size=(3, 3), alpha=alpha,norm_layer=norm_layer, padding=1)

        self.ocb2 = OctaveCB(width, planes * self.expansion, kernel_size=(3, 3), alpha=alpha,stride=stride, groups=groups,
                             norm_layer=norm_layer, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        if self.first:
            x_h_res, x_l_res = self.ocb1(x)
            x_h, x_l = self.ocb2((x_h_res, x_l_res))
        else:
            x_h_res, x_l_res = x
            x_h, x_l = self.ocb1((x_h_res, x_l_res))
            x_h, x_l = self.ocb2((x_h, x_l))

        if self.downsample is not None:
            x_h_res, x_l_res = self.downsample((x_h_res, x_l_res))

        x_h += x_h_res
        x_l += x_l_res

        x_h = self.relu(x_h)
        x_l = self.relu(x_l)

        return x_h, x_l


class BottleneckLast(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,alpha=0.5, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckLast, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Last means the end of two branch
        self.ocb1 = OctaveCBR(inplanes, width, kernel_size=(3, 3),alpha=alpha, padding=1)
        self.ocb2 = OctaveCB(width, planes * self.expansion, kernel_size=(3, 3),alpha=alpha, padding=1, stride=stride,
                             groups=groups, norm_layer=norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride

    def forward(self, x):

        x_h_res, x_l_res = x
        x_h, x_l = self.ocb1((x_h_res, x_l_res))

        x_h, x_l = self.ocb2((x_h, x_l))

        if self.downsample is not None:
            x_h_res = self.downsample((x_h_res, x_l_res))
        x_l = self.upsample(x_l)
        x_h = torch.cat((x_h, x_l), dim=1)
        x_h += x_h_res
        x_h = self.relu(x_h)

        return x_h


class BottleneckOrigin(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckOrigin, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)
        #
        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class OCtaveResNet(nn.Module):

    def __init__(self, block, layers, in_ch=3, num_classes=9, alpha=0.5,
                 zero_init_residual=False,groups=1, width_per_group=64, norm_layer=None,
                 axiss=144, channels=256, kernel_size=3, patch_size=(2, 2)
                 ):
        super(OCtaveResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_ch, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0],alpha=alpha, stride=1, norm_layer=norm_layer, First=True)

        self.conv_out_1 = LastOCtaveCBR(64, 64, kernel_size=(1, 1),alpha=alpha, stride=1, padding=0,  ##############
                                        groups=1, norm_layer=norm_layer)

        self.layer2 = self._make_layer(block, 128, layers[1],alpha=alpha, stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2],alpha=alpha, stride=2, norm_layer=norm_layer)
        self.conv_out_2 = LastOCtaveCBR(256, 256, kernel_size=(1, 1),alpha=alpha, stride=1, padding=0,  ##############
                                        groups=1, norm_layer=norm_layer)
        # self.layer4 = self._make_last_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        # self.conv_out_3 = LastOCtaveCBR(256, 256, kernel_size=(1, 1), stride=1, padding=0,  ##############
        #                                 groups=1, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 最后一层替换为MViT
        L = [2, 4, 6, 8, 10]
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(axiss, L[2], channels, kernel_size, patch_size, int(axiss * 2)))

        # 池化后的1x1卷积
        self.channelconv1 = nn.Conv2d(64, 256, 1, stride=1)
        self.channelconv2 = nn.Conv2d(256, 256, 1, stride=1)
        self.channelconv3 = nn.Conv2d(256, 256, 1, stride=1)

        # 通过 Pooling 层将高宽降低为 1x1,[b,128,1,1]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpoolatten = nn.AdaptiveAvgPool2d(4)

        self.SpatialAttention = SpatialAttention(64)
        self.CoordinateAttention = CoordinateAttention(256)
        self.ChannelAttention = ChannelAttention(256)

        self.convmvt = conv_1x1_bn(channels, int(channels * 4))

        self.pool = nn.AvgPool2d(2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                else:
                    pass
                # elif isinstance(m, BasicBlock):
                # nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks,alpha=0.5, stride=2, norm_layer=None, First=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                OctaveCB(in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),alpha=alpha,
                         stride=stride, padding=0)
            )

        layers = []
        layers.append(block(self.inplanes, planes,alpha, stride, downsample, self.groups,
                            self.base_width, norm_layer, First))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,alpha, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_last_layer(self, block, planes, blocks, stride=1, norm_layer=None):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                LastOCtaveCB(in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=(1, 1),
                             stride=stride, padding=0)
            )

        layers = []
        layers.append(BottleneckLast(self.inplanes, planes, stride, downsample, self.groups,
                                     self.base_width, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(BottleneckOrigin(self.inplanes, planes, groups=self.groups,
                                           base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_h1, x_l1 = self.layer1(x)

        #空间注意力
        h_low = self.conv_out_1((x_h1, x_l1))
        h_low = self.SpatialAttention(h_low)
        h_low = self.avgpoolatten(h_low)
        h_low = self.channelconv1(h_low)

        x_h2, x_l2 = self.layer2((x_h1, x_l1))

        x_h3, x_l3 = self.layer3((x_h2, x_l2))
        # #坐标注意力
        h_mid = self.conv_out_2((x_h3, x_l3))
        h_mid = self.CoordinateAttention(h_mid)
        h_mid = self.avgpoolatten(h_mid)
        h_mid = self.channelconv2(h_mid)

        x_h4 = self.mvit[0](h_mid)  #mvt,channelatten
        h_high = x_h4
        h_high = self.ChannelAttention(h_high)
        h_high = self.avgpoolatten(h_high)
        h_high = self.channelconv3(h_high)

        out_x = h_low + h_mid + h_high
        out_x = self.avgpool(out_x)
        out_x = out_x.view(out_x.size(0), -1)
        out_x = self.fc(out_x)

        return out_x


def model18(in_ch, num_classes,alpha=0.5):
    model = OCtaveResNet(Bottleneck, [2, 2, 2, 2],alpha=alpha, in_ch=in_ch, num_classes=num_classes)
    return model


def model34(in_ch, cls,alpha=0.5):
    model = OCtaveResNet(Bottleneck, [3, 4, 6, 3],alpha=alpha, in_ch=in_ch, num_classes=cls)
    return model


from thop import profile
from thop import clever_format


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input):
    flops = profile(model, inputs=(input,))[0] * 2
    flops = clever_format(flops, "%.2f")
    return flops


if __name__ == '__main__':
    img = torch.randn(1, 1, 64, 64)

    model = model18(in_ch=1, num_classes=10,alpha=0.5)
    out = model(img)
    print(out.shape)
    from thop import profile
    flops, params = profile(model, inputs=(img, ))
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M
