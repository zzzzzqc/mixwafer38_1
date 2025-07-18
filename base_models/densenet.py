import torch
import torch.nn as nn
import torchvision

# from torchsummary import summary

# 可选择的densenet模型
__all__ = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet264']



def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )




# BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv
class _DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            # growth_rate：增长率。一层产生多少个特征图
            nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)




class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size, drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        # 随着layer层数的增加，每增加一层，输入的特征图就增加一倍growth_rate
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)




# BN+1×1Conv+2×2AveragePooling
class _TransitionLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace, out_channels=plance, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition_layer(x)


class DenseNet(nn.Module):
    def __init__(self,inputchannels = 3, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16], num_classes=10):
        super(DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=inputchannels, places=init_channels)

        blocks * 4

        # 第一次执行特征的维度来自于前面的特征提取
        num_features = init_channels

        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        # 第1个transition 执行 _TransitionLayer（256,128）
        self.transition1 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        # num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        num_features = num_features // 2

        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition2 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        num_features = num_features // 2

        # 第3个DenseBlock有24个DenseLayer, 执行DenseBlock（24,256,32,4）
        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（1024,512）
        self.transition3 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        # num_features减少为原来的一半，执行第3回合之后，第4个DenseBlock的输入的feature应该是：num_features = 512
        num_features = num_features // 2

        # 第4个DenseBlock有16个DenseLayer, 执行DenseBlock（16,512,32,4）
        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def DenseNet121(num_classes,input_channels,):
    return DenseNet(inputchannels=input_channels,init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16],num_classes=num_classes)


def DenseNet169(num_classes,input_channels,):
    return DenseNet(inputchannels=input_channels,init_channels=64, growth_rate=32, blocks=[6, 12, 32, 32],num_classes=num_classes)


def DenseNet201(num_classes,input_channels,):
    return DenseNet(inputchannels=input_channels,init_channels=64, growth_rate=32, blocks=[6, 12, 48, 32],num_classes=num_classes)


def DenseNet264(num_classes,input_channels):
    return DenseNet(inputchannels=input_channels,init_channels=64, growth_rate=32, blocks=[6, 12, 64, 48],num_classes=num_classes)


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

    model = DenseNet121(38,1)
    out = model(img)
    print(out.shape)
    from thop import profile
    flops, params = profile(model, inputs=(img, ))
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M