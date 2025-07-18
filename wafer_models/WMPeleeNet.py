import torch
import torch.nn as nn
import torch.nn.functional as F

# todo  复现 "Wafer map defect patterns classification based on a lightweight network and data augmentation"

class StemBlock(nn.Module):
    def __init__(self,in_ch):
        super(StemBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        concat = torch.cat([branch1, branch2], dim=1)
        out = self.fusion(concat)
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = torch.cat([branch1, branch2], dim=1)  # 输出通道数: 2 * growth_rate
        return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            layer = DenseLayer(channels, growth_rate)
            self.layers.append(layer)
            channels += 2 * growth_rate  # 每层增加 2*growth_rate 个通道

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)  # Dense连接
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        # Channel attention
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(spatial_att))
        x = x * spatial_att
        return x

class WMPeleeNet(nn.Module):
    def __init__(self,in_ch=1, num_classes=9):
        super(WMPeleeNet, self).__init__()
        # 配置参数 (根据论文Table 1调整)
        growth_rate1 = 24  # DenseBlock1增长率
        growth_rate2 = 32  # DenseBlock2增长率
        
        # Stem Block
        self.stem = StemBlock(in_ch=in_ch)  # 输出: 56x56x32
        
        # DenseBlock1 (2层)
        self.dense1 = DenseBlock(32, num_layers=2, growth_rate=growth_rate1)  # 输出: 56x56x[32 + 2*(24*2)] = 56x56x128
        
        # TransitionLayer1 (包含下采样)
        self.trans1 = TransitionLayer(128, 128, downsample=True)  # 输出: 28x28x128
        
        # DenseBlock2 (2层)
        self.dense2 = DenseBlock(128, num_layers=2, growth_rate=growth_rate2)  # 输出: 28x28x[128 + 2*(32*2)] = 28x28x256
        
        # TransitionLayer2 (不进行下采样)
        self.trans2 = TransitionLayer(256, 256, downsample=False)  # 输出: 28x28x256
        
        # CBAM注意力模块
        self.cbam = CBAM(256)
        
        # 分类头
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)          # 56x56x32
        x = self.dense1(x)        # 56x56x128
        x = self.trans1(x)        # 28x28x128
        x = self.dense2(x)        # 28x28x256
        x = self.trans2(x)        # 28x28x256
        x = self.cbam(x)          # 28x28x256
        x = self.global_avg_pool(x)  # 1x1x256
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


from thop import profile
from thop import clever_format

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input):
    flops = profile(model, inputs=(input,))[0] * 2
    flops = clever_format(flops, "%.2f")
    return flops

# 测试网络
if __name__ == "__main__":
    model = WMPeleeNet(in_ch=1,num_classes=38)
    x = torch.randn(1, 1, 64, 64)
    print("Mixed模型输出尺寸:", model(x).shape)
    from thop import profile
    flops, params = profile(model, inputs=(x, ))
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M
    # total = count_parameters(model)
    # flops = count_flops(model, x)
    # print("Number of parameter: %.2fM" % (total/1e6))
    # print(flops)