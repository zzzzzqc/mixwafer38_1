import torch
import torch.nn as nn
import torch.nn.functional as F

# SpatialAttention，低层
class SpatialAttention(nn.Module):
    def __init__(self, in_channel):
        super(SpatialAttention, self).__init__()
        self.in_channel = in_channel
        self.maxpool = nn.MaxPool2d(1, 1)
        self.avgpool = nn.AvgPool2d(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv_l1 = nn.Conv2d(in_channels=int(self.in_channel//2), out_channels=1, kernel_size=(3,1), stride=1, padding=(1,0))
        self.conv_l2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,3), stride=1, padding=(0,1))
        self.conv_r1 = nn.Conv2d(in_channels=int(self.in_channel//2), out_channels=1, kernel_size=(1,3), stride=1, padding=(0,1))
        self.conv_r2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,1), stride=1, padding=(1,0))
        self.conv_layer = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=3, groups=2, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.act = nn.ReLU()

    def forward(self, inputs):
        identity = inputs
        x = inputs

        output_data = self.conv_layer(x)
        x_l = output_data[:, :int(self.in_channel//2), :, :]
        x_r = output_data[:, int(self.in_channel//2):, :, :]

        x_l = self.conv_l1(x_l)
        x_l = self.bn(x_l)
        x_l = self.act(x_l)
        x_l = self.conv_l2(x_l)
        x_l = self.bn(x_l)
        x_l = self.act(x_l)
        x_l = self.avgpool(x_l)

        x_r = self.conv_r1(x_r)
        x_r = self.bn(x_r)
        x_r = self.act(x_r)
        x_r = self.conv_r2(x_r)
        x_r = self.bn(x_r)
        x_r = self.act(x_r)
        x_r = self.maxpool(x_r)

        x = torch.add(x_l, x_r)
        x = self.sigmoid(x)
        x = identity * x
        return x

# CoordinateAttention(中间层)
class CoordinateAttention(nn.Module):
    def __init__(self, in_ch, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_ch // reduction)
        self.conv1 = nn.Conv2d(in_ch, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, in_ch, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_ch, kernel_size=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out

# ChannelAttention(高层)
class ChannelAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.shared_MLP = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(gate_channels, int(gate_channels//reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(gate_channels//reduction_ratio), gate_channels),
            nn.ReLU()
        )

    def Channel_Attention(self, x):
        b, c, h, w = x.size()

        x_m = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        x_a = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))


        mlp_m = self.shared_MLP(x_m)
        mlp_a = self.shared_MLP(x_a)

        # 确保 MLP 的输出尺寸正确
        mlp_m = mlp_m.view(b, c, 1, 1)
        mlp_a = mlp_a.view(b, c, 1, 1)

        cc = torch.add(mlp_a, mlp_m)
        mc = self.sigmoid(cc)
        return mc

    def forward(self, x):
        identity = x
        mc = self.Channel_Attention(x)
        out = mc * identity

        return out

# 测试代码
if __name__ == "__main__":
    # 创建一个随机初始化的输入张量
    input_tensor = torch.randn(1, 256, 32, 32)

    # 创建每个注意力层的实例
    spatial_attention = SpatialAttention(in_channel=256)
    coord_attention = CoordinateAttention(in_ch=256)
    channel_attention = ChannelAttention(gate_channels = 256)

    # 通过每个层进行前向传播
    spatial_output = spatial_attention(input_tensor)
    coord_output = coord_attention(input_tensor)
    channel_output = channel_attention(input_tensor)

    # 打印输出形状以确认是否正确
    print("SpatialAttention output shape:", spatial_output.shape)
    print("CoordinateAttention output shape:", coord_output.shape)
    print("ChannelAttention output shape:", channel_output.shape)