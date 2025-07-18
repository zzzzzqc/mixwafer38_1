import torch
import torch.nn as nn
import torch.nn.functional as F

# todo 复现" WaferSegClassNet - A light-weight network for classification and segmentation of semiconductor wafer defects "

class SeparableConv2d(nn.Module):
    """PyTorch实现TensorFlow的SeparableConv2D"""
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', activation=None):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if self.activation == 'relu':
            x = F.relu(x)
        return x

class ConvBlock(nn.Module):
    """下采样卷积块 含MaxPool"""
    def __init__(self, in_channels, conv_channels, kernel_size=(3,3), 
                 pool_stride=(2,2), dropout_rate=0.1, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.sep_conv = SeparableConv2d(conv_channels, conv_channels, kernel_size, padding, activation='relu')
        self.bn2 = nn.BatchNorm2d(conv_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_stride, stride=pool_stride)
        
    def forward(self, x):
        # 第一层（保留作为跳跃连接）
        layer1 = F.relu(self.conv1(x))
        layer2 = self.bn1(layer1)
        layer3 = self.dropout(layer2)
        # 可分离卷积层
        layer4 = self.sep_conv(layer3)
        layer5 = self.bn2(layer4)
        layer6 = self.pool(layer5)
        return layer1, layer6  # (skip_connection, output)

class GapConvBlock(ConvBlock):
    """下采样卷积块（含AvgPool）"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = nn.AvgPool2d(kernel_size=self.pool.kernel_size, stride=self.pool.stride)

class TerminalConvBlock(nn.Module):
    """终端卷积块（无池化）"""
    def __init__(self, in_channels, conv_channels, kernel_size=(3,3), dropout_rate=0.1, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.sep_conv = SeparableConv2d(conv_channels, conv_channels, kernel_size, padding, activation='relu')
        self.bn2 = nn.BatchNorm2d(conv_channels)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.sep_conv(x)
        x = self.bn2(x)
        return x

class TransposeConvBlock(nn.Module):
    """上采样转置卷积块"""
    def __init__(self, in_channels, skip_channels, conv_channels, 
                 kernel_size=(3,3), transpose_kernel_size=(2,2), 
                 dropout_rate=0.1, padding='same', transpose_strides=(2,2)):
        super().__init__()
        # 转置卷积（上采样）
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels, conv_channels, transpose_kernel_size, 
            stride=transpose_strides, padding=padding
        )
        # 合并后的卷积块
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_channels + skip_channels, conv_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(conv_channels),
            nn.Dropout2d(dropout_rate),
            SeparableConv2d(conv_channels, conv_channels, kernel_size, padding, activation='relu'),
            nn.BatchNorm2d(conv_channels)
        )
        
    def forward(self, x, skip):
        # 上采样
        x = self.transpose_conv(x)
        # 拼接跳跃连接
        x = torch.cat([x, skip], dim=1)  # 沿通道维度拼接
        x = self.conv_block(x)
        return x

class Encoder(nn.Module):
    """特征提取编码器"""
    def __init__(self, in_channels=3):
        super().__init__()
        # 下采样路径
        self.down1 = ConvBlock(in_channels, 8)
        self.down2 = ConvBlock(8, 16)
        self.down3 = ConvBlock(16, 16)
        self.down4 = GapConvBlock(16, 32)
        self.terminal = TerminalConvBlock(32, 64)
        
    def forward(self, x):
        # 获取各层输出和跳跃连接
        s1, d1 = self.down1(x)
        s2, d2 = self.down2(d1)
        s3, d3 = self.down3(d2)
        s4, d4 = self.down4(d3)
        t1 = self.terminal(d4)
        return s1, s2, s3, s4, t1

class Decoder(nn.Module):
    """分割掩码解码器"""
    def __init__(self, encoder,SEG_CLS=1):
        super().__init__()
        # 上采样路径
        self.up1 = TransposeConvBlock(64, 32, 32)  # 输入64, 跳跃32, 输出32
        self.up2 = TransposeConvBlock(32, 16, 16)
        self.up3 = TransposeConvBlock(16, 16, 16)
        self.up4 = TransposeConvBlock(16, 8, 8)
        # 最终分割输出层
        self.seg_out = nn.Sequential(
            nn.Conv2d(8, out_channels=SEG_CLS, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, t1, skips):
        s1, s2, s3, s4 = skips
        u1 = self.up1(t1, s4)
        u2 = self.up2(u1, s3)
        u3 = self.up3(u2, s2)
        u4 = self.up4(u3, s1)
        return self.seg_out(u4)

class WaferSegClassNet(nn.Module):
    """完整模型（含分类和分割双输出）"""
    def __init__(self, in_channels=3, num_class=1000,frozen = False):
        super().__init__()
        # 编码器
        self.frozen = frozen
        self.encoder = Encoder(in_channels)
        
        # 解码器（分割）遵循原论文设计，在涉及到分类分割时解码器已经提前训练好，这里直接保留分类网络，不使用分割网络
        self.decoder = Decoder(self.encoder)
        
        # 分类头
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_class)
        )
        
        # 冻结编码器
        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 编码器前向传播
        s1, s2, s3, s4, t1 = self.encoder(x)
        
        # 分割解码
        # seg_out = self.decoder(t1, [s1, s2, s3, s4])
        
        # 分类预测
        cls_out = self.cls_head(t1)
        
        # return seg_out, cls_out
        return cls_out



from thop import profile
from thop import clever_format

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input):
    flops = profile(model, inputs=(input,))[0] * 2
    flops = clever_format(flops, "%.2f")
    return flops

# 使用示例
if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaferSegClassNet(in_channels=1, num_class=38)
    # 测试前向传播
    x = torch.randn(1, 1, 64, 64)
    print("Mixed模型输出尺寸:", model(x).shape)
    from thop import profile
    flops, params = profile(model, inputs=(x, ))
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M
    # total = count_parameters(model)
    # flops = count_flops(model, x)
    # print("Number of parameter: %.2fM" % (total/1e6))
    # print(flops)