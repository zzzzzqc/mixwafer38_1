import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, axis, fn):
        super().__init__()
        self.norm = nn.LayerNorm(axis)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, axis, hidden_axis, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(axis, hidden_axis),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_axis, axis),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, axis, heads=8, axis_head=64, dropout=0.1):
        super().__init__()
        inner_axis = axis_head * heads
        project_out = not (heads == 1 and axis_head == axis)

        self.heads = heads
        self.scale = axis_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(axis, inner_axis * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_axis, axis),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        b, p, n, hd = q.shape
        q = q.view(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        k = k.view(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)
        v = v.view(b, p, n, self.heads, -1).permute(0, 1, 3, 2, 4)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)

        out = attn.matmul(v).permute(0, 1, 3, 2, 4).view(b, p, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, axis, depth, heads, axis_head, mlp_axis, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(axis, Attention(axis, heads, axis_head, dropout)),
                PreNorm(axis, FeedForward(axis, mlp_axis, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, axis, depth, channel, kernel_size, patch_size, mlp_axis, dropout=0.2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, axis)

        self.transformer = Transformer(axis, depth, 1, 32, mlp_axis, dropout)

        self.conv3 = conv_1x1_bn(axis, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        # Global representations
        n, c, h, w = x.shape

        # 确保张量是连续的
        x = x.permute(0, 3, 1, 2).contiguous().reshape((n, self.ph * self.pw, -1, c))
        x = self.transformer(x)
        x = x.reshape((n, h, -1, c)).permute(0,3,1, 2).contiguous()
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


if __name__ == "__main__":
    # 设置随机种子以便于结果的复现
    torch.manual_seed(0)

    # 创建一个随机初始化的输入张量，假设 batch size 是 1，通道数是 64，图像大小是 224x224
    input_tensor = torch.randn(1, 64, 224, 224)

    # 实例化 MobileViTBlock
    # 假设的参数：axis=64, depth=4, channel=64, kernel_size=3, patch_size=(4, 4), mlp_axis=256, dropout=0.2
    mobilevit_block = MobileViTBlock(axis=64, depth=4, channel=64, kernel_size=3, patch_size=(4, 4), mlp_axis=256, dropout=0.2)

    # 将输入张量传递给 MobileViTBlock 并获取输出
    output_tensor = mobilevit_block(input_tensor)

    # 打印输出张量的形状以确认是否正确
    print("Output tensor shape:", output_tensor.shape)