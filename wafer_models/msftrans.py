import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

# todo 复现 "Mixed-Type Wafer Defect Recognition With Multi-Scale Information Fusion Transformer"

class PABlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2, dilation_rate=2):
        super().__init__()
        reduced_channels = in_channels // reduction_ratio
        
        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, 
                              padding=dilation_rate, dilation=dilation_rate)
        self.conv3 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.bn2 = nn.BatchNorm2d(reduced_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        
        # 通道压缩
        x = F.relu(self.bn1(self.conv1(x)))
        # 空洞卷积捕获空间特征
        x = F.relu(self.bn2(self.conv2(x)))
        # 通道恢复
        x = self.bn3(self.conv3(x))
        # 生成注意力权重
        attention_weights = self.sigmoid(x)
        
        # 残差连接
        return (1 + attention_weights) * identity

class MSFNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # 第一个模块
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pa_block1 = PABlock(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个模块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pa_block2 = PABlock(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        # 输入: (B, 1, 52, 52)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pa_block1(x)
        x = self.pool1(x)  # (B, 32, 26, 26)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pa_block2(x)
        x = self.pool2(x)  # (B, 64, 13, 13)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding size must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        # 残差连接 + 层归一化
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LearnableThreshold(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 初始化为0.5
        self.threshold = nn.Parameter(torch.ones(num_classes) * 0.5)
        self.label_cardinality = None
        
    def set_label_cardinality(self, labels):
        # labels: (B, num_classes)
        self.label_cardinality = labels.sum() / labels.size(0)
        
    def forward(self, logits):
        # 计算T-loss
        t_loss = self.label_cardinality - (logits.sigmoid() > self.threshold).float().mean()
        return logits, t_loss

class MSFTransformer(nn.Module):
    def __init__(self, num_classes=8,image_size = 64, embed_dim=64, num_heads=8, num_layers=8):
        super().__init__()
        # MSF-Network
        self.msf_network = MSFNetwork()
        
        # Transformer
        self.fea_size = image_size//4
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1,  self.fea_size * self.fea_size  + 1, embed_dim))
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # cls
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # LearnableThreshold
        self.threshold = LearnableThreshold(num_classes)
        
    def forward(self, x, labels=None):
        # MSF-Network
        x = self.msf_network(x)  
        # print('after msf_network', x.shape)
        
        # flatten 
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c') 
        
        # cls_token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)

        x = torch.cat([cls_tokens, x], dim=1)  

        # pos_embed
        x = x + self.pos_embed
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # cls_head
        x = self.norm(x)
        cls_token = x[:, 0]  
        logits = self.head(cls_token)  # (B, num_classes)
        
    
        if labels is not None:
            self.threshold.set_label_cardinality(labels)
            logits, t_loss = self.threshold(logits)
            return logits, t_loss
        
        return logits

if __name__ == '__main__':
    model = MSFTransformer(num_classes=38,image_size=64, embed_dim=64, num_heads=8, num_layers=6)
    x = torch.randn(1, 1, 64, 64)
    print("模型输出尺寸:", model(x).shape)

    
    from thop import profile
    flops, params = profile(model, inputs=(x, ))
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M

    pass