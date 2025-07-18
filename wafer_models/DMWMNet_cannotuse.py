import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DropBlock2d

# todo 复现"   ""


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        return x * self.sigmoid(out.unsqueeze(-1).unsqueeze(-1))

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(concat))
        return x * att

class StemBlock(nn.Module):
    """Stem Block for initial downsampling and feature extraction"""
    def __init__(self, in_channels=1, out_channels=32):
        super(StemBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.path1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        )
        self.path2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.path1(x)
        p2 = self.path2(x)
        x = torch.cat([p1, p2], dim=1)
        return self.final_conv(x)

class DenseBlock(nn.Module):
    """Improved Dense Block with triple pathway structure"""
    def __init__(self, in_channels, loop=2):
        super(DenseBlock, self).__init__()
        self.loop = loop
        self.pathways = nn.ModuleList([
            # Skip connection
            nn.Identity(),
            # Small-scale features path
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels*2, kernel_size=1),
                DropBlock2d(0.1),
                nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
                DropBlock2d(0.1)
            ),
            # Large-scale features path
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels*2, kernel_size=1),
                DropBlock2d(0.1),
                nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
                DropBlock2d(0.1),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                DropBlock2d(0.1)
            )
        ])
        
    def forward(self, x):
        for _ in range(self.loop):
            outs = [path(x) for path in self.pathways]
            # Sum all pathways (element-wise addition)
            x = torch.stack(outs, dim=0).sum(dim=0)
        return x

class TransitionLayer(nn.Module):
    """Transition Layer for feature compression"""
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)

class BasicBlock(nn.Module):
    """Fundamental building block of DMWMNet"""
    def __init__(self, in_channels=1):
        super(BasicBlock, self).__init__()
        # Stem Block
        self.stem = StemBlock(in_channels, 32)
        
        # Dense Block 1
        self.dense1 = DenseBlock(32)
        self.trans1 = TransitionLayer(32, 256)
        
        # Dense Block 2
        self.dense2 = DenseBlock(256)
        self.trans2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256)
        )
        
        # CBAM Attention
        self.cbam = CBAM(256)
        
    def forward(self, x):
        x = self.stem(x)         # [B, 32, 13, 13]
        x = self.dense1(x)       # [B, 32, 13, 13]
        x = self.trans1(x)       # [B, 256, 6, 6]
        x = self.dense2(x)       # [B, 256, 6, 6]
        x = self.trans2(x)       # [B, 256, 6, 6]
        return self.cbam(x)      # [B, 256, 6, 6]

class Branch1(nn.Module):
    """Branch 1: Basic Defect Discrimination & Defect Type Detection"""
    def __init__(self):
        super(Branch1, self).__init__()
        # Stage 1: Basic defect discrimination
        self.basic_block = BasicBlock()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.basic_discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(8)  # 8 basic defects
        ])
        
        # Stage 2: Defect type detection
        self.type_detector = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 38),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Stage 1
        features = self.basic_block(x)
        pooled = self.gap(features).squeeze(-1).squeeze(-1)
        basic_preds = [discriminator(pooled) for discriminator in self.basic_discriminators]
        basic_preds = torch.cat(basic_preds, dim=1)  # [B, 8]
        
        # Stage 2
        type_pred = self.type_detector(basic_preds)  # [B, 38]
        return basic_preds, type_pred

class Branch2(nn.Module):
    """Branch 2: Defect Number Detection & Defect Type Detection"""
    def __init__(self):
        super(Branch2, self).__init__()
        # Stage 1: Defect number detection
        self.basic_block1 = BasicBlock()
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.num_detector = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Softmax(dim=1)
        )
        
        # Stage 2: Defect type detection
        self.embedding = nn.Embedding(5, 52*52)
        self.basic_block2 = BasicBlock()
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.type_detector = nn.Sequential(
            nn.Linear(512, 64),  # 256 (block1) + 256 (block2) = 512
            nn.ReLU(),
            nn.Linear(64, 38),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Stage 1
        feat1 = self.basic_block1(x)
        pooled1 = self.gap1(feat1).squeeze(-1).squeeze(-1)
        num_pred = self.num_detector(pooled1)  # [B, 5]
        
        # Process defect number info
        num_class = torch.argmax(num_pred, dim=1)  # [B]
        num_embed = self.embedding(num_class)  # [B, 52*52]
        num_embed = num_embed.view(-1, 1, 52, 52)  # [B, 1, 52, 52]
        
        # Stage 2
        combined = torch.cat([x, num_embed], dim=1)  # [B, 2, 52, 52]
        feat2 = self.basic_block2(combined)
        
        # Concatenate features from both stages
        concat_feat = torch.cat([
            self.gap1(feat1).squeeze(-1).squeeze(-1),
            self.gap2(feat2).squeeze(-1).squeeze(-1)
        ], dim=1)  # [B, 512]
        
        type_pred = self.type_detector(concat_feat)  # [B, 38]
        return num_pred, type_pred

class FusionClassifier(nn.Module):
    """Fusion Classifier for combining branch outputs"""
    def __init__(self):
        super(FusionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(38, 38),
            nn.Softmax(dim=1)
        )
    
    def forward(self, pred1, pred2):
        # Element-wise sum of predictions
        fused = pred1 + pred2
        return self.fc(fused)

class DMWMNet(nn.Module):
    """Complete DMWMNet Architecture"""
    def __init__(self):
        super(DMWMNet, self).__init__()
        self.branch1 = Branch1()
        self.branch2 = Branch2()
        self.fusion = FusionClassifier()
        
    def forward(self, x):
        # Branch 1 outputs
        basic_preds, type_pred1 = self.branch1(x)
        
        # Branch 2 outputs
        num_pred, type_pred2 = self.branch2(x)
        
        # Fusion output
        final_type = self.fusion(type_pred1, type_pred2)
        
        return {
            'basic_defects': basic_preds,     # [B, 8]
            'defect_num': num_pred,           # [B, 5]
            'type_branch1': type_pred1,       # [B, 38]
            'type_branch2': type_pred2,       # [B, 38]
            'final_type': final_type          # [B, 38]
        }

class FocalLoss(nn.Module):
    """Composite Focal Loss for DMWMNet"""
    def __init__(self, alpha=0.75, gamma=2, weights=[5, 1, 38]):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights  # [λ1, λ2, λ3]
        
    def forward(self, outputs, targets):
        # Unpack targets
        num_target, basic_targets, type_target = targets
        
        # Defect number loss (5 classes)
        ce_num = F.cross_entropy(outputs['defect_num'], num_target, reduction='none')
        pt_num = torch.exp(-ce_num)
        loss_num = (self.alpha * (1-pt_num)**self.gamma * ce_num).mean()
        
        # Basic defects loss (8 binary tasks)
        loss_basic = 0
        for i in range(8):
            # Binary focal loss for each defect type
            pred = outputs['basic_defects'][:, i]
            target = basic_targets[:, i]
            bce = F.binary_cross_entropy(pred, target, reduction='none')
            pt = torch.exp(-bce)
            loss_basic += (self.alpha * (1-pt)**self.gamma * bce).mean()
        
        # Defect type losses (3 components)
        def focal_loss(pred, target):
            ce = F.cross_entropy(pred, target, reduction='none')
            pt = torch.exp(-ce)
            return (self.alpha * (1-pt)**self.gamma * ce).mean()
        
        loss_type1 = focal_loss(outputs['type_branch1'], type_target)
        loss_type2 = focal_loss(outputs['type_branch2'], type_target)
        loss_final = focal_loss(outputs['final_type'], type_target)
        loss_type = loss_type1 + loss_type2 + loss_final
        
        # Composite loss
        total_loss = (self.weights[0] * loss_num + 
                      self.weights[1] * loss_basic + 
                      self.weights[2] * loss_type)
        
        return total_loss

# 示例用法
if __name__ == "__main__":
    # 输入: 批量大小 x 1通道 x 52x52
    input_tensor = torch.randn(16, 1, 52, 52)
    
    # 目标输出
    num_target = torch.randint(0, 5, (16,))       # 缺陷数量 (5类)
    basic_targets = torch.randint(0, 2, (16, 8))  # 8种基础缺陷 (二值标签)
    type_target = torch.randint(0, 38, (16,))     # 缺陷类型 (38类)
    
    # 初始化模型
    model = DMWMNet()
    criterion = FocalLoss()
    
    # 前向传播
    outputs = model(input_tensor)
    
    # 计算损失
    loss = criterion(outputs, (num_target, basic_targets, type_target))
    print(f"模型输出结构: ")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    print(f"总损失: {loss.item():.4f}")