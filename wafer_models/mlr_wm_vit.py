import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP

class MobileOneStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, padding=0),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.GELU()
        )
        self.layer3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class SpatialAttentionEnhancedGhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.ghost_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, 
                      padding=1, groups=out_channels // 2),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        )
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ghost_out = self.ghost_conv(x)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        attention = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        return ghost_out * attention + x

class ConvolutionalAdditiveSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        
        self.spatial_op = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.channel_op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    
    def forward(self, x):
        q, k, v = torch.chunk(self.to_qkv(x), 3, dim=1)
        
        q_so = self.spatial_op(q)
        k_co = self.channel_op(k)
        
        attn = self.dw_conv(q_so + k_co)
        return attn * v

class ECASABlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.integration = nn.Sequential(
            SpatialAttentionEnhancedGhostConv(dim, dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU()
        )
        self.casa = ConvolutionalAdditiveSelfAttention(dim)
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(dim),
            SpatialAttentionEnhancedGhostConv(dim, dim),
            nn.GELU(),
            SpatialAttentionEnhancedGhostConv(dim, dim)
        )
    
    def forward(self, x):
        # Integration
        res = x
        x = self.integration(x)
        x = x + res
        
        # CASA
        x = x + self.casa(x)
        
        # MLP
        res = x
        x = self.mlp(x)
        x = x + res
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        branch_outs = []
        for branch in self.branches[:4]:
            branch_outs.append(branch(x))
        
        # For global average pooling branch
        pool_out = self.branches[4](x)
        pool_out = F.interpolate(pool_out, size=x.shape[2:], mode='bilinear', align_corners=True)
        branch_outs.append(pool_out)
        
        out = torch.cat(branch_outs, dim=1)
        return self.fusion(out)

class BasicBlock(nn.Module):
    def __init__(self, in_channels=1, stem_channels=64, dim=256, num_stages=4):
        super().__init__()
        self.stem = MobileOneStem(in_channels, stem_channels)
        
        self.stages = nn.ModuleList()
        current_channels = stem_channels
        for i in range(num_stages):
            self.stages.append(ECASABlock(current_channels))
            # Patch embedding (downsampling)
            if i < num_stages - 1:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(current_channels * 2)
                ))
                current_channels *= 2
        
        self.aspp = ASPP(current_channels, dim)
    
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.aspp(x)
        return x

class MultiLevelRelayClassifier(nn.Module):
    def __init__(self, in_features, num_basic_defects=8, num_defect_types=38, num_defect_nums=5):
        super().__init__()
        self.basic_defect_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, num_basic_defects),
            nn.Sigmoid()
        )
        
        # Level 2 classifiers
        self.defect_num_classifier = nn.Sequential(
            nn.Linear(in_features + num_basic_defects, 256),
            nn.ReLU(),
            nn.Linear(256, num_defect_nums)
        )
        
        self.defect_type_classifier1 = nn.Sequential(
            nn.Linear(in_features + num_basic_defects, 512),
            nn.ReLU(),
            nn.Linear(512, num_defect_types)
        )
        
        # Level 3 classifier
        self.defect_type_classifier2 = nn.Sequential(
            nn.Linear(in_features + num_defect_nums, 512),
            nn.ReLU(),
            nn.Linear(512, num_defect_types)
        )
        
        # Final fusion
        self.final_classifier = nn.Linear(num_defect_types, num_defect_types)
    
    def forward(self, features):
        # Level 1: Basic defects
        basic_defects = self.basic_defect_classifier(features)
        
        # Level 2: Defect number and type (first prediction)
        feat_vec = F.adaptive_avg_pool2d(features, 1).flatten(1)
        feat_basic = torch.cat([feat_vec, basic_defects], dim=1)
        
        defect_num = self.defect_num_classifier(feat_basic)
        defect_type1 = self.defect_type_classifier1(feat_basic)
        
        # Level 3: Defect type (second prediction)
        defect_num_onehot = F.one_hot(torch.argmax(defect_num, dim=1), num_classes=5).float()
        feat_num = torch.cat([feat_vec, defect_num_onehot], dim=1)
        defect_type2 = self.defect_type_classifier2(feat_num)
        
        # Final fusion
        fused_type = defect_type1 + defect_type2
        final_type = self.final_classifier(fused_type)
        
        return basic_defects, defect_num, defect_type1, defect_type2, final_type

class MLRWMViT(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 stem_channels=64,
                 feature_dim=256,
                 num_basic_defects=8,
                 num_defect_types=38,
                 num_defect_nums=5):
        super().__init__()
        self.basic_block = BasicBlock(in_channels, stem_channels, feature_dim)
        self.classifier = MultiLevelRelayClassifier(
            feature_dim, num_basic_defects, num_defect_types, num_defect_nums
        )
    
    def forward(self, x):
        features = self.basic_block(x)
        outputs = self.classifier(features)
        return outputs

class MLRWMViTLoss(nn.Module):
    def __init__(self, alpha1=1, alpha2=5, alpha3=38):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        basic_pred, num_pred, type_pred1, type_pred2, final_type = outputs
        basic_target, num_target, type_target = targets
        
        # Basic defect loss (8 binary classifications)
        basic_loss = 0
        for i in range(8):
            basic_loss += self.bce_loss(basic_pred[:, i], basic_target[:, i])
        basic_loss *= self.alpha1
        
        # Defect number loss
        num_loss = self.ce_loss(num_pred, num_target) * self.alpha2
        
        # Defect type losses
        type_loss1 = self.ce_loss(type_pred1, type_target)
        type_loss2 = self.ce_loss(type_pred2, type_target)
        final_type_loss = self.ce_loss(final_type, type_target)
        
        type_loss = self.alpha3 * (type_loss1 + type_loss2 + 2 * final_type_loss)
        
        total_loss = basic_loss + num_loss + type_loss
        return total_loss
    
from thop import profile
from thop import clever_format
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input):
    flops = profile(model, inputs=(input,))[0] * 2
    flops = clever_format(flops, "%.2f")
    return flops

if __name__ == '__main__':
    model = MLRWMViT(
    in_channels=1,        # 输入通道（灰度图）
    stem_channels=64,     # Stem输出通道
    feature_dim=256,      # 特征维度
    num_basic_defects=8,  # 基本缺陷类别数
    num_defect_types=38,  # 总缺陷类型数
    num_defect_nums=38    # 缺陷数量类别数(0-4)
)
    x = torch.randn(1, 1, 64, 64)
    print("模型输出尺寸:", model(x).shape)
    total = count_parameters(model)
    flops = count_flops(model, x)
    print("Number of parameter: %.2fM" % (total/1e6))
    print(flops)
    pass