import torch
import torch.nn as nn
import torch.nn.functional as F

class FACActiveContour(nn.Module):
    def __init__(self, lambda_param=0.1, theta=1.0, beta=1.0, max_iter=100):
        super().__init__()
        self.lambda_param = lambda_param
        self.theta = theta
        self.beta = beta
        self.max_iter = max_iter
        
    def forward(self, f):
        """快速全局最小化主动轮廓分割
        Args:
            f: 输入图像 [B, 1, H, W] 值域[0,1]
        Returns:
            u: 分割结果 [B, 1, H, W] 值域[0,1]
        """
        # 计算边缘指示函数 g(x) = 1/(1 + beta*|∇f|^2)
        grad_x = F.conv2d(f, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float(), padding=1)
        grad_y = F.conv2d(f, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float(), padding=1)
        grad_mag = (grad_x**2 + grad_y**2).sqrt()
        g = 1 / (1 + self.beta * grad_mag**2)
        
        # 初始化变量
        batch, _, h, w = f.shape
        u = torch.clone(f)
        v = torch.zeros_like(f)
        p1 = torch.zeros((batch, 1, h, w), device=f.device)
        p2 = torch.zeros((batch, 1, h, w), device=f.device)
        
        # 迭代优化 (FAC对偶方法)
        for _ in range(self.max_iter):
            # 更新对偶变量 p
            div_p = self.divergence(p1, p2)
            grad_div_p = self.gradient(div_p - (f - v)/self.theta)
            
            denom = 1 + (1/8) * torch.sqrt(grad_div_p[:, 0:1]**2 + grad_div_p[:, 1:2]**2) / g
            p1 = (p1 + (1/8) * grad_div_p[:, 0:1]) / denom
            p2 = (p2 + (1/8) * grad_div_p[:, 1:2]) / denom
            
            # 更新原始变量 u
            div_p = self.divergence(p1, p2)
            u = f - v - self.theta * div_p
            
            # 更新辅助变量 v
            diff = f - u
            v = torch.where(diff > self.theta * self.lambda_param, 
                            diff - self.theta * self.lambda_param,
                            torch.where(diff < -self.theta * self.lambda_param,
                                       diff + self.theta * self.lambda_param,
                                       torch.zeros_like(diff)))
        
        return u
    
    def divergence(self, p1, p2):
        kernel_x = torch.tensor([[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]]).float()
        kernel_y = torch.tensor([[[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]]).float()
        div_p = F.conv2d(p1, kernel_x.to(p1.device), padding=1) + \
                F.conv2d(p2, kernel_y.to(p2.device), padding=1)
        return div_p
    
    def gradient(self, u):
        kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float() / 8.0
        kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float() / 8.0
        grad_x = F.conv2d(u, kernel_x.to(u.device), padding=1)
        grad_y = F.conv2d(u, kernel_y.to(u.device), padding=1)
        return torch.cat([grad_x, grad_y], dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        # 深度卷积 (depthwise)
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # groups=in_channels 实现深度卷积
            bias=False
        )
        # 点卷积 (pointwise)
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class AC_LDN(nn.Module):
    def __init__(self, num_classes=38, use_separable=False):
        super().__init__()
        self.use_separable = use_separable
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Depthwise卷积层
        self.depthwise = DepthWiseConv(16, 16, kernel_size=3, stride=3, padding=1)
        
        # Separable版本特有层
        if use_separable:
            self.pointwise = nn.Conv2d(16, 16, kernel_size=1)
            self.bn_point = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d(3, stride=2, padding=0)
        
        # 公共卷积层
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Separable版本的第二池化层
        if use_separable:
            self.pool2 = nn.MaxPool2d(3, stride=2, padding=0)
        
        # 1x1卷积层
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # 分类层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 输入: [B, 1, 52, 52]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 16, 52, 52]
        x = F.relu(self.depthwise(x))        # [B, 16, 18, 18]
        
        if self.use_separable:
            x = F.relu(self.bn_point(self.pointwise(x)))  # [B, 16, 18, 18]
            x = F.relu(self.bn2(self.conv2(x)))           # [B, 16, 18, 18]
            x = self.pool1(x)                # [B, 16, 8, 8]
        
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 64, 8, 8] (标准) 或 [B, 64, 8, 8] (Sep)
        
        if self.use_separable:
            x = self.pool2(x)                # [B, 64, 3, 3]
        
        x = F.relu(self.bn4(self.conv4(x)))  # [B, 128, 3, 3] (Sep) 或 [B, 128, 8, 8] (标准)
        x = self.global_pool(x)              # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)            # [B, 128]
        x = self.dropout(x)
        x = self.fc(x)                       # [B, num_classes]
        return x
    

class FAC_ACLDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fac_seg = FACActiveContour()
        self.acldn = AC_LDN()
        
    def forward(self, x):
        # 预处理: 归一化到[0,1]
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # FAC主动轮廓分割
        seg_map = self.fac_seg(x_normalized)
        
        # 二值化分割结果 (可选)
        binary_seg = (seg_map > 0.5).float()
        
        # AC-LDN分类
        output = self.acldn(binary_seg)
        return output
    
from thop import profile
from thop import clever_format
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input):
    flops = profile(model, inputs=(input,))[0] * 2
    flops = clever_format(flops, "%.2f")
    return flops

if __name__ == "__main__":
    model = FAC_ACLDN()
    # 测试前向传播
    x = torch.randn(1, 1, 52, 52)
    print("模型输出尺寸:", model(x).shape)
    from thop import profile
    flops, params = profile(model, inputs=(x, ))
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M
    total = count_parameters(model)
    flops = count_flops(model, x)
    print("Number of parameter: %.2fM" % (total/1e6))
    print(flops)
    model = AC_LDN()
    x = torch.randn(1, 1, 52, 52)
    print("模型输出尺寸:", model(x).shape)
    from thop import profile
    flops, params = profile(model, inputs=(x, ))
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M
    total = count_parameters(model)
    flops = count_flops(model, x)
    print("Number of parameter: %.2fM" % (total/1e6))
    print(flops)
