import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.ops import DeformConv2d

# todo 复现" Mixed-type wafer defect detection based on multi-branch feature enhanced residual module  "

class DeformConv2d(nn.Module):
    def __init__(self, 
                 inc, 
                 outc, 
                 kernel_size=3, 
                 padding=1, 
                 stride=1, 
                 bias=None, 
                 modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable 
            Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, #该卷积用于最终的卷积
                              outc, 
                              kernel_size=kernel_size, 
                              stride=kernel_size, 
                              bias=bias)

        self.p_conv = nn.Conv2d(inc, #该卷积用于从input中学习offset
                                2*kernel_size*kernel_size, 
                                kernel_size=3, 
                                padding=1, 
                                stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation #该部分是DeformableConv V2版本的，可以暂时不看
        if modulation:
            self.m_conv = nn.Conv2d(inc, 
                                    kernel_size*kernel_size, 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
    def forward(self, x):
        offset = self.p_conv(x) 
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out
        
    def _get_p_n(self, N, dtype): #求
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset



class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
 
        hidden_channels = oup // ratio
        new_channels = hidden_channels*(ratio-1)
 
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, hidden_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False) if relu else nn.Sequential(),
        )
 
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(hidden_channels, new_channels, dw_size, 1, dw_size//2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=False) if relu else nn.Sequential(),
        )
 
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels // reduction_ratio, channels//2, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        weights_map = self.shared_mlp((avg_out + max_out))
        weights = self.sigmoid(weights_map)
        return weights

class ResDeformCE(nn.Module):
    """ResDeformCE块 可变形卷积+通道注意力+残差连接"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResDeformCE, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=False),
        )
        self.channel_att = ChannelAttention(out_channels)
        self.deform_conv = nn.Sequential(
            DeformConv2d(in_channels//2, in_channels//2, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=False))
        
        self.conv2 =nn.Sequential(
            nn.Conv2d(in_channels//2, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


        # 如果输入输出通道数不同，使用1x1卷积调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        weights = self.channel_att(x)
        x1 = self.conv1(x)
        x2 = self.deform_conv(x1)
        out = self.conv2(x2*weights)
        out = out + residual  # 使用非原地加法
        return out


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
    

class SpatialEnhancedGhost(nn.Module):
    """空间特征增强的Ghost卷积块"""
    def __init__(self, in_channels, out_channels,kernel_size=1, ratio=2):
        super(SpatialEnhancedGhost, self).__init__()
        self.ghost = GhostModule(in_channels, out_channels,kernel_size=kernel_size, ratio=ratio)
        
        # 空间特征增强：使用大核卷积扩大感受野
        self.spatial_enhance = SpatialAttention()
        
    def forward(self, x):
        x = self.ghost(x)
        enhanced = self.spatial_enhance(x)
        return torch.cat([x, enhanced], dim=1)


class FeatureFusionWeightGenerator(nn.Module):

    def __init__(self, in_channels=128,out_channels=128, reduction_ratio=16):
        super(FeatureFusionWeightGenerator, self).__init__()

        self.ResDeformCE1 = ResDeformCE(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.sp1=SpatialEnhancedGhost(in_channels, out_channels//2,kernel_size=5)

        #  u1+u2  后池化+全连接层
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()

        u1 =self.ResDeformCE1(x)
        # print(u1.shape)
        u2 = self.sp1(x) 
        # print(u2.shape)

        fea = u1+u2
        # print(fea.shape)

        se = self.gmp(fea).view(b,c)
        out = self.fc(se).view(b, c, 1, 1)

        return u1,u2,out


class MERModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MERModule, self).__init__()
        # 分支1: ResDeformCE
        self.branch1 = ResDeformCE(in_channels, out_channels)
        
        # 分支2: 空间增强Ghost卷积
        self.branch2 = SpatialEnhancedGhost(in_channels, out_channels // 2)
        
        # 特征融合权重生成器
        self.fusion_generator = FeatureFusionWeightGenerator(out_channels)
        
        # 调整通道数的卷积
        self.conv_adjust = nn.Conv2d(out_channels , out_channels, kernel_size=1)
        
    def forward(self, x):

        u1,u2,out = self.fusion_generator(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)

        b11 = torch.add(b1, u1)
        b111 = torch.mul(b11, out)

        b22 = torch.add(b2, u2)
        b222 = torch.mul(b22, out)

        out = torch.add(b111, b222)
        # print('out',out.shape)
        # 调整通道数
        out = self.conv_adjust(out)
        return out

class ConvBlock1(nn.Module):
    """ConvBlock1 提取浅层特征"""
    def __init__(self,in_ch):
        super(ConvBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=11, stride=3, padding=5, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ConvBlock2(nn.Module):
    """ConvBlock2 提取深层特征"""
    def __init__(self):
        super(ConvBlock2, self).__init__()
        self.conv1 = nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class MERNet(nn.Module):
    """MER-Net主网络结构"""
    def __init__(self,in_channels=3, num_classes=38):
        super(MERNet, self).__init__()
        self.conv_block1 = ConvBlock1(in_channels)
        self.mer_module = MERModule(128, 128)
        self.conv_block2 = ConvBlock2()
        # 全连接分类层
        self.fc = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.mer_module(x)
        x = self.conv_block2(x)
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

if __name__ == '__main__':
    model = MERNet(in_channels = 1,num_classes=38)
    x = torch.randn(1, 1, 64, 64)
    print("模型输出尺寸:", model(x).shape)
    total = count_parameters(model)
    flops = count_flops(model, x)
    print("Number of parameter: %.2fM" % (total/1e6))
    print(flops)
    from thop import profile
    flops, params = profile(model, inputs=(x, ))
    print('flops  0.91           para  6.31,')
    print(f'flops is {flops*2/1e9} G,params is {params/1e6} M') #flops单位G，para单位M
    pass