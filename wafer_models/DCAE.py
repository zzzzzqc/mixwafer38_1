import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from scipy.stats import laplace
import copy
import random

# 可变形卷积模块
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 常规卷积层用于特征提取
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # 偏移量生成卷积层
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, stride=stride, padding=padding)
        
        # 初始化偏移量卷积的权重和偏置
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # 生成偏移量
        offset = self.offset_conv(x)
        
        # 采样网格生成
        batch_size, _, height, width = offset.size()
        grid_h, grid_w = torch.meshgrid(
            torch.arange(0, height * self.stride, self.stride),
            torch.arange(0, width * self.stride, self.stride)
        grid = torch.stack((grid_w, grid_h), 2).float().to(x.device)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # 调整偏移量形状 [b, 2*k*k, h, w] -> [b, h, w, k*k, 2]
        offset = offset.permute(0, 2, 3, 1).contiguous()
        offset = offset.view(batch_size, height, width, -1, 2)
        
        # 生成采样位置
        k = self.kernel_size
        center = (k - 1) / 2
        kernel_offset = torch.tensor([(i - center, j - center) for i in range(k) for j in range(k)]).to(x.device)
        kernel_offset = kernel_offset.view(1, 1, 1, k*k, 2)
        
        # 计算采样点坐标
        sample_points = grid.unsqueeze(3) + kernel_offset + offset
        
        # 双线性插值采样
        x = F.grid_sample(x, sample_points, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 常规卷积
        return self.conv(x)

# 可变形卷积自编码器 (DCAE)
class DCAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(DCAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            DeformableConv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            DeformableConv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            DeformableConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 潜在空间
        self.latent_fc = nn.Linear(128*12*12, latent_dim)
        
        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, 128*12*12)
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DeformableConv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            DeformableConv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            DeformableConv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.latent_fc(x)

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 128, 12, 12)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# Vision Transformer (ViT) 组件
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=6, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.proj(x)  # [b, n_patches, embed_dim]
        
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_token, x], dim=1)  # [b, n_patches+1, embed_dim]
        x += self.pos_embed
        
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, embed_dim * mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=96, patch_size=6, in_chans=3, num_classes=9, embed_dim=128, depth=3, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

# SDeeeFL 框架实现
class SDeeeFLClient:
    def __init__(self, client_id, data, num_classes=9):
        self.client_id = client_id
        self.data = data  # (images, labels)
        self.num_classes = num_classes
        
        # 初始化模型
        self.dcae = DCAE()
        self.vit = VisionTransformer()
        
        # 优化器
        self.dcae_optim = optim.Adam(self.dcae.parameters(), lr=0.001)
        self.vit_optim = optim.SGD(self.vit.parameters(), lr=0.0001, momentum=0.9)
        
        # 客户端状态
        self.model_weights = None
        self.neighbors = []
        self.shared_db = None
        self.local_db = None
    
    def build_shared_database(self, epsilon=0.05, sensitivity=1.0):
        """构建差分隐私共享数据库"""
        images, labels = self.data
        num_samples = int(len(images) * epsilon)
        
        # 随机选择样本
        indices = np.random.choice(len(images), num_samples, replace=False)
        selected_images = images[indices]
        selected_labels = labels[indices]
        
        # 添加拉普拉斯噪声
        noise = laplace.rvs(loc=0, scale=sensitivity/epsilon, size=selected_images.shape)
        noisy_images = selected_images + noise
        noisy_images = np.clip(noisy_images, 0, 1)
        
        self.shared_db = (noisy_images, selected_labels)
        return self.shared_db
    
    def augment_data(self, target_samples=1000):
        """使用DCAE增强数据以解决类别不平衡"""
        images, labels = self.data
        
        # 计算每个类别的样本数量
        class_counts = np.bincount(labels)
        minority_classes = np.where(class_counts < target_samples)[0]
        
        augmented_images = []
        augmented_labels = []
        
        for cls in minority_classes:
            cls_indices = np.where(labels == cls)[0]
            num_needed = target_samples - len(cls_indices)
            
            if num_needed <= 0:
                continue
                
            # 使用DCAE生成新样本
            cls_images = images[cls_indices]
            num_batches = int(np.ceil(num_needed / len(cls_images)))
            
            for _ in range(num_batches):
                # 添加噪声到潜在空间
                latent = self.dcae.encode(torch.tensor(cls_images).float())
                noise = torch.randn_like(latent) * 0.1
                noisy_latent = latent + noise
                
                # 解码生成新样本
                new_images = self.dcae.decode(noisy_latent).detach().numpy()
                augmented_images.append(new_images)
                augmented_labels.extend([cls] * len(cls_images))
        
        # 合并原始数据和增强数据
        if augmented_images:
            augmented_images = np.concatenate(augmented_images, axis=0)
            augmented_labels = np.array(augmented_labels)
            
            all_images = np.concatenate([images, augmented_images], axis=0)
            all_labels = np.concatenate([labels, augmented_labels], axis=0)
        else:
            all_images, all_labels = images, labels
        
        self.local_db = (all_images, all_labels)
        return self.local_db
    
    def local_train(self, epochs=10, batch_size=32):
        """在本地数据上训练模型"""
        if self.local_db is None:
            self.augment_data()
        
        images, labels = self.local_db
        dataset = torch.utils.data.TensorDataset(torch.tensor(images).float(), 
                                               torch.tensor(labels).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练DCAE
        self.dcae.train()
        for epoch in range(epochs):
            for batch_imgs, _ in loader:
                self.dcae_optim.zero_grad()
                recon = self.dcae(batch_imgs)
                loss = F.mse_loss(recon, batch_imgs)
                loss.backward()
                self.dcae_optim.step()
        
        # 训练ViT
        self.vit.train()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for batch_imgs, batch_labels in loader:
                self.vit_optim.zero_grad()
                outputs = self.vit(batch_imgs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                self.vit_optim.step()
        
        # 保存模型权重
        self.model_weights = {
            'dcae': copy.deepcopy(self.dcae.state_dict()),
            'vit': copy.deepcopy(self.vit.state_dict())
        }
    
    def update_model(self, global_weights):
        """更新本地模型权重"""
        self.dcae.load_state_dict(global_weights['dcae'])
        self.vit.load_state_dict(global_weights['vit'])
    
    def aggregate(self, neighbor_weights):
        """聚合邻居模型权重"""
        if not neighbor_weights:
            return self.model_weights
        
        # 平均聚合
        aggregated_weights = {}
        for key in self.model_weights.keys():
            weights_sum = torch.zeros_like(self.model_weights[key])
            for weight in neighbor_weights:
                weights_sum += weight[key]
            aggregated_weights[key] = weights_sum / len(neighbor_weights)
        
        return aggregated_weights

class SDeeeFLServer:
    def __init__(self, num_clients, connection_prob=0.5):
        self.num_clients = num_clients
        self.connection_prob = connection_prob
        self.topology = self.generate_topology()
        self.global_model = {
            'dcae': DCAE().state_dict(),
            'vit': VisionTransformer().state_dict()
        }
    
    def generate_topology(self):
        """使用Erdos-Renyi方法生成随机拓扑"""
        topology = np.zeros((self.num_clients, self.num_clients))
        for i in range(self.num_clients):
            for j in range(i+1, self.num_clients):
                if np.random.rand() < self.connection_prob:
                    topology[i, j] = 1
                    topology[j, i] = 1
        return topology
    
    def build_global_shared_db(self, clients, epsilon=0.05):
        """构建全局共享数据库"""
        all_images = []
        all_labels = []
        
        for client in clients:
            shared_images, shared_labels = client.build_shared_database(epsilon)
            all_images.append(shared_images)
            all_labels.append(shared_labels)
        
        all_images = np.concatenate(all_images, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_images, all_labels
    
    def train_global_model(self, shared_db, epochs=5, batch_size=32):
        """在共享数据库上训练全局模型"""
        images, labels = shared_db
        dataset = torch.utils.data.TensorDataset(torch.tensor(images).float(), 
                                               torch.tensor(labels).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        vit = VisionTransformer()
        optimizer = optim.SGD(vit.parameters(), lr=0.00005, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        vit.train()
        for epoch in range(epochs):
            for batch_imgs, batch_labels in loader:
                optimizer.zero_grad()
                outputs = vit(batch_imgs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
        # 更新全局模型
        self.global_model['vit'] = copy.deepcopy(vit.state_dict())
    
    def federated_round(self, clients, communication_rounds=20):
        """执行一轮联邦学习"""
        # 步骤1: 分发全局模型
        for client in clients:
            client.update_model(self.global_model)
        
        # 步骤2: 客户端本地训练
        for client in clients:
            client.local_train()
        
        # 步骤3: 基于拓扑的模型聚合
        for client in clients:
            neighbor_indices = [i for i, connected in enumerate(self.topology[client.client_id]) 
                              if connected and i != client.client_id]
            neighbor_weights = [clients[i].model_weights for i in neighbor_indices]
            
            # 聚合邻居模型权重
            aggregated_weights = client.aggregate(neighbor_weights)
            client.model_weights = aggregated_weights
        
        # 更新全局模型（简单平均所有客户端模型）
        avg_weights = {}
        for key in self.global_model.keys():
            weights_sum = torch.zeros_like(self.global_model[key])
            for client in clients:
                weights_sum += client.model_weights[key]
            avg_weights[key] = weights_sum / len(clients)
        
        self.global_model = avg_weights
        return self.global_model

# 模拟实验
def simulate_sdeeefl(num_clients=4, num_rounds=10, connection_prob=0.5):
    # 创建客户端和服务器
    server = SDeeeFLServer(num_clients, connection_prob)
    clients = []
    
    # 模拟不同客户端的数据 (实际应用中应使用真实数据)
    for i in range(num_clients):
        # 模拟数据：96x96晶圆图像，9个类别
        num_samples = np.random.randint(500, 1000)
        images = np.random.rand(num_samples, 3, 96, 96).astype(np.float32)
        labels = np.random.randint(0, 9, num_samples)
        clients.append(SDeeeFLClient(i, (images, labels)))
    
    # 构建全局共享数据库
    shared_db = server.build_global_shared_db(clients)
    
    # 在共享数据库上训练全局模型
    server.train_global_model(shared_db)
    
    # 联邦学习过程
    all_accuracies = []
    for round in range(num_rounds):
        # 更新拓扑（可选，论文中是时变拓扑）
        server.topology = server.generate_topology()
        
        # 执行一轮联邦学习
        global_weights = server.federated_round(clients)
        
        # 评估全局模型性能
        accuracy = evaluate_global_model(global_weights, shared_db)
        all_accuracies.append(accuracy)
        print(f"Round {round+1}/{num_rounds}, Global Accuracy: {accuracy:.4f}")
    
    return all_accuracies

def evaluate_global_model(global_weights, test_data):
    """评估全局模型性能"""
    test_images, test_labels = test_data
    dataset = torch.utils.data.TensorDataset(torch.tensor(test_images).float(), 
                                           torch.tensor(test_labels).long())
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    vit = VisionTransformer()
    vit.load_state_dict(global_weights['vit'])
    vit.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = vit(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# 可视化函数
def visualize_dcae_performance(dcae, sample_images):
    """可视化DCAE生成效果"""
    dcae.eval()
    with torch.no_grad():
        reconstructed = dcae(torch.tensor(sample_images).float())
    
    plt.figure(figsize=(10, 4))
    for i in range(5):
        # 原始图像
        plt.subplot(2, 5, i+1)
        plt.imshow(sample_images[i].transpose(1, 2, 0))
        plt.title("Original")
        plt.axis('off')
        
        # 重建图像
        plt.subplot(2, 5, i+6)
        plt.imshow(reconstructed[i].permute(1, 2, 0).numpy())
        plt.title("Reconstructed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 模拟DCAE性能可视化
    sample_images = np.random.rand(5, 3, 96, 96)
    dcae = DCAE()
    visualize_dcae_performance(dcae, sample_images)
    
    # 运行SDeeeFL模拟
    print("Starting SDeeeFL Simulation...")
    accuracies = simulate_sdeeefl(num_clients=4, num_rounds=10)
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, marker='o')
    plt.title("Global Model Accuracy during Federated Training")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()