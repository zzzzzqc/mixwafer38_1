from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from label_maping import label_mapping


class WaferDefectDataset(Dataset):
    def __init__(self, images, labels, mode='train'):
        self.images = images
        self.labels = labels
        self.mode = mode
        self.label_mapping = label_mapping if label_mapping is not None else {}

        # Define transformations for train and other sets
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),  # Convert numpy array to PIL Image
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(0.2),  # Horizontal flip with probability 0.2
                transforms.RandomVerticalFlip(0.2),  # Vertical flip with probability 0.2
                transforms.RandomRotation(30),  # Random rotation
                transforms.ToTensor(),  # Convert PIL Image back to tensor
                transforms.Normalize([0.5], [0.5])  # Normalize the image tensor
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 2.0  # 归一化到 [0, 1]
        # label = self.labels[idx]
        original_label = ''.join(map(str, self.labels[idx]))  # 将one-hot编码转换为字符串形式

        # 更新标签为新的38分类标签
        new_label = self.update_label(original_label)
        # 将图像数据转换为 PyTorch 张量，并添加通道维度 (C, H, W)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 52, 52)
        label_tensor = torch.tensor(new_label, dtype=torch.float32)

        # Apply transformation if defined
        if self.transform is not None:
            image_tensor = image_tensor.squeeze(0)
            image_tensor = self.transform(image_tensor.numpy())  # Apply transform and convert back to tensor
            if image_tensor.dim() == 2:  # If the tensor does not have a channel dimension after transformation
                image_tensor = image_tensor.unsqueeze(0)

        return image_tensor, label_tensor

    def update_label(self, original_label_str):
        """
        根据新的标签映射规则更新标签。
        :param original_label_str: 原始的one-hot编码标签 (8维)，作为字符串
        :return: 新的one-hot编码标签 (38维)
        """
        # 使用label_mapping将原始标签字符串映射到新的索引
        if original_label_str in self.label_mapping:
            new_indices = self.label_mapping[original_label_str]
        else:
            # 如果没有找到对应的映射，默认保持原样或设置为未知类别
            new_indices = [0]  # 假设0是“未知”或“未定义”的类别

        # 创建新的one-hot编码标签
        new_label = np.zeros(38, dtype=np.float32)
        new_label[new_indices] = 1.0  # 多标签情况下的处理

        return new_label



def get_train_test_loaders(datapath,
                               batch_size=128,
                               test_size=0.2,
                               random_state=42):
    data = np.load(datapath)
    # 获取数据和标签
    images = data['arr_0']
    labels = data['arr_1']

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels)


    train_dataset = WaferDefectDataset(train_images, train_labels, mode='train')

    test_dataset = WaferDefectDataset(test_images, test_labels, mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_dataloader,test_dataloader


def get_train_val_test_loaders(datapath,
                               batch_size=128,
                               test_size=0.2,
                               random_state=42):
    data = np.load(datapath)
    # 获取数据和标签
    images = data['arr_0']
    labels = data['arr_1']


    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels)
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.25, random_state=42, stratify=train_labels)

    train_dataset = WaferDefectDataset(train_images, train_labels, mode='train')
    val_dataset = WaferDefectDataset(val_images, val_labels, mode='val')
    test_dataset = WaferDefectDataset(test_images, test_labels, mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_dataloader,val_dataloader, test_dataloader


if __name__ == '__main__':
    import os
    import zipfile
    zip_file_path = 'dataset_mix38/Wafer_Map_Datasets.npz.zip'
    extracted_npz_path = 'dataset_mix38/Wafer_Map_Datasets.npz'

    # 解压文件（如果未解压）
    if not os.path.exists(extracted_npz_path):
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f"压缩包文件不存在: {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall('dataset_mix38/')
    datapath = 'dataset_mix38/Wafer_Map_Datasets.npz'
    # 设置超参数
    batch_size = 1
    num_workers = 4
    train_DataLoader,val_DataLoader, test_DataLoader = get_train_val_test_loaders(datapath,
                                                                                  batch_size=128,
                                                                                  test_size=0.2,
                                                                                  random_state=42)

    # 测试数据加载器
    for phase in ['train', 'test']:
        dataloader = train_DataLoader if phase == 'train' else test_DataLoader

        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"{phase.capitalize()} Batch {batch_idx + 1}")
            print("Images shape:", images.shape)
            print("Labels shape:", labels.shape)
            print("batch_labels", labels)
            # 只打印前两个批次的数据
            if batch_idx >= 0:
                break
