import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

class data_loader:

    def __init__(self, dataset_path, vis=False, target_size=None, interpolation_mode='bilinear'):
        """Initialize the data_loader class

        Args:
            dataset_path (str): Path to the dataset file
            vis (bool): Whether to enable visualization
            target_size (tuple): Target size (H, W) for resizing images
            interpolation_mode (str): Interpolation mode for resizing (e.g., 'bilinear', 'nearest')
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} not found")

        self.dataset_path = dataset_path
        self.is_vis = vis
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

        # Mapping from label to defect types
        self.label_keys = ["Center", "Donut", "Edge_Loc", "Edge_Ring",
                           "Loc", "Near_Full", "Scratch", "Random"]  # 修正后的标签列表
        
        self.show_split = True

    def load_data(self):
        """Load data into self.train and self.label"""
        self.data = np.load(self.dataset_path)
        self.train = self.data["arr_0"]
        self.label = self.data["arr_1"]

        print(f"MixedWM38: {self.label.shape[0]} wafers loaded")

    def prep_data(self):
        """Split data and convert to PyTorch tensors"""
        # Split dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.train, self.label, test_size=0.2, random_state=42)

        # Add channel dimension and convert to PyTorch tensors
        # Convert from (N, H, W) to (N, 1, H, W)   .............normalization can '/2'
        self.x_train = torch.from_numpy(np.expand_dims(self.x_train, axis=1)).float()
        self.x_test = torch.from_numpy(np.expand_dims(self.x_test, axis=1)).float()

        # Resize images if target_size is specified
        if self.target_size is not None:
            # Resize using specified interpolation mode
            self.x_train = F.interpolate(
                self.x_train,
                size=self.target_size,
                mode=self.interpolation_mode,
                align_corners=False if self.interpolation_mode == 'bilinear' else None
            )
            self.x_test = F.interpolate(
                self.x_test,
                size=self.target_size,
                mode=self.interpolation_mode,
                align_corners=False if self.interpolation_mode == 'bilinear' else None
            )
        if self.show_split:
            self.print_class_distribution(self.y_train, "Train")  
            self.print_class_distribution(self.y_test, "Test")    
            self.print_label_counts(self.y_train, "Train")       
            self.print_label_counts(self.y_test, "Test")        

        # Convert labels to PyTorch tensors
        self.y_train = torch.from_numpy(np.array(self.y_train)).float()
        self.y_test = torch.from_numpy(np.array(self.y_test)).float()

    def read_label(self, label):
        """Convert label to human-readable defect type string"""
        if np.sum(label) == 0:
            return 'Normal wafer'

        defect_types = []
        for i, value in enumerate(label):
            if value == 1:
                defect_types.append(self.label_keys[i])

        return ', '.join(defect_types)  

    def see_wafer(self, wafer_num):
        """Visualize a wafer and its label (using original size)"""
        if wafer_num < 0 or wafer_num >= len(self.train):
            raise ValueError(
                f"Invalid wafer index: {wafer_num}. "
                f"Valid range: 0-{len(self.train)-1}"
            )
        # Convert to numpy for visualization
        image = self.train[wafer_num]
        label = self.label[wafer_num]
        print("Defect types =", self.read_label(label))
        print("Labeled as:", label)
        plt.title(f"wafer #{wafer_num} (Original size: {image.shape})")
        plt.imshow(image, cmap='gray')
        plt.colorbar()
        plt.show()

    def get_data(self):
        """Return processed data for training/evaluation"""
        self.load_data()
        self.prep_data()

        train_dataset = TensorDataset(self.x_train, self.y_train)
        test_dataset = TensorDataset(self.x_test, self.y_test)

        return train_dataset, test_dataset
    
    def print_class_distribution(self, labels, set_name="Train"):
        print(f"\n[{set_name}] Class distribution:")
        total_samples = labels.shape[0]
        for i, class_name in enumerate(self.label_keys):
            count = int(np.sum(labels[:, i]))
            print(f"{class_name:<12}: {count:>5} ({count/total_samples:.1%})")
        normal_count = np.sum(np.all(labels == 0, axis=1))
        print(f"{'Normal':<12}: {normal_count:>5} ({normal_count/total_samples:.1%})")

    def print_label_counts(self, labels, set_name="Train"):
        label_counts = np.sum(labels, axis=1)
        count_dict = {i:0 for i in range(5)}
        for cnt in label_counts:
            count_dict[4 if cnt >=4 else cnt] += 1
        
        print(f"\n[{set_name}] Label counts per sample:")
        for k in sorted(count_dict):
            if k < 4:
                print(f"{k} label{'s' if k>1 else ''}:".ljust(12),
                      f"{count_dict[k]:>5} ({count_dict[k]/len(labels):.1%})")
            elif k ==4:
                print("4+ labels:".ljust(12),
                      f"{count_dict[k]:>5} ({count_dict[k]/len(labels):.1%})")


if __name__ == '__main__':
    # 使用示例
    dataset_path = r'D:\0000mywork\mix_wafer_38\dataset_mix38\Wafer_Map_Datasets.npz'

    # 初始化时指定调整尺寸和插值方式
    target_size = (64, 64)  # 目标尺寸
    A1 = data_loader(dataset_path,
                     vis=True,
                     target_size=target_size,
                     interpolation_mode='bilinear')

    train_dataset, test_dataset = A1.get_data()

    # 查看调整后的张量形状
    print("Resized training data shape:", A1.x_train.shape)
    print("Resized test data shape:", A1.x_test.shape)

    # 查看原始尺寸的晶圆图像（注意显示的是原始数据）
    A1.see_wafer(20005)