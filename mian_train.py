# encoding: utf-8
from __future__ import print_function, division

import copy
import torch
import torch.nn as nn
import numpy as np
import math
import os
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import get_train_val_test_loaders,get_train_test_loaders

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from sklearn.metrics import confusion_matrix
from datetime import datetime
import torch.nn.init as init
import warnings

from wafer_models import msftrans

warnings.simplefilter(action='ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
from plotConfusionMatrix import plot_Matrix


def test_my_model(model, classes, dataloader_valid, save_dir='./cm/'):
    model.eval()  # 将模型设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, labels in dataloader_valid:
            if labels.dim() == 2:  # 检查是否是 one-hot 编码
                labels = torch.argmax(labels, dim=1)
            img = img.cuda()
            labels = labels.cuda()
            outputs = model(img)
            _, preds = torch.max(outputs.data, 1)
            # 保存预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Accuracy: {accuracy:.8f}')
    print(f'Precision: {precision:.8f}')
    print(f'Recall: {recall:.8f}')
    print(f'F1 Score: {f1:.8f}')
    # 计算混淆矩阵
    confusion = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(20, 20))
    # plt.rcParams['font.sans-serif'] = 'SimHei'  # 中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.xlabel('Predicted Label')
    plt.ylabel('真实值')
    sns.heatmap(confusion, annot=True, fmt='d', cmap=plt.cm.Blues)

    # plt.title('混淆矩阵')
    # plt.tight_layout()
    if save_dir:
        # 检查目录是否存在，不存在则创建
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：年月日_小时分钟秒
        eps_fname = f'confusion_matrix_{now}.eps'  # .eps 文件名
        plt.savefig(os.path.join(save_dir, eps_fname), format='eps', dpi=300)
        print(f'Confusion matrix saved to {os.path.join(save_dir, eps_fname)}')
    else:
        # 如果未指定保存目录，可以保存到当前工作目录
        now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：年月日_小时分钟秒
        eps_fname = f'confusion_matrix_{now}.eps'  # .eps 文件名
        plt.savefig(eps_fname, format='eps', dpi=300)
        print(f'Confusion matrix saved to {eps_fname}')
    # plt.show()
    plot_Matrix(confusion, classes, title='Confusion matrix', save_dir='confusion_matrix')


def train_model(model, criterion, optimizer, scheduler, start_time, num_epochs):

    # todo 错误捕获，其余时候注释
    # torch.autograd.set_detect_anomaly(True)

    best_model_wts = None
    best_acc = 0.0
    train_loss = []
    test_loss = []
    train_f1_micro = []
    train_f1_macro = []
    test_f1_micro = []
    test_f1_macro = []
    lr_list = []
    epoch_list = []
    since = time.time()
    save_dir = os.path.join('./result', start_time)
    os.makedirs(save_dir, exist_ok=True)  # 如果目录已经存在，则不会报错
    for epoch in range(num_epochs):
        lr_list.append(scheduler.get_last_lr())
        epoch_list.append(epoch)
        print('Epoch {}/{}, lr {}'.format(epoch + 1, num_epochs, scheduler.get_last_lr()))
        print('-' * 20)
        # Each epoch has a training and validation phase

        for phase in ['train', 'test']:
            if phase == 'train':
                print('model is training')
                model.train()
            else:
                print('model is evaling')
                model.eval()

            running_loss, running_corrects, total_samples = 0.0, 0, 0
            all_preds, all_labels = [], []

            for inputs, labels in data_loaders[phase]:
                # 数据移动到设备上
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(labels.shape)
                if labels.dim() == 2:  # 检查是否是 one-hot 编码
                    labels = torch.argmax(labels, dim=1)
                # 清零梯度
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).float()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 统计信息
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(torch.softmax(outputs, dim=1), 1)
                corrects = (preds == labels)
                running_corrects += corrects.sum().item()
                total_samples += inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # print(f'dataloder down')
            # 计算损失、准确率和F1分数
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples
            all_preds, all_labels = np.array(all_preds), np.array(all_labels)
            f1_micro = f1_score(all_labels, all_preds, average='micro')
            f1_macro = f1_score(all_labels, all_preds, average='macro')

            if phase == 'train':
                train_loss.append(epoch_loss)
                # 可以选择记录不同的F1分数
                train_f1_micro.append(f1_micro)
                train_f1_macro.append(f1_macro)
                scheduler.step()
            else:
                test_loss.append(epoch_loss)
                test_f1_micro.append(f1_micro)
                test_f1_macro.append(f1_macro)

            print('{} Loss: {:.4f} {} Acc: {:.4f} F1 Micro: {:.4f} F1 Macro: {:.4f}'.format(
                phase, epoch_loss, phase, epoch_acc, f1_micro, f1_macro))

            if phase == 'test' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                # best_model_wts = model.state_dict()
                best_model_wts = copy.deepcopy(model.state_dict())  # 使用深拷贝保存当前模型参数
            if phase == 'test' and epoch_acc > 0.99:
                # 判断是否需要保存模型
                save_path = os.path.join(save_dir, f'{start_time}_model_{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
        # torch.save(model.state_dict(), f'./result/{start_time}' + '_model_' + str(epoch + 1) + '.pth')
    time_elapsed = time.time() - since

    # save_file('fea_train', train_loss, './result/')
    # plt_lr_decay(epoch_list, lr_list)
    print('Training complete in {:.0f}s'.format(time_elapsed))
    print('Best val Acc: {:4f}:'.format(best_acc))
    # now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：年月日_小时分钟秒
    save_path = os.path.join(save_dir, f'{start_time}_best_model.pth')
    torch.save(best_model_wts, save_path)
    print(f'Model saved to: {save_path}')
    model.load_state_dict(torch.load(save_path), False)
    print(f'Model loaded from: {save_path}')

    return model


def weight_init(m):
    """Kaiming 初始化 + BatchNorm 初始化"""
    # init.xavier_normal_(m.weight)
    if isinstance(m, nn.Conv2d):
        # Kaiming 初始化（针对 ReLU 激活函数）
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:  # 如果卷积层有偏置项，初始化为0
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # BatchNorm 初始化：权重为1，偏置为0
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def test_model_output(model_factory, input_shape=(1, 1, 64, 64)):
    try:
        model = model_factory()
        print(f"[INFO] Model initialized: {model.__class__.__name__}")

        x = torch.randn(input_shape)  # 随机输入
        output = model(x)

        if isinstance(output, tuple):
            output = output[0]  # 如果模型返回多个输出，取第一个

        print(f"[SUCCESS] Forward pass OK. Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed for model: {model_factory.__qualname__}")
        print(f"Error: {e}")
        return False


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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 定义训练参数
    dataset_path = 'dataset_mix38/Wafer_Map_Datasets.npz'
    num_epochs = 100
    batch_size = 128
    test_size = 0.2

    # train_DataLoader, val_DataLoader, test_DataLoader = get_train_val_test_loaders(dataset_path,
    #                                                                                batch_size=128,
    #                                                                                test_size=0.2,
    #                                                                                random_state=42)
    train_DataLoader,  test_DataLoader = get_train_test_loaders(dataset_path,
                                                                                   batch_size=128,
                                                                                   test_size=0.2,
                                                                                   random_state=42)
    data_loaders = {'train': train_DataLoader,
                    'test': test_DataLoader,

                    }

    classes = ('Normal', 'Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random', 'C+EL',
               'C+ER', 'C+L', 'C+S', 'D+EL', 'D+ER', 'D+L', 'D+S', 'EL+L', 'EL+S', 'ER+L', 'ER+S', 'L+S',
               'C+EL+L', 'C+EL+S', 'C+ER+L', 'C+ER+S', 'C+L+S', 'D+EL+L', 'D+EL+S', 'D+ER+L', 'D+ER+S',
               'D+L+S', 'EL+L+S', 'ER+L+S', 'C+L+EL+S', 'C+L+ER+S', 'D+L+EL+S', 'D+L+ER+S')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from model import model18
    from base_models import resnet, densenet, vit, swintransformer, mobilvit
    from wafer_models import dc_net, msftrans, WSCN, WMPeleeNet, cwdr, mernet

    model_factories = [
        lambda: resnet.ResNet18(1, 38),
        lambda: densenet.DenseNet121(38, 1),
        #lambda: dc_net.DCNet(1,38),# 内存4090不够128bs
        lambda: WSCN.WaferSegClassNet(in_channels=1, num_class=38),
        lambda: WMPeleeNet.WMPeleeNet(in_ch=1, num_classes=38),
        lambda: cwdr.CWDR_model(num_of_classes=38),
        lambda: mernet.MERNet(in_channels=1, num_classes=38),
        #lambda: vit.Vision_transformer_base(1, 38), # 内存4090不够128bs
        #lambda: swintransformer.swin_tiny_patch4_window7_224(38, 1),
        lambda: mobilvit.mobilevit_xxs(38, 1),
        lambda: msftrans.MSFTransformer(num_classes=38, image_size=64, embed_dim=64, num_heads=8, num_layers=6),
        lambda: model18(in_ch=1, num_classes=38,alpha=0.875),
        lambda: model18(in_ch=1, num_classes=38,alpha=0.75),
        lambda: model18(in_ch=1, num_classes=38,alpha=0.625),
        lambda: model18(in_ch=1, num_classes=38,alpha=0.5),
        lambda: model18(in_ch=1, num_classes=38,alpha=0.375),
        lambda: model18(in_ch=1, num_classes=38,alpha=0.25),
        lambda: model18(in_ch=1, num_classes=38,alpha=0.125),
    ]



    model_names = [
        # "ResNet18","DenseNet121",
        # "DCNet",
        # "WaferSegClassNet","WMPeleeNet","CWDR_model",
        # "MERNet",
        # "Vision_transformer_base",
        # "swin_tiny_patch4_window7_224",
        # "mobilevit_xxs","MSFTransformer",
        # "proposed",
        "proposed_0.875", "proposed_0.75", "proposed_0.625", "proposed_0.5","proposed_0.375", "proposed_0.25", "proposed_0.125",
        # "proposed_only_spa", "proposed_only_freq",
        # "proposed_vallina", "proposed_only_spa_vallina", "proposedonly_freq_vallina",
        # "CWDR_model"
    ]

    # results = {}
    # for i, factory in enumerate(model_factories):
    #     input_shape = (1, 1, 64, 64)
    #     result = test_model_output(factory, input_shape)
    #     results[f"Model {i + 1}"] = {
    #         "status": "Passed" if result else "Failed",
    #         "input_shape": input_shape,
    #     }
    # 输出结果汇总
    # print("\n=== 测试结果汇总 ===")
    # for key, value in results.items():
    #     print(f"{key}: {value['status']}, Input Shape: {value['input_shape']}")
    # time.sleep(6000)

    for i, (model_factory, model_name) in enumerate(zip(model_factories, model_names)):
        print(f"\n===== 开始训练 {model_name}=====")

        # 初始化模型
        model = model_factory()
        weight_init(model)
        model = model.to(device)

        # 定义优化器和损失函数（可根据模型调整参数）
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # total_steps = num_epochs * len(train_DataLoader)
        # lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=total_steps)
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        start_time = datetime.now().strftime(f"{model_name}_mc_%Y%m%d_%H%M%S")

        # 训练模型
        print(f'Start training at {start_time}')
        trained_model = train_model(
            model, criterion, optimizer, lr_schedule,
            start_time, num_epochs=num_epochs
        )
        print(f'Finished training {model_name} at {datetime.now().strftime("%Y%m%d_%H%M%S")}')

        test_my_model(trained_model, classes, test_DataLoader)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if i < 5:
            time.sleep(30)

    time.sleep(600)
    os.system("/usr/bin/shutdown")
