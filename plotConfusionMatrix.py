import os

import numpy as np
# import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime

from matplotlib.colors import ListedColormap
# from sklearn.metrics import multilabel_confusion_matrix
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置


def confusion_matrix(pred, Y, number_class, save_path=None):
    confusion_matrice = np.zeros((number_class, number_class))
    for prediction, real_value in zip(pred, Y):
        prediction = int(prediction)
        real_value = int(real_value)
        confusion_matrice[prediction][real_value] += 1

    if save_path is not None:
        np.savetxt(save_path, confusion_matrice, fmt='%d', delimiter='\t')
        print(f"Confusion matrix saved to: {save_path}")

    return np.array(confusion_matrice)

def plot_Matrix(cm, classes, title=None, save_dir = None, cmap=plt.cm.Blues):
    plt.rc('font', family='DejaVu Sans', size='8')  # 设置字体样式、大小

    # 过滤小于 1 的值
    cm_filtered = np.where(cm < 1, 0, cm)

    # 归一化混淆矩阵（用于颜色映射）
    cm_normalized = cm_filtered.astype('float') / cm_filtered.sum(axis=1)[:, np.newaxis]
    cm_normalized[np.isnan(cm_normalized)] = 0  # 处理可能的 NaN 值

    # 根据混淆矩阵的行列数动态调整图像尺寸
    n_rows, n_cols = cm.shape
    cell_size = 0.15  # 每个单元格的大小（单位：英寸）
    fig_width = n_cols * cell_size
    fig_height = n_rows * cell_size

    # 创建图像
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # 设置图像尺寸

    # 自定义颜色映射
    colors = ['white', 'lightblue']
    cmap_custom = ListedColormap(colors)

    # 使用归一化后的值绘制混淆矩阵
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap_custom, vmin=0, vmax=1)

    # 设置坐标轴标签和标题
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 计算颜色阈值
    thresh = 1 # 归一化后的阈值

    # 标注原始数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm_filtered[i, j] > 0:  # 只标注大于 0 的值
                ax.text(j, i, f'{cm_filtered[i, j]:.0f}',  # 显示原始数值，不保留小数
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black",  # 动态设置文本颜色
                        fontsize=6)  # 设置文本字体大小

    # 调整布局
    fig.tight_layout()
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

    # 动态显示图像
    # plt.ion()  # 开启交互模式
    # plt.show(block=False)  # 非阻塞显示图像
    # plt.pause(20)  # 暂停 20 秒
    # plt.close()  # 关闭图像
# def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
#     plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小
#
#     # 按列进行归一化
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     cm[np.isnan(cm)] = 0  # 或者使用其他合适的值
#     print("Normalized confusion matrix")
#     str_cm = cm.astype(np.str_).tolist()
#     for row in str_cm:
#         print('\t'.join(row))
#     # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             if int(cm[i, j] * 100 + 0.5) <= 1:
#                 cm[i, j] = 0
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带
#
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='Actual',
#            xlabel='Predicted')
#
#     # 通过绘制格网，模拟每个单元格的边框
#     ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
#     ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
#     ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
#     ax.tick_params(which="minor", bottom=False, left=False)
#
#     # 将x轴上的lables旋转45度
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # 标注百分比信息
#     fmt = 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             if int(cm[i, j] * 100 + 0.5) > 0:
#                 ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
#                         ha="center", va="center",
#                         color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     plt.savefig('./result/' + 'confusion_matrix' + '.eps', dpi=300)
#     plt.ion()
#     plt.pause(500)
#     plt.close()


def plot_Matrix_with_number(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小

    # 打印原始混淆矩阵（如果需要）
    print("Confusion matrix with absolute numbers")
    str_cm = cm.astype(np.str_).tolist()
    for row in str_cm:
        print('\t'.join(row))


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注具体数量信息
    fmt = 'd'  # 使用'd'格式化整数
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] != 0:  # 只有当值不是0时才进行标注
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig('./result/' + 'confusion_matrix_with_numbers' + '.eps', dpi=300)
    plt.show()  # 显示图像，替代了原来的plt.pause(500)和plt.close()


def convert_labels(real_labels, pre_labels):
    real_labels =[p.astype(int) for p in real_labels]  # True → 1, False → 0
    pre_labels =[p.astype(int) for p in pre_labels]
    real_labels = np.array(real_labels)  # 形状应为 (N, 8)
    pre_labels = np.array(pre_labels)    # 形状应为 (N, 8)
    new_real_labels = convert_one_hot_to_index(real_labels)
    new_pre_labels = convert_one_hot_to_index(pre_labels)
    return new_real_labels, new_pre_labels

def convert_one_hot_to_index(label_array):
    labels = []
    for i in range(label_array.shape[0]):
        label_str = ''.join(map(str, label_array[i].astype(int)))
        idx = update_label_to_index(label_str)
        labels.append(idx)
    return np.array(labels)

from label_maping import *

def update_label_to_index(original_label_str):
    """
    根据新的标签映射规则将8维one-hot编码转换为单个索引
    :param original_label_str: 原始的one-hot编码标签 (8维)，作为字符串
    :return: 对应的单标签索引 (0-37)
    """
    # 使用label_mapping将原始标签字符串映射到新的索引
    if original_label_str in label_mapping:
        # 从列表中取出唯一的索引值
        new_indices = label_mapping[original_label_str]
        if len(new_indices) != 1:
            raise ValueError(f"标签映射应该只包含一个索引，但发现: {new_indices}")
        return new_indices[0]  # 返回单个整数索引
    else:
        # 如果没有找到对应的映射，默认设为未知类别(0)
        return 0