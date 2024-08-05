"""
File: anotc_fetch_eeg_emotion.py
Author: xiales
Date: 2024-08-05
Description: 用于处理匿名科创地面站导出的.csv文件中的数据 处理脑电情绪识别标签(每个样本连续标签值) 并将其保存为 csv 和 npy 格式
"""

import os
import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def fetch_data_from_file(filename, num_channels, num_samples, label_column, retain_tail=False):
    """
    从CSV文件中提取数据段，并返回数据和标签。

    参数：
    filename (str): CSV文件的路径。
    num_channels (int): 数据的通道数。
    num_samples (int): 每个数据段的样本数。
    label_column (int): 标签所在的列索引。
    retain_tail (bool): 是否保留尾部不足num_samples的数据段。

    返回：
    X (np.ndarray): 提取的数据，形状为 (data_size, num_channels, num_samples)。
    y (np.ndarray): 对应的标签，形状为 (data_size,)。
    """

    # 检查文件路径是否有效
    if not os.path.isfile(filename):
        raise ValueError(f"The file path is invalid. Please check the file path. \r\nError path: {filename}")

    # 读取CSV文件并跳过前两行
    dataframe = pd.read_csv(filename, skiprows=2, header=None)

    # 提取标签列
    labels = dataframe.iloc[:, label_column].values

    # 提取有效数据列
    data = dataframe.iloc[:, :num_channels].values

    # 初始化数据和标签列表
    X = []  # 初始化X为空列表，用于存储数据段
    y = []  # 初始化y为空列表，用于存储对应的标签

    # 初始化开始索引
    start_idx = None  # start_idx用于记录当前非零标签段的起始索引

    # 遍历标签数据
    for i in range(len(labels)):
        if labels[i] != 0 and start_idx is None:
            # 遇到非零标签且开始索引未设置时，设置开始索引
            start_idx = i
        elif labels[i] == 0 and start_idx is not None:
            # 遇到零标签且开始索引已设置时，处理前面的段
            if i - start_idx >= num_samples or (retain_tail and i > start_idx):
                # 确保段长度不小于num_samples，或retain_tail为True时保留尾部数据
                end_idx = start_idx + num_samples  # 计算数据段的结束索引
                while end_idx <= i or (retain_tail and start_idx < i):
                    if end_idx <= i:
                        # 提取数据段并转置以符合 (data_size, num_channels, num_samples)
                        segment = data[start_idx:end_idx].T  # 转置数据段
                        X.append(segment)  # 添加数据段到X
                        y.append(labels[start_idx])  # 添加对应的标签到y
                    elif retain_tail:
                        # 保留尾部数据时，确保片段长度为 num_samples
                        tail_segment = data[start_idx:i]  # 获取尾部数据段
                        needed_len = num_samples - len(tail_segment)  # 计算需要补足的长度
                        if start_idx + needed_len <= len(data):
                            segment = np.vstack((tail_segment, data[i:i + needed_len])).T  # 合并并转置数据段
                            X.append(segment)  # 添加数据段到X
                            y.append(labels[start_idx])  # 添加对应的标签到y
                    start_idx += num_samples  # 更新开始索引
                    end_idx = start_idx + num_samples  # 更新结束索引
            # 重置开始索引
            start_idx = None

    # 处理文件尾部仍然有标签的情况
    if start_idx is not None and len(labels) - start_idx >= num_samples:
        end_idx = start_idx + num_samples  # 计算数据段的结束索引
        while end_idx <= len(labels):
            # 提取数据段并转置以符合 (data_size, num_channels, num_samples)
            segment = data[start_idx:end_idx].T  # 转置数据段
            X.append(segment)  # 添加数据段到X
            y.append(labels[start_idx])  # 添加对应的标签到y
            start_idx += num_samples  # 更新开始索引
            end_idx = start_idx + num_samples  # 更新结束索引

    # 检查是否有有效数据
    if len(X) == 0 or len(y) == 0:
        print(f"No valid data found in the file: {os.path.basename(filename)}")  # 输出无有效数据的文件名
        return None, None  # 返回None

    # 转换为 Numpy 数组
    X = np.array(X)  # 将X转换为Numpy数组
    y = np.array(y)  # 将y转换为Numpy数组

    return X, y  # 返回数据和标签


def fetch_data_from_folder(folder_path, num_channels, num_samples, label_column, save_to_file, retain_tail=False):
    """
    从文件夹中的所有CSV文件中提取数据段，并返回数据和标签。

    参数：
    folder_path (str): 包含CSV文件的文件夹路径。
    num_channels (int): 数据的通道数。
    num_samples (int): 每个数据段的样本数。
    label_column (int): 标签所在的列索引。
    save_to_file (bool): 是否将数据保存到文件。
    retain_tail (bool): 是否保留尾部不足num_samples的数据段。

    返回：
    all_X (np.ndarray): 提取的所有数据，形状为 (total_data_size, num_channels, num_samples)。
    all_y (np.ndarray): 提取的所有标签，形状为 (total_data_size,)。
    """

    # 获取文件夹中所有CSV文件的路径
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # 初始化所有文件的数据和标签列表
    all_X = []  # 初始化all_X为空列表，用于存储所有文件的数据段
    all_y = []  # 初始化all_y为空列表，用于存储所有文件的标签

    # 处理每个CSV文件
    for file in csv_files:
        # 从当前文件中提取数据和标签
        X, y = fetch_data_from_file(file, num_channels, num_samples, label_column, retain_tail)
        if X is not None and y is not None:
            all_X.append(X)  # 将当前文件的数据添加到all_X
            all_y.append(y)  # 将当前文件的标签添加到all_y

    # 将所有文件的数据和标签组合成单个Numpy数组
    if len(all_X) > 0 and len(all_y) > 0:
        all_X = np.concatenate(all_X, axis=0)  # 将所有数据段沿第0维拼接
        all_y = np.concatenate(all_y, axis=0)  # 将所有标签沿第0维拼接
    else:
        all_X, all_y = None, None  # 如果没有有效数据，返回None
        print("No valid data found in any file in the folder.")  # 输出无有效数据的提示

    # 保存数据到文件
    if save_to_file:
        # save_file_path = os.path.join(folder_path, 'numpy_data_file')  # 设置保存路径
        save_file_path = folder_path
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)  # 如果路径不存在，创建路径
        np.save(os.path.join(save_file_path, 'X.npy'), all_X)  # 保存数据到X.npy
        np.save(os.path.join(save_file_path, 'y.npy'), all_y)  # 保存标签到y.npy
        print(f"Data saved to {save_file_path}")  # 输出保存路径

    return all_X, all_y  # 返回所有数据和标签

def fetch_eeg_emotion_loader(data_path, batch_size=64, is_print=True, default_type=torch.float32):
    """
    加载EEG情感数据集，创建数据加载器。

    参数：
    data_path (str): 数据路径，应包含 'X.npy' 和 'y.npy' 文件。
    batch_size (int): 数据加载器的批量大小，默认为64。
    is_print (bool): 是否打印数据集信息，默认为True。
    default_type (torch.dtype): 数据类型，默认为torch.float32。

    返回：
    train_loader (DataLoader): 训练集的数据加载器。
    val_loader (DataLoader): 验证集的数据加载器。
    test_loader (DataLoader): 测试集的数据加载器。
    """
    X = np.load(os.path.join(data_path, 'X.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))

    # 将 NumPy 数组转换为 PyTorch 张量
    X = torch.tensor(X, dtype=default_type)
    y = torch.tensor(y, dtype=torch.long)
    # 使用 train_test_split 将数据集拆分为训练集和验证集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if is_print:
        print(f"X shape: {X.shape}")  # 打印 X 的形状
        print(f"y shape: {y.shape}")  # 打印 y 的形状
        unique, counts = np.unique(y, return_counts=True)  # 获取唯一值及其数量
        for label, count in zip(unique, counts):
            print(f"Label: {label}, Count: {count}")  # 打印每个标签的数量

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    
    folder_path = "D:/Downloads"

    num_channels = 8
    num_samples = 250
    label_column = 8
    save_to_file = True
    retain_tail = True

    X, y = fetch_data_from_folder(folder_path, num_channels, num_samples, label_column, save_to_file, retain_tail)

    train_loader, val_loader, test_loader = fetch_eeg_emotion_loader(data_path=folder_path, batch_size=32, is_print=True)





