""""
File: data_fetch_tools.py
Author: xiales
Date: 2024-07-31
Description: 用于各个模型的数据提取
"""



import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split



def deap_loader_fetch(batch_size=64, test_size=0.2, random_state=42, is_print=False, default_type=torch.float32):
    """
    加载DEAP数据集, 并返回训练集和测试集以及验证集
    """
    print("fetch DEAP dataset...")
    # 加载数据集
    # dataset_name = "deap"
    dataset_name = "deap_class4"
    # databases_out_directory = r"E:/Databases/OutData/DEAP/ACSE/"  # Windows 目录
    databases_out_directory = r"/bigdisk/322xcq/Databases/OutData/DEAP/ACSE/"  # school-gpu 目录
    filename = databases_out_directory + dataset_name + ".npz"
    data = np.load(filename)
    X = data['X']  # 形状: (38400, 14, 256)
    y = data['y']  # 形状: (38400,) 标签: (0, 1, 2, 3)
    # 将 NumPy 数组转换为 PyTorch 张量
    X = torch.tensor(X, dtype=default_type)
    y = torch.tensor(y, dtype=torch.long)

    # X = torch.rand(38400, 14, 256)  # 生成随机数据
    # y = torch.randint(0, 4, (38400,))  # 生成随机标签


    # 打印数据形状和标签类型
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y unique: {np.unique(y)}")
    # 使用 train_test_split 将数据集拆分为训练集和测试集和验证集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=random_state)
    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if is_print:
        # 打印训练集的大小和形状
        print(f'Training set size: {len(train_dataset)}')
        print(f'Training set data shape: {train_dataset.data.shape}')
        print(f'Training set labels shape: {train_dataset.targets.shape}')
        # 打印训练集的标签唯一值
        train_labels = train_dataset.targets
        print(f'Unique labels in training set: {train_labels.unique()}')

    print("fetch data complete")

    return train_loader, test_loader, val_loader


if __name__ == '__main__':
    train_loader, test_loader, val_loader = deap_loader_fetch()

    # 获取一个批次的数据
    data_iter = iter(train_loader)
    inputs, labels = next(data_iter)
    # 查看输入数据的形状
    print(f"Input shape: {inputs.shape}")
    # 查看标签的形状
    print(f"Labels shape: {labels.shape}")

    print(f"Input shape[0]: {inputs.shape[0]}")
    print(f"Input shape[1]: {inputs.shape[1]}")
    print(f"Input shape[2]: {inputs.shape[2]}")
    print(f"labels uqique: {len(labels.unique())}")

