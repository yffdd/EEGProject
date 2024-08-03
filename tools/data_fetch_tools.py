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



def deap_loader_fetch_acse(batch_size=64, test_size=0.2, random_state=42, is_print=False, default_type=torch.float32):
    """
    加载DEAP数据集, 并返回训练集和测试集以及验证集
    """
    print("fetch DEAP dataset...")
    # 加载数据集
    # dataset_name = "deap"
    dataset_name = "deap_class4"
    databases_out_directory = r"E:/Databases/OutData/DEAP/ACSE/"  # Windows 目录
    # databases_out_directory = r"/bigdisk/322xcq/Databases/OutData/DEAP/ACSE/"  # school-gpu 目录
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if is_print:
        # 打印训练集的大小和形状
        print(f'training set size:          {len(train_dataset)}')
        print(f'training set data shape:    {X_train.shape}')
        print(f'training set labels shape:  {y_train.shape}')
        # 打印训练集的标签唯一值
        print(f'training set unique labels: {y_train.unique()}')

    print("fetch data complete")

    return train_loader, test_loader, val_loader


def deap_loader_fetch_sgmc(batch_size=64, is_print=False, default_type=torch.float32):


    print("fetch DEAP dataset...")
    # 提取数据集
    databases_out_directory = "E:/Databases/OutData/DEAP/SGMC/"  # Windows目录
    # databases_out_directory = r"/bigdisk/322xcq/Databases/OutData/DEAP/SGMC/"  # school-gpu 目录
    X_train = np.load(databases_out_directory + 'x_train_DEAP.npy')
    X_test = np.load(databases_out_directory + 'x_test_DEAP.npy')
    X_val = np.load(databases_out_directory + 'x_val_DEAP.npy')
    y_train = np.load(databases_out_directory + 'y_train_DEAP.npy')
    y_test = np.load(databases_out_directory + 'y_test_DEAP.npy')
    y_val = np.load(databases_out_directory + 'y_val_DEAP.npy')

    # 将 numpy 数组转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=default_type)
    X_test = torch.tensor(X_test, dtype=default_type)
    X_val = torch.tensor(X_val, dtype=default_type)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if is_print:
        # 打印训练集的大小和形状
        print(f'training set size:          {len(train_dataset)}')
        print(f'training set data shape:    {X_train.shape}')
        print(f'training set labels shape:  {y_train.shape}')
        # 打印训练集的标签唯一值
        print(f'training set unique labels: {y_train.unique()}')

    print("fetch data complete")

    return train_loader, test_loader, val_loader


def eeg_movement_loader_fetch(batch_size=64, is_print=False, default_type=torch.float32):
    data_path = "E:/Databases/OutData/EEG-MOVEMENT/EEG movement test data 240424/data_slice"
    X = np.load(os.path.join(data_path, 'X.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    # 将 NumPy 数组转换为 PyTorch 张量
    X = torch.tensor(X, dtype=default_type)
    y = torch.tensor(y, dtype=torch.long)
    X = X.unsqueeze(1) # 添加一个额外的维度，以匹配模型输入的形状

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if is_print:
        print(f'training set size:          {len(train_dataset)}')
        print(f'training set data shape:    {X_test.shape}')
        print(f'training set labels shape:  {y_test.shape}')
        # 打印测试集的标签唯一值
        print(f'training set unique labels: {np.unique(y_test)}')

    return train_loader, test_loader, val_loader

if __name__ == '__main__':
    train_loader, test_loader, val_loader = eeg_movement_loader_fetch(is_print=True)

    
