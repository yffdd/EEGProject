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


# 设置系统的Dataset根目录
# sys_dataset_root_dir = 'E:/'  # xiales-pc Windows系统路径
sys_dataset_root_dir = '/bigdisk/322xcq/'  # 服务器系统路径


def deap_loader_fetch_acse(batch_size=64, test_size=0.2, random_state=42, is_print=True, default_type=torch.float32):
    """
    加载DEAP数据集, 并返回训练集和测试集以及验证集
    """
    print("fetch DEAP dataset...")
    # 加载数据集
    dataset_name = "deap"
    databases_out_directory = os.path.join(sys_dataset_root_dir, "Databases/OutData/DEAP/ACSE/")
    filename = databases_out_directory + dataset_name + ".npz"
    data = np.load(filename)
    X = data['X']  # 形状: (38400, 14, 256)
    y = data['y']  # 形状: (38400,) 标签: (0, 1, 2, 3)
    # 将 NumPy 数组转换为 PyTorch 张量
    X = torch.tensor(X, dtype=default_type)
    y = torch.tensor(y, dtype=torch.long)

    # X = torch.rand(38400, 14, 256)  # 生成随机数据
    # y = torch.randint(0, 4, (38400,))  # 生成随机标签

    # # 打印数据形状和标签类型
    # print(f"X shape: {X.shape}")
    # print(f"y shape: {y.shape}")
    # print(f"y unique: {np.unique(y)}")

    # 使用 train_test_split 将数据集拆分为训练集和测试集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if is_print:
        # 打印训练集的大小和形状
        print(f'training set size:          {len(train_dataset)}')
        print(f'training set data shape:    {X_train.shape}')
        print(f'training set labels shape:  {y_train.shape}')
        # 打印训练集的标签唯一值
        print(f'training set unique labels: {y_train.unique()}')

    print("fetch data complete")

    return train_loader, val_loader, test_loader


def deap_loader_fetch_sgmc(batch_size=64, is_print=True, default_type=torch.float32):

    print("fetch DEAP dataset...")
    # 提取数据集
    databases_out_directory = os.path.join(sys_dataset_root_dir, "Databases/OutData/DEAP/SGMC/")  # 数据集输出目录
    X_train = np.load(databases_out_directory + 'x_train_DEAP.npy')
    X_test = np.load(databases_out_directory + 'x_test_DEAP.npy')
    X_val = np.load(databases_out_directory + 'x_val_DEAP.npy')
    y_train = np.load(databases_out_directory + 'y_train_DEAP.npy')
    y_test = np.load(databases_out_directory + 'y_test_DEAP.npy')
    y_val = np.load(databases_out_directory + 'y_val_DEAP.npy')

    # 将 numpy 数组转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=default_type)
    X_val = torch.tensor(X_val, dtype=default_type)
    X_test = torch.tensor(X_test, dtype=default_type)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if is_print:
        # 打印训练集的大小和形状
        print(f'training set size:          {len(train_dataset)}')
        print(f'training set data shape:    {X_train.shape}')
        print(f'training set labels shape:  {y_train.shape}')
        # 打印训练集的标签唯一值
        print(f'training set unique labels: {y_train.unique()}')

    print("fetch data complete")

    return train_loader, val_loader, test_loader


def eeg_movement_loader_fetch(batch_size=64, is_print=True, default_type=torch.float32):

    databases_out_directory = os.path.join(sys_dataset_root_dir, "/Databases/OutData/EEG_Movement/EEG movement data 240424/data_slice")  # 数据集输出目录
    X = np.load(os.path.join(databases_out_directory, 'X.npy'))
    y = np.load(os.path.join(databases_out_directory, 'y.npy'))
    # 将 NumPy 数组转换为 PyTorch 张量
    X = torch.tensor(X, dtype=default_type)
    y = torch.tensor(y, dtype=torch.long)
    X = X.unsqueeze(1) # 添加一个额外的维度，以匹配模型输入的形状

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
        print(f'training set size:          {len(train_dataset)}')
        print(f'training set data shape:    {X_train.shape}')
        print(f'training set labels shape:  {X_train.shape}')
        # 打印测试集的标签唯一值
        print(f'training set unique labels: {np.unique(y_train)}')

    return train_loader, val_loader, test_loader


def select_eeg_channels(X, channel_names, channels_to_use):
    """
    选择脑电特定的通道。
    
    参数：
    X (ndarray): 输入信号，形状为 (num_subjects, num_channels, num_samples)。
    channel_names (list): 所有通道的名称列表。
    channels_to_use (list): 需要选择的通道名称列表。
    
    返回：
    ndarray: 选择通道后的信号，形状为 (num_subjects, len(channels_to_use), num_samples)。
    """
    # 初始化通道索引列表
    channel_index_list = list()
    # 查找需要使用的通道索引
    for i in range(len(channels_to_use)):
        # 如果通道在实际通道名称列表中，添加其索引到通道索引列表中
        if channels_to_use[i] in channel_names:
            channel_index_list.append(channel_names.index(channels_to_use[i]))
        else:
            # 如果通道不在实际通道名称列表中，发出警告
            print(' Channel ' + channels_to_use[i] + ' could not be found in the list of actual channels')
    # 初始化选择通道后的数组，形状为 (num_subjects, len(self.channels_to_use), num_samples)
    X_selected_channels = np.zeros((X.shape[0], len(channels_to_use), X.shape[2]))
    # 遍历每个通道及其索引，将对应的通道数据复制到选择后的数组中
    for channel, channel_index in enumerate(channel_index_list):
        X_selected_channels[:, channel, :] = X[:, channel_index, :]
    # 更新复制的数据为选择通道后的数据
    X = X_selected_channels
    # 返回选择通道后的信号
    return X


def transform_labels(y, label_transform_dict):
    """
    根据标签转换字典，将旧标签转换为新标签。

    参数：
    y (np.ndarray): 包含旧标签的NumPy数组。
    label_change_dict (dict): 标签转换字典，其中键是旧的标签值，值是新的标签值。

    返回：
    np.ndarray: 转换后的标签数组。
    """
    # 创建一个与 y 形状相同的空数组，用于存储新标签
    new_y = np.empty_like(y)

    # 遍历旧标签，转换为新标签
    for old_label, new_label in label_transform_dict.items():
        new_y[y == old_label] = new_label

    return new_y


def seed_loader_fetch(batch_size=64, test_size=0.2, random_state=42, is_print=True, default_type=torch.float32):
    """
    加载SEED数据集, 并返回训练集和测试集以及验证集
    """
    print("fetch SEED dataset...")

    channel_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
                     'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
                     'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                     'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                     'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                     'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
                     'O2', 'OB1']
    channels_to_use = ['FP1', 'FP2', 'FZ', 'C3', 'CZ', 'C4', 'O1', 'O2']


    # 加载数据集
    databases_out_directory = os.path.join(sys_dataset_root_dir, "Databases/OutData/SEED/CNN/")  # 数据集输出目录
    X = np.load(os.path.join(databases_out_directory, 'X.npy'))
    y = np.load(os.path.join(databases_out_directory, 'y.npy'))

    # 标签转换, 将标签转换为[0, 1, 2, 3]
    label_transform_dict = {-1:0, 0:1, 1:2}
    y = transform_labels(y, label_transform_dict)

    # 选择需要的通道
    # X = select_eeg_channels(X, channel_names, channels_to_use)

    # 将 NumPy 数组转换为 PyTorch 张量
    X = torch.tensor(X, dtype=default_type)
    y = torch.tensor(y, dtype=torch.long)
    # X = X.unsqueeze(0) # 添加一个额外的维度，以匹配模型输入的形状

    # 使用 train_test_split 将数据集拆分为训练集和测试集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if is_print:
        # 打印训练集的大小和形状
        print(f'training set size:          {len(train_dataset)}')
        print(f'training set data shape:    {X_train.shape}')
        print(f'training set labels shape:  {y_train.shape}')
        # 打印训练集的标签唯一值
        print(f'training set unique labels: {y_train.unique()}')

    print("fetch data complete")

    return train_loader, val_loader, test_loader

if __name__ == '__main__':

    # train_loader, val_loader, test_loader = eeg_movement_loader_fetch(is_print=True)
    train_loader, val_loader, test_loader, val_loader = seed_loader_fetch()

    
