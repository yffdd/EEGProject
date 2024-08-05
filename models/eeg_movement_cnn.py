"""
File: eeg_movement_cnn.py
Author: xiales
Date: 2024-08-01
Description: 脑电情绪识别的CNN模型
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm, tgrange


def eeg_movement_loader_fetch(data_path, batch_size=64, is_print=True, default_type=torch.float32):
    """
    加载EEG运动意识数据集
    :param data_path: 数据集路径 应该包含 X.npy 和 y.npy 文件
    """
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
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if is_print:
        print(f'training set size:          {len(train_dataset)}')
        print(f'training set data shape:    {X_test.shape}')
        print(f'training set labels shape:  {y_test.shape}')
        # 打印测试集的标签唯一值
        print(f'training set unique labels: {np.unique(y_test)}')

    return train_loader, val_loader, test_loader



class DeepConvNet(nn.Module):
    """
    input: (batch_size, num_channels, num_samples)
    output: (batch_size, num_classes)

    Example Usage:
    # Initialize the model
    data_iter = iter(train_loader)              # Create an iterator for the train_loader to get data in batches
    inputs, labels = next(data_iter)            # Get one batch of data from the iterator (inputs and labels)
    model = cnn_models.AdversarialCNN_DeepConvNet(
        batch_size=train_loader.batch_size,     # Set the batch size from the train_loader
        num_channels=inputs.shape[1],           # Set the number of channels from the input shape
        num_samples=inputs.shape[2],            # Set the number of samples from the input shape
        num_classes=len(labels.unique()),       # Set the number of classes by counting unique labels
        learning_rate=learning_rate             # Set the learning rate
    )
    model.to(device)                            # Move model to the specified device (CPU/GPU)
    optimizer = model.optimizer                 # Get the optimizer defined within the model
    criterion = model.criterion                 # Get the loss function defined within the model
    """

    def __init__(self, batch_size=64, num_channels=14, num_samples=256, num_classes=4, learning_rate=0.001, model_name='AdversarialCNN_DeepConvNet'):
        """
        Args:
        - batch_size (int): Size of each batch of data.
        - num_channels (int): Number of input channels (e.g., EEG channels).
        - num_samples (int): Number of samples per channel (e.g., length of the time series).
        - num_classes (int): Number of output classes (e.g., number of classes for classification).
        - learning_rate (float): Learning rate for the optimizer.
        - model_name (str): Name of the model.
        """
        super(DeepConvNet, self).__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        self.conv1 = nn.Conv2d(1, 25, (1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(25, 25, (self.num_channels, 1), padding=0, bias=False)  # 对于 (chans, 1) 的卷积核不需要padding
        self.bn1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.dropout = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(25, 50, (1, 5), padding=(0, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1)
        
        self.conv4 = nn.Conv2d(50, 100, (1, 5), padding=(0, 2), bias=False)
        self.bn3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1)
        
        self.conv5 = nn.Conv2d(100, 200, (1, 5), padding=(0, 2), bias=False)
        self.bn4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1)
        
        self.fc = nn.Linear(200 * ((self.num_samples // 16)), self.num_classes)

        # Initialize optimizer and loss function
        self.module_name = model_name
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加一个通道维度，使形状变为 (batch_size, 1, num_channels, num_samples)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # 展平张量
        x = self.fc(x)
        
        return x
    

if __name__ == '__main__':
    
    # 检查是否有可用的 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device: {device}')

    # 设置超参数
    epochs = 100
    batch_size = 64
    learning_rate = 0.001

    is_save_model = False  # 是否保存模型
    is_plot = False  # 是否绘制训练集的准确率与损失曲线

    # 加载数据
    data_path = "E:/Databases/OutData/EEG movement/EEG movement data 240424/data_slice"  # 提取数据的目录
    train_loader, val_loader, test_loader= eeg_movement_loader_fetch(data_path=data_path, batch_size=batch_size)
    
    # 初始化模型
    num_channels = 1
    num_samples = 1000
    model = DeepConvNet(
        batch_size=batch_size,              # Set the batch size from the train_loader
        num_channels=num_channels,          # Set the number of channels from the input shape
        num_samples=num_samples,            # Set the number of samples from the input shape
        num_classes=3,                      # Set the number of classes by counting unique labels
        learning_rate=learning_rate         # Set the learning rate
    )
    model.to(device)

    # 定义优化器 和 损失函数
    optimizer = model.optimizer
    criterion = model.criterion

    # 初始化列表以存储每个 epoch 的 loss 和 accuracy
    train_losses = []
    train_accuracies = []

    # 训练模型
    print("training begins...")  # 开始训练
    for epoch in range(epochs):
        running_loss = 0.0  # 初始化损失值
        running_corrects = 0  # 初始化正确预测数
        total = 0  # 初始化总数
        for i, data in enumerate(train_loader, 0):  # 遍历训练数据集
            inputs, labels = data  # 获取输入数据和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到 GPU

            optimizer.zero_grad()  # 将梯度缓存清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            running_loss += loss.item()  # 累积损失
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)  # 更新样本总数
            running_corrects += (predicted == labels).sum().item()  # 更新正确预测数
        epoch_accuracy = running_corrects / total  # 计算准确率
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)  # 记录平均损失
        train_accuracies.append(epoch_accuracy)  # 记录准确率
        print(f"Epoch [{epoch}/{epochs}], train accuracy: {100*epoch_accuracy:.2f}%, train loss: {epoch_loss:.4f}")  # 打印准确率和平均损失
    print('training completed')  # 训练完成

    if is_save_model:
        # 保存模型
        if not os.path.exists('models_save'):
            os.makedirs('models_save')
        torch.save(model.state_dict(), "models_save/" + model.model_name + "_model.pth")

    if is_plot:
        # 绘制训练损失和准确率图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        # 绘制训练损失
        ax1.plot(train_losses, label='Training Loss', color='tab:blue')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss over Epochs')
        ax1.legend()
        # 绘制训练准确率
        ax2.plot(train_accuracies, label='Training Accuracy', color='tab:red')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy over Epochs')
        ax2.legend()
        plt.tight_layout()  # 调整布局
        plt.show()

    # 测试模型
    correct = 0  # 初始化正确预测数
    total = 0  # 初始化总数
    with torch.no_grad():  # 禁用梯度计算
        for data in test_loader:  # 遍历测试数据集
            inputs, labels = data  # 获取输入数据和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到 GPU
            outputs = model(inputs)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 更新总数
            correct += (predicted == labels).sum().item()  # 更新正确预测数
    test_accuracy = correct / total  # 计算测试准确率
    print(f"test accuracy: {100*test_accuracy:.2f}%")  # 打印测试集上的准确率



