""""
File: test.py
Author: xiales
Date: 2024-07-31
Description: 用于各种测试的主程序
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

# 自定义模块
from models import cnn_model
from tools import data_fetch_tools
from tools import plot_tools

# 检查是否有可用的 GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
epochs = 1000
batch_size = 64
learning_rate = 0.005


# 数据提取
train_loader, test_loader, val_loader = data_fetch_tools.deap_loader_fetch(batch_size=batch_size)


# 定义模型
model = cnn_model.CnnC6F2(in_channels=14, num_classes=4).to(device)
model_name = 'CnnC6F2'
# print(model)  # 打印网络结构
# 定义优化器为 Adam
optimizer = optim.Adam(
    model.parameters(),                # 需要优化的模型参数
    lr=learning_rate,                  # 学习率
    betas=(0.9, 0.999),                # beta_1 和 beta_2
    eps=1e-7,                          # 防止除零错误的平滑项，默认值是 1e-8
    weight_decay=0                     # 权重衰减，通常用于 L2 正则化，默认值是 0
)
# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()


# # 定义模型
# model = cnn_model.CnnC2F2(in_channels=14, num_classes=4).to(device)
# model_name = 'CnnC2F2'
# # print(model)  # 打印网络结构
# # 定义损失函数为交叉熵损失
# criterion = nn.CrossEntropyLoss()
# # 定义优化器为随机梯度下降，学习率为0.01，动量为0.9
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# 加载保存点
savepoint = torch.load("models_save/" + model_name + "_model_savepoint.pth")
model.load_state_dict(savepoint['model_state_dict'])
optimizer.load_state_dict(savepoint['optimizer_state_dict'])
start_epoch = savepoint['epoch'] + 1
# running_loss = savepoint['loss']
# learning_rate = savepoint['learning_rate']
# batch_size = savepoint['batch_size']
# train_losses = savepoint['train_losses']
# train_accuracies = savepoint['train_accuracies']

# plot_tools.plot_training_metrics(train_losses=train_losses, train_accuracies=train_accuracies, is_save=True, save_name=model_name + "_training_metrics")

# 测试模型
correct = 0  # 初始化正确预测数
total = 0  # 初始化总数
with torch.no_grad():  # 禁用梯度计算
    for data in test_loader:  # 遍历测试数据集
        images, labels = data  # 获取输入数据和标签
        images, labels = images.to(device), labels.to(device)  # 将数据移到 GPU

        outputs = model(images)  # 前向传播
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)  # 更新总数
        correct += (predicted == labels).sum().item()  # 更新正确预测数

print(f'Accuracy of the modelwork on the test data: {100* correct / total:.2f}%')  # 打印测试集上的准确率



