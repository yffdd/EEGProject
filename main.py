""""
File: main.py
Author: xiales
Date: 2024-07-30
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
from tqdm import tqdm, tgrange

# 自定义模块
from models import cnn_models
from tools import data_fetch_tools
from tools import plot_tools

# 更改torch默认数据类型为64位浮点数
default_type = torch.float64
torch.set_default_dtype(default_type)

# 检查是否有可用的 GPU
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f'Using device: {device}')
epochs = 100
batch_size = 64
learning_rate = 0.001


# 数据提取
train_loader, test_loader, val_loader = data_fetch_tools.deap_loader_fetch(batch_size=batch_size)

print("model initialization...")


# 定义模型
model = cnn_models.CnnC6F2(in_channels=14, num_classes=3).to(device)
model_name = model.module_name
# print(model)  # 打印网络结构
# 定义优化器为 Adam
optimizer = torch.optim.Adam(
    model.parameters(),                # 需要优化的模型参数
    lr=learning_rate,                  # 学习率
    betas=(0.9, 0.999),                # beta_1 和 beta_2
    eps=1e-7,                          # 防止除零错误的平滑项，默认值是 1e-8
    weight_decay=0                     # 权重衰减，通常用于 L2 正则化，默认值是 0
)
# 定义损失函数为交叉熵损失
criterion = torch.nn.CrossEntropyLoss()


# # 定义模型
# model = cnn_model.CnnC2F2(in_channels=14, num_classes=4).to(device)
# model_name = model.module_name
# # print(model)  # 打印网络结构
# # 定义损失函数为交叉熵损失
# criterion = torch.nn.CrossEntropyLoss()
# # 定义优化器为随机梯度下降，学习率为0.01，动量为0.9
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# # 定义模型
# model = cnn_model.CNNECG().to(device)
# model_name = model.module_name
# # 定义优化器为 Adam
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
# # 定义损失函数
# criterion = torch.nn.CrossEntropyLoss()


# # Initialize the model
# model = cnn_model.AdversarialCNN_DeepConvNet(in_channels=14, num_classes=3, sequence_length=256, learning_rate=0.001)
# model.to(device)
# model_name = model.module_name
# optimizer = model.optimizer
# criterion = model.criterion



print("model initialization complete")

# 初始化列表以存储每个 epoch 的 loss 和 accuracy
train_losses = []
train_accuracies = []

# 训练模型
print("start training...")
for epoch in range(1, epochs+1):
    running_loss = 0.0  # 初始化损失值
    running_corrects = 0  # 初始化正确预测数
    total = 0  # 初始化总数
    for i, data in enumerate(train_loader, 0):  # 遍历训练数据集
        inputs, labels = data  # 获取输入数据和标签
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到 GPU 或 CPI

        optimizer.zero_grad()  # 将梯度缓存清零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        # 梯度剪裁, 以确保梯度的范数不会超过 max_norm. 可以防止梯度爆炸，使训练过程更加稳定
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # 更新参数
        running_loss += loss.item()  # 累积损失
        
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)  # 更新样本总数
        running_corrects += (predicted == labels).sum().item()  # 更新正确预测数

    accuracy = running_corrects / total  # 计算准确率
    train_losses.append(running_loss / len(train_loader))  # 记录平均损失
    train_accuracies.append(accuracy)  # 记录准确率
    print(f"[{epoch}/{epochs}] loss: {running_loss / len(train_loader):.4f}, accuracy: {100* accuracy:.2f}%")  # 打印平均损失和准确率
print('Finished Training')  # 训练完成



# 确保文件夹存在
if not os.path.exists('models_save'):
    os.makedirs('models_save')
# 保存模型状态
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_loss,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
}
torch.save(checkpoint, "models_save/" + model_name + "_model_checkpoint.pth")


plot_tools.plot_training_metrics(train_losses=train_losses, train_accuracies=train_accuracies, is_save=True, save_name=model_name + "_training_metrics")


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

