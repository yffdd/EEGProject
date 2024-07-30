"""
File: deap_eeg_emotion_classification.py
Author: xiales
Date: 2024-07-30
Description: This script implements a convolutional neural network (CNN) for emotion classification using the DEAP EEG dataset.
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

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
epochs = 10
learning_rate = 0.001

# 加载数据集
dataset_name = "deap"
databases_out_directory = r"E:/Databases/OutData/DEAP/ACSE/"
filename = databases_out_directory + dataset_name + ".npz"
data = np.load(filename)
X = data['X']  # 形状: (38400, 14, 256)
y = data['y']  # 形状: (38400,) 标签: (0, 1, 2, 3)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y unique: {np.unique(y)}")

# 将 NumPy 数组转换为 PyTorch 张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 使用 train_test_split 将数据集拆分为训练集和测试集和验证集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
val_dataset = TensorDataset(X_val, y_val)
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
valloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# # 打印训练集的大小和形状
# print(f'Training set size: {len(trainset)}')
# print(f'Training set data shape: {trainset.data.shape}')
# print(f'Training set labels shape: {trainset.targets.shape}')
# # 打印训练集的标签唯一值
# train_labels = trainset.targets
# print(f'Unique labels in training set: {train_labels.unique()}')


# 定义卷积神经网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义第一个卷积层，输入通道为14，输出通道为32，卷积核大小为3x3
        # 输入: [batch_size, 14, 256]
        # 输出: [batch_size, 32, 254]
        self.conv1 = nn.Conv1d(14, 32, kernel_size=3, stride=1)
        
        # 定义第一个最大池化层，池化窗口大小为2
        # 输入: [batch_size, 32, 254]
        # 输出: [batch_size, 32, 127]
        self.pool1 = nn.MaxPool1d(2)
        
        # 定义第二个卷积层，输入通道为32，输出通道为64，卷积核大小为3x3
        # 输入: [batch_size, 32, 127]
        # 输出: [batch_size, 64, 125]
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        
        # 定义第二个最大池化层，池化窗口大小为2
        # 输入: [batch_size, 64, 125]
        # 输出: [batch_size, 64, 62]
        self.pool2 = nn.MaxPool1d(2)
        
        # 定义 dropout 层，丢弃概率为0.25
        self.dropout1 = nn.Dropout(0.25)
        
        # 定义第一个全连接层，将输入特征数 64*62 转换为128
        # 输入: [batch_size, 64*62]
        # 输出: [batch_size, 128]
        self.fc1 = nn.Linear(64 * 62, 128)
        
        # 定义 dropout 层，丢弃概率为0.5
        self.dropout2 = nn.Dropout(0.5)
        
        # 定义第二个全连接层，将输入特征数 128 转换为4（4个情绪类别）
        # 输入: [batch_size, 128]
        # 输出: [batch_size, 4]
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)  # 第一个卷积层
        x = torch.relu(x)  # ReLU 激活函数
        x = self.pool1(x)  # 第一个最大池化层
        x = self.conv2(x)  # 第二个卷积层
        x = torch.relu(x)  # ReLU 激活函数
        x = self.pool2(x)  # 第二个最大池化层
        x = self.dropout1(x)  # Dropout 层
        x = x.view(x.size(0), -1)  # 将特征展平成一维向量
        x = self.fc1(x)  # 第一个全连接层
        x = torch.relu(x)  # ReLU 激活函数
        x = self.dropout2(x)  # Dropout 层
        x = self.fc2(x)  # 第二个全连接层
        return x

model = Model().to(device)  # 实例化网络并移到 GPU
# print(model)  # 打印网络结构

# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()

# 定义优化器为随机梯度下降，学习率为0.01，动量为0.9
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# 初始化列表以存储每个 epoch 的 loss 和 accuracy
train_losses = []
train_accuracies = []

# 训练模型
for epoch in range(epochs):  # 训练10个epoch
    running_loss = 0.0  # 初始化损失值
    running_corrects = 0  # 初始化正确预测数
    total = 0  # 初始化总数
    for i, data in enumerate(trainloader, 0):  # 遍历训练数据集
        inputs, labels = data  # 获取输入数据和标签
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到 GPU

        optimizer.zero_grad()  # 将梯度缓存清零
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        # 梯度剪裁, 以确保梯度的范数不会超过 max_norm. 可以防止梯度爆炸，使训练过程更加稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # 更新参数
        running_loss += loss.item()  # 累积损失
        
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)  # 更新样本总数
        running_corrects += (predicted == labels).sum().item()  # 更新正确预测数

    accuracy = running_corrects / total  # 计算准确率
    train_losses.append(running_loss / len(trainloader))  # 记录平均损失
    train_accuracies.append(accuracy)  # 记录准确率
    print(f"[{epoch + 1}/{epochs}] loss: {running_loss / len(trainloader):.4f}, accuracy: {100* accuracy:.2f}%")  # 打印平均损失和准确率
print('Finished Training')  # 训练完成

# 确保文件夹存在
if not os.path.exists('models_save'):
    os.makedirs('models_save')
# 保存模型参数和优化器状态
savepoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_loss
}
torch.save(savepoint, 'models_save/deap_cnn_model_savepoint.pth')


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

# 确保文件夹存在
if not os.path.exists('plot_save'):
    os.makedirs('plot_save')
# 保存图像
plt.savefig('plot_save/deap_cnn_train_metrics.png')

plt.tight_layout()  # 调整布局
plt.show()

# 测试模型
correct = 0  # 初始化正确预测数
total = 0  # 初始化总数
with torch.no_grad():  # 禁用梯度计算
    for data in testloader:  # 遍历测试数据集
        images, labels = data  # 获取输入数据和标签
        images, labels = images.to(device), labels.to(device)  # 将数据移到 GPU

        outputs = model(images)  # 前向传播
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)  # 更新总数
        correct += (predicted == labels).sum().item()  # 更新正确预测数

print(f'Accuracy of the modelwork on the test data: {correct / total:.2f}')  # 打印测试集上的准确率
