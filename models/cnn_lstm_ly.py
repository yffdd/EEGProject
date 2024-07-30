""""
File: cnn_lstm_ly.py
Author: xiales
Date: 2024-07-30
Description: 由李颖的 Tensorflow 框架 CNN-LSTM 模型修改为 PyTorch 框架
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 加载数据
filename = "E:/Databases/OutData/DEAP/ACSE/deap.npz"
filedata = np.load(filename)
data = filedata['X']
labels = filedata['y']

# 数据转换为PyTorch张量
data = torch.tensor(data, dtype=torch.float32).to(device)
labels = torch.tensor(labels, dtype=torch.long).to(device)

# 将数据集拆分为训练集和测试集
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# 创建DataLoader
batch_size = 64
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 辅助函数来计算展平后的特征数量
def calculate_flattened_size(model, input_size):
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_size).to(device)
        output = model.forward_conv(dummy_input)
    return output.shape[1] * output.shape[2]

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv1d(in_channels=data.shape[1], out_channels=128, kernel_size=50, stride=3, padding=25)
        self.BachNorm1 = nn.BatchNorm1d(128)
        self.MaxPooling1 = nn.MaxPool1d(kernel_size=2, stride=3)

        self.layer2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.BachNorm2 = nn.BatchNorm1d(32)
        self.MaxPooling2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, stride=1, padding=5)
        self.BachNorm3 = nn.BatchNorm1d(32)

        self.layer4 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.BachNorm4 = nn.BatchNorm1d(128)
        self.MaxPooling3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer5 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.BachNorm5 = nn.BatchNorm1d(512)

        self.layer6 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.BachNorm6 = nn.BatchNorm1d(128)

        self.flat = nn.Flatten()
        self.dense_input_size = self._get_conv_output(data.shape[1], data.shape[2])
        self.dense = nn.Linear(self.dense_input_size, 512)
        self.Dropout = nn.Dropout(0.1)
        self.outputSoftmax = nn.Linear(512, 4)  # 假设有 4 个类别

    def _get_conv_output(self, shape1, shape2):
        with torch.no_grad():
            dummy_input = torch.zeros(1, shape1, shape2).to(device)
            output = self.forward_conv(dummy_input)
        return output.shape[1] * output.shape[2]

    def forward_conv(self, x):
        x = F.relu(self.layer1(x))
        x = self.BachNorm1(x)
        x = self.MaxPooling1(x)

        x = F.relu(self.layer2(x))
        x = self.BachNorm2(x)
        x = self.MaxPooling2(x)

        x = F.relu(self.layer3(x))
        x = self.BachNorm3(x)

        x = F.relu(self.layer4(x))
        x = self.BachNorm4(x)
        x = self.MaxPooling3(x)

        x = F.relu(self.layer5(x))
        x = self.BachNorm5(x)

        x = F.relu(self.layer6(x))
        x = self.BachNorm6(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.flat(x)
        x = F.relu(self.dense(x))
        x = self.Dropout(x)
        output = F.softmax(self.outputSoftmax(x), dim=1)
        return output
    
# 创建模型、定义损失函数和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# 在测试集上评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')
