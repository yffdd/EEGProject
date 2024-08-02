""""
File: cnn_model.py
Author: xiales
Date: 2024-07-31
Description: CNN 模型
"""


import torch
import torch.nn as nn


# 定义卷积神经网络模型
class CnnC2F2(nn.Module):
    """
    input: (batch_size, num_features, sequence_length)
    output: (batch_size, num_classes)

    # 定义模型
    model = cnn_model.CnnC2F2(in_channels=14, num_classes=4).to(device)
    model_name = model.module_name
    # 定义损失函数为交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    # 定义优化器为随机梯度下降，学习率为0.01，动量为0.9
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    """

    def __init__(self, in_channels=14, num_classes=4):
        super(CnnC2F2, self).__init__()
        self.module_name = "CnnC2F2"
        # 定义第一个卷积层，输入通道为14，输出通道为32，卷积核大小为3x3
        # 输入: [batch_size, 14, 256]
        # 输出: [batch_size, 32, 254]
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=1)
        
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
        self.fc2 = nn.Linear(128, num_classes)

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




class CnnC6F2(nn.Module):
    """
    Reference: CNN.py -- Li Yin

    input: (batch_size, num_features, sequence_length)
    output: (batch_size, num_classes)

    # 定义模型
    model = cnn_model.CNN(in_channels=in_channels, num_classes=num_classes).to(device)
    model_name = model.module_name
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
    """
    def __init__(self, in_channels=14, num_classes=4):
        super(CnnC6F2, self).__init__()
        self.module_name = "CnnC6F2"
        # 要保持输出长度与输入长度相同，可以使用以下公式计算填充量：Padding= ((Kernel_Size-1) - (Stride-1)) / 2
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=50, stride=3, padding=24)  # Adjust padding
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=3)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, stride=1, padding=3)  # Adjust padding
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, stride=1, padding=5)  # Adjust padding
        self.bn3 = nn.BatchNorm1d(32)
        
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding=2)  # Adjust padding
        self.bn4 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=5, stride=1, padding=2)  # Adjust padding
        self.bn5 = nn.BatchNorm1d(512)
        
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)  # Adjust padding
        self.bn6 = nn.BatchNorm1d(128)
        

        # sequence_length = 128 -> some_size = 2
        # sequence_length = 256 -> some_size = 4
        # sequence_length = 50 -> some_size = 7
        # sequence_length = 512 -> some_size = 7
        # sequence_length = 1000 -> some_size = 14
        # sequence_length = 1024 -> some_size = 14
        # sequence_length = 2000 -> some_size = 28
        # sequence_length = 2048 -> some_size = 29
        some_size = 4  # This is just an example, you need to calculate the actual size based on the input shape and the effects of all layers
        self.fc1 = nn.Linear(128 * some_size, 512)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool3(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = torch.relu(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = torch.softmax(x, dim=1)
        return output




class CNN_LSTM(nn.Module):
    """
    Reference: Peng Junwen

    input: (batch_size, num_features, sequence_length)
    output: (batch_size, num_classes)
    
    # 定义模型
    model = cnn_model.CNN_LSTM(in_channels=in_channels, num_classes=num_classes).to(device)
    model_name = model.module_name
    # 定义优化器为 Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # 定义损失函数
    criterion = nn.BCELoss()

    """


    def __init__(self, input_channels=14, hidden_size=256, num_layers=3, num_classes=4):
        super(CNN_LSTM, self).__init__()
        self.module_name = "CNN_LSTM"
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1, stride=1),
        )

        # LSTM layers
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.5)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x 的形状为 [batch_size, features]

        # # 添加时间步维度，将形状变为 [batch_size, 1, features]
        # x = x.unsqueeze(1)

        # # 转置为 [batch_size, features, 1]，以匹配Conv1d的输入要求
        # x = x.transpose(1, 2)

        # CNN feature extraction
        x = self.cnn_layers(x)

        # 为LSTM准备输入 (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # 从 [batch_size, features, time] 变为 [batch_size, time, features]

        # LSTM layers
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # 我们取最后一个时间步的输出
        out = out[:, -1, :]

        # Fully connected layers
        out = self.fc(out)

        # # Activation
        # out = self.sigmoid(out)

        return out



class CNNECG(nn.Module):
    def __init__(self):
        super(CNNECG, self).__init__()
        self.module_name = "CNNECG"
        self.features = nn.Sequential(
            nn.Conv1d(14, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8192, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.unsqueeze(1)  # 添加通道维度
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    


class AdversarialCNN(nn.Module):
    """
    Ref: https://github.com/philipph77/acse-framework
    Paper: Exploiting Multiple EEG Data Domains with Adversarial Learning

    # 定义模型
    model = cnn_model.AdversarialCNN(in_channels=14, num_classes=4, samples=128)
    model_name = model.module_name
    # 定义优化器为 Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # 定义损失函数为交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()

    """

    def __init__(self, in_channels=14, num_classes=4, samples=256, model_name='DeepConvNet'):
        super(AdversarialCNN, self).__init__()
        self.module_name = model_name
        self.conv1 = nn.Conv2d(1, 25, (1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(25, 25, (in_channels, 1), padding=0, bias=False)  # 对于 (chans, 1) 的卷积核不需要padding
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
        
        self.fc = nn.Linear(200 * ((samples // 16)), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加一个通道维度，使形状变为 (38400, 1, 14, 256)
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
    

