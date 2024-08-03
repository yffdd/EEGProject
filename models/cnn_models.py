""""
File: cnn_models.py
Author: xiales
Date: 2024-07-31
Description: CNN 模型

data shape: (batch_size, num_channels, num_samples)
labes shape: (batch_size,)
output shape: (batch_size, num_classes)
    batch_size: 批次大小，表示每次处理的样本数量。
    num_channels: EEG信号的通道数，通常由电极数量决定。
    num_samples: 每个通道的样本数，通常与时间点数量相关，取决于信号的采样频率和时间窗口长度。

"""


import torch
import torch.nn as nn


class CnnC2F2(nn.Module):
    """
    input: (batch_size, num_channels, num_samples)
    output: (batch_size, num_classes)


    Example Usage:
    # Initialize the model
    data_iter = iter(train_loader)              # Create an iterator for the train_loader to get data in batches
    inputs, labels = next(data_iter)            # Get one batch of data from the iterator (inputs and labels)
    model = cnn_models.CnnC2F2(
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

    def __init__(self, batch_size=64, num_channels=14, num_samples=256, num_classes=4, learning_rate=0.001, model_name='CnnC6F2'):
        """
        Args:
        - batch_size (int): Size of each batch of data.
        - num_channels (int): Number of input channels (e.g., EEG channels).
        - num_samples (int): Number of samples per channel (e.g., length of the time series).
        - num_classes (int): Number of output classes (e.g., number of classes for classification).
        - learning_rate (float): Learning rate for the optimizer.
        - model_name (str): Name of the model.
        """
        super(CnnC2F2, self).__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_name = model_name

        # 定义第一个卷积层，输入通道为14，输出通道为32，卷积核大小为3x3
        # 输入: [batch_size, 14, 256]
        # 输出: [batch_size, 32, 254]
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, stride=1)
        
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
        self.fc2 = nn.Linear(128, self.num_classes)

        # Initialize optimizer and loss function
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()

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

    input: (batch_size, num_channels, num_samples)
    output: (batch_size, num_classes)

    Example Usage:
    # Initialize the model
    data_iter = iter(train_loader)              # Create an iterator for the train_loader to get data in batches
    inputs, labels = next(data_iter)            # Get one batch of data from the iterator (inputs and labels)
    model = cnn_models.CnnC6F2(
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
    def __init__(self, batch_size=64, num_channels=14, num_samples=256, num_classes=4, learning_rate=0.001, model_name='CnnC6F2'):
        """
        Args:
        - batch_size (int): Size of each batch of data.
        - num_channels (int): Number of input channels (e.g., EEG channels).
        - num_samples (int): Number of samples per channel (e.g., length of the time series).
        - num_classes (int): Number of output classes (e.g., number of classes for classification).
        - learning_rate (float): Learning rate for the optimizer.
        - model_name (str): Name of the model.
        """
        super(CnnC6F2, self).__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_name = model_name

        # 要保持输出长度与输入长度相同，可以使用以下公式计算填充量：Padding= ((Kernel_Size-1) - (Stride-1)) / 2
        self.conv1 = nn.Conv1d(in_channels=self.num_channels, out_channels=128, kernel_size=50, stride=3, padding=24)  # Adjust padding
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
        

        # num_samples = 128 -> some_size = 2
        # num_samples = 256 -> some_size = 4
        # num_samples = 50 -> some_size = 7
        # num_samples = 512 -> some_size = 7
        # num_samples = 1000 -> some_size = 14
        # num_samples = 1024 -> some_size = 14
        # num_samples = 2000 -> some_size = 28
        # num_samples = 2048 -> some_size = 29
        # 定义 num_samples 和 some_size 之间的映射
        if num_samples == 128:
            some_size = 2
        elif num_samples == 256:
            some_size = 4
        elif num_samples == 50:
            some_size = 7
        elif num_samples == 512:
            some_size = 7
        elif num_samples == 1000:
            some_size = 14
        elif num_samples == 1024:
            some_size = 14
        elif num_samples == 2000:
            some_size = 28
        elif num_samples == 2048:
            some_size = 29
        else:
            some_size = 2  # 默认值
        self.fc1 = nn.Linear(128 * some_size, 512)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, self.num_classes)

        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.parameters(),              # 需要优化的模型参数
            lr=self.learning_rate,          # 学习率
            betas=(0.9, 0.999),             # beta_1 和 beta_2
            eps=1e-7,                       # 防止除零错误的平滑项，默认值是 1e-8
            weight_decay=0                  # 权重衰减，通常用于 L2 正则化，默认值是 0
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        
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

    input: (batch_size, num_channels, num_samples)
    output: (batch_size, num_classes)
    
    Example Usage:
    # Initialize the model
    data_iter = iter(train_loader)              # Create an iterator for the train_loader to get data in batches
    inputs, labels = next(data_iter)            # Get one batch of data from the iterator (inputs and labels)
    model = cnn_models.CNN_LSTM(
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

    def __init__(self, batch_size=64, num_channels=14, num_samples=256, num_classes=4, learning_rate=0.001, hidden_size=256, num_layers=3, model_name='CNN_LSTM'):
        """
        Args:
        - batch_size (int): Size of each batch of data.
        - num_channels (int): Number of input channels (e.g., EEG channels).
        - num_samples (int): Number of samples per channel (e.g., length of the time series).
        - num_classes (int): Number of output classes (e.g., number of classes for classification).
        - learning_rate (float): Learning rate for the optimizer.
        - model_name (str): Name of the model.
        """
        super(CNN_LSTM, self).__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = model_name

        # CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(self.num_channels, 64, kernel_size=5, stride=1, padding=2),
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

        # Initialize optimizer and loss function
        self.module_name = model_name
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        self.criterion = torch.nn.CrossEntropyLoss()

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
    """
    Reference: Peng Junwen

    input: (batch_size, num_channels, num_samples)
    output: (batch_size, num_classes)

    Example Usage:
    # Initialize the model
    data_iter = iter(train_loader)              # Create an iterator for the train_loader to get data in batches
    inputs, labels = next(data_iter)            # Get one batch of data from the iterator (inputs and labels)
    model = cnn_models.CNNECG(
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
    def __init__(self, batch_size=64, num_channels=14, num_samples=256, num_classes=4, learning_rate=0.001, model_name='CNNECG'):
        """
        Args:
        - batch_size (int): Size of each batch of data.
        - num_channels (int): Number of input channels (e.g., EEG channels).
        - num_samples (int): Number of samples per channel (e.g., length of the time series).
        - num_classes (int): Number of output classes (e.g., number of classes for classification).
        - learning_rate (float): Learning rate for the optimizer.
        - model_name (str): Name of the model.
        """
        super(CNNECG, self).__init__()
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_name = model_name
        
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
            nn.Dropout(),  # 默认p值为 0.5
            nn.Linear(8192, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
            nn.Sigmoid()
        )

        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # x = x.unsqueeze(1)  # 添加通道维度
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    


class AdversarialCNN_DeepConvNet(nn.Module):
    """
    Ref: https://github.com/philipph77/acse-framework
    Paper: Exploiting Multiple EEG Data Domains with Adversarial Learning

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
        super(AdversarialCNN_DeepConvNet, self).__init__()
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
    

