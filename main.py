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
from tools import fetch_tools, plot_tools, train_tools, anotc_fetch_eeg_emotion

# 设置系统的Dataset根目录
sys_dataset_root_dir = 'E:/'  # xiales-pc Windows系统路径
# sys_dataset_root_dir = '/bigdisk/322xcq/'  # 服务器系统路径


# 检查是否有可用的 GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')
# 设置超参数
epochs = 30
batch_size = 24
learning_rate = 0.0001

# 数据提取
train_loader, val_loader, test_loader = fetch_tools.seed_loader_fetch(batch_size=batch_size, is_print=True)

# save_path = "E:/Databases/OutData/EEG_Emotion/not_preprocess"
# train_loader, val_loader, test_loader = anotc_fetch_eeg_emotion.fetch_data_loader(data_path=save_path, batch_size=32, is_print=True)

# Initialize the model
print("model initialization...")
data_iter = iter(train_loader)              # Create an iterator for the train_loader to get data in batches
inputs, labels = next(data_iter)            # Get one batch of data from the iterator (inputs and labels)
model = cnn_models.ConvNet(
    batch_size=train_loader.batch_size,     # Set the batch size from the train_loader
    num_channels=inputs.shape[1],           # Set the number of channels from the input shape
    num_samples=inputs.shape[2],            # Set the number of samples from the input shape
    num_classes=len(labels.unique()),       # Set the number of classes by counting unique labels
    learning_rate=learning_rate             # Set the learning rate
)
model.to(device)                            # Move model to the specified device (CPU/GPU)
optimizer = model.optimizer                 # Get the optimizer defined within the model
criterion = model.criterion                 # Get the loss function defined within the model
print("model initialization complete")
scheduler = model.scheduler                 # Get the learning rate scheduler defined within the model


# training model
checkpoint = train_tools.train_model(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=epochs,
    device=device,
    clip_grad=False,
    use_val=True,
    save_model=True,
    use_early_stopping=False,
    patience=10,
    metric='val_loss',
    scheduler=scheduler
)

# # 绘制准确率与损失函数图像
# plot_tools.plot_training_metrics(train_losses=checkpoint["train_losses"], train_accuracies=checkpoint["train_accuracies"], is_save=True, save_name=model.model_name + "_training_metrics")

# 测试模型
train_tools.test_model(checkpoint=checkpoint, model=model, criterion=criterion, test_loader=test_loader, device=device)

