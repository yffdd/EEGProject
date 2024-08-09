# coding:UTF-8
'''
对原始的 eeg 信号，使用 CNN 进行情感分类。
Created by Xiao Guowen.
'''
from seed_tools import build_preprocessed_eeg_dataset_CNN, RawEEGDataset, subject_independent_data_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

# 设置系统的Dataset根目录
# sys_dataset_root_dir = 'E:/'  # xiales-pc Windows系统路径
sys_dataset_root_dir = '/bigdisk/322xcq/'  # 学校服务器系统路径

# 加载数据，整理成所需要的格式
folder_path = os.path.join(sys_dataset_root_dir, "Databases/RawData/SEED/SEED/Preprocessed_EEG/")
print("folder_path: ", folder_path)
feature_vector_dict, label_dict = build_preprocessed_eeg_dataset_CNN(folder_path)
train_feature, train_label, test_feature, test_label = subject_independent_data_split(feature_vector_dict, label_dict,
                                                                                      {'2', '6', '9'})

print(f"features: {np.array(train_feature).shape}")
print(f"label: {np.array(train_label).shape}")

desire_shape = [1, 62, 200]
train_data = RawEEGDataset(train_feature, train_label, desire_shape)
test_data = RawEEGDataset(test_feature, test_label, desire_shape)



# 超参数设置
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
num_epochs = 30
num_classes = 3
batch_size = 24
learning_rate = 0.0001

# Data loader
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)


# 获取第一个批次的数据
for features, labels in train_data_loader:
    print(f"Feature batch shape: {features.shape}")
    print(f"Label batch shape: {labels.shape}")
    break  # 只查看第一个批次的数据

# 定义卷积网络结构
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(128 * 7 * 25, 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)


# Train the model
def train():
    # 初始化列表以存储每个 epoch 的 loss 和 accuracy
    train_losses = []
    train_accuracies = []
    
    # writer = SummaryWriter('../log')
    total_step = len(train_data_loader)
    batch_cnt = 0
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化运行损失
        correct_predictions = 0  # 初始化正确预测的计数
        total_samples = 0  # 初始化样本总数
        for i, (features, labels) in enumerate(train_data_loader):
            features = features.to(device)
            labels = labels.to(device)
            print(f"features.shape: {features.shape}")
            print(f"labels.shape: {labels.shape}")
            outputs = model(features)
            print(f"outputs.shape: {outputs.shape}")
            loss = criterion(outputs, labels)
            batch_cnt += 1
            # writer.add_scalar('train_loss', loss, batch_cnt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 累加损失
            _, preds  = torch.max(outputs, 1)  # 获取预测结果
            correct_predictions += torch.sum((preds  == labels)).item()  # 计算正确预测的数量
            total_samples += labels.size(0)  # 更新样本总数

            if (i + 1) % 500 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct_predictions / total_samples)*100))
        scheduler.step()
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_data_loader)
        epoch_accuracy = correct_predictions / total_samples  # 计算准确率

        train_losses.append(epoch_loss)  # 记录平均损失
        train_accuracies.append(epoch_accuracy)  # 记录准确率
        print(f"Epoch [{epoch}/{num_epochs}], train accuracy: {100*epoch_accuracy:.2f}%, train loss: {epoch_loss:.4f}")  # 打印准确率和平均损失
        test()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
    }
    save_model_path = "../models_save/raw_eeg_model_checkpoint.pth"  # 保存模型的路径
    torch.save(checkpoint, save_model_path)

    # torch.save(model.state_dict(), '../model/model.ckpt')


# Test the model
def test(is_load=False):
    if is_load:
        save_model_path = "../models_save/raw_eeg_model_checkpoint.pth"  # 保存模型的路径
        checkpoint = torch.load(save_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # model.load_state_dict(torch.load('../model/model.ckpt'))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_data_loader:
            features = features.to(device)
            labels = labels.to(device)
            output = model(features)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy is {}%'.format(100 * correct / total))


train()
test(is_load=False)
