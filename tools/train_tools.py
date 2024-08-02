"""
File: train_tools.py
Author: xiales
Date: 2024-08-02
Description: 用于不同训练方式的脚本函数

"""


import os
import torch


def training_model(model, train_loader, device, epochs=100, clip_grad=False, save_model=True):
    """
    训练模型并在每个 epoch 后记录损失和准确率。

    参数:
    - model (nn.Module): 要训练的模型。
    - train_loader (DataLoader): 训练数据加载器。
    - device (torch.device): 模型和数据所在的设备(CPU 或 GPU)。
    - epochs (int): 训练的 epoch 数量。
    - clip_grad (bool): 是否应用梯度剪裁。
    - save_model (bool): 是否保存训练后的模型。

    返回:
    - checkpoint (dict): 包含模型状态、优化器状态等信息的字典。
    """
    # 初始化列表以存储每个 epoch 的 loss 和 accuracy
    train_losses = []
    train_accuracies = []
    print("training begins...")
    
    for epoch in range(1, epochs+1):
        running_loss = 0.0  # 初始化损失值
        running_corrects = 0  # 初始化正确预测数
        total = 0  # 初始化总数
        model.train()  # 设置模型为训练模式
        for i, data in enumerate(train_loader, 0):  # 遍历训练数据集
            inputs, labels = data  # 获取输入数据和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到 GPU 或 CPU

            model.optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            loss = model.criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            if clip_grad:
                # 梯度剪裁, 以确保梯度的范数不会超过 max_norm. 可以防止梯度爆炸，使训练过程更加稳定
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.optimizer.step()  # 更新模型参数
            running_loss += loss.item()  # 累积损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 更新样本总数
            running_corrects += (predicted == labels).sum().item()  # 更新正确预测数

        # 计算平均损失和准确率
        accuracy = running_corrects / total  # 计算准确率
        train_losses.append(running_loss / len(train_loader))  # 记录平均损失
        train_accuracies.append(accuracy)  # 记录准确率
        print(f"[{epoch}/{epochs}] loss: {running_loss / len(train_loader):.4f}, accuracy: {100* accuracy:.2f}%")  # 打印平均损失和准确率
    print('training completed')  # 训练完成

    # 保存模型状态
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'loss': running_loss,
        'learning_rate': model.learning_rate,
        'batch_size': model.batch_size,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
    }

    # 保存模型
    if save_model:
        # 确保文件夹存在
        if not os.path.exists('models_save'):
            os.makedirs('models_save')
        save_model_path = "models_save/" + model.model_name + "_model_checkpoint.pth"
        try:
            torch.save(checkpoint, save_model_path)
            print(f"Model saved to: {save_model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
        print(f"save model to: {save_model_path}")

    return checkpoint

def test_model(model, test_loader, device):
    """
    测试模型在测试数据集上的准确率。

    参数:
    - model (nn.Module): 要测试的模型。
    - test_loader (DataLoader): 测试数据加载器。
    - device (torch.device): 模型和数据所在的设备（CPU 或 GPU）。

    返回:
    - None
    """
    correct = 0  # 初始化正确预测数
    total = 0  # 初始化总数
    model.eval()  # 设置模型为评估模式（禁用 dropout 和 batch normalization）

    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in test_loader:  # 遍历测试数据集
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到 GPU

            outputs = model(inputs)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 更新总数
            correct += (predicted == labels).sum().item()  # 更新正确预测数

    # 处理总数为 0 的情况，避免除以 0
    if total > 0:
        accuracy = 100 * correct / total
        print(f'Accuracy of the modelwork on the test data: {accuracy:.2f}%')
    else:
        print('The test dataset is empty and accuracy cannot be calculated')




