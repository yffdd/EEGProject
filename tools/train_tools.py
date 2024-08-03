"""
File: train_tools.py
Author: xiales
Date: 2024-08-02
Description: 用于不同训练方式的脚本函数

"""



import sys
import os
# 将当前脚本的上一级目录添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的路径
relative_path = os.path.abspath(os.path.join(current_dir, ".."))  # 拼接上一级路径 (工程路径)
sys.path.append(relative_path)  # 将工程路径添加到系统路径中

import torch
from models import cnn_models
from tools import data_fetch_tools, plot_tools


def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, device, clip_grad=False, use_val=True, save_model=True, use_early_stopping=True, patience=5, metric='val_loss'):
    """
    训练模型并在每个 epoch 后记录损失和准确率。

    参数:
    - model (nn.Module): 要训练的模型。
    - optimizer (torch.optim.Optimizer): 优化器。
    - criterion (nn.Module): 损失函数。
    - train_loader (DataLoader): 训练数据加载器。
    - val_loader (DataLoader): 验证数据加载器。
    - epochs (int): 训练的 epoch 数量。
    - device (torch.device): 模型和数据所在的设备(CPU 或 GPU)。
    - clip_grad (bool): 是否应用梯度剪裁。默认值为 False。
    - use_val (bool): 是否在每个 epoch 后使用验证集进行评估。默认值为 True。
    - save_model (bool): 是否在训练完成后保存模型。默认值为 True。
    - use_early_stopping (bool): 是否使用早停机制。默认值为 True。
    - patience (int): 早停的耐心值，默认值为 5。
    - metric (str): 用于早停的指标，默认值为 'val_loss'。可选值为 'val_loss' 或 'val_acc'。

    返回:
    - checkpoint (dict): 包含模型状态、优化器状态和训练参数等信息的字典。
    """
    print("training begins...")  # 开始训练
    # 初始化列表以存储每个 epoch 的 loss 和 accuracy
    train_losses = []
    train_accuracies = []
    # 保存模型的路径
    if save_model:
        # 确保保存模型的文件夹存在
        if not os.path.exists('models_save'):
            os.makedirs('models_save')
    save_model_path = "models_save/" + model.model_name + "_model_checkpoint.pth"  # 保存模型的路径
    # 初始化早停计数器
    best_metric = float('inf') if metric == 'val_loss' else 0.0
    epochs_no_improve = 0
    checkpoint = {}
    
    for epoch in range(1, epochs+1):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化运行损失
        correct_predictions = 0  # 初始化正确预测的计数
        total_samples = 0  # 初始化样本总数
        for inputs, labels in train_loader:  # 获取输入数据和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到 GPU 或 CPU
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            if clip_grad:
                # 梯度剪裁, 以确保梯度的范数不会超过 max_norm. 可以防止梯度爆炸，使训练过程更加稳定
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # 更新模型参数
            running_loss += loss.item()  # 累加损失
            _, preds  = torch.max(outputs, 1)  # 获取预测结果
            correct_predictions += torch.sum((preds  == labels)).item()  # 计算正确预测的数量
            total_samples += labels.size(0)  # 更新样本总数

        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        accuracy = correct_predictions / total_samples  # 计算准确率
        train_losses.append(epoch_loss)  # 记录平均损失
        train_accuracies.append(epoch_accuracy)  # 记录准确率

        if use_val:
            # 评估模型在验证集上的性能
            val_loss, val_accuracy = evaluate_model(model=model, criterion=criterion, val_loader=val_loader, device=device)
            print(f"Epoch [{epoch}/{epochs}], train accuracy: {100*accuracy:.2f}%, train loss: {epoch_loss:.4f}, val accuracy: {100*val_accuracy:.2f}%, val loss: {val_loss:.4f}")

            if use_early_stopping:
                # 检查验证集损失是否有所改善
                if metric == 'val_loss' and val_loss < best_metric:
                    best_metric = val_loss
                    epochs_no_improve = 0
                    # 保存最佳模型的检查点
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
                        torch.save(checkpoint, save_model_path)
                elif metric == 'val_accuracy' and val_accuracy > best_metric:
                    best_metric = val_accuracy
                    epochs_no_improve = 0
                    # 保存最佳模型的检查点
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
                        torch.save(checkpoint, save_model_path)
                else:
                    epochs_no_improve += 1
                    # 如果验证损失未改善超过耐心值，停止训练
                    if epochs_no_improve >= patience:
                        print('Early stopping!')
                        break
        else:
            print(f"Epoch [{epoch}/{epochs}], train accuracy: {100*accuracy:.2f}%, train loss: {epoch_loss:.4f}")  # 打印准确率和平均损失
    
    # 如果没有使用早停机制保存模型, 则在训练完成后保存模型
    if not use_early_stopping:
    # 保存模型的检查点
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
            torch.save(checkpoint, save_model_path)

    print('training completed')  # 训练完成
    if save_model:
        print(f"model saved to: {save_model_path}")
    return checkpoint


def evaluate_model(model, criterion, val_loader, device):
    """
    评估模型在验证集上的性能。

    参数:
    - model (nn.Module): 要评估的模型。
    - criterion (nn.Module): 损失函数。
    - val_loader (DataLoader): 验证数据加载器。
    - device (torch.device): 模型和数据所在的设备 (CPU 或 GPU)。

    返回:
    - val_loss (float): 验证集上的平均损失。
    - val_accuracy (float): 验证集上的准确率。
    """
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0  # 初始化运行损失
    correct_predictions = 0  # 初始化正确预测的计数
    total_samples = 0  # 初始化样本总数
    
    # 禁用梯度计算，节省内存和计算资源
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            running_loss += loss.item()  # 累加损失
            
            _, preds = torch.max(outputs, 1)  # 获取预测结果
            correct_predictions += torch.sum(preds == labels).item()  # 计算正确预测的数量
            total_samples += labels.size(0)  # 累加样本总数
    
    # 计算平均损失和准确率
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct_predictions / total_samples
    
    return val_loss, val_accuracy


def test_model(checkpoint, model, criterion, test_loader, device):
    """
    测试模型在测试数据集上的准确率。

    参数:
    - checkpoint (dict): 训练好的模型的检查点，包括模型状态和优化器状态。
    - model (nn.Module): 要测试的模型。
    - criterion (nn.Module): 损失函数。
    - test_loader (DataLoader): 测试数据加载器。
    - device (torch.device): 模型和数据所在的设备(CPU 或 GPU)。

    返回:
    - None
    """
    # 将模型的状态从 checkpoint 加载到模型中
    # checkpoint = torch.load("models_save/" + model.model_name + "_model_checkpoint.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置模型为评估模式 (禁用 dropout 和 batch normalization)
    running_loss = 0.0  # 初始化累计损失
    correct_predictions = 0  # 初始化正确预测的计数
    total_samples = 0  # 初始化样本总数
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in test_loader:  # 遍历测试数据集
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到 GPU
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            running_loss += loss.item()  # 累加损失
            _, preds = torch.max(outputs, 1)  # 获取预测结果
            correct_predictions += torch.sum((preds  == labels)).item()  # 计算正确预测的数量
            total_samples += labels.size(0)  # 更新总数

    # 计算平均损失和准确率
    test_loss = running_loss / len(test_loader)
    test_accuracy = correct_predictions / total_samples
    print(f"test accuracy: {test_accuracy:.4f}, test loss: {test_loss:.4f}")


def train_model_simple(model, optimizer, criterion, train_loader, val_loader, epochs, device, clip_grad=False, use_val=True, save_model=True):
    """
    训练模型并在每个 epoch 后记录损失和准确率。

    参数:
    - model (nn.Module): 要训练的模型。
    - optimizer (torch.optim.Optimizer): 优化器。
    - criterion (nn.Module): 损失函数。
    - train_loader (DataLoader): 训练数据加载器。
    - val_loader (DataLoader): 验证数据加载器。
    - epochs (int): 训练的 epoch 数量。
    - device (torch.device): 模型和数据所在的设备(CPU 或 GPU)。
    - clip_grad (bool): 是否应用梯度剪裁。默认值为 False。
    - use_val (bool): 是否在每个 epoch 后使用验证集进行评估。默认值为 True。
    - save_model (bool): 是否在训练完成后保存模型。默认值为 True。

    返回:
    - checkpoint (dict): 包含模型状态、优化器状态和训练参数等信息的字典。
    """
    print("training begins...")
    # 初始化列表以存储每个 epoch 的 loss 和 accuracy
    train_losses = []
    train_accuracies = []
    
    for epoch in range(1, epochs+1):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化运行损失
        correct_predictions = 0  # 初始化正确预测的计数
        total_samples = 0  # 初始化样本总数
        for inputs, labels in train_loader:  # 获取输入数据和标签
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到 GPU 或 CPU
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            if clip_grad:
                # 梯度剪裁, 以确保梯度的范数不会超过 max_norm. 可以防止梯度爆炸，使训练过程更加稳定
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # 更新模型参数
            running_loss += loss.item()  # 累加损失
            _, preds  = torch.max(outputs, 1)  # 获取预测结果
            correct_predictions += torch.sum((preds  == labels)).item()  # 计算正确预测的数量
            total_samples += labels.size(0)  # 更新样本总数

        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        accuracy = correct_predictions / total_samples  # 计算准确率
        train_losses.append(epoch_loss)  # 记录平均损失
        train_accuracies.append(epoch_accuracy)  # 记录准确率

        if use_val:
            # 评估模型在验证集上的性能
            val_loss, val_accuracy = evaluate_model(model=model, criterion=criterion, val_loader=val_loader, device=device)
            print(f"Epoch [{epoch}/{epochs}], train accuracy: {100*accuracy:.2f}%, train loss: {epoch_loss:.4f}, val accuracy: {100*val_accuracy:.2f}%, val loss: {val_loss:.4f}")
        else:
            print(f"Epoch [{epoch}/{epochs}], train accuracy: {100*accuracy:.2f}%, train loss: {epoch_loss:.4f}")  # 打印准确率和平均损失

    # 保存模型的检查点
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
            print(f"model saved to: {save_model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    print('training completed')  # 训练完成
    print(f"model saved to: {save_model_path}")
    return checkpoint


if __name__  == '__main__':

    # 检查是否有可用的 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using device: {device}')
    epochs = 100
    batch_size = 128
    learning_rate = 0.001

    # 数据提取
    train_loader, test_loader, val_loader = data_fetch_tools.deap_loader_fetch(batch_size=batch_size)

    # Initialize the model
    print("model initialization...")
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
    print("model initialization complete")

    # 测试模型
    checkpoint = torch.load("models_save/" + model.model_name + "_model_checkpoint.pth")
    test_model(checkpoint=checkpoint, model=model, test_loader=test_loader, device=device)


