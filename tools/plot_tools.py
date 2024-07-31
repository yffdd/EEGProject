"""
File: plot_tools.py
Author: xiales
Date: 2024-07-30
Description: 用于绘图的工具函数
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import csv

def plot_training_metrics(train_losses, train_accuracies, title_loss='Training Loss over Epochs', title_accuracy='Training Accuracy over Epochs', is_save=False, save_name="training_metrics"):
    """
    绘制训练损失和准确率图像。

    参数:
    - train_losses: 训练损失列表。
    - train_accuracies: 训练准确率列表。
    - title_loss: 训练损失图像的标题（默认值为 'Training Loss over Epochs'）。
    - title_accuracy: 训练准确率图像的标题（默认值为 'Training Accuracy over Epochs'）。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制训练损失
    ax1.plot(train_losses, label='Training Loss', color='tab:blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(title_loss)
    ax1.legend()
    
    # 绘制训练准确率
    ax2.plot(train_accuracies, label='Training Accuracy', color='tab:red')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(title_accuracy)
    ax2.legend()
    
    if is_save:
        # 保存图像
        if not os.path.exists('plot_save'):
            os.makedirs('plot_save')
        plt.savefig('plot_save/' + save_name + ".png", dpi=300)
        # 保存训练损失和准确率到 CSV 文件
        with open('plot_save/' + save_name + ".csv", mode='w', newline='') as file:  # 打开文件以写入模式
            writer = csv.writer(file)  # 创建一个csv写入对象
            writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy'])  # 写入表头
            for epoch, (loss, accuracy) in enumerate(zip(train_losses, train_accuracies), start=1):  # 遍历每个epoch及其对应的损失和准确率
                writer.writerow([epoch, loss, accuracy])  # 写入每个epoch的损失和准确率

    plt.tight_layout()  # 调整布局
    plt.show()


def plot(data_array, start_index=0, length=0, chx=0, fs=500):
    """
    绘制二维数据数组中的数据。 图例在图形内部，当同一张图片中图例较多时显示效果较好

    :param data_array: 输入的二维numpy数组，其中每一行代表一个数据集。
    :param start_index: 每个数据集开始绘制的索引，默认从0开始。
    :param length: 从start_index开始要绘制的长度，如果为0则绘制从start_index到数据集末尾的所有数据。
    :param chx: 如果为0，则绘制data_array中的所有数据集。如果不为0，则只绘制第chx个数据集。

    这个函数创建一个图表，并可以选择性地绘制所有数据集或一个特定的数据集。
    可以指定绘制数据的起始索引和长度。如果没有提供长度（或长度为0），
    将从起始索引绘制到数据集的末尾。
    """

    # 确定绘图的数量
    num_datasets = data_array.shape[0]

    # 创建图形和轴对象
    plt.figure(figsize=(10, 6))  # 可以调整图形的大小

    if chx == 0:
        # 绘制每一列数据
        for i in range(num_datasets):
            if length > 0 and length <= (data_array.shape[1] - start_index):
                data_to_plot = data_array[i, start_index:length]
            else:
                data_to_plot = data_array[i, :]

            # 纵坐标变换
            data_to_plot = ((data_to_plot/((2<<23)-1)) / 24) * 4.5
            data_to_plot = np.array(data_to_plot) * 1000 * 1000  # 单位uV

            plt.plot(data_to_plot, label=f'Data {i + 1}')

    else:
        if length > 0 and length <= (data_array.shape[1] - start_index):
            data_to_plot = data_array[chx - 1, start_index:length]
        else:
            data_to_plot = data_array[chx - 1, :]
            
        # 纵坐标变换
        data_to_plot = ((data_to_plot/((2<<23)-1)) / 24) * 4.5
        data_to_plot = np.array(data_to_plot) * 1000 * 1000  # 单位uV
        
        time_axis = np.arange(0, length) / fs
        plt.plot(time_axis, data_to_plot, color='black', label=f'Data {chx}')
        # plt.plot(data_to_plot, label=f'Data {chx}')

    # 添加图例
    # plt.legend()

    # 添加标题和轴标签
    plt.title('Plot')
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage(uV)')

    # 自动调整坐标轴范围
    plt.autoscale(enable=True, axis='both', tight=None)

    # 显示图形
    plt.show()



def calculate_the_spectrum(signal, fs):
    """
    计算输入信号的频谱。

    :param signal: 输入的一维信号列表。
    :param fs: 采样率，单位为Hz。
    :return: 频率和对应的频谱幅值。
    """
    # 去除直流分量
    signal = signal - np.mean(signal)
    
    # 计算信号的FFT
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)
    spectrum = np.fft.fft(signal)
    
    # 只取前半部分频率和幅值，因为FFT结果是对称的
    freqs = freqs[:n//2]
    spectrum = np.abs(spectrum[:n//2])
    
    return freqs, spectrum


def plot_spectrum(signal, fs, title='wave', freq_end=None, rec_axis=None, gain=24):
    """
    绘制信号的原始波形和频谱图，并添加总标题。

    :param signal: 输入的一维信号列表。
    :param fs: 采样率，单位为Hz。
    :param title: 图像的总标题。
    :param freq_end: 频谱图的最大横坐标频率，单位为Hz。
    :param rec_axis: 矩形框的坐标范围列表，格式为 [x_min, x_max, y_min, y_max]。
    :param gain: ADS1299增益放大倍数。
    """
    # 计算频谱
    freqs, spectrum = calculate_the_spectrum(signal, fs)
    
    # 创建图形和两个子图对象
    fig, (ax_waveform, ax_spectrum) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 纵坐标变换
    plot_data = ((signal/((2<<23)-1)) / gain) * 4.5
    plot_data = np.array(plot_data) * 1000  # 单位mV
    # 绘制原始波形
    time_axis = np.arange(len(plot_data)) / fs
    ax_waveform.plot(time_axis, plot_data, color='black')
    ax_waveform.set_title('Original Waveform')
    ax_waveform.set_xlabel('Time (s)')
    ax_waveform.set_ylabel('Voltage (mV)')
    
    # 绘制频谱图
    if freq_end is not None:
        idx = np.where(freqs <= freq_end)[0]
        freqs = freqs[idx]
        spectrum = spectrum[idx]

    # 绘制频谱图（使用plot函数）
    ax_spectrum.plot(freqs, spectrum, color='black')
    ax_spectrum.set_title('Frequency Spectrum')
    ax_spectrum.set_xlabel('Frequency (Hz)')
    ax_spectrum.set_ylabel('Magnitude')

    # # 绘制频谱图（使用stem函数）
    # markerline, stemlines, baseline = ax_spectrum.stem(freqs, spectrum, linefmt='black', markerfmt='ko', basefmt=' ')
    # plt.setp(markerline, 'markersize', 1)
    # plt.setp(stemlines, 'linewidth', 1)
    # ax_spectrum.set_title('Frequency Spectrum')
    # ax_spectrum.set_xlabel('Frequency (Hz)')
    # ax_spectrum.set_ylabel('Magnitude')

    # 添加矩形框
    if rec_axis is not None:
        rect = Rectangle((rec_axis[0], rec_axis[2]), rec_axis[1] - rec_axis[0], rec_axis[3] - rec_axis[2], 
                         linewidth=1, edgecolor='r', linestyle='--', facecolor='none')
        ax_spectrum.add_patch(rect)

    # 添加总标题
    fig.suptitle(title, fontsize=16)
    
    # 调整子图之间的间距
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(top=0.90)
    
    # 显示图形
    plt.show()




