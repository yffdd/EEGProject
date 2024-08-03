"""
File: anotc_fetch_eeg_movement.py
Author: xiales
Date: 2024-08-01
Description: 用于处理匿名科创地面站导出的.csv文件中的数据 并将其保存为 csv 和 npy 格式
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt

def data_fetch(filename, ch_num=0, length=0):
    """
    从csv文件中提取数据

    :param filename: 文件的路径
    :param ch_num: 要提取的通道数目，即一共要提取多少列数据。如果ch_num=0，则提取有效数据不为0的所有列(第3行开始为有效数据)。如果ch_num不等于0，则提取前ch_num列非零和非空数据。
    :param length: 要提取的每列数据的长度。如果length=0，则提取每列数据的所有非空数据。如果length不等于0，则提取每列数据中有效数据(第3行开始为有效数据)的前length个数据。
    :return: 返回一个二维数组，即返回文件中每列数据 返回格式shape代表[通道n, 通道n的数据]
    """

    # 检查文件路径是否有效
    if not os.path.isfile(filename):
        raise ValueError("The file path is invalid. Please check the file path.")

    # 读取CSV文件
    file_data = pd.read_csv(filename, header=None, low_memory=False)

    # 跳过前两行，从第三行开始读取数据
    file_data = file_data.iloc[2:]

    # 初始化结果列表
    data = []

    # 确定要提取的列数
    columns_to_extract = ch_num if ch_num != 0 else file_data.shape[1]

    # 遍历每列数据
    for i in range(columns_to_extract):
        # 使用 pandas.to_numeric 转换为数字，非数字的转换为 NaN
        # column_data = file_data.iloc[:, i]
        column_data = pd.to_numeric(file_data.iloc[:, i], errors='coerce')
        # 如果length不为0，则只取每列的前length个数据
        if length != 0:
            column_data = column_data.iloc[:length]

        # 将列转换成列表
        column_list = column_data.tolist()

        # 排除全部为0或者nan的列
        if any(x != 0 and pd.notna(x) for x in column_list):
            data.append(column_list)

    # 转换数据为 numpy 数组，
    data_array = np.array(data).astype(float)
    # 填充 NaN 为 0
    data_array = np.nan_to_num(data_array)

    # 将结果以二维数组形式返回
    return data_array


def find_indices_by_numbers(data_array, numbers):
    """
    在数组中查找指定数字的索引值。

    :param data_array: 输入的一维numpy数组
    :param numbers: 一个包含需要查找的数字的列表
    :return: 一个字典，键是要查找的数字，值是该数字在数组中的索引列表 返回格式为{数字n1:数字n1的索引值列表; 数字n2:数字n2的索引值列表; ...}
    """
    indices_dict = {}
    for number in numbers:
        indices = np.where(data_array == number)[0]
        indices_dict[number] = indices.tolist()

    return indices_dict


def slice_by_indices(data, indices, length):
    """
    根据索引值和长度从二维数组的每行中截取片段。

    :param data: 输入的二维numpy数组
    :param indices: 索引值数组
    :param length: 截取的长度
    :return: 一个三维numpy数组，包含从每行中截取的片段 返回格式shape代表[通道n, 片段m, 通道n片段m的数据]
    """
    # 确定每个片段的结尾位置不超过行的长度
    valid_indices = [i for i in indices if i + length <= data.shape[1]]

    # 初始化结果列表
    result = []

    # 遍历二维数组的每一行
    for row in data:
        # 每一行根据有效索引值截取片段
        segments = [row[i:i + length] for i in valid_indices]
        # 将片段添加到结果列表中
        result.append(segments)

    # 将结果列表转换为三维numpy数组
    return np.array(result)


def plot_data_legend_inside(data_array, start_index=0, length=0, chx=0):
    """
    绘制二维数据数组中的数据。 图例在图形内部，当同一张图片中图例较多时显示效果较好

    :param data_array: 输入的二维numpy数组，其中每一行代表一个数据集。
    :param start_index: 每个数据集开始绘制的索引，默认从0开始。
    :param length: 从start_index开始要绘制的长度，如果为0则绘制从start_index到数据集末尾的所有数据。
    :param chx: 如果为0，则绘制data_array中的所有数据集。如果不为0，则只绘制第chx个数据集。

    这个函数创建一个图表，并可以选择性地绘制所有数据集或一个特定的数据集。
    可以指定绘制数据的起始索引和长度。如果没有提供长度(或长度为0)将从起始索引绘制到数据集的末尾。
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

            plt.plot(data_to_plot, label=f'Data {i + 1}')

    else:
        if length > 0 and length <= (data_array.shape[1] - start_index):
            data_to_plot = data_array[chx - 1, start_index:length]
        else:
            data_to_plot = data_array[chx - 1, :]
        plt.plot(data_to_plot, label=f'Data {chx}')
    # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('Data Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # 自动调整坐标轴范围
    plt.autoscale(enable=True, axis='both', tight=None)
    # 显示图形
    plt.show()


def plot_data_legend_external(data_array, start_index=0, length=0, chx=0):
    """
    绘制二维数据数组中的数据。图例在图形外部，当同一张图片中图例较少时显示效果较好

    :param data_array: 输入的二维numpy数组，其中每一行代表一个数据集。
    :param start_index: 每个数据集开始绘制的索引，默认从0开始。
    :param length: 从start_index开始要绘制的长度，如果为0则绘制从start_index到数据集末尾的所有数据。
    :param chx: 如果为0，则绘制data_array中的所有数据集。如果不为0，则只绘制第chx个数据集。

    这个函数创建一个图表，并可以选择性地绘制所有数据集或一个特定的数据集。
    可以指定绘制数据的起始索引和长度。如果没有提供长度(或长度为0)将从起始索引绘制到数据集的末尾。
    """

    # 确定绘图的数量
    num_datasets = data_array.shape[0]
    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(10, 6))  # 使用subplots创建图形和轴
    if chx == 0:
        # 绘制每一列数据
        for i in range(num_datasets):
            if length > 0 and length <= (data_array.shape[1] - start_index):
                data_to_plot = data_array[i, start_index:start_index + length]  # 这里是修正过的索引
            else:
                data_to_plot = data_array[i, :]

            ax.plot(data_to_plot, label=f'Data {i + 1}')  # 使用ax对象调用plot
    else:
        if length > 0 and length <= (data_array.shape[1] - start_index):
            data_to_plot = data_array[chx - 1, start_index:start_index + length]  # 这里是修正过的索引
        else:
            data_to_plot = data_array[chx - 1, :]
        ax.plot(data_to_plot, label=f'Data {chx}')  # 使用ax对象调用plot

    # 添加图例，现在指定在ax对象上
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
    # 添加标题和轴标签
    ax.set_title('Data Plot')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    # 自动调整坐标轴范围
    ax.autoscale(enable=True, axis='both', tight=None)
    # 调整布局以适应图例
    plt.tight_layout(rect=[0, 0, 0.76, 1])  # 调整图形大小以适应图例
    # 显示图形
    plt.show()


def generate_slice_data_files(filelist, out_data_path, ch_num=2, slice_to_nums=[4, [1,2]], length=1000):
    """
    生成切片后的数据文件。生成后的文件保存在工作空间的 data_slice 目录下

    :param filelist: 需要切片的数据文件列表，一个一维的字符串列表
    :param ch_num: 要提取的通道数目，即一共要提取多少列数据。如果ch_num=0，则提取有效数据不为0的所有列(第3行开始为有效数据)。如果ch_num不等于0，则提取前ch_num列非零和非空数据。
    :param slice_to_nums: 一个包含切片标志位信息二维列表. slice_to_nums[0]为标志位所在的列, slice_to_nums[1]为包含需要查找的数字的列表
    :param length: 每个片段截取的长度
    :return: None
    """
    print("gererate slice data files...")
    # 保存数据路径
    if not os.path.exists(out_data_path):  # 如果 data_slice 文件夹不存在则创建 data_slice 文件夹
        os.makedirs(out_data_path)

    # 提取的标签数据 extend后shape代表[标签i, 文件j, 片段k, 标签i文件j片段k的数据]
    label_data = [[[] for i in range(len(filelist))] for j in range(len(slice_to_nums[1]))]
    # 提取数据到 label_data. shape代表[标签i, 文件j, 片段k, 标签i文件j片段k的数据]
    for label_index, label_value in enumerate(slice_to_nums[1]):
        for file_index, filename in enumerate(filelist):
            data = data_fetch(filename=filename)  # 获取文件中的数据
            """
            如果需要对原始数据进行滤波, 则在此处进行
            注意不要处理包含标志位的列
            """
            slice_indices = find_indices_by_numbers(data_array=data[slice_to_nums[0]], numbers=slice_to_nums[1])  # 获取需要切片数据的索引
            slice_data = slice_by_indices(data=data, indices=slice_indices[label_value], length=length)  # 获取切片的结果 shape代表[通道n, 片段m, 标签x通道n片段m的数据]
            current_label_data = []  # 清空 label_data 列表用于当前标签的数据
            # 迭代通道数量，合并所有切片数据
            for ch in range(ch_num):
                current_label_data.extend(slice_data[ch].tolist())  # 将标签 slice_to_nums[1][] 通道ch 的切片片段加到 label_data 中
            # 将当前标签的所有数据片段存入 label_data
            label_data[label_index][file_index].extend(current_label_data)  # 将合并后的数据添加到 label_data 中
    # 将数据写入文件
    # 遍历标签, 将每个标签单独作为一个文件写入
    for label_index, label_value in enumerate(slice_to_nums[1]):
        # 最终写入文件的数据 extend后shape代表[标签j, 片段k, 文件i标签j片段k的数据]
        data_write = []
        for file_index, filename in enumerate(filelist):
            data_write.extend(label_data[label_index][file_index])
        # 创建DataFrame
        df = pd.DataFrame(np.array(data_write))
        # 将DataFrame保存为CSV文件
        out_path = os.path.join(out_data_path, "data_" + str(label_value) + ".csv")
        df.to_csv(out_path, index=False)  # index=False表示不保存行索引
    print(f"slice data file created successfully. save to: {out_data_path}")


def generate_empty_label_data_files(filelist, out_data_path, ch_num=2, length=1000):
    """
    生成空标签的数据文件，并且将数据文件存入 "data_slice/data_0.csv" 文件中

    :param filelist: 需要切片的数据文件列表，一个一维的字符串列表
    :param ch_num: 要提取的通道数目，即一共要提取多少列数据。如果ch_num=0，则提取有效数据不为0的所有列(第3行开始为有效数据)。如果ch_num不等于0，则提取前ch_num列非零和非空数据。
    :param length: 要提取的每个数据片段的长度
    :return: 
    """
    print("generate empty label data files...")
    # 保存数据路径
    if not os.path.exists(out_data_path):  # 如果 data_slice 文件夹不存在则创建 data_slice 文件夹
        os.makedirs(out_data_path)
    # 读取数据
    data_write = []  # 创建用来存放需要写入文件的数据的空列表
    for file_index, filename in enumerate(filelist):
        data = data_fetch(filename=filename)  # 获取文件中的数据
        data_num = data.shape[1] // length  # 计算需要提取的片段数量 取整数
        # 遍历前 ch_num 个通道
        for ch in range(ch_num):
            channel_data = []  # 当前通道收集数据片段
            # 提取该通道的所有片段
            for segment_start in range(data_num):
                start_index = segment_start * length
                end_index = start_index + length
                # 将片段数据添加到当前通道的数据列表
                channel_data.append(data[ch, start_index:end_index])
            # 将当前通道的所有数据片段追加到data_write中
            data_write.extend(channel_data)
    # 将数据写入文件
    for file_index, filename in enumerate(filelist):
        # 创建DataFrame
        df = pd.DataFrame(np.array(data_write))
        # 将DataFrame保存为CSV文件
        out_path = os.path.join(out_data_path, "data_0.csv")
        df.to_csv(out_path, index=False)  # index=False表示不保存行索引
    print(f"data file with empty label was created successfully. save to: {out_data_path}")

def create_label_file(out_data_path, filedict={0:"data_0.csv", 1:"data_1.csv", 2:"data_2.csv"}):
    """
    根据切片后的文件生成对应的标签文件

    :filedict: 包含数据文件标签值和文件名称的一个字典. key为标签值, value为对应标签值的数据文件
    """
    print("create label file...")
    # 保存数据路径
    if not os.path.exists(out_data_path):  # 如果 data_slice 文件夹不存在则创建 data_slice 文件夹
        os.makedirs(out_data_path)
    # 创建完整路径字典
    full_path_filedict = {key: os.path.join(out_data_path, filename) for key, filename in filedict.items()}
    # 遍历文件并生成对应的标签文件
    for index, (key, value) in enumerate(full_path_filedict.items()):
        file_data = pd.read_csv(value, header=None, low_memory=False)  # 读取文件数据
        label_length = file_data.shape[0] - 1  # 计算标签的长度
        label_list = [key] * label_length  # 创建列表
        df = pd.DataFrame(np.array(label_list))  # 转换为 DataFrame
        base_name, ext = os.path.splitext(value)  # 分离文件名和扩展名
        new_filename = f"{base_name}_label{ext}"  # 重新构建文件名，加上"_label"后再加上原来的扩展名
        out_path = os.path.join(out_data_path, new_filename)
        df.to_csv(out_path, index=False)  # 保存为 CSV 文件
    print(f"label file generated successfully. save to: {out_data_path}")


def plot_data_subset(original_data, filtered_data, start_sample, end_sample):
    '''
    绘制原始数据和滤波后数据的比较图，但只显示指定样本区间。
    :param original_data: 原始数据数组（多通道）
    :param filtered_data: 滤波后的数据数组（多通道）
    :param start_sample: 要开始显示的样本索引
    :param end_sample: 要结束显示的样本索引
    '''
    channels = original_data.shape[0]
    fig, axs = plt.subplots(channels, 1, figsize=(10, 8), sharex=True)

    for i in range(channels):
        axs[i].plot(original_data[i, start_sample:end_sample], label='Original', color='blue', alpha=0.5)
        axs[i].plot(filtered_data[i, start_sample:end_sample], label='Filtered', color='red', alpha=0.8)
        axs[i].set_title(f'Channel {i + 1}')
        axs[i].legend()

    plt.xlabel('Sample Number')
    plt.tight_layout()
    plt.show()

def plot_data_subset(original_data, filtered_data, start_sample, end_sample):
    '''
    绘制原始数据和滤波后数据的比较图，但只显示指定样本区间。
    :param original_data: 原始数据数组（多通道）
    :param filtered_data: 滤波后的数据数组（多通道）
    :param start_sample: 要开始显示的样本索引
    :param end_sample: 要结束显示的样本索引
    '''
    channels = original_data.shape[0]
    fig, axs = plt.subplots(channels, 1, figsize=(10, 8), sharex=True)

    for i in range(channels):
        axs[i].plot(original_data[i, start_sample:end_sample], label='Original', color='blue', alpha=0.5)
        axs[i].plot(filtered_data[i, start_sample:end_sample], label='Filtered', color='red', alpha=0.8)
        axs[i].set_title(f'Channel {i + 1}')
        axs[i].legend()

    plt.xlabel('Sample Number')
    plt.tight_layout()
    plt.show()

def load_data_and_labels(data_label_dict, save_data=False, save_path=None):
    """
    读取数据和标签文件，并将它们组合成适用于深度学习的特征矩阵和标签向量。
    可选地将组合后的数据保存为 .npy 文件。

    参数:
    - data_label_dict (dict): 数据和标签文件的字典，键是数据文件名，值是标签文件名。
    - save_data (bool): 是否保存组合后的数据和标签，默认为 False。
    - save_path (str): 保存组合后的数据和标签的路径，默认为 None。

    返回:
    - X (numpy.ndarray): 组合后的特征矩阵。
    - y (numpy.ndarray): 组合后的标签向量。
    """
    X_list = []
    y_list = []
    # 获取数据与标签列表
    for data_file, label_file in data_label_dict.items():
        # 读取数据和标签文件
        data = pd.read_csv(data_file, header=None)
        labels = pd.read_csv(label_file, header=None)
        # 转换为 NumPy 数组
        X = data.to_numpy()
        y = labels.to_numpy().flatten()  # 确保标签是 1D 向量
        X_list.append(X)
        y_list.append(y)
    # 将所有数据和标签组合在一起
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    # 保存数据
    if save_data:
        if not save_path:
            # 从字典中提取任意一个文件路径
            save_path = next(iter(data_label_dict.keys()))
            # 获取上一级路径
            save_path = os.path.dirname(save_path)
        # 保存数据
        np.save(save_path + '/X.npy', X)
        np.save(save_path + '/y.npy', y)
    return X, y


if __name__ == "__main__":

    # 原始数据路径
    raw_data_path = "E:/Databases/RawData/EEG-MOVEMENT/EEG movement test data 240424"
    # 保存数据路径
    out_data_path = "E:/Databases/OutData/EEG-MOVEMENT/EEG movement test data 240424/data_slice"
    ch_num = 4
    slice_to_nums = [8, [1,2]]
    length = 1000

    # 生成切片后的数据文件。生成后的文件保存在工作空间的 data_slice 目录下
    filelist_movement = [
        os.path.join(raw_data_path, "wangpeng_movement_10min.csv"),
        os.path.join(raw_data_path, "xiachuanqian_movement_10min_a.csv"),
        os.path.join(raw_data_path, "xiachuanqian_movement_10min_b.csv"),
    ]
    generate_slice_data_files(filelist=filelist_movement, out_data_path=out_data_path, ch_num=ch_num, slice_to_nums=slice_to_nums, length=length)

    # 生成空标签的数据文件，并且将数据文件存入 "data_slice/data_0.csv" 文件中
    filelist_sit_still = [
        os.path.join(raw_data_path, "wangpeng_sit_still_5min.csv"),
        os.path.join(raw_data_path, "xiachuanqian_sit_still_5min.csv"),
        os.path.join(raw_data_path, "liying_sit_still_5min.csv"),
    ]
    generate_empty_label_data_files(filelist=filelist_sit_still, out_data_path=out_data_path, ch_num=ch_num, length=length)

    # 根据切片后的文件生成对应的标签文件
    data_file_dict={
        0:"data_0.csv",
        1:"data_1.csv",
        2:"data_2.csv"
    }
    create_label_file(filedict=data_file_dict, out_data_path=out_data_path)

    # 处理后的数据和标签文件组合成的字典
    data_label_dict = {
        "data_0.csv": "data_0_label.csv",
        "data_1.csv": "data_1_label.csv",
        "data_2.csv": "data_2_label.csv"
    }
    # 获取完整路径
    data_label_dict = {os.path.join(out_data_path, key): os.path.join(out_data_path, filename) for key, filename in data_label_dict.items()}
    # 获取数据和标签 并保存为 npy 文件
    X, y = load_data_and_labels(data_label_dict=data_label_dict, save_data=True)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y unique: {np.unique(y)}")



