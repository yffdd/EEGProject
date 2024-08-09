"""
File: deap_database_fatch.py
Author: xiales
Date: 2024-08-09
Description: This script is used to preprocess the EEG data from SEED databases.

channel_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
                     'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
                     'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                     'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                     'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                     'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
                     'O2', 'OB1']
channels_to_use = ['FP1', 'FP2', 'FZ', 'C3', 'CZ', 'C4', 'O1', 'O2']
"""
import os
import numpy as np
import scipy.io as scio

# 设置系统的Dataset根目录
sys_dataset_root_dir = 'E:/'  # xiales-pc Windows系统路径
# sys_dataset_root_dir = '/bigdisk/322xcq/'  # 服务器系统路径


def fatch_seed_dataset_CNN(folder_path, num_samples=200, is_save=False, save_path=None, returned_dict=False):
    '''
    提取SEED数据集中的脑电数据，并将其转化为CNN网络所对应的数据格式
    预处理后的 EEG 数据维度为 62 * N，其中 62 为通道数量，N 为采样点个数(已下采样到 200 Hz)
    此函数将预处理后的 EEG 信号转化为 CNN 网络所对应的数据格式，即 62 * num_samples 的二维输入（每 1 秒的信号作为一个样本）
    并区分不同 trial 的数据，最终汇总所有数据用于模型训练，并提供保存选项

    :param folder_path: str, Preprocessed_EEG 文件夹的路径。
    :param num_samples: int, 每个样本所包含的采样点数，默认为 200(即 1 秒的信号)
    :param is_save: bool, 是否保存处理后的特征向量和标签，默认为 False
    :param save_path: str, 保存处理后数据的文件夹路径，默认为 'data'
    :param returned_dict: bool, 是否返回字典，默认为 false
    :return: X, y, feature_vector_dict, label_dict
             - X: numpy.ndarray, 所有样本的特征向量，总形状为 (样本数, 62, num_samples)
             - y: numpy.ndarray, 所有样本的标签，总形状为 (样本数,)
             - feature_vector_dict: dict, 每个被试者的特征向量字典
                - key: str, 被试者的名称 (subject_name)
                - value: dict, 包含被试者的 trial 特征向量
                  - key: str, trial 编号
                  - value: numpy.ndarray, 对应 trial 的特征向量，形状为 (num_trials, 62, num_samples)
             - label_dict: dict, 每个被试者的标签字典，
                - key: str, 被试者的名称 (subject_name)
                - value: dict, 包含被试者的 trial 标签
                  - key: str, trial 编号
                  - value: numpy.ndarray, 对应 trial 的标签，形状为 (num_trials,)
    '''

    # 初始化存放特征向量和标签的字典
    feature_vector_dict = {}
    label_dict = {}
    # 从标签文件 'label.mat' 中获取标签
    labels = scio.loadmat(os.path.join(folder_path, 'label.mat'), verify_compressed_data_integrity=False)
    labels = labels['label'][0]  # 标签 labels['label'] 为 1 * N 的向量，N 为 trial 的数量. labels值为 1 for positive, 0 for neutral, -1 for negative
    try:
        # 获取文件夹中所有 .mat 文件的路径
        all_mat_file = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        # 记录处理的文件数量
        file_cnt = 0
        # 遍历文件夹中所有的文件
        for file_name in all_mat_file:
            file_cnt += 1
            print(f"Processing file: {file_name}. Degree of progress: {file_cnt}/{len(all_mat_file)}")
            # 加载 .mat 文件内容, all_trials_dict 是一个字典, 包含多个试验的 EEG 数据 
            # all_trials_dict.keys() = ['__header__', '__version__', '__globals__', 'ww_eeg1', 'ww_eeg2', 'ww_eeg3', 'ww_eeg4', 'ww_eeg5', 'ww_eeg6', 
            #                           'ww_eeg7', 'ww_eeg8', 'ww_eeg9', 'ww_eeg10', 'ww_eeg11', 'ww_eeg12', 'ww_eeg13', 'ww_eeg14', 'ww_eeg15']
            all_trials_dict = scio.loadmat(os.path.join(folder_path, file_name), verify_compressed_data_integrity=False)
            # 获取实验名称，即文件名去除扩展名
            subject_name = file_name.split('.')[0]
            # 初始化每个 trial 的特征向量和标签字典
            feature_vector_trial_dict = {}
            label_trial_dict = {}
            # 遍历字典中的所有键, 通常每个键对应一个 EEG 信号
            for key in all_trials_dict.keys():
                # 如果键名中不包含 'eeg', 则跳过
                if 'eeg' not in key:
                    continue
                # 初始化当前 trial 的特征向量和标签列表
                feature_vector_list = []
                label_list = []
                # 获取当前 trial 的 EEG 数据，cur_trial 的 shape 为 (62,  N), 其中 62 是通道数, N 是总的采样点数
                cur_trial = all_trials_dict[key]
                # 获取当前 trial 的数据长度, 即总的采样点的数量 N
                length = len(cur_trial[0])
                # 通过滑动窗口的方式，每 200 个采样点截取一个样本
                start_index = 0
                while start_index + num_samples <= length:
                    # 将 shape 为 (62, num_samples) 的二维片段添加到特征向量列表中
                    feature_vector_list.append(np.asarray(cur_trial[:, start_index:start_index + num_samples]))
                    # 获取片段对应的标签，并将其转换为 -1, 0, 1 的格式
                    raw_label = labels[int(key.split('_')[-1][3:]) - 1]
                    # 将标签添加到标签列表中
                    label_list.append(raw_label)
                    # 滑动窗口右移 num_samples 个采样点
                    start_index += num_samples
                # 获取 trial 的编号
                trial = key.split('_')[1][3:]
                # 将当前 trial 的特征向量列表和标签列表转换为 numpy 数组, 并存入字典
                feature_vector_trial_dict[trial] = np.asarray(feature_vector_list)
                label_trial_dict[trial] = np.asarray(label_list)
            # 将当前实验的特征向量和标签字典存入总的字典中
            feature_vector_dict[subject_name] = feature_vector_trial_dict
            label_dict[subject_name] = label_trial_dict
    # 如果文件夹路径或文件有误，捕获并打印错误信息
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")

    # 初始化空列表，用于存储所有样本的特征向量和标签
    X = []
    y = []
    # 遍历每个被试者(subject)
    for subject in feature_vector_dict.keys():
        # 遍历该被试者下的每个 trial
        for trial in feature_vector_dict[subject].keys():
            # 将当前 trial 的所有样本特征向量添加到 X 列表中
            X.extend(feature_vector_dict[subject][trial])
            # 将当前 trial 的所有样本标签添加到 y 列表中
            y.extend(label_dict[subject][trial])
    # 将特征向量列表 X 和 y 转换为 numpy 数组，方便后续的模型输入
    X = np.array(X)
    y = np.array(y)
    # 如果没有指定保存路径，使用默认路径 'data'
    if save_path is None:
        save_path = 'data'
    # 如果保存路径不存在，则创建该路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_save:
        # 保存特征向量和标签到文件
        np.save(os.path.join(save_path, 'X.npy'), X)
        np.save(os.path.join(save_path, 'y.npy'), y)
        print(f"Data saved to {save_path}")
    if returned_dict:
        # 返回包含特征向量和标签的字典 (暂存留做备用)
        return X, y, feature_vector_dict, label_dict
    else:
        # 返回特征向量和标签的 numpy 数组
        return X, y


if __name__ == '__main__':
    # 加载数据
    databases_out_directory = os.path.join(sys_dataset_root_dir, "Databases/OutData/SEED/CNN/")  # 数据集输出目录
    folder_path = os.path.join(sys_dataset_root_dir, "Databases/RawData/SEED/SEED/Preprocessed_EEG/")
    X, y = fatch_seed_dataset_CNN(folder_path=folder_path, num_samples=200, is_save=True, save_path=databases_out_directory, returned_dict=False)