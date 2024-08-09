# import scipy.io as sio
# import os

# seed_Preprocessed_EEG_path = "E:\Databases\RawData\SEED\SEED\Preprocessed_EEG"
# file_path = os.path.join(seed_Preprocessed_EEG_path, "1_20131027.mat")
# # 加载.mat文件
# data = sio.loadmat(file_path)
# # 查看文件中的所有变量
# print(f"keys: {data.keys()}")
# # 提取EEG数据和标签
# eeg_data = data['djc_eeg12']  # EEG数据
# # labels = data['labels']      # 标签
# print(f"eeg_data shape: {eeg_data.shape}")




# seed_ExtractedFeatures_path = "E:\Databases\RawData\SEED\SEED\ExtractedFeatures"
# file_path = os.path.join(seed_ExtractedFeatures_path, "1_20131027.mat")
# # 加载.mat文件
# data = sio.loadmat(file_path)
# # 查看文件中的所有变量
# print(f"keys: {data.keys()}")

import os
from seed_tools import build_preprocessed_eeg_dataset_CNN, RawEEGDataset, subject_independent_data_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

# 设置系统的Dataset根目录
sys_dataset_root_dir = 'E:/'  # xiales-pc Windows系统路径
# sys_dataset_root_dir = '/bigdisk/322xcq/'  # 学校服务器系统路径

channel_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
                     'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
                     'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                     'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                     'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                     'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
                     'O2', 'OB1']
channels_to_use = ['FP1', 'FP2', 'FZ', 'C3', 'CZ', 'C4', 'O1', 'O2']

# 加载数据，整理成所需要的格式
folder_path = "E:/Databases/RawData/SEED/SEED/Preprocessed_EEG"
feature_vector_dict, label_dict = build_preprocessed_eeg_dataset_CNN(folder_path)
train_feature, train_label, test_feature, test_label = subject_independent_data_split(feature_vector_dict, label_dict,
                                                                                      {'2', '6', '9'})

desire_shape = [1, 62, 200]
train_data = RawEEGDataset(train_feature, train_label, desire_shape)
test_data = RawEEGDataset(test_feature, test_label, desire_shape)

train_feature_temp = np.array(train_feature)
print(f"train_feature shape: {train_feature_temp.shape}")
train_label_temp = np.array(train_label)
print(f"train_label_temp shape: {train_label_temp.shape}")
# 保存数据和标签到本地的.npy文件
databases_out_directory = os.path.join(sys_dataset_root_dir, "Databases/OutData/SEED/SEED-Emotion-Recognition/")
np.save(os.path.join(databases_out_directory, 'X_train.npy'), train_feature_temp)
np.save(os.path.join(databases_out_directory, 'y_train.npy'), train_label_temp)

test_feature_temp = np.array(test_feature)
print(f"test_feature shape: {test_feature_temp.shape}")
test_label_temp = np.array(test_label)
print(f"test_label shape: {test_label_temp.shape}")


# 保存数据和标签到本地的.npy文件
databases_out_directory = os.path.join(sys_dataset_root_dir, "Databases/OutData/SEED/SEED-Emotion-Recognition/")
np.save(os.path.join(databases_out_directory, 'X_test.npy'), test_feature_temp)
np.save(os.path.join(databases_out_directory, 'y_test.npy'), test_label_temp)

print('Data and labels saved as .npy files.')