"""
File: eeg_databases_pre.py
Author: xiales
Date: 2024-07-20
Description: This script is used to preprocess the EEG data from different databases.
"""

import os
import sklearn as sk
import torch
import numpy as np
import sys
import scipy.io as sio
from scipy.signal import resample
from tqdm import tqdm, trange
import warnings
import pickle
from scipy.signal import butter, filtfilt, sosfilt, sosfreqz
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# For preprocessing
deap_config = {
    "dataset_name": "deap",
    "databases_root_directory": r"E:/Databases/RawData/DEAP/data_preprocessed_python",
    "databases_out_directory": r"E:/Databases/OutData/DEAP/ACSE/",
    # "databases_root_directory": r"/bigdisk/322xcq/Databases/RawData/DEAP/data_preprocessed_python",
    # "databases_out_directory": r"/bigdisk/322xcq/Databases/OutData/DEAP/ACSE/",
    "sampling_rate": 128,
    "resampling_rate": 128,
    "channel_names": ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2', 'CZ', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
    # "channels_to_use": ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
    "channels_to_use": ['FP1', 'FP2', 'FZ', 'C3', 'CZ', 'C4', 'O1', 'O2'],
    "baseline_removal_window": 3,
    "cutoff_frequencies": [4,40],
    "seconds_to_use": 60,
    "window_size": 2,
    "window_overlap": 0,
    "progress_bar": True
}


class SignalProcessor:
    def __init__(self, fs=128):
        super(SignalProcessor, self).__init__()
        self.fs = fs

    def plot_frequency_spectrum(self, signal, fs=None, title='Frequency Spectrum'):
        """
        计算并绘制一维数组的频谱图。
        参数：
        signal (ndarray): 输入的一维数组。
        fs (int): 采样频率。
        返回：
        None
        """
        if fs is None:
            fs = self.fs
        # 计算傅里叶变换
        spectrum = np.fft.fft(signal)
        # 获取频率分量
        freqs = np.fft.fftfreq(len(signal), 1/fs)
        # 只取前半部分频率，因为频谱是对称的
        freqs = freqs[:len(freqs)//2]
        spectrum = np.abs(spectrum[:len(spectrum)//2])
        
        # 绘制频谱图
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, spectrum)
        plt.title(title)
        plt.X_cp_cp_cp_cp_cplabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()

        return freqs, spectrum

    def highpass_filter(self, signal, fs=None, cutoff=4, order=5):
        """
        对信号进行高通滤波。
        参数：
        signal (ndarray): 输入信号。
        cutoff (float): 截止频率。
        fs (int): 采样频率。
        order (int): 滤波器阶数。
        返回：
        ndarray: 高通滤波后的信号。
        """
        if fs is None:
            fs = self.fs
        nyquist = 0.5 * fs  # 计算奈奎斯特频率
        normal_cutoff = cutoff / nyquist  # 将截止频率归一化到[0, 1]之间
        b, a = butter(order, normal_cutoff, btype='high', analog=False)  # 设计高通Butterworth滤波器
        filtered_signal = filtfilt(b, a, signal)  # 使用filtfilt函数应用滤波器，避免相位失真
        return filtered_signal  # 返回滤波后的信号
    
    def lowpass_filter(self, signal, fs=None, cutoff=40, order=5):
        """
        对信号进行低通滤波。
        参数：
        signal (ndarray): 输入信号。
        cutoff (float): 截止频率。
        fs (int): 采样频率。
        order (int): 滤波器阶数。
        返回：
        ndarray: 低通滤波后的信号。
        """
        if fs is None:
            fs = self.fs
        nyquist = 0.5 * fs  # 计算奈奎斯特频率
        normal_cutoff = cutoff / nyquist  # 将截止频率归一化到[0, 1]之间
        b, a = butter(order, normal_cutoff, btype='low', analog=False)  # 设计低通Butterworth滤波器
        filtered_signal = filtfilt(b, a, signal)  # 使用filtfilt函数应用滤波器，避免相位失真
        return filtered_signal  # 返回滤波后的信号
    
    def bandpass_filter(self, signal, fs=None, lowcut=40, highcut=4, order=5):
        """
        对信号进行带通滤波。
        参数：
        signal (ndarray): 输入信号。
        lowcut (float): 低截止频率。
        highcut (float): 高截止频率。
        fs (int): 采样频率。
        order (int): 滤波器阶数。
        返回：
        ndarray: 带通滤波后的信号。
        """
        if fs is None:
            fs = self.fs
        nyquist = 0.5 * fs  # 计算奈奎斯特频率
        low = lowcut / nyquist  # 将低截止频率归一化到[0, 1]之间
        high = highcut / nyquist  # 将高截止频率归一化到[0, 1]之间
        b, a = butter(order, [low, high], btype='band')  # 设计带通Butterworth滤波器
        filtered_signal = filtfilt(b, a, signal)  # 使用filtfilt函数应用滤波器，避免相位失真
        return filtered_signal  # 返回滤波后的信号
    
    def notch_filter(self, signal, fs=None, cutoff=50, quality_factor=30):
        """
        对信号进行陷波滤波。
        参数：
        signal (ndarray): 输入信号。
        cutoff (float): 陷波频率。
        fs (int): 采样频率。
        quality_factor (float): 品质因数。
        返回：
        ndarray: 陷波滤波后的信号。
        """
        if fs is None:
            fs = self.fs
        nyquist = 0.5 * fs  # 计算奈奎斯特频率
        normal_cutoff = cutoff / nyquist  # 将陷波频率归一化到[0, 1]之间
        b, a = butter(2, [normal_cutoff - normal_cutoff / quality_factor, 
                        normal_cutoff + normal_cutoff / quality_factor], btype='bandstop')  # 设计带阻Butterworth滤波器
        filtered_signal = filtfilt(b, a, signal)  # 使用filtfilt函数应用滤波器，避免相位失真
        return filtered_signal  # 返回滤波后的信号
    
    def baseline_adjustments(self, signal):
        """
        基线调整
        参数：
        signal (ndarray): 输入信号。
        返回：
        ndarray: 调整后的信号。
        """
        base_line = signal.sum(0) / len(signal)
        signal = signal - base_line
        return signal
    
    def plot_signal_and_spectrum(self, signal, fs=None, suptitle='Signal and Spectrum'):
        """
        绘制信号和频谱图。
        
        参数：
        signal (ndarray): 输入的一维数组。
        fs (int): 采样频率。如果未提供，使用默认的 self.fs。
        suptitle (str): 图的总标题。
        
        返回：
        None
        """
        if fs is None:
            fs = self.fs

        # 计算傅里叶变换
        spectrum = np.fft.fft(signal)
        # 获取频率分量
        freqs = np.fft.fftfreq(len(signal), 1/fs)
        # 只取前半部分频率，因为频谱是对称的
        freqs = freqs[:len(freqs)//2]
        spectrum = np.abs(spectrum[:len(spectrum)//2])
        # 创建图形
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        # 绘制信号
        axs[0].plot(signal)
        axs[0].set_title('Signal')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Amplitude')
        # 绘制频谱图
        axs[1].plot(freqs, spectrum)
        axs[1].set_title('Spectrum')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Amplitude')
        # 添加总标题
        plt.suptitle(suptitle)
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以确保标题不重叠
        plt.show()

class DatabasesPreprocessing(SignalProcessor):
    def __init__(self, config):
        super(DatabasesPreprocessing, self).__init__()
        self.dataset_name = config["dataset_name"]
        self.databases_root_directory = config["databases_root_directory"]
        self.databases_out_directory = config["databases_out_directory"]
        self.sampling_rate = config["sampling_rate"]
        self.resampling_rate = config["resampling_rate"]
        self.channel_names = config["channel_names"]
        self.channels_to_use = config["channels_to_use"]
        self.baseline_removal_window = config["baseline_removal_window"]
        self.cutoff_frequencies = config["cutoff_frequencies"]
        self.seconds_to_use = config["seconds_to_use"]
        self.window_size = config["window_size"]
        self.window_overlap = config["window_overlap"]
        self.progress_bar = config["progress_bar"]
        self.convert_labels_to_nnp = False
        super().__init__(fs=self.sampling_rate)

    def load_deap_data(self, data_path, rm_baseline=True):
        """
        加载DEAP数据集
        参数:
        data_path (str): 存储DEAP数据集的目录路径
        rm_baseline (bool): 是否移除前3s休息时的基线数据
        返回: 
        x_data, y_labels, subjects, trials, sessions = load_deap_data(data_path=data_path)
        tuple: 包含所有被试的EEG和其他生理信号数据，以及对应的情绪标签数据。
            x_data_32 (ndarray): 移除基线前形状为 (1280, 32, 8064) 的EEG数据。 移除基线后形状为 (1280, 32, 7680) 的EEG数据。
            y_data (ndarray): 形状为 (1280, 4) 的情绪标签数据。
        """
        # 获取目录中所有以.dat结尾的文件名
        files = [f for f in os.listdir(data_path) if f.endswith('.dat')]
        # 初始化存储所有数据和标签的列表
        raw_data = []  # 形状为 (num_subjects, num_trials, num_channels, num_samples)
        raw_labels = []  # 形状为 (num_subjects, num_trials, num_labels)
        subjects = []  # 存储每个试验的受试者编号
        trials = []  # 存储每个受试者的每次试验编号
        sessions = []  # 存储每个试验的会话编号
        # 是否显示进度条
        if self.progress_bar:
            files_iter = tqdm(files, desc="Loading DEAP data")
        else:
            files_iter = files
        # 遍历每个文件
        for file in files_iter:
            # 获取每个文件的完整路径
            file_path = os.path.join(data_path, file)
            # 打开并读取文件中的数据
            with open(file_path, 'rb') as f:
                # 使用latin1编码读取文件
                data = pickle.load(f, encoding='latin1')
                # 提取EEG和其他生理信号数据，并添加到 raw_data 列表中
                raw_data.append(data['data'])
                # 提取情绪标签数据，并添加到 raw_labels 列表中
                raw_labels.append(data['labels'])
                # # 添加受试者和试验信息
                subject_id = int(file[1:3])  # 提取受试者编号
                subjects.extend([subject_id] * 40)  # 每个受试者有 40 次试验
                trials.extend(range(1, 41))  # 每个受试者的试验编号从 1 到 40
                sessions.extend([1] * 40)  # 假设每个受试者只有一个会话
        # 将 raw_data 的形状从 (32, 40, 40, 8064) 改为 (1280, 40, 8064)
        x_data = np.array(raw_data).reshape(32*40, 40, 8064)
        # 选择前 32 个通道并去除前3s准备时间变为 (1280, 32, 7680)
        if rm_baseline:
            x_data_32 = x_data[:, :32, self.baseline_removal_window*self.sampling_rate:]
        else:
            x_data_32 = x_data[:, :32, :]
        # 将 raw_labels 的形状从 (32, 40, 4) 改为 (1280, 4)
        y_labels = np.array(raw_labels).reshape(32*40, 4)
        # 将结果转换为NumPy数组并返回
        return np.array(x_data_32), np.array(y_labels), np.array(subjects), np.array(trials), np.array(sessions)
    
    def deap_filter(self, X):
        # 复制输入信号，以保护原始数据
        X_cp = np.copy(X)
        if self.progress_bar:
            ep_id = trange(X_cp.shape[0], desc='Baseline adjustments and filter')
        else:
            ep_id = range(X_cp.shape[0])
        for experiment_id in ep_id:
            for channel_id in range(X_cp.shape[1]):
                # 基线调整
                X_cp[experiment_id, channel_id, :] = self.baseline_adjustments(signal=X_cp[experiment_id, channel_id, :])
                # 滤波
                X_cp[experiment_id, channel_id, :] = self.highpass_filter(signal=X_cp[experiment_id, channel_id, :], cutoff=self.cutoff_frequencies[0])
                X_cp[experiment_id, channel_id, :] = self.lowpass_filter(signal=X_cp[experiment_id, channel_id, :], cutoff=self.cutoff_frequencies[1])
        
        # 重采样
        if not (self.resampling_rate == self.sampling_rate) and not (self.resampling_rate == 0):
            X_cp = self.resampling(X=X_cp, sampling_rate=self.sampling_rate, resampling_rate=self.resampling_rate)
        # 选择合适的通道
        X_cp = self.select_channels(X=X_cp)
        

        return X_cp
    
    def resampling(self, X, sampling_rate, resampling_rate):
        """
        对信号进行重采样。
        参数：
        X (ndarray): 输入信号，形状为 (num_subjects, num_trials, num_samples)。
        sampling_rate (int): 原始采样率。
        resampling_rate (int): 目标采样率。
        返回：
        ndarray: 重采样后的信号。
        """
        # 复制输入信号，以保护原始数据
        X_cp = np.copy(X)
        # 检查是否需要重采样（目标采样率不为零且不同于原始采样率）
        if not(resampling_rate == 0) and not(resampling_rate == sampling_rate):
            # 计算重采样后的新长度
            new_length = int(X_cp.shape[2] / sampling_rate * resampling_rate)
            # 初始化重采样后的数组，形状为 (num_subjects, num_trials, new_length)
            X_resampled = np.zeros((X_cp.shape[0], X_cp.shape[1], new_length))
            # 判断是否需要显示进度条
            if self.progress_bar:
                ep_id = trange(X_cp.shape[0], desc='Downsampling')
            else:
                ep_id = range(X_cp.shape[0])
            # 遍历每个实验
            for experiment_id in ep_id:
                # 遍历每个通道
                for channel_id in range(X_cp.shape[1]):
                    # 对每个通道的数据进行重采样
                    X_resampled[experiment_id, channel_id, :] = resample(X_cp[experiment_id, channel_id, :], new_length)
            # 更新复制的数据为重采样后的数据
            X_cp = X_resampled
        # 返回重采样后的信号
        return X_cp
    
    def select_channels(self, X):
        """
        选择特定的通道。
        
        参数：
        X (ndarray): 输入信号，形状为 (num_subjects, num_channels, num_samples)。
        
        返回：
        ndarray: 选择通道后的信号，形状为 (num_subjects, len(self.channels_to_use), num_samples)。
        """
        # 复制输入信号，以保护原始数据
        X_cp = np.copy(X)
        # 初始化通道索引列表
        channel_index_list = list()
        # 查找需要使用的通道索引
        for i in range(len(self.channels_to_use)):
            # 如果通道在实际通道名称列表中，添加其索引到通道索引列表中
            if self.channels_to_use[i] in self.channel_names:
                channel_index_list.append(self.channel_names.index(self.channels_to_use[i]))
            else:
                # 如果通道不在实际通道名称列表中，发出警告
                warnings.warn(' Channel ' + self.channels_to_use[i] + ' could not be found in the list of actual channels')
        # 初始化选择通道后的数组，形状为 (num_subjects, len(self.channels_to_use), num_samples)
        X_selected_channels = np.zeros((X_cp.shape[0], len(self.channels_to_use), X_cp.shape[2]))
        # 判断是否需要显示进度条
        if self.progress_bar:
            channel_index_list_en = tqdm(enumerate(channel_index_list), desc="Select channels")
        else:
            channel_index_list_en = enumerate(channel_index_list)
        # 遍历每个通道及其索引，将对应的通道数据复制到选择后的数组中
        for channel, channel_index in channel_index_list_en:
            X_selected_channels[:, channel, :] = X_cp[:, channel_index, :]
        # 更新复制的数据为选择通道后的数据
        X_cp = X_selected_channels
        # 返回选择通道后的信号
        return X_cp
    
    def deap_label_conversion(self, X, y, subjects, trials, sessions, convert_labels_to_nnp=False, plot_en=False):
        """
        转换 DEAP 数据集的标签，并对数据进行窗口化处理。
        参数：
        - X: 输入的信号数据。
        - y: 输入的标签数据。
        - subjects: 受试者编号。
        - trials: 试验编号。
        - sessions: 会话编号。
        - convert_labels_to_nnp: 是否转换为 NNP 标签。
        - plot_en: 是否启用绘图。
        返回：
        - X_cp: 窗口化处理后的信号数据。
        - y_cp: 转换后的标签数据。
        - subjects: 处理后的受试者编号。
        - trials: 处理后的试验编号。
        - sessions: 处理后的会话编号。
        """
        # 复制输入信号, 保护原始数据
        X_cp = np.copy(X)
        y_cp = np.copy(y)
        # 计算每个窗口的点数和每个窗口的重叠点数
        num_points_per_window = self.window_size * self.resampling_rate
        num_points_overlap = self.window_overlap * self.resampling_rate
        # 计算窗口滑动的步长
        stride = num_points_per_window - num_points_overlap
        # 初始化起始和结束索引列表
        start_index = [0]
        end_index = [num_points_per_window]
        # 初始化每个实验的窗口数
        num_windows_per_exp = 1
        # 计算每个实验的窗口数以及相应的起始和结束索引
        while(end_index[-1] + stride <= X_cp.shape[2]):
            num_windows_per_exp += 1
            start_index.append(start_index[-1] + stride)
            end_index.append(end_index[-1] + stride)
        # 初始化切割后的数据数组
        X_cut = np.zeros((num_windows_per_exp * X_cp.shape[0], X_cp.shape[1], num_points_per_window))
        y_cut = np.zeros((num_windows_per_exp * X_cp.shape[0], 4))
        subjects_cut = np.zeros(num_windows_per_exp * X_cp.shape[0])
        trials_cut = np.zeros(num_windows_per_exp * X_cp.shape[0])
        sessions_cut = np.zeros(num_windows_per_exp * X_cp.shape[0])
        # 判断是否需要显示进度条
        if self.progress_bar:
            exp_id_range = trange(X_cp.shape[0], desc="Cutting data")
        else:
            exp_id_range = range(X_cp.shape[0])
        # 遍历每个实验
        for exp_id in exp_id_range:
            # 遍历每个窗口
            for window_id in range(len(start_index)):
                # 根据窗口的起始和结束索引切割数据
                X_cut[exp_id * num_windows_per_exp + window_id, :, :] = X_cp[exp_id, :, start_index[window_id]:end_index[window_id]]
                # 复制标签、会话、受试者和试验信息
                y_cut[exp_id * num_windows_per_exp + window_id, :] = y_cp[exp_id, :]
                subjects_cut[exp_id * num_windows_per_exp + window_id] = subjects[exp_id]
                trials_cut[exp_id * num_windows_per_exp + window_id] = trials[exp_id]
                sessions_cut[exp_id * num_windows_per_exp + window_id] = sessions[exp_id]
        # 更新原始数据为切割后的数据
        X_cp = X_cut
        y_cp = y_cut
        subjects = subjects_cut
        trials = trials_cut
        sessions = sessions_cut
        # 创建 MinMaxScaler 对象，用于将数据缩放到 [-1, 1] 范围
        scaler = MinMaxScaler(feature_range=(-1,1))
        # 将情绪价度(愉快度)标签缩放到 [-1, 1] 范围
        valence = scaler.fit_transform(y_cp[:,0].reshape(-1,1))
        # 将唤醒度标签缩放到 [-1, 1] 范围
        arousal = scaler.fit_transform(y_cp[:,1].reshape(-1,1))
        # 将缩放后的愉快度和唤醒度标签按列连接成一个新的数据数组
        datapoints = np.concatenate((valence, arousal), axis=1)
        # 使用 KMeans 对数据进行聚类，指定聚类数量为 4，随机种子为 7
        kmeans = KMeans(n_clusters=4, random_state=7, n_init=10).fit(datapoints)
        # 对聚类中心点按与四个角（右下角、左下角、左上角、右上角）的距离进行排序
        # 并分别对应悲伤 (sad)、恐惧 (fear)、中性 (neutral)、快乐 (happy) 标签
        sad_label, fear_label, neutral_label, happy_label = self.sort_centeroids(centeroids=kmeans.cluster_centers_)
        if plot_en:
            # 绘制愉快度（Valence）和唤醒度（Arousal）的散点图，并使用 KMeans 聚类的标签进行着色
            plt.scatter(valence, arousal, c=kmeans.labels_, edgecolors='none')
            plt.xlabel('Valence')  # 设置 x 轴标签
            plt.ylabel('Arousal')  # 设置 y 轴标签
            plt.title('DEAP - NNP Label via K-Means')  # 设置图表标题
            plt.show()  # 显示图表
        # 找到每个情绪类别对应的数据点索引
        idx_sad = np.where(kmeans.labels_ == sad_label)  # 找到悲伤类标签的数据点索引
        idx_fear = np.where(kmeans.labels_ == fear_label)  # 找到恐惧类标签的数据点索引
        idx_neutral = np.where(kmeans.labels_ == neutral_label)  # 找到中性类标签的数据点索引
        idx_happy = np.where(kmeans.labels_ == happy_label)  # 找到快乐类标签的数据点索引
        # 初始化新的标签数组，形状与 y_cp 相同
        Y_nnp = np.zeros(y_cp.shape[0],)
        # 根据索引设置新的标签值
        # 将情绪分为3类：负面情绪(-1)  中性情绪(0)  正面情绪(1)
        Y_nnp[idx_sad] = -1  # 将悲伤类标签设为 -1
        Y_nnp[idx_fear] = -1  # 将恐惧类标签设为 -1
        Y_nnp[idx_neutral] = 0  # 将中性类标签设为 0
        Y_nnp[idx_happy] = 1  # 将快乐类标签设为 1
        # 如果需要转换，则将 y_cp 更新为新的标签
        if convert_labels_to_nnp:
            y_cp = Y_nnp
            # 统计每个情绪类别的数据点数量，并打印结果
            print('Negative(-1): %i -- neutral(0): %i -- Positive(1): %i'
                % (np.count_nonzero(Y_nnp == -1),  # 统计恐惧类标签的数据点数量
                    np.count_nonzero(Y_nnp == 0),   # 统计悲伤类标签的数据点数量
                    np.count_nonzero(Y_nnp == 1)))  # 统计中性类标签的数据点数量
        else:
            y_cp = kmeans.labels_
            # 统计每个情绪类别的数据点数量，并打印结果
            print('Fear(%i): %i -- Sad(%i): %i -- Neutral(%i): %i -- Happy(%i): %i'
                % (fear_label,    np.count_nonzero(kmeans.labels_ == fear_label),  # 统计恐惧类标签的数据点数量
                    sad_label,      np.count_nonzero(kmeans.labels_ == sad_label),   # 统计悲伤类标签的数据点数量
                    neutral_label,  np.count_nonzero(kmeans.labels_ == neutral_label),  # 统计中性类标签的数据点数量
                    happy_label,    np.count_nonzero(kmeans.labels_ == happy_label)))  # 统计快乐类标签的数据点数量
        return X_cp, y_cp, subjects, trials, sessions

    def sort_centeroids(self, centeroids):
        """
        计算给定中心点到四个角（右下角、左下角、左上角、右上角）的距离，
        并返回按以下顺序排列的索引：
        - 最靠近右下角的点的索引
        - 最靠近左下角的点的索引
        - 最靠近左上角的点的索引
        - 最靠近右上角的点的索引
        参数：
        centeroids (ndarray): 中心点数组，形状为 (num_points, 2)，每行表示一个点的坐标 (x, y)。
        返回：
        tuple: 包含四个整数的元组，表示最靠近右下角、左下角、左上角和右上角的点的索引。
        """
        # 初始化距离数组
        distance_br = np.zeros(centeroids.shape[0])  # 到右下角的距离
        distance_bl = np.zeros(centeroids.shape[0])  # 到左下角的距离
        distance_tl = np.zeros(centeroids.shape[0])  # 到左上角的距离
        distance_tr = np.zeros(centeroids.shape[0])  # 到右上角的距离
        # 计算每个中心点到四个角的距离
        for i in range(centeroids.shape[0]):
            distance_br[i] = abs((1 - centeroids[i, 0])**2 + (-1 - centeroids[i, 1])**2)
            distance_bl[i] = abs((-1 - centeroids[i, 0])**2 + (-1 - centeroids[i, 1])**2)
            distance_tl[i] = abs((-1 - centeroids[i, 0])**2 + (1 - centeroids[i, 1])**2)
            distance_tr[i] = abs((1 - centeroids[i, 0])**2 + (1 - centeroids[i, 1])**2)
        # 找到距离四个角最近的点的索引
        br_idx = np.argmin(distance_br)  # 最靠近右下角的点的索引
        bl_idx = np.argmin(distance_bl)  # 最靠近左下角的点的索引
        tl_idx = np.argmin(distance_tl)  # 最靠近左上角的点的索引
        tr_idx = np.argmin(distance_tr)  # 最靠近右上角的点的索引
        # 返回索引
        return br_idx, bl_idx, tl_idx, tr_idx
    

def classify_emotions_valence_arousal(y):
    """
    根据 Valence 和 Arousal 的值进行情绪分类
    参数：
    y (ndarray): 情绪标签，形状为 (num_samples, num_labels)
    返回：
    labels (ndarray): 分类后的标签，形状为 (num_samples,)
    """
    # 初始化分类标签
    labels = np.zeros(y.shape[0], dtype=int)
    
    for i in range(y.shape[0]):
        valence = y[i, 0]
        arousal = y[i, 1]
        
        if valence > 5 and arousal > 5:
            labels[i] = 1  # happy
        elif valence <= 5 and arousal <= 5:
            labels[i] = 2  # sad
        elif valence <= 5 and arousal > 5:
            labels[i] = 3  # fear
        elif valence > 5 and arousal <= 5:
            labels[i] = 0  # neutral
    
    return labels

def slice_data(X, y, num_samples):
    """
    将数据和标签进行切片
    参数：
    X (ndarray): 输入数据，形状为 (num_samples, num_channels, num_timepoints)
    y (ndarray): 输入标签，形状为 (num_samples,)
    num_samples (int): 每个切片的样本数
    返回：
    X_sliced (ndarray): 切片后的数据
    y_sliced (ndarray): 切片后的标签
    """
    # 初始化切片后的数据和标签列表
    X_sliced = []
    y_sliced = []
    # 遍历每个样本
    for i in range(X.shape[0]):
        # 获取当前样本的数据和标签
        data = X[i]
        label = y[i]
        # 计算切片数量，忽略不足一个切片的部分
        num_slices = data.shape[1] // num_samples
        # 进行切片
        for j in range(num_slices):
            start_idx = j * num_samples
            end_idx = (j + 1) * num_samples
            # 追加切片后的数据和标签
            X_sliced.append(data[:, start_idx:end_idx])
            y_sliced.append(label)
    # 转换为ndarray
    X_sliced = np.array(X_sliced)
    y_sliced = np.array(y_sliced)
    
    return X_sliced, y_sliced

def deap_preprocessing(DatabasesPre):
    dp = DatabasesPre
    # 加载数据
    x_data, y_labels, subjects, trials, sessions = dp.load_deap_data(data_path=dp.databases_root_directory)
    # 保护原始数据
    X_cp = np.copy(x_data)
    y_cp = np.copy(y_labels)
    # 基线调整, 滤波, 重采样, 通道选择
    X_cp = dp.deap_filter(X=X_cp)
    # 情绪分类
    y_cp = classify_emotions_valence_arousal(y_cp)
    # 进行数据切片
    num_samples_per_slice = deap_config["sampling_rate"]  # 例如，每个切片包含256个样本点
    X_cp, y_cp = slice_data(X_cp, y_cp, num_samples_per_slice)
    # 分割数据集
    print("Splitting dataset .....")
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_cp, y_cp, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42)
    S_train, S_temp, P_train, P_temp = train_test_split(subjects, sessions, test_size=0.2, random_state=42)
    S_val, S_test, P_val, P_test = train_test_split(S_temp, P_temp, test_size=0.2, random_state=42)
    # 调整session的ID, 调整 sessions 的 ID 是为了在后续分析时方便区分不同的数据集
    S_train = S_train + 100 * 1
    S_val = S_val + 100 * 2
    S_test = S_test + 100 * 3
    # 保存数据
    print("Saving data ......")
    np.savez_compressed(
                        file = dp.databases_out_directory + dp.dataset_name,
                        X=X_cp,
                        y=y_cp,
                        subject = subjects,
                        trial = trials,
                        session = sessions,
                        dataset_name = dp.dataset_name,
                        sampling_rate = dp.sampling_rate,
                        downsampling_rate = dp.resampling_rate,
                        baseline_removal_window = dp.baseline_removal_window,
                        channel_names = dp.channels_to_use,
                        seconds_to_use = dp.seconds_to_use,
                        window_size = dp.window_size,
                        window_overlap = dp.window_overlap,
                        cutoff_frequencies = dp.cutoff_frequencies,
                        X_train=X_train, X_val=X_val, X_test=X_test,
                        Y_train=Y_train, Y_val=Y_val, Y_test=Y_test,
                        S_train=S_train, S_val=S_val, S_test=S_test,
                        P_train=P_train, P_val=P_val, P_test=P_test
    )
    print(f"Saved File to {dp.databases_out_directory + dp.dataset_name}.npz")

def deap_dataset_load(filename):
    """
    filename = dp.databases_out_directory + dp.dataset_name + ".npz"
    """
    # 加载DEAP数据集
    print("Loading dataset ......")
    data = np.load(filename)
    # 展示数据集内容
    # print(data.files)
    X = data['X']
    y = data['y']
    X_train = data['X_train']
    X_val = data['X_val']
    X_test =data['X_test']
    Y_train = data['Y_train']
    Y_val = data['Y_val']
    Y_test = data['Y_test']
    S_train = data['S_train']
    S_val = data['S_val']
    S_test = data['S_test']
    P_train = data['P_train']
    P_val = data['P_val']
    P_test = data['P_test']
    print("Loading complete")

    # print(f"X shape: {X.shape}")
    # print(f"y shape: {y.shape}")
    # print(f"y unique: {np.unique(y)}")

if __name__ == "__main__":
    
    # 定义一个数据预处理的类
    DeapDP = DatabasesPreprocessing(config=deap_config)
    # 对 DEAP 数据集进行预处理并保存处理后的数据集
    deap_preprocessing(DatabasesPre=DeapDP)

