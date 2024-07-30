from scipy.signal import butter, filtfilt, iirnotch


def apply_bandstop_filter(signal, lowcut, highcut, fs, order=2):
    """
    设计并应用带阻滤波器.
    
    :param signal: 输入信号
    :param lowcut: 带阻滤波器低截止频率
    :param highcut: 带阻滤波器高截止频率
    :param fs: 采样频率
    :param order: 滤波器阶数
    :return: 滤波后的信号
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    
    # 应用带阻滤波器
    y = filtfilt(b, a, signal)
    return y


def apply_lowpass_filter(data, cutoff, fs, order=2):
    """
    设计并应用低通滤波器.
    
    :param data: 输入信号
    :param cutoff: 截止频率
    :param fs: 采样频率
    :param order: 滤波器阶数
    :return: 滤波后的信号
    """
    # 设计低通滤波器
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # 应用低通滤波器
    y = filtfilt(b, a, data)
    return y


def apply_highpass_filter(data, cutoff, fs, order=2):
    """
    设计并应用高通滤波器.
    
    :param data: 输入信号
    :param cutoff: 截止频率
    :param fs: 采样频率
    :param order: 滤波器阶数
    :return: 滤波后的信号
    """
    # 设计高通滤波器
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    # 应用高通滤波器
    y = filtfilt(b, a, data)
    return y


def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    应用带通滤波器以通过特定频率范围的信号。

    参数:
    - data: 输入信号数据（数组类型）。
    - lowcut: 带通滤波器的低截止频率(Hz)。
    - highcut: 带通滤波器的高截止频率(Hz)。
    - fs: 采样频率(Hz)。
    - order: 滤波器的阶数(默认为4)。

    返回:
    - filtered_data: 应用带通滤波器后的信号。
    """

    # 计算带通滤波器的参数
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyquist  # 低截止频率归一化
    high = highcut / nyquist  # 高截止频率归一化
    # 设计带通滤波器
    b, a = butter(order, [low, high], btype='band')

    # 对数据应用滤波器
    y = filtfilt(b, a, data)  # 零相位滤波
    return y


def apply_notch_filter(data, notch_freq, fs, Q=4):
    """
    应用陷波滤波器以去除特定频率的噪声。

    参数:
    - data: 输入信号数据（数组类型）。
    - freq: 需要去除的频率(Hz)。
    - fs: 采样频率(Hz)。
    - order: 滤波器的阶数(默认为4)。

    返回:
    - y: 应用陷波滤波器后的信号。
    """

    # 计算陷波滤波器的参数
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = notch_freq / nyquist  # 归一化的陷波频率
    b, a = iirnotch(normal_cutoff, Q=Q)  # 设计陷波滤波器

    # 对数据应用滤波器
    y = filtfilt(b, a, data)  # 零相位滤波
    return y

