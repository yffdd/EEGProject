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
dataset_name = "deap_class4"
databases_root_directory = r"/bigdisk/322xcq/Databases/RawData/DEAP/data_preprocessed_python"
databases_out_directory = r"/bigdisk/322xcq/Databases/OutData/DEAP/ACSE/"
baseline_removal_window = 3
cutoff_frequencies = [4,40]
seconds_to_use = 60
downsampling_rate = 128
channels_to_use = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
window_size = 2
window_overlap = 0
convert_labels_to_nnp = False
save_plots_to_file = True

sampling_rate = 128
channel_names = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2', 'CZ', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

X = np.empty((1280,32,8064))
X_len = np.empty((1280,), dtype=int)
Y = np.empty((1280,4))
trial = np.empty((1280), dtype=np.int8)
subject = np.empty((1280,), dtype=np.int8)
session = np.ones((1280,), dtype=np.int8)


import sys
np.set_printoptions(threshold=sys.maxsize)
file_id = 0
for index, filename in tqdm(enumerate(sorted(os.listdir(databases_root_directory)),start=1)):
    if filename.endswith(".dat"):
        temp_file = pickle.load(open(os.path.join(databases_root_directory, filename), 'rb'), encoding='iso-8859-1')
        X_len[file_id*40:file_id*40+40] = np.ones((40,))*temp_file['data'].shape[2]
        X[file_id*40:file_id*40+40,:,:] = temp_file['data'][:,:32,:]
        Y[file_id*40:file_id*40+40,:] = temp_file['labels']
        subject[file_id*40:file_id*40+40] = np.ones((40,), dtype=np.int8)* (file_id+1)
        trial[file_id*40:file_id*40+40] = np.arange(1,41)
        file_id = file_id +1


print("Shape of the Time Series Array X: " + str(X.shape))
print("Shape of the Time Series Array Y: " + str(Y.shape))
print("Unique Session Indices: " + str(np.unique(session)))
print("Unique Subject Indices: " + str(np.unique(subject)))
print("Unique Trial Indices: " + str(np.unique(trial)))
print("Minimum length of Timeseries: " + str(min(X_len)))
print("Maximum length of Timeseries: " + str(max(X_len)))


# X_raw is later used for plotting, if you don't want to see the plots, you can uncomment this line
X_raw = X.copy()

if not(baseline_removal_window==0):
    baseline_datapoints = baseline_removal_window * sampling_rate
    baseline = X[:,:,:baseline_datapoints].sum(2) / baseline_datapoints
    for timestep in trange(X.shape[2]):
        X[:,:,timestep] = X[:,:,timestep] - baseline

from scipy.signal import butter, sosfilt, sosfreqz

def butter_bandpass(lowcut, highcut, fs, btype='band', order=5):
        nyq = 0.5 * fs
        if btype == 'bandpass':
            low = lowcut / nyq
            high = highcut / nyq
            sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
        elif btype == 'highpass':
            low = lowcut / nyq
            sos = butter(order, low, analog=False, btype='highpass', output='sos')
        elif btype == 'lowpass':
            high = highcut / nyq
            sos = butter(order, high, analog=False, btype='lowpass', output='sos')
        return sos

def butter_bandpass_filter(X, lowcut, highcut, fs, btype='bandpass', order=5):
        sos = butter_bandpass(lowcut, highcut, fs, btype=btype, order=order)
        X = sosfilt(sos, X)
        return X


if not(cutoff_frequencies[0] == None):
    if not(cutoff_frequencies[1] == None):
        btype='bandpass'
    else:
        btype='highpass'
elif not (cutoff_frequencies[1] == None):
        btype='lowpass'

for experiment_id in trange(X.shape[0]):
    for channel_id in range(X.shape[1]):
        X[experiment_id, channel_id, :] = butter_bandpass_filter(
                                                        X[experiment_id, channel_id, :],
                                                        cutoff_frequencies[0],
                                                        cutoff_frequencies[1],
                                                        sampling_rate,
                                                        btype=btype,
                                                        order=5)


print(f"x: {X.shape}")


if not(seconds_to_use == None):
    num_sample_points_to_use = seconds_to_use * sampling_rate
    X_selected = np.zeros((X.shape[0], X.shape[1], num_sample_points_to_use))
    for exp_id in trange(len(X_len)):
        X_selected[exp_id,:,:] = X[exp_id,:,X_len[exp_id]-num_sample_points_to_use:X_len[exp_id]]
    X = X_selected


if not(downsampling_rate == 0) and not(downsampling_rate == sampling_rate):
    new_length = int(X.shape[2] / sampling_rate * downsampling_rate)
    X_downsampled = np.zeros((X.shape[0], X.shape[1], new_length))
    for experiment_id in trange(X.shape[0]):
        for channel_id in range(X.shape[1]):
            X_downsampled[experiment_id, channel_id, :] = resample(X[experiment_id, channel_id, :], new_length)
    X = X_downsampled

if channels_to_use == None:
    channels_to_use = channel_names
print(channels_to_use)


channel_index_list = list()
for i in range(len(channels_to_use)):
    if channels_to_use[i] in channel_names:
        channel_index_list.append(channel_names.index(channels_to_use[i]))
    else:
        warnings.warn(' Channel ' + channels_to_use[i] +' could not be found in the list of actual channels')
print(channel_index_list)
print(len(channel_index_list))


X_selected_channels = np.zeros((X.shape[0], len(channels_to_use), X.shape[2]))
for channel in trange(len(channel_index_list)):
    X_selected_channels[:,channel,:] = X[:,channel_index_list[channel],:]
X = X_selected_channels



# 计算每个窗口的点数和每个窗口的重叠点数
num_points_per_window = window_size * downsampling_rate
num_points_overlap = window_overlap * downsampling_rate

# 计算窗口滑动的步长
stride = num_points_per_window - num_points_overlap

# 初始化起始和结束索引列表
start_index = [0]
end_index = [num_points_per_window]

# 初始化每个实验的窗口数
num_windows_per_exp = 1

# 计算每个实验的窗口数以及相应的起始和结束索引
while(end_index[-1] + stride < X.shape[2]):
    num_windows_per_exp += 1
    start_index.append(start_index[-1] + stride)
    end_index.append(end_index[-1] + stride)

# 初始化切割后的数据数组
X_cut = np.zeros((num_windows_per_exp * X.shape[0], X.shape[1], num_points_per_window))
Y_cut = np.zeros((num_windows_per_exp * X.shape[0], 4))
session_cut = np.zeros(num_windows_per_exp * X.shape[0])
subject_cut = np.zeros(num_windows_per_exp * X.shape[0])
trial_cut = np.zeros(num_windows_per_exp * X.shape[0])

# 遍历每个实验
for exp_id in trange(X.shape[0]):
    # 遍历每个窗口
    for window_id in range(len(start_index)):
        # 根据窗口的起始和结束索引切割数据
        X_cut[exp_id * num_windows_per_exp + window_id, :, :] = X[exp_id, :, start_index[window_id]:end_index[window_id]]
        # 复制标签、会话、受试者和试验信息
        Y_cut[exp_id * num_windows_per_exp + window_id, :] = Y[exp_id, :]
        session_cut[exp_id * num_windows_per_exp + window_id] = session[exp_id]
        subject_cut[exp_id * num_windows_per_exp + window_id] = subject[exp_id]
        trial_cut[exp_id * num_windows_per_exp + window_id] = trial[exp_id]

# 更新原始数据为切割后的数据
X = X_cut
Y = Y_cut
session = session_cut
subject = subject_cut
trial = trial_cut


def sort_centeroids(centeroids):
    # calculates the distance between given centeroids
    # and the corners bottom-right, bottom-left, top-left, top-right
    # and returns them in the order that the point closest to the br corner is returned first,
    # the one closest to the bl second,
    # the one clostest to the tl third,
    # and the one closest to the top right fourth
    
    distance_br = np.zeros(centeroids.shape[0])
    distance_bl = np.zeros(centeroids.shape[0])
    distance_tl = np.zeros(centeroids.shape[0])
    distance_tr = np.zeros(centeroids.shape[0])
    
    for i in range(centeroids.shape[0]):
        distance_br[i] = abs((1-centeroids[i,0])**2 + (-1-centeroids[i,1])**2)
        distance_bl[i] = abs((-1-centeroids[i,0])**2 + (-1-centeroids[i,1])**2)
        distance_tl[i] = abs((-1-centeroids[i,0])**2 + (1-centeroids[i,1])**2)
        distance_tr[i] = abs((1-centeroids[i,0])**2 + (1-centeroids[i,1])**2)
    
    br_idx = np.argmin(distance_br)
    bl_idx = np.argmin(distance_bl)
    tl_idx = np.argmin(distance_tl)
    tr_idx = np.argmin(distance_tr)
    
    return br_idx, bl_idx, tl_idx, tr_idx


scaler = MinMaxScaler(feature_range=(-1,1))
valence = scaler.fit_transform(Y[:,0].reshape(-1,1))
arousal = scaler.fit_transform(Y[:,1].reshape(-1,1))
datapoints = np.concatenate((valence,arousal),1)


kmeans = KMeans(n_clusters=4, random_state=7).fit(datapoints)

sad_label, fear_label, neutral_label, happy_label = sort_centeroids(kmeans.cluster_centers_)


plt.scatter(valence,arousal, c=kmeans.labels_, edgecolors='none')
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.title('DEAP - NNP Label via K-Means')
plt.show()

print('Fear(%i): %i -- Sad(%i): %i -- Neutral(%i): %i -- Happy(%i): %i'
                % (fear_label,    np.count_nonzero(kmeans.labels_ == fear_label),  # 统计恐惧类标签的数据点数量
                    sad_label,      np.count_nonzero(kmeans.labels_ == sad_label),   # 统计悲伤类标签的数据点数量
                    neutral_label,  np.count_nonzero(kmeans.labels_ == neutral_label),  # 统计中性类标签的数据点数量
                    happy_label,    np.count_nonzero(kmeans.labels_ == happy_label)))  # 统计快乐类标签的数据点数量

idx_sad = np.where(kmeans.labels_==sad_label)
idx_fear = np.where(kmeans.labels_==fear_label)
idx_neutral = np.where(kmeans.labels_==neutral_label)
idx_happy = np.where(kmeans.labels_==happy_label)





Y_nnp = np.zeros(Y.shape[0],)
Y_nnp[idx_sad] = -1
Y_nnp[idx_fear] = -1
Y_nnp[idx_neutral] = 0
Y_nnp[idx_happy] = 1



if convert_labels_to_nnp:
    Y = Y_nnp



np.savez_compressed(
                    databases_out_directory + dataset_name + '.npz',
                    X=X,
                    y=Y,
                    session = session,
                    subject = subject,
                    trial = trial,
                    downsampling_rate = downsampling_rate,
                    channel_names = channels_to_use,
                    window_size=window_size,
                    window_overlap = window_overlap,
                    cutoff_frequencies = cutoff_frequencies,
                    baseline_removal_window = baseline_removal_window,
                    seconds_to_use = seconds_to_use
                    )
print('Saved File')


