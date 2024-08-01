"""
File: CNN.py
Author: 彭俊文
Date: 2024-08-01
Description: CNN-LSTM 模型
"""

import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, hinge_loss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
import torch.distributed


DIR_DATA = "B:/ruanzhu/CNN-SVM/WESAD/dataset/"
DIR_NET_SAVING = "B:/ruanzhu/CNN-SVM/WESAD/net/"
DIR_RESULTS = "B:/ruanzhu/CNN-SVM/WESAD/data/"
# 设置随机种子以确保结果可复现
manualSeed = 1
torch.manual_seed(manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)
g = torch.Generator()
g.manual_seed(manualSeed)
def suppr(dic):
    """ 从字典中删除极端值（由于异常峰值检测引起的无关数据）
    """
    # 计算特征的 99% 和 1% 分位数
    bornemax = np.quantile(dic["features"], 0.99, axis=0)
    bornemin = np.quantile(dic["features"], 0.01, axis=0)

    # 找到异常值的索引
    indicesmauvais = \
    np.where(np.sum(np.add(bornemin > np.array(dic["features"]), np.array(dic["features"]) > bornemax), axis=1) > 0)[0]
    k = 0
    # 删除这些异常值
    for i in indicesmauvais:
        del dic["features"][i - k]
        del dic["label"][i - k]
        k += 1

    return dic

def extract_ds_from_dict(data):
    """ 删除每种状态的无关数据并返回数据集字典
    """
    Letat = []
    # 将数据分为四个状态：中性、压力、娱乐、冥想
    for i in range(0, 4):
        dictio = {}
        features = [data["features"][j] for j in np.where(np.array(data["label"]) == i + 1)[0]]
        label = [data["label"][j] for j in np.where(np.array(data["label"]) == i + 1)[0]]
        dictio["features"] = features
        dictio["label"] = label
        Letat.append(dictio.copy())
    # 分别处理四个状态的数据
    neutr = Letat[0]
    stress = Letat[1]
    amu = Letat[2]
    med = Letat[3]
    neutr = suppr(neutr)
    stress = suppr(stress)
    amu = suppr(amu)
    med = suppr(med)
    features = []
    label = []
    dict_id = {}

    # 合并所有状态的数据
    for m in range(0, 4):
        dictio = Letat[m]
        features += [x for x in dictio["features"]]
        label += [x for x in dictio["label"]]

    dict_id["features"] = features
    dict_id["label"] = label
    return dict_id.copy()


def conf_mat(net, datal, trsh):
    """ 根据数据加载器和阈值计算两个混淆矩阵（一个是压力/非压力分类，一个是状态分类）
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = datal[0].float().to(device)
    y = net(x).squeeze()
    pred = (y > trsh).int()  # 根据阈值进行二分类预测
    label = datal[1].float().to(device).view(-1).int()
    num = datal[2].float().to(device).int()
    comp = torch.eq(label, pred).int()
    mat_label = np.zeros((2, 4))
    mat_nolbl = np.zeros((2, 2))

    # 构建混淆矩阵
    for i in range(0, 4):
        tens = torch.where(num == i + 1, 1, 0)
        numtot = torch.sum(tens).item()
        num_G = torch.sum(torch.where(torch.mul(tens, comp) == 1, 1, 0)).item()

        if i == 1:
            mat_nolbl[0, 0] += num_G
            mat_nolbl[1, 0] += numtot - num_G
            mat_label[0, i] = num_G
            mat_label[1, i] = numtot - num_G
        else:
            mat_nolbl[1, 1] += num_G
            mat_nolbl[0, 1] += numtot - num_G
            mat_label[1, i] = num_G
            mat_label[0, i] = numtot - num_G

    return mat_label, mat_nolbl


def fusion_dic(list_dic):
    """ 合并字典（数据集），每个字典代表一个个体的数据集
    """
    features = []
    label = []
    dic_f = {}

    for dic in list_dic:
        features += dic["features"]
        label += dic["label"]

    dic_f["features"] = features
    dic_f["label"] = label
    return dic_f

def proportion(dic, indice, prop):
    """ 返回一个平衡的数据集（按比例缩小）以便于平衡训练
    """
    tot = len(indice)
    features = [dic["features"][j] for j in indice[::int(np.ceil(tot / prop))]]
    label = [dic["label"][j] for j in indice[::int(np.ceil(tot / prop))]]
    return features, label

def eq_dic(dic):
    """ 返回一个平衡的数据集，确保中性/压力/娱乐/冥想条件的数据量相同
    """
    indice_neutr = np.where(np.array(dic["label"]) == 1)[0]
    indice_stress = np.where(np.array(dic["label"]) == 2)[0]
    indice_amu = np.where(np.array(dic["label"]) == 3)[0]
    indice_med = np.where(np.array(dic["label"]) == 4)[0]
    nbr_neutr = len(indice_neutr)
    nbr_stress = len(indice_stress)
    nbr_amu = len(indice_amu)
    nbr_med = len(indice_med)
    prop = min([3 * nbr_neutr, nbr_stress, 3 * nbr_amu, 3 * nbr_med])
    prop_stress = prop
    prop_neutr = int(0.333 * prop)
    prop_amu = int(0.333 * prop)
    prop_med = int(0.333 * prop)
    features = []
    label = []
    dic_f = {}
    tempf, templ = proportion(dic, indice_neutr, prop_neutr)
    features += tempf
    label += templ
    tempf, templ = proportion(dic, indice_stress, prop_stress)
    features += tempf
    label += templ
    tempf, templ = proportion(dic, indice_amu, prop_amu)
    features += tempf
    label += templ
    tempf, templ = proportion(dic, indice_med, prop_med)
    features += tempf
    label += templ
    dic_f["features"] = features
    dic_f["label"] = label
    return dic_f
class ds_wesad(Dataset):
    """ 定义 WESAD 数据集对象 (特征向量; 是否有压力(0/1); 情感状态(0;1;2;3)

    0 平静
    1 紧张
    2 兴奋
    3 放松
    """

    def __init__(self, dic):
        self.samples = []
        self.dic = dic
        for i in range(0, len(dic["label"])):
            num = dic["label"][i]
            stress = num == 2
            x = np.array(dic["features"][i])
            self.samples.append((x, int(stress), num))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, id):
        return self.samples[id]

"""处理交叉验证过程，为交叉验证生成多个训练和验证数据字典"""

name_list = ['WESADWESADECG_S2.json', 'WESADWESADECG_S3.json', 'WESADWESADECG_S4.json', 'WESADWESADECG_S5.json', 'WESADWESADECG_S6.json', 'WESADWESADECG_S7.json',
 'WESADWESADECG_S8.json', 'WESADWESADECG_S9.json', 'WESADWESADECG_S10.json', 'WESADWESADECG_S11.json', 'WESADWESADECG_S13.json', 'WESADWESADECG_S14.json',
 'WESADWESADECG_S15.json', 'WESADWESADECG_S16.json']

# assert (len(name_list)==14)  ONE Subject for testing S17
list_dic_ds = []
cntr = 0

for k in range(0, 14):
    for j in range(k, 14):
        if k != j:
            f1 = open(DIR_DATA + name_list[k])
            f2 = open(DIR_DATA + name_list[j])
            data_1 = json.load(f1)
            data_2 = json.load(f2)
            dic_3 = fusion_dic([data_1, data_2])
            dic_v = extract_ds_from_dict(dic_3)  # 创建由2个受试者的数据组成的验证字典（由于传感器故障/不可能的值，某些受试者的数据被删除）
            L = []

            for i in range(0, len(name_list)):
                if (i != k and i != j):
                    f = open(DIR_DATA + name_list[i])
                    data = json.load(f)
                    dic = eq_dic(data)
                    L.append(dic)

            assert (len(L) == 12)
            dic_4 = fusion_dic(L)
            dic_t = extract_ds_from_dict(dic_4)
            list_dic_ds.append([dic_t, dic_v, k, j])  # 创建由12个受试者的数据组成的训练字典，所有数据都经过平衡并删除了不可能的（物理上）值
            cntr += 1
            if cntr % 10 == 0:
                print(cntr)

"""为每个数据字典生成多个数据集对象（第一个是训练数据集，第二个是验证数据集）"""
list_ds = []
cntr = 0
for sample in list_dic_ds:
    list_ds.append([ds_wesad(sample[0]), ds_wesad(sample[1]), sample[2], sample[3]])  # [训练数据集; 验证数据集（来自受试者k和j）; k; j]
    cntr += 1


""" DNN 模型 """

def init_weight(m):
    """权重初始化"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)    # 对线性层的权重进行Xavier均匀分布初始化
        m.bias.data.fill_(0.01)              # 将偏置初始化为0.01
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)               # 对BatchNorm层的权重初始化为1
        m.bias.data.zero_()                  # 将偏置初始化为0


import torch
import torch.nn as nn
import numpy as np
import random

class CNNLSTMECG(nn.Module):
    def __init__(self, ngpu, input_channels=12, hidden_size=256, num_layers=3, num_classes=1):
        super(CNNLSTMECG, self).__init__()
        self.ngpu = ngpu
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=1, stride=1),
        )

        # LSTM layers
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.5)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x 的形状为 [batch_size, features]

        # 添加时间步维度，将形状变为 [batch_size, 1, features]
        x = x.unsqueeze(1)

        # 转置为 [batch_size, features, 1]，以匹配Conv1d的输入要求
        x = x.transpose(1, 2)

        # CNN feature extraction
        x = self.cnn_layers(x)

        # 为LSTM准备输入 (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # 从 [batch_size, features, time] 变为 [batch_size, time, features]

        # LSTM layers
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # 我们取最后一个时间步的输出
        out = out[:, -1, :]

        # Fully connected layers
        out = self.fc(out)

        # Activation
        out = self.sigmoid(out)

        return out
# 初始化权重函数也需要相应修改
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

# 训练函数保持不变，只需要将ClassifierECG替换为CNNECG
def training(net, dataloader_t, dataloader_v, num_epochs, j, k):
    Loss = []
    Lossv = []
    for epoch in range(num_epochs):
        net.train()  # 确保在每个epoch开始时设置为训练模式
        L_t = []
        L_v = []
        for i, dataj in enumerate(dataloader_t, 0):
            optimizer.zero_grad()  # 使用optimizer.zero_grad()替代net.zero_grad()
            x = dataj[0].float().to(device)
            yhat = dataj[1].float().to(device)
            yhat = yhat.view(-1, 1)
            y = net(x)
            err_t = nn.BCELoss()(y.float(), yhat.float())
            err_t.backward()
            optimizer.step()
            L_t.append(err_t.item())

        net.eval()
        with torch.no_grad():
            for i, dataj in enumerate(dataloader_v, 0):
                x = dataj[0].float().to(device)
                yhat = dataj[1].float().to(device)
                yhat = yhat.view(-1, 1)
                y = net(x)
                err_v = nn.BCELoss()(y.float(), yhat.float())
                L_v.append(err_v.item())

        err = np.mean(L_t)
        errv = np.mean(L_v)
        Loss.append(err)
        Lossv.append(errv)
        torch.save(net.state_dict(), DIR_NET_SAVING + f"net_{j}_{k}_epoch_{epoch}.pth")

    return [Lossv, np.argmin(Lossv)]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



# 主程序
num_workers = 0
batch_size = 48
ngpu = 1
input_channels = 12  # 新增：指定输入通道数
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

L = []
cntr = 0

for ds in tqdm(list_ds):
    # 在主程序中
    net = CNNLSTMECG(ngpu, input_channels=input_channels).to(device)  # 修改：指定输入通道数
    net.apply(init_weights)
    lr = 0.0001
    beta1 = 0.9
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    dataset_t = ds[0]
    dataset_v = ds[1]
    k = ds[2]
    j = ds[3]
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               worker_init_fn=seed_worker, generator=g, drop_last=True)
    dataloader_v = torch.utils.data.DataLoader(dataset_v, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               worker_init_fn=seed_worker, generator=g, drop_last=True)

    L.append(training(net, dataloader_t, dataloader_v, 15, j, k))
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


with open(DIR_RESULTS + "results.json", "w") as file:
    json.dump(L, file, cls=NpEncoder)

with open(DIR_RESULTS + "results.json", "r") as file:
    L = json.load(file)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

num_workers = 0
batch_size = 32
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

"""
该单元格计算每个fold最佳模型的混淆矩阵（无标签和有标签）=>（网络预测为行标签，真实情感状态为列标签）
绘制的矩阵是所有混淆矩阵的均值+/-标准差（所有fold）

该单元格还计算每个fold模型的准确率、精确率、召回率和F1得分，并将这些指标存储在列表中
"""
confusionlabelmean = np.zeros((2, 4))
confusionmean = np.zeros((2, 2))
acc_list = []
prec_list = []
recall_list = []
f1_list = []
confusion_label_list = []
confusion_list = []

for n in range(0, len(L)):
    if n % 10 == 0:
        print(n)
    k = list_ds[n][2]
    j = list_ds[n][3]
    dataset_t = list_ds[n][0]
    dataset_v = list_ds[n][1]
    epch = np.argmin(L[n][0])
    net = CNNLSTMECG(ngpu, input_channels=input_channels).to(device)  # 修改：指定输入通道数
    lr = 0.0001
    beta1 = 0.9
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    net.load_state_dict(torch.load(DIR_NET_SAVING + "net_" + str(j) + "_" + str(k) + "_epoch_" + str(epch) + ".pth"))
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               worker_init_fn=seed_worker, generator=g, drop_last=True)
    dataloader_v = torch.utils.data.DataLoader(dataset_v, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               worker_init_fn=seed_worker, generator=g, drop_last=True)

    trsh = 0.5
    net.eval()
    confusionlabel = np.zeros((2, 4))
    confusion = np.zeros((2, 2))
    length_dsv = 0
    for i, datal in enumerate(dataloader_v, 0):
        confusionlabelt, confusiont = conf_mat(net, datal, trsh)
        confusion += confusiont
        confusionlabel += confusionlabelt
        length_dsv += batch_size

    TP = confusion[0, 0]
    TN = confusion[1, 1]
    FN = confusion[1, 0]
    FP = confusion[0, 1]
    acc = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = (2 * recall * precision) / (recall + precision)
    acc_list.append(acc)
    prec_list.append(precision)
    recall_list.append(recall)
    f1_list.append(F1score)
    confusion_label_list.append(100 * confusionlabel / length_dsv)
    confusion_list.append(100 * confusion / length_dsv)

# 首先计算了两个混淆矩阵的均值和标准差，然后使用Seaborn库的heatmap函数绘制了两个混淆矩阵的热图。接着，它使用Matplotlib库的hist函数绘制了准确率、精确度、召回率和F1分数的直方图，并在图表上添加了均值和标准差的文本。最后，使用plt.show()函数显示所有图表。
# 计算混淆矩阵的均值，并保留三位小数
confusionmean = np.round(np.mean(confusion_list, axis=0), 3)

# 计算混淆矩阵标签的均值，并保留三位小数
confusionlabelmean = np.round(np.mean(confusion_label_list, axis=0), 3)

# 计算混淆矩阵的标准差，并保留三位小数
confusionstd = np.round(np.std(confusion_list, axis=0), 3)

# 计算混淆矩阵标签的标准差，并保留三位小数
confusionlabelstd = np.round(np.std(confusion_label_list, axis=0), 3)

# 将混淆矩阵的均值和标准差组合成字符串形式，并重新调整为原始形状
annot_confusion = np.array([str(a)+"+/-"+str(b) for a, b in zip(confusionmean.reshape(-1).tolist(), confusionstd.reshape(-1).tolist())]).reshape(confusionmean.shape)

# 将混淆矩阵标签的均值和标准差组合成字符串形式，并重新调整为原始形状
annot_confusion_label = np.array([str(a)+"+/-"+str(b) for a, b in zip(confusionlabelmean.reshape(-1).tolist(), confusionlabelstd.reshape(-1).tolist())]).reshape(confusionlabelmean.shape)

# 设置x轴和y轴的标签
x_axis_confl = ['peacefu', 'nervous', 'excited', 'Relax']  # 情感状态标签
y_axis_confl = ['pressure', 'no pressure']  # 压力状态标签
x_axis_conf = ['pressure', 'no pressure']  # x轴标签
y_axis_conf = ['pressure', 'no pressure']  # y轴标签

# 设置图表的尺寸
sns.set(rc={"figure.figsize": (15, 5)})
# 创建一个包含两个子图的图表
fig, axs = plt.subplots(ncols=2, figsize=(33, 9))

# 绘制第一个混淆矩阵的热图，展示压力/无压力的分类结果
sns.heatmap(confusionmean.astype('int32'), xticklabels=x_axis_conf, yticklabels=y_axis_conf, annot=annot_confusion, ax=axs[0], fmt='')
axs[0].set_xlabel('The true situation')  # 设置x轴标签
axs[0].set_ylabel('Prediction')  # 设置y轴标签
axs[0].title.set_text('Confusion Matrix label : pressure/No pressure')  # 设置标题

# 绘制第二个混淆矩阵的热图，展示情绪状态的分类结果
sns.heatmap(confusionlabelmean.astype('int32'), xticklabels=x_axis_confl, yticklabels=y_axis_confl, annot=annot_confusion_label, ax=axs[1], fmt='')
axs[1].set_xlabel('The true situation')  # 设置x轴标签
axs[1].set_ylabel('Prediction')  # 设置y轴标签
axs[1].title.set_text('Confusion Matrix label : emotionnal state')  # 设置标题
plt.show()

# 绘制准确率的直方图
mes = acc_list
plt.hist(mes, bins=20)
plt.title('Accuracy of the models')  # 设置标题
plt.xlabel('Accuracy')  # 设置x轴标签
plt.ylabel('Number of models')  # 设置y轴标签
txtm = "mean: " + str(round(np.mean(mes), 3))  # 计算均值并格式化
txtstd = "std: " + str(round(np.std(mes), 3))  # 计算标准差并格式化
plt.text(0.85, 12, txtm)  # 在图表上添加均值文本
plt.text(0.85, 11, txtstd)  # 在图表上添加标准差文本
plt.show()  # 显示图表

# 重复上述步骤，分别绘制精确度、召回率和F1分数的直方图，并添加均值和标准差的文本
# ...
mes=prec_list
plt.hist(mes,bins=20)
plt.title('Precision of all best models')
plt.xlabel('Precision')
plt.ylabel('Number of models')
txtm="mean: " +str(round(np.mean(mes),3))
txtstd="std: " +str(round(np.std(mes),3))
plt.text(0.5, 10, txtm)
plt.text(0.5, 9, txtstd)
plt.show()

mes=recall_list
plt.hist(mes,bins=20)
plt.title('Recall of all best models')
plt.xlabel('Recall')
plt.ylabel('Number of models')
txtm="mean: " +str(round(np.mean(mes),3))
txtstd="std: " +str(round(np.std(mes),3))
plt.text(0.5, 19, txtm)
plt.text(0.5, 18, txtstd)
plt.show()

mes=f1_list
plt.hist(mes,bins=20)
plt.title('F1 score of all best models')
plt.xlabel('F1 score')
plt.ylabel('Number of models')
txtm="mean: " +str(round(np.mean(mes),3))
txtstd="std: " +str(round(np.std(mes),3))
plt.text(0.5, 12, txtm)
plt.text(0.5, 11, txtstd)
plt.show()
