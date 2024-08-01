"""
File: CNN.py
Author: 李颖
Date: 2024-07-01
Description: TensorFlow实现卷积神经网络
"""

import tensorflow as tf


# =======================================直接调用的方式=======================================
'''
卷积层1: 包含128个滤波器，滤波器大小为50，步长为3，使用ReLU激活函数。
批归一化层1: 对卷积层1的输出进行批归一化处理。
最大池化层1: 池化窗口大小为2，步长为3，对批归一化层1的输出进行池化操作。
卷积层2: 包含32个滤波器，滤波器大小为7，步长为1，使用ReLU激活函数。
批归一化层2: 对卷积层2的输出进行批归一化处理。
最大池化层2: 池化窗口大小为2，步长为2，对批归一化层2的输出进行池化操作。
卷积层3: 包含32个滤波器，滤波器大小为10，步长为1，使用ReLU激活函数。
卷积层4: 包含128个滤波器，滤波器大小为5，步长为2，使用ReLU激活函数。
最大池化层3: 池化窗口大小为2，步长为2，对卷积层4的输出进行池化操作。
卷积层5: 包含512个滤波器，滤波器大小为5，步长为1，使用ReLU激活函数。
卷积层6: 包含128个滤波器，滤波器大小为3，步长为1，使用ReLU激活函数。
展开层: 将卷积层6的输出展平为一维向量。
全连接层1: 包含512个神经元，使用ReLU激活函数。
Dropout层: 以0.1的概率对全连接层1进行Dropout操作。
全连接层2: 输出层，包含7个神经元，使用softmax激活函数，用于进行心电异常的分类预测。
'''

def CNN1d(input_ecg):
    layer1 = tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same', activation=tf.nn.relu)(
        input_ecg)
    BachNorm = tf.keras.layers.BatchNormalization()(layer1)
    MaxPooling1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(BachNorm)
    layer2 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(
        MaxPooling1)
    BachNorm = tf.keras.layers.BatchNormalization()(layer2)
    MaxPooling2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(BachNorm)
    layer3 = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(
        MaxPooling2)
    layer4 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)(
        layer3)
    MaxPooling3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(layer4)
    layer5 = tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(
        MaxPooling3)
    layer6 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(
        layer5)
    flat = tf.keras.layers.Flatten()(layer6)
    x = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)(flat)
    x = tf.keras.layers.Dropout(rate=0.1)(x)
    label_ecg = tf.keras.layers.Dense(units=3, activation=tf.nn.softmax)(x)
    return label_ecg


# ======================================================自定义类的方式==============================
'''
这个模型的输入形状取决于数据集的特征数量和时间步数，而输出形状是根据分类任务的类别数量而定。假设输入数据的形状为 (batch_size, seq_length, num_features)，其中: 

batch_size 是每个批次中的样本数量，
seq_length 是每个样本序列的长度（或时间步数），
num_features 是每个时间步上的特征数量。
那么模型的输入形状即为 (batch_size, seq_length, num_features)。

输出形状取决于分类任务的类别数量，通常是一个具有 num_classes 个元素的向量，其中每个元素表示样本属于对应类别的概率。因此，输出形状为 (batch_size, num_classes)。

综上所述，这个模型的输入形状是 (batch_size, seq_length, num_features)，输出形状是 (batch_size, num_classes)。
'''


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same',
                                             activation=tf.nn.relu)
        self.BachNorm1 = tf.keras.layers.BatchNormalization()
        self.MaxPooling1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)
        self.layer2 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same',
                                             activation=tf.nn.relu)
        self.BachNorm2 = tf.keras.layers.BatchNormalization()
        self.MaxPooling2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.layer3 = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same',
                                             activation=tf.nn.relu)
        self.BachNorm3 = tf.keras.layers.BatchNormalization()
        self.layer4 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same',
                                             activation=tf.nn.relu)
        self.BachNorm4 = tf.keras.layers.BatchNormalization()
        self.MaxPooling3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.layer5 = tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same',
                                             activation=tf.nn.relu)
        self.BachNorm5 = tf.keras.layers.BatchNormalization()
        self.layer6 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same',
                                             activation=tf.nn.relu)
        self.BachNorm6 = tf.keras.layers.BatchNormalization()
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.Dropout = tf.keras.layers.Dropout(rate=0.1)
        self.outputSoftmax = tf.keras.layers.Dense(units=3, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.BachNorm1(x)
        x = self.MaxPooling1(x)
        x = self.layer2(x)
        x = self.BachNorm2(x)
        x = self.MaxPooling2(x)
        x = self.layer3(x)
        x = self.BachNorm3(x)
        x = self.layer4(x)
        x = self.BachNorm4(x)
        x = self.MaxPooling3(x)
        x = self.layer5(x)
        x = self.BachNorm5(x)
        x = self.layer6(x)
        x = self.BachNorm6(x)
        x = self.flat(x)
        x = self.dense(x)
        x = self.Dropout(x)
        output = self.outputSoftmax(x)
        return output

# class CNN(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same',
#                                              activation=tf.nn.relu)
#         self.BachNorm1 = tf.keras.layers.BatchNormalization()
#         self.MaxPooling1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)
#         self.layer2 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same',
#                                              activation=tf.nn.relu)
#         self.BachNorm2 = tf.keras.layers.BatchNormalization()
#         self.MaxPooling2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
#         self.layer3 = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same',
#                                              activation=tf.nn.relu)
#         self.layer4 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same',
#                                              activation=tf.nn.relu)
#         self.MaxPooling3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
#         self.layer5 = tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same',
#                                              activation=tf.nn.relu)
#         self.layer6 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same',
#                                              activation=tf.nn.relu)
#         self.flat = tf.keras.layers.Flatten()
#         self.dense = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
#         self.Dropout = tf.keras.layers.Dropout(rate=0.1)
#         self.outputSoftmax = tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)
#
#     def call(self, inputs):
#         x = self.layer1(inputs)
#         x = self.BachNorm1(x)
#         x = self.MaxPooling1(x)
#         x = self.layer2(x)
#         x = self.BachNorm2(x)
#         x = self.MaxPooling2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.MaxPooling3(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.flat(x)
#         x = self.dense(x)
#         x = self.Dropout(x)
#         output = self.outputSoftmax(x)
#         return output

# ======================================================sequential的方式==============================================
def SeqCNN():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=3),
        tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)
    ])
