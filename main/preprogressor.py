# -*- coding: utf-8 -*-

# @Time  : 2020/1/8 下午2:38
# @Author : fl
# @Project : HaierDataMining
# @FileName: 1

import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import AllKNN

from haier_data_mining.utils.helpers import Config

data_path = '../data'


class PreProgressor:
    def __init__(self, config):
        self.config = config

    def run(self):
        raw_data_path = os.path.join(data_path, 'raw')
        files = os.listdir(raw_data_path)
        files = [f for f in files if (f.endswith('.xlsx') or f.endswith('.xls'))]
        all_train_data = None
        all_test_data = None
        for f in files:
            file_path = os.path.join(raw_data_path, f)
            # 加载原始数据
            data = pd.read_excel(file_path)
            # 每一个取前10W条数据
            data = data[0:100000]
            # 去掉时间列
            data = data.drop('数据时间', axis=1)
            # 归一化到[0,1]
            data = (data - data.min()) / (data.max() - data.min())
            # 删除NAN列
            data = data.dropna(axis=1)
            # 转化为numpy数组
            data = data.values
            # 进行平衡化处理
            X = data[:, :-1]
            y = data[:, -1]
            smote_enn = SMOTEENN(random_state=0)
            X_resampled, y_resampled = smote_enn.fit_resample(X, y)
            data_resampled = np.hstack((X_resampled, y_resampled.reshape(len(y_resampled), 1)))
            # 75%作为训练数据，25%作为测试数据
            train_data, test_data = train_test_split(data_resampled)
            if all_train_data is None:
                all_train_data = train_data
                all_test_data = test_data
            else:
                all_train_data = np.vstack((all_train_data, train_data))
                all_test_data = np.vstack((all_test_data, test_data))
            file_name = f.split('.')[0] + '.npy'
            # 按设备保存文件
            np.save(os.path.join(data_path, 'train', file_name), train_data)
            np.save(os.path.join(data_path, 'test', file_name), test_data)
            print(file_name)
        # 合并所有场
        # all_data = np.vstack((all_train_data, all_test_data))
        # X = all_data[:, :-1]
        # y = all_data[:, -1]
        # all_knn = SMOTEENN(random_state=0)
        # X_resampled, y_resampled = all_knn.fit_resample(X, y)
        # print(X.shape)
        # print(y.shape)
        # print(X_resampled.shape)
        # print(y_resampled.shape)
        # all_data_resampled = np.hstack((X_resampled, y_resampled.reshape(len(y_resampled), 1)))
        # print(all_data_resampled.shape)
        # train_data, test_data = train_test_split(all_data_resampled)
        np.save(os.path.join(data_path, 'all', 'train', 'all_combine_sampling.npy'), all_train_data)
        np.save(os.path.join(data_path, 'all', 'test', 'all_combine_sampling.npy'), all_test_data)


# config = Config('config.yaml')
# progressor = PreProgressor(config)
# progressor.run()
# data = np.load(os.path.join('../data', 'all', 'test', 'all_combine_sampling.npy'))
# y = data[:, -1]
# print(sorted(Counter(y).items()))
# raw_data_path = os.path.join(data_path, 'raw', '山水大酒店2.csv')
# data = pd.read_csv(raw_data_path)
# data = data[20000:30000]
# data = data[data.columns[1]]
# value = data.values
# value = value.reshape(len(value), 1)
# test_data, train_data = train_test_split(value, test_size=0.5, shuffle=False)
# print(train_data.shape)
# print(test_data.shape)
# np.save(os.path.join(data_path, 'train', 'F3.npy'), train_data)
# np.save(os.path.join(data_path, 'validate', 'F3.npy'), train_data)
# np.save(os.path.join(data_path, 'test', 'F3.npy'), test_data)
# iris = load_iris()
# data = iris.data
# target = iris.target
# target = np.reshape(target, (-1, 1))
# data = np.hstack((data, target))
# 同一个厂平均分为五分
# data = np.load(os.path.join(data_path, 'raw', 'balanced_all.npy'))
# scaler = MinMaxScaler()
# data = scaler.fit_transform(data)
# train_data, test_data = train_test_split(data)
# for i in range(5):
#     partial_train_data = train_data[i * int(len(train_data) / 5):(i + 1) * int(len(train_data) / 5)]
#     partial_test_data = test_data[i * int(len(test_data) / 5):(i + 1) * int(len(test_data) / 5)]
#     print(len(partial_test_data))
#     print(len(partial_train_data))
#     np.save(os.path.join(data_path, 'train', 'F' + str(i) + '.npy'), partial_train_data)
#     np.save(os.path.join(data_path, 'test', 'F' + str(i) + '.npy'), partial_test_data)
# 四个不同厂
def prepare_data(name):
    data = np.load(os.path.join(data_path, 'raw', name))
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    train_data, test_data = train_test_split(data)
    print(name)
    print(train_data.shape)
    print(test_data.shape)
    np.save(os.path.join(data_path, 'train', 'Hetero_' + name), train_data)
    np.save(os.path.join(data_path, 'test', 'Hetero_' + name), test_data)


# prepare_data('F1_Equal.npy')
# prepare_data('F2_Equal.npy')
# prepare_data('F3_Equal.npy')
# prepare_data('F4_Equal.npy')

# #数据量相同四个厂合并,
F1_Equal = np.load(os.path.join(data_path, 'raw', 'F1_Equal.npy'))
F2_Equal = np.load(os.path.join(data_path, 'raw', 'F2_Equal.npy'))
F3_Equal = np.load(os.path.join(data_path, 'raw', 'F3_Equal.npy'))
F4_Equal = np.load(os.path.join(data_path, 'raw', 'F4_Equal.npy'))
All_Equal = np.vstack((F1_Equal, F2_Equal, F3_Equal, F4_Equal))
scaler = MinMaxScaler()
data = scaler.fit_transform(All_Equal)
train_data, test_data = train_test_split(data)
print(train_data.shape)
print(test_data.shape)
np.save(os.path.join(data_path, 'all', 'train', 'Hetero_All_Equal.npy'), train_data)
np.save(os.path.join(data_path, 'all', 'test', 'Hetero_All_Equal.npy'), test_data)
# # 数据量不同四个厂合并,
# F1 = np.load(os.path.join(data_path, 'raw', 'F1.npy'))
# F2 = np.load(os.path.join(data_path, 'raw', 'F2.npy'))
# F3 = np.load(os.path.join(data_path, 'raw', 'F3.npy'))
# F4 = np.load(os.path.join(data_path, 'raw', 'F4.npy'))
# All = np.vstack((F1, F2, F3, F4))
# scaler = MinMaxScaler()
# data = scaler.fit_transform(All)
# train_data, test_data = train_test_split(data)
# print(train_data.shape)
# print(test_data.shape)
# np.save(os.path.join(data_path, 'all', 'train', 'Hetero_All.npy'), train_data)
# np.save(os.path.join(data_path, 'all', 'test', 'Hetero_All.npy'), test_data)
