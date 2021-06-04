# -*- coding: utf-8 -*-

# @Time  : 2020/2/14 下午4:25
# @Author : upcbdipt
# @Project : CDW_FedAvg
# @FileName: ploting

import matplotlib.pyplot as plt
import os
import numpy as np

from scipy.spatial import distance

path = os.path.join('..', 'results')


def plot(client):
    feature_path = os.path.join('data', 'reduce', client + '.npy')
    target_path = os.path.join('data', 'test', client + '.npy')
    save_path = os.path.join('results', client + '.pdf')
    feature = np.load(feature_path)
    target = np.load(target_path)
    data_1 = np.hstack((feature, target[:, -1].reshape(-1, 1)))
    np.random.shuffle(data_1)
    data_1 = data_1[:1000]
    normal = data_1[data_1[:, -1] == 0]
    abnormal = data_1[data_1[:, -1] == 1]
    plt.scatter(normal[:, 0], normal[:, 1], c='b', s=8)
    plt.scatter(abnormal[:, 0], abnormal[:, 1], c='r', s=8)
    plt.legend(['normal', 'abnormal'])
    plt.savefig(save_path)
    plt.close('all')
