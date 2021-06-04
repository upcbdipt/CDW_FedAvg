# -*- coding: utf-8 -*-

# @Time  : 2020/1/8 下午2:38
# @Author : upcbdipt
# @Project : CDW_FedAvg
# @FileName: preprogressor

import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


data_path = 'data'


def prepare_client_data(name):
    data = np.load(os.path.join(data_path, 'raw', name))
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    train_data, test_data = train_test_split(data)
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    np.save(os.path.join(train_path, 'Hetero_' + name), train_data)
    np.save(os.path.join(test_path, 'Hetero_' + name), test_data)


def prepare_server_data(clients):
    # merge data from all clients
    clients_data = []
    for client in clients:
        client_data = np.load(os.path.join(data_path, 'raw', client))
        clients_data.append(client_data)
    All_Equal = np.vstack(clients_data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(All_Equal)
    train_data, test_data = train_test_split(data)
    print(train_data.shape)
    print(test_data.shape)
    train_path = os.path.join(data_path, 'all', 'train')
    test_path = os.path.join(data_path, 'all', 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    np.save(os.path.join(train_path, 'Hetero_All_Equal.npy'), train_data)
    np.save(os.path.join(test_path, 'Hetero_All_Equal.npy'), test_data)
