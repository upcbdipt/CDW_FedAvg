# -*- coding: utf-8 -*-

# @Time  : 2021/6/4 下午8:38
# @Author : upcbdipt
# @Project : CDW_FedAvg
# @FileName: 3_plot

import os

from main.ploting import plot_train_data

if __name__ == '__main__':
    data_dir = os.path.join('data', 'train')
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.npy')]
    for f in files:
        factory_name = f.split('.')[0]
        plot_train_data(factory_name)
