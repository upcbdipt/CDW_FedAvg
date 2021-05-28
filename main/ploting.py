# -*- coding: utf-8 -*-

# @Time  : 2020/2/14 下午4:25
# @Author : fl
# @Project : HaierDataMining
# @FileName: ploting

import matplotlib.pyplot as plt
import os
import numpy as np
import random
import scipy.stats as stats
from scipy.spatial import distance

path = os.path.join('..', 'results')


def plot_value(client, model, use_set):
    set_path = os.path.join(path, model, use_set)
    true_value_path = os.path.join(set_path, 'labels', client + '.npy')
    predicted_value_path = os.path.join(set_path, 'preds', client + '.npy')
    score_value_path = os.path.join(set_path, 'scores', client + '.npy')
    true_value = np.load(true_value_path)
    print(true_value.shape)
    predicted_value = np.load(predicted_value_path)
    print(predicted_value.shape)
    score_value = np.load(score_value_path)
    print(score_value.shape)
    plt.close('all')
    cmap = plt.get_cmap('inferno')
    grid = 0
    grid += true_value.shape[1]
    grid += predicted_value.shape[1]
    grid += 1  # score
    fig, axes = plt.subplots(grid, 1, figsize=(15, 1.5 * grid))
    i = 0
    c = cmap(i / grid)
    axes[i].set_title('true_value')
    for col in range(true_value.shape[1]):
        axes[i].plot(true_value[:, col], color=c)
        i += 1
    c = cmap(i / grid)
    axes[i].set_title('predict_value')
    for col in range(predicted_value.shape[1]):
        axes[i].plot(predicted_value[:, col, :], color=c)
        i += 1
    c = cmap(i / grid)
    axes[i].set_title('scores')
    axes[i].plot(score_value, color=c)

    save_pic(fig, set_path, client)


def save_pic(fig, model_path, client, extension='pdf'):
    pic_path = os.path.join(model_path, f'{client}.{extension}')
    fig.savefig(pic_path)


# plot_value('F3', 'lstm', 'test')
# fig, ax = plt.subplots()
# data = np.load(os.path.join('..', 'data', 'all', 'train', '1.npy'))
# ax.plot(data)
# fig.savefig(os.path.join(path, '2.pdf'))
# x=[]
# for i in range(10):
#     x.append(i)
# x=np.array(x)
# x=x.reshape((len(x),1))
# x=np.expand_dims(x, axis=0)
# input=x[:, :-2, :]
# target_data = []
# for l in range(2 - 1):
#     target_data += [x[:, 1 + l:-2 + 1 + l, :]]
# target_data += [x[:, 2:, :]]
# target_data=np.stack(target_data, axis=3)
# print(input.shape)
# print(target_data.shape)
files_path = os.path.join('..', 'data', 'reduce')

i = 0
markerlist = ['*', 'o', '1', 'v', '.', 'p']
all_data = None
for file in os.listdir(files_path):
    if file.endswith('.npy'):
        data = np.load(os.path.join(files_path, file))
        # np.random.shuffle(data)
        # data = data[:500]
        # data = data[:, :-1]
        if all_data is None:
            all_data = data
        else:
            all_data = np.vstack((all_data, data))
print(all_data.shape)
np.random.shuffle(all_data)
all_data = all_data
all_data_mean = np.mean(all_data, axis=0)
print(all_data_mean)
for file in os.listdir(files_path):
    if file.endswith('.npy'):
        data = np.load(os.path.join(files_path, file))
        # np.random.shuffle(data)
        # data = data[:500]
        dist = np.linalg.norm(data - all_data_mean)
        # kl = stats.entropy(data,all_data)
        # bc = np.sum(np.sqrt(data*all_data))
        # b = -np.log(bc)

        print(file)
        print('欧氏距离：' + str(dist))
        # print('KL散度:'+str(kl))
        # print('巴氏距离:' + str(b))
        plt.scatter(x=data[:, 0], y=data[:, 1], label=file.split('.')[0], marker=markerlist[i], s=3)
        i += 1
#plt.scatter(x=mean[0], y=mean[1], label='mean', marker=markerlist[i], s=100)
plt.legend()
plt.savefig(os.path.join('..', 'results', 'ae', '3.23.pdf'))

data_dict = {}

for file in os.listdir(files_path):
    if file.endswith('.npy'):
        data = np.load(os.path.join(files_path, file))
        client = file.split('.')[0]
        print(file)
        mean = np.mean(data, axis=0)
        data_dict[client] = np.mean(data, axis=0)
for client in list(data_dict.keys()):
    print(client)
    dist = np.linalg.norm(all_data - data_dict[client])
    print(dist)

data_list = []

for file in os.listdir(files_path):
    if file.endswith('.npy'):
        data = np.load(os.path.join(files_path, file))
        client = file.split('.')[0]
        print(file)
        mean = np.mean(data, axis=0)
        data_list.append(np.mean(data, axis=0))
data_array = np.array(data_list)
dist = distance.cdist(data_array, data_array, metric='euclidean')
print(dist)
print(dist.sum(axis=1))
print(all_data.shape)
