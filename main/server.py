# -*- coding: utf-8 -*-

# @Time  : 2020/1/13 上午10:30
# @Author : fl
# @Project : HaierDataMining
# @FileName: server

import numpy as np


class Server:
    def __init__(self, config, client_model):
        self.config = config
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.updates = []

    def select_clients(self, my_round, possible_clients, num_clients=5):
        """随机选择num_clients 个客户端

        在函数中, num_clients 设置为
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: 服务器能选择的客户端数目
            num_clients: 选择的客户端数目; default 5
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        for c in self.selected_clients:
            print(c.id)
        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """每个客户端训练自己的模型

        如果clients=None在self.selected_clients训练模型，每个客户端按照给定的epoch和batch_size来训练

        Args:
            clients: 客户端列表
            num_epochs: 训练的epoch数
            batch_size: 训练的批次大小
            minibatch: 使用minibatch sgd算法时客户端数据分片大小，使用FedAvg是为None
        Return:
            bytes_written: 每个客户端发往服务器的字节数
                dictionary with client ids as keys and integer values.
            client computations: 每个客户端的每秒浮点运算次数
                dictionary with client ids as keys and integer values.
            bytes_read: 每个客户端从服务器读取的字节数
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {self.config.bytes_written_key: 0,
                   self.config.bytes_read_key: 0,
                   self.config.local_computations_key: 0} for c in clients}
        for c in clients:
            # 将客户端模型设置为平均后的模型
            c.model.set_params(self.model)
            comp, num_samples, update = c.train(num_epochs, batch_size, minibatch)

            sys_metrics[c.id][self.config.bytes_read_key] += c.model.size
            sys_metrics[c.id][self.config.bytes_written_key] += c.model.size
            sys_metrics[c.id][self.config.local_computations_key] = comp

            centroid_dist = 1/c.centroid_dist

            self.updates.append((num_samples, update, centroid_dist))

        return sys_metrics

    def update_model(self):
        total_weight = 0.
        total_var = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model, client_data_centroid_dist) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]
        # 平均后的服务器端模型
        self.model = averaged_soln
        self.updates = []

    def test_model(self, clients_to_test, set_to_use='test'):
        """对给定的客户端测试其模型，如果clients_to_test =None就用self.selected_clients来测试模型

        Args:
            clients_to_test: 客户端对象列表
            set_to_use: 用来测试的数据集.  ['train', 'test']二者之一.
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics

        return metrics

    def get_clients_info(self, clients):
        """对给定的客户端返回id,hierarchies 和 num_samples

         如果 clients=None，返回self.selected_clients的信息

        Args:
            clients: 客户端对象列表
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """保存服务器模型checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess = self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()
