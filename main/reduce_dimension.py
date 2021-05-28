import os
import numpy as np
import random
import tensorflow as tf

from haier_data_mining.model.autoencoder import AutoEncoderModel
import haier_data_mining.metrics.writer as metrics_writer
from haier_data_mining.utils.model_utils import read_data
from haier_data_mining.ae_client import Client


class ReduceDimension:
    def __init__(self, config):
        self.config = config

    def run(self):
        random_seed = self.config.random_seed
        # 设置随机数种子
        random.seed(1 + random_seed)
        np.random.seed(12 + random_seed)
        tf.set_random_seed(123 + random_seed)
        # 屏蔽tf警告
        tf.logging.set_verbosity(tf.logging.WARN)
        # 加载模型
        lr = self.config.lr
        n_hidden = self.config.n_hidden
        tf.reset_default_graph()
        client_model1 = AutoEncoderModel(config=self. -config, seed=random_seed, lr=lr, n_hidden=n_hidden)
        client_model2 = AutoEncoderModel(config=self.config, seed=random_seed, lr=lr, n_hidden=n_hidden)
        client_model3 = AutoEncoderModel(config=self.config, seed=random_seed, lr=lr, n_hidden=n_hidden)
        client_model4 = AutoEncoderModel(config=self.config, seed=random_seed, lr=lr, n_hidden=n_hidden)
        client_model5 = AutoEncoderModel(config=self.config, seed=random_seed, lr=lr, n_hidden=n_hidden)
        client_model6 = AutoEncoderModel(config=self.config, seed=random_seed, lr=lr, n_hidden=n_hidden)
        client_models = [client_model1, client_model2, client_model3, client_model4, client_model5, client_model6]
        # 准备客户端
        train_data_dir = os.path.join('data', 'train')
        test_data_dir = os.path.join('data', 'test')
        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
        if len(groups) == 0:
            groups = [[] for _ in users]
        clients = []
        i = 0
        for u, g in zip(users, groups):
            clients.append(Client(u, g, train_data[u], test_data[u], client_models[i]))
            i += 1
        client_ids, client_groups, client_num_samples = get_clients_info(clients)
        print('Clients in Total: %d' % len(clients))
        # 初始化状态
        print('--- Random Initialization ---')
        # 用来保存状态
        stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, self.config)
        sys_writer_fn = get_sys_writer_function(self.config)
        print_stats(0, clients, client_num_samples, stat_writer_fn)
        num_rounds = self.config.num_rounds
        eval_every = self.config.eval_every
        clients_per_round = self.config.clients_per_round
        num_epochs = self.config.num_epochs
        batch_size = self.config.batch_size
        # 模拟训练
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

            # 当前轮选择的客户端
            client_ids, client_groups, client_num_samples = get_clients_info(clients)

            # Simulate server model training on selected clients' data
            sys_metrics = train_model(config=self.config, num_epochs=num_epochs, batch_size=batch_size,
                                      minibatch=None, clients=clients)
            sys_writer_fn(i + 1, client_ids, sys_metrics, client_groups, client_num_samples)

            # Test model
            if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
                print_stats(i + 1, clients, client_num_samples, stat_writer_fn)
        encoders, _ = test_model(clients, 'test')
        for client in clients:
            print(encoders[client.id].shape)
            np.save(os.path.join('data', 'reduce', str(client.id + '.npy')), encoders[client.id])

        for client in client_models:
            client.close()


def train_model(config, num_epochs=1, batch_size=10, minibatch=None, clients=None):
    sys_metrics = {
        c.id: {config.bytes_written_key: 0,
               config.bytes_read_key: 0,
               config.local_computations_key: 0} for c in clients}
    for c in clients:
        # 每个客户端进行训练
        comp, num_samples, update = c.train(num_epochs, batch_size, minibatch)
        sys_metrics[c.id][config.bytes_read_key] += c.model.size
        sys_metrics[c.id][config.bytes_written_key] += c.model.size
        sys_metrics[c.id][config.local_computations_key] = comp
    return sys_metrics


def get_clients_info(clients):
    """对给定的客户端返回id,hierarchies 和 num_samples

     如果 clients=None，返回self.selected_clients的信息

    Args:
        clients: 客户端对象列表
    """

    ids = [c.id for c in clients]
    groups = {c.id: c.group for c in clients}
    num_samples = {c.id: c.num_samples for c in clients}
    return ids, groups, num_samples


def get_stat_writer_function(ids, groups, num_samples, config):
    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, config.metrics_dir,
            '{}_{}'.format(config.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(config):
    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', config.metrics_dir,
            '{}_{}'.format(config.metrics_name, 'sys'))

    return writer_fn


def print_stats(
        num_round, clients, num_samples, writer, use_val_set=False):
    _, train_stat_metrics = test_model(clients, set_to_use='train')  # 每一个客户端的loss
    print_metrics(train_stat_metrics, num_samples, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    _, test_stat_metrics = test_model(clients, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)


def test_model(clients_to_test, set_to_use='test'):
    """对给定的客户端测试其模型，如果clients_to_test =None就用self.selected_clients来测试模型

    Args:
        clients_to_test: 客户端对象列表
        set_to_use: 用来测试的数据集.  ['train', 'test']二者之一.
    """
    metrics = {}
    encoders = {}

    for client in clients_to_test:
        encoder, c_metrics = client.test(set_to_use)
        metrics[client.id] = c_metrics
        encoders[client.id] = encoder

    return encoders, metrics


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    # for metric in metric_names:
    #     ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
    #     print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
    #           % (prefix + metric,
    #              np.average(ordered_metric, weights=ordered_weights),
    #              np.percentile(ordered_metric, 10),
    #              np.percentile(ordered_metric, 50),
    #              np.percentile(ordered_metric, 90)))
