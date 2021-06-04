# -*- coding: utf-8 -*-

# @Time  : 2020/1/14 上午10:30
# @Author : upcbdipt
# @Project : CDW_FedAvg
# @FileName: reduce_dimension

import os
import numpy as np
import random
import tensorflow as tf

from main.model.autoencoder import AutoEncoderModel
import main.metrics.writer as metrics_writer
from main.utils.model_utils import read_data
from main.ae_client import Client


class ReduceDimension:
    def __init__(self, config):
        self.config = config

    def run(self):
        random_seed = self.config.random_seed
        random.seed(1 + random_seed)
        np.random.seed(12 + random_seed)
        tf.set_random_seed(123 + random_seed)
        tf.logging.set_verbosity(tf.logging.WARN)
        # load model
        lr = self.config.lr
        n_hidden = self.config.n_hidden
        tf.reset_default_graph()

        # client
        train_data_dir = os.path.join('data', 'train')
        test_data_dir = os.path.join('data', 'test')
        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
        if len(groups) == 0:
            groups = [[] for _ in users]
        clients = []
        i = 0
        for u, g in zip(users, groups):
            client_model = AutoEncoderModel(config=self.config, seed=random_seed, lr=lr, n_hidden=n_hidden)
            clients.append(Client(u, g, train_data[u], test_data[u], client_model))
            i += 1
        client_ids, client_groups, client_num_samples = get_clients_info(clients)
        print('Clients in Total: %d' % len(clients))
        # Initialization
        print('--- Random Initialization ---')
        # save state
        stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, self.config)
        sys_writer_fn = get_sys_writer_function(self.config)
        print_stats(0, clients, client_num_samples, stat_writer_fn)
        num_rounds = self.config.num_rounds
        eval_every = self.config.eval_every
        clients_per_round = self.config.clients_per_round
        num_epochs = self.config.num_epochs
        batch_size = self.config.batch_size
        # training
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

            # Choose client
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
            reduce_path = os.path.join('data', 'reduce')
            if not os.path.exists(reduce_path):
                os.makedirs(reduce_path)
            np.save(os.path.join(reduce_path, str(client.id + '.npy')), encoders[client.id])
            client.model.close()


def train_model(config, num_epochs=1, batch_size=10, minibatch=None, clients=None):
    sys_metrics = {
        c.id: {config.bytes_written_key: 0,
               config.bytes_read_key: 0,
               config.local_computations_key: 0} for c in clients}
    for c in clients:
        # train
        comp, num_samples, update = c.train(num_epochs, batch_size, minibatch)
        sys_metrics[c.id][config.bytes_read_key] += c.model.size
        sys_metrics[c.id][config.bytes_written_key] += c.model.size
        sys_metrics[c.id][config.local_computations_key] = comp
    return sys_metrics


def get_clients_info(clients):
    """Returns the ids, hierarchies and num_samples for the given clients.

     Returns info about self.selected_clients if clients=None;

    Args:
        clients: list of Client objects.
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
    """Tests self.model on given clients.
    Tests model on self.selected_clients if clients_to_test=None.

    Args:
        clients_to_test: list of Client objects.
        set_to_use: dataset to test on. Should be in ['train', 'test'].
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
