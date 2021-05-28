# -*- coding: utf-8 -*-

# @Time  : 2020/2/14 下午7:58
# @Author : fl
# @Project : HaierDataMining
# @FileName: rnn


import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn

from haier_data_mining.model.model import Model


class RNNModel(Model):
    def __init__(self, config, seed, lr, seq_len, num_classes, n_hidden):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        super(RNNModel, self).__init__(config=config, seed=seed, lr=lr)

    def create_model(self):
        features = tf.placeholder(tf.float32, [None, self.config.seq_len, self.config.n_dimension], name='features')
        labels = tf.placeholder(tf.float32, [None, self.config.len_out], name='labels')

        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, features, dtype=tf.float32)
        pred = tf.layers.dense(inputs=outputs[:, -1, :], units=self.config.len_out, name='preds')
        loss = tf.reduce_mean(tf.square(pred-labels))
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return features, labels, train_op, loss, pred

    def process_x(self, raw_x_batch):
        x_batch = raw_x_batch
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):

        y_batch = raw_y_batch
        y_batch = np.array(y_batch)
        return y_batch
