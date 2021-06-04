# -*- coding: utf-8 -*-

# @Time  : 2020/1/14 下午8:04
# @Author : upcbdipt
# @Project : CDW_FedAvg
# @FileName: autoencoder

import numpy as np
import tensorflow as tf

from main.model.model_ae import Model
from main.utils.model_utils import one_hot


class AutoEncoderModel(Model):
    def __init__(self, config, seed, lr, n_hidden):
        self.n_hidden = n_hidden
        super(AutoEncoderModel, self).__init__(config=config, seed=seed, lr=lr)

    def encoder_op(self, X):
        layer_1 = tf.nn.sigmoid(tf.layers.dense(inputs=X, units=self.n_hidden))
        layer_2 = tf.nn.sigmoid(tf.layers.dense(inputs=layer_1, units=self.config.n_hidden))
        layer_3 = tf.nn.sigmoid(tf.layers.dense(inputs=layer_2, units=2))
        return layer_3

    def decoder_op(self, X):
        layer_1 = tf.nn.sigmoid(tf.layers.dense(inputs=X, units=self.n_hidden))
        layer_2 = tf.nn.sigmoid(tf.layers.dense(inputs=layer_1, units=self.config.n_hidden))
        layer_3 = tf.nn.sigmoid(tf.layers.dense(inputs=layer_2, units=self.config.n_dimension))
        return layer_3

    def create_model(self):
        features = tf.placeholder(dtype=tf.float32, shape=[None, self.config.n_dimension], name='features')
        encoder = self.encoder_op(features)
        pred = self.decoder_op(encoder)

        loss = tf.reduce_mean(tf.square(pred - features))

        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return features, encoder, train_op, loss

    def process_x(self, raw_x_batch):
        x_batch = raw_x_batch
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [one_hot(c, self.config.n_label) for c in raw_y_batch]
        return y_batch
