# -*- coding: utf-8 -*-

# @Time  : 2020/1/14 下午8:04
# @Author : fl
# @Project : HaierDataMining
# @FileName: lstm

import numpy as np
import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn

from haier_data_mining.model.model import Model
from haier_data_mining.utils.model_utils import one_hot


class NeuralNetworksModel(Model):
    def __init__(self, config, seed, lr, n_hidden):
        self.n_hidden = n_hidden
        super(NeuralNetworksModel, self).__init__(config=config, seed=seed, lr=lr)

    def create_model(self):
        features = tf.placeholder(dtype=tf.float32, shape=[None, self.config.n_dimension], name='features')
        labels = tf.placeholder(dtype=tf.float32, shape=[None, self.config.n_label], name='labels')
        hidden1 = tf.nn.relu6(tf.layers.dense(inputs=features, units=self.n_hidden))
        hidden2 = tf.nn.relu6(tf.layers.dense(inputs=hidden1, units=self.n_hidden))
        hidden3 = tf.layers.dense(inputs=hidden2, units=self.config.n_label)
        pred = tf.nn.softmax(hidden3)
        # 交叉熵损失函数
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        predictions = tf.argmax(pred, 1)
        actuals = tf.argmax(labels, 1)

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)

        tp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        tn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )

        fp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        fn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )

        correct_pred = tf.equal(predictions, actuals)
        eval_metric_ops = tf.count_nonzero(correct_pred)

        return features, labels, train_op, eval_metric_ops, loss, tp_op, tn_op, fp_op, fn_op

    def process_x(self, raw_x_batch):
        x_batch = raw_x_batch
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        # y_batch = [int(e) for e in raw_y_batch]
        # y_batch = [val_to_vec(self.num_classes, e) for e in y_batch]
        # y_batch = np.array(y_batch)
        y_batch = [one_hot(c, self.config.n_label) for c in raw_y_batch]
        return y_batch
