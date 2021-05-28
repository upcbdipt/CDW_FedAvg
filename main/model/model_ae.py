# -*- coding: utf-8 -*-

# @Time  : 2020/1/14 下午4:23
# @Author : fl
# @Project : HaierDataMining
# @FileName: model

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from haier_data_mining.utils.model_utils import batch_data
from haier_data_mining.utils.tf_utils import graph_size
from haier_data_mining.model.KNN import KNN


class Model(ABC):

    def __init__(self, config, seed, lr, optimizer=None):
        self.lr = lr
        self.seed = seed
        self._optimizer = optimizer
        self.config = config
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + self.seed)
            # self.features, self.labels, self.train_op, self.eval_metric_ops, self.loss, self.tp_op, \
            # self.tn_op, self.fp_op, self.fn_op = self.create_model()
            self.features, self.encoder, self.train_op, self.loss = self.create_model()
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        self.size = graph_size(self.graph)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

        np.random.seed(self.seed)

    def set_params(self, model_params):
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, model_params):
                variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        return self._optimizer

    @abstractmethod
    def create_model(self):
        """创建模型.

        Returns:
            A 4-tuple consisting of:
                features: A placeholder for the samples' features.
                labels: A placeholder for the samples' labels.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
        """
        return None, None, None, None, None

    def train(self, data, num_epochs=1, batch_size=10):
        """
        训练模型

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        for _ in range(num_epochs):
            self.run_epoch(data, batch_size)

        update = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return comp, update

    def run_epoch(self, data, batch_size):

        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            with self.graph.as_default():
                # self.sess.run(self.train_op,
                #               feed_dict={
                #                   self.features: input_data,
                #                   self.labels: target_data
                #               })
                self.sess.run(self.train_op,
                              feed_dict={
                                  self.features: input_data
                              })

    def test(self, data):
        """
        在给定数据上测试模型.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        with self.graph.as_default():
            # tp, tn, fp, fn, tot_acc, loss = self.sess.run(
            #     [self.tp_op, self.tn_op, self.fp_op, self.fn_op, self.eval_metric_ops, self.loss],
            #     feed_dict={self.features: x_vecs, self.labels: labels}
            # )
            encoder, loss = self.sess.run(
                [self.encoder, self.loss],
                feed_dict={self.features: x_vecs}
            )
        # tpr = float(tp) / (float(tp) + float(fn))
        # fpr = float(fp) / (float(fp) + float(tn))
        # fnr = float(fn) / (float(tp) + float(fn))
        # recall = tpr
        # precision = float(tp) / (float(tp) + float(fp))
        # acc = float(tot_acc) / x_vecs.shape[0]
        # return {self.config.accuracy_key: acc, 'loss': loss, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        #         'precision': precision, 'recall': recall}
        return encoder, {'loss': loss}

    def close(self):
        self.sess.close()

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass


class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        with self.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for v in all_vars:
                val = self.model.sess.run(v)
                var_vals[v.name] = val
        for c in clients:
            with c.model.graph.as_default():
                all_vars = tf.trainable_variables()
                for v in all_vars:
                    v.load(var_vals[v.name], c.model.sess)

    def save(self, path='checkpoints/model.ckpt'):
        return self.model.saver.save(self.model.sess, path)

    def close(self):
        self.model.close()


class BaseModel:
    def __init__(self, config):
        self._config = config
        self._name = None
        self._model = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


class LogisticRegressionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = config.LR
        self.model = LogisticRegression()


class RandomForestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = config.random_forest
        self.model = RandomForestClassifier()


class DecisionTreeModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = config.decision_tree
        self.model = DecisionTreeClassifier()


class IsolationForestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = config.isolation_forest
        self.model = IsolationForest()


class KNeighborsModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = config.k_neighbors
        self.model = KNeighborsClassifier(n_neighbors=100)


class SVMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = config.svm
        self.model = SVC(kernel='rbf', class_weight='balanced')
