import numpy as np
from math import sqrt
from collections import Counter


class KNN:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return

    def predict(self, X_predict):
        """给定待预测数据集X_predict, 返回表示X_predict的结果向量"""
        assert self.X is not None and self.y is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self.X.shape[1], \
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x,返回x的预测结果值"""
        assert x.shape[0] == self.X.shape[1], \
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self.X]
        nearest = np.argsort(distances)

        topK_y = [self.y[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]
