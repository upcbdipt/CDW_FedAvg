from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import joblib
import numpy as np
import os
import time
from haier_data_mining.model.model import LogisticRegressionModel, RandomForestModel, IsolationForestModel, \
    DecisionTreeModel, KNeighborsModel, SVMModel
from haier_data_mining.utils.helpers import Config


class Classifier:
    def __init__(self, config, model_name='logistic_regression'):
        """
        针对空调设备的故障检测类
        Args:
            config(object): 配置类
            model(str): 使用的模型名称

        """
        self.config = config
        self.model_list = [LogisticRegressionModel(self.config), RandomForestModel(self.config),
                           IsolationForestModel(self.config), DecisionTreeModel(self.config),
                           KNeighborsModel(self.config), SVMModel(self.config)]
        self.__create_model(model_name)
        self.__load_data()

    def __create_model(self, model_name):
        for _model in self.model_list:
            if _model.name == model_name:
                self.model = _model
                return
        print('没有对应模型，采用默认的逻辑回归模型')
        self.model = LogisticRegressionModel(self.config)

    def __load_data(self):
        # 加载数据
        train_data_path = os.path.join('data', 'all', 'train', 'Hetero_All_Equal.npy1')
        print('加载训练数据......')
        train_data = np.load(train_data_path)
        print(train_data.shape)
        test_data_path = os.path.join('data', 'all', 'test', 'Hetero_All_Equal.npy1')
        print('加载测试数据......')
        test_data = np.load(test_data_path)
        print(test_data.shape)
        self.train_x = train_data[:, :-1]  # 前n-1维为数据的特征,最后一维为时序数据的标签
        self.train_y = train_data[:, -1]
        self.test_x = test_data[:, :-1]
        self.test_y = test_data[:, -1]
        print('数据加载完毕')

    def __save_model(self):
        # 保存模型
        model_path = os.path.join('data', 'model', self.model.name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(self.model.model, os.path.join('data', 'model', self.model.name, str(int(time.time())) + '.m'))
        return model_path

    def __load_model(self, name=None):
        # 加载之前的模型
        self.model.model = joblib.load(name)

    def run(self, train=True, model_time=''):
        if train:
            print('使用' + self.model.name + '进行训练......')
            self.model.model.fit(X=self.train_x, y=self.train_y)
            print('训练完毕')
            # path = self.__save_model()
            # print('模型保存至'+path)
        else:
            model_name = os.path.join('data', 'model', self.model.name, model_time + '.m')
            # self.__load_model(model_name)
            # print('加载模型'+model_name)
        print('开始进行测试......')
        predict = self.model.model.predict(self.test_x)
        print('混淆矩阵')
        print(confusion_matrix(self.test_y, predict))
        print('准确率')
        print(accuracy_score(self.test_y, predict))
        print('精确率')
        print(precision_score(self.test_y, predict))
        print('召回率')
        print(recall_score(self.test_y, predict))
        print('F1分数')
        print(f1_score(self.test_y, predict))
