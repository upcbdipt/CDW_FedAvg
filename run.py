# -*- coding: utf-8 -*-RandomForestClassifier

# @Time  : 2020/1/11 上午9:12
# @Author : fl
# @Project : HaierDataMining
# @FileName: run.py

from haier_data_mining.classifier import Classifier
from haier_data_mining.detector import Detector
from haier_data_mining.utils.helpers import Config
from haier_data_mining.reduce_dimension import ReduceDimension
if __name__ == '__main__':
    config_path = 'config.yaml'
    config = Config(config_path)
    detector = Detector(config)
    detector.run(model_type=config.Fed_NN)
    # reduceDimension = ReduceDimension(config)
    # reduceDimension.run()
    # classifier = Classifier(config=config, model_name=config.logistic_regression)
    # classifier.run()
