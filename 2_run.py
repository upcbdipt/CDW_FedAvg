# -*- coding: utf-8

# @Time  : 2020/1/11 上午9:12
# @Author : upcbdipt
# @Project : CDW_FedAvg
# @FileName: 2_run

from main.detector import Detector
from main.utils.helpers import Config

if __name__ == '__main__':
    config_path = 'config.yaml'
    config = Config(config_path)
    detector = Detector(config)
    detector.run(model_type=config.Fed_NN)

