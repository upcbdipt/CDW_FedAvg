# -*- coding: utf-8 -*-

# @Time  : 2021/6/4 上午11:41
# @Author : upcbdipt
# @Project : CDW_FedAvg
# @FileName: 2_plot

from main.reduce_dimension import ReduceDimension
from main.utils.helpers import Config

if __name__ == '__main__':
    config_path = 'config.yaml'
    config = Config(config_path)
    reduceDimension = ReduceDimension(config)
    reduceDimension.run()




