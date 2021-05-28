# -*- coding: utf-8 -*-

# @Time  : 2020/1/7 晚上10:00
# @Author : fl
# @Project : HaierDataMining
# @FileName: helpers.py

import logging
import yaml
import json
import sys
import os

logger = logging.getLogger('haier')
sys.path.append('../haier_data_mining')


class Config:
    """从config.yaml加载全局变量

    """

    def __init__(self, path_to_config):

        self.path_to_config = path_to_config

        if os.path.isfile(path_to_config):
            pass
        else:
            self.path_to_config = '../{}'.format(self.path_to_config)

        with open(self.path_to_config, "r", encoding='utf8') as f:
            self.dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)

        for k, v in self.dictionary.items():
            setattr(self, k, v)

    def build_group_lookup(self, path_to_groupings):

        channel_group_lookup = {}

        with open(path_to_groupings, "r") as f:
            groupings = json.loads(f.read())

            for subsystem in groupings.keys():
                for subgroup in groupings[subsystem].keys():
                    for chan in groupings[subsystem][subgroup]:
                        channel_group_lookup[chan["key"]] = {}
                        channel_group_lookup[chan["key"]]["subsystem"] = subsystem
                        channel_group_lookup[chan["key"]]["subgroup"] = subgroup

        return channel_group_lookup


def make_dirs(_id):
    """根据时间id创建存储数据文件夹

    """

    config = Config("config.yaml")

    if not config.train or not config.predict:
        if not os.path.isdir('data/%s' % config.use_id):
            raise ValueError(
                "Run ID {} is not valid. If loading prior models or predictions, must provide valid ID.".format(_id))

    paths = ['data', 'data/%s' % _id, 'data/logs', 'data/%s/models' % _id, 'data/%s/smoothed_errors' % _id,
             'data/%s/y_hat' % _id]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)


def setup_logging():
    """配置日志参数跟踪参数设置、训练和评估

    Returns:
        logger (obj): 日志对象
    """

    logger = logging.getLogger('haier')
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    return logger
