# -*- coding: utf-8 -*-
# @Time    : 2020/6/26 11:10
# @Author  : JRQ
# @FileName: data_process.py

from six.moves import cPickle as pickle
import platform
from abc import ABC, abstractmethod
from global_config import GlobalConfig
import numpy as np

__all__ = ["DataProcess"]


class DataProcess(ABC):
    """
    DataProcess is a base class offering some basic data processing methods
    """

    def __init__(self, config: GlobalConfig):
        # super(DataProcess, self).__init__()
        self.config = config

    # 读取文件
    def load_pickle(self, f):
        version = platform.python_version_tuple()  # 取python版本号
        if version[0] == '2':
            return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
        elif version[0] == '3':
            return pickle.load(f, encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))

    def convert_number_to_one_hot(self, Y):
        """
        this method converts the labels with real numbers to one-zero array
        """
        data_num = int(Y.shape[0])
        one_hit = np.zeros((data_num, self.config.output_num))
        for i in range(data_num):
            one_hit[i][int(Y[i])] = 1
        return one_hit

    @abstractmethod
    def add_noise(self):
        """
        this method flips original labels and assigns new noisy labels to samples
        the noisy data will be saved
        """
        pass
