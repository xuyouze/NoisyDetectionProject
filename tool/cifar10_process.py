# -*- coding: utf-8 -*-
# @Time    : 2020/6/26 11:14
# @Author  : JRQ
# @FileName: cifar10_process.py

from .data_process import DataProcess
from global_config import GlobalConfig
import numpy as np
import os

__all__ = ["CIFAR10Process"]


class CIFAR10Process(DataProcess):
    def __init__(self, config: GlobalConfig):
        super(CIFAR10Process, self).__init__(config)

    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = self.load_pickle(f)  # dict类型
            X = datadict['data']  # X, ndarray, 像素值
            Y = datadict['labels']  # Y, list, 标签, 分类

            # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
            # transpose，转置
            # astype，复制，同时指定类型
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    def load_CIFAR10(self, ROOT):
        """ load all of cifar """
        xs = []  # list
        ys = []

        # 训练集batch 1～5
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
            ys.append(Y)
        Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
        Ytr = np.concatenate(ys)
        del X, Y

        # 测试集
        Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    def add_noise(self):
        X_raw, Y_raw, X_test, Y_test = self.load_CIFAR10(self.config.dir_raw_data)  # obtain the raw data

        # convert the original labels to one hot
        Y_raw = self.convert_number_to_one_hot(Y_raw)

        # convert the test data to one hot
        Y_test = self.convert_number_to_one_hot(Y_test)

        self.flip_and_save_data(X_raw, Y_raw, X_test, Y_test)

    def flip_and_save_data(self, X_raw, Y_raw, X_test, Y_test):
        train_data_num = int(Y_raw.shape[0])
        test_data_num = int(Y_test.shape[0])
        for ratio in self.config.noise_ratio:
            slot = int(10 / int(10 * ratio))
            train_data_dict = {"data": X_raw, "raw_label": Y_raw}
            test_data_dict = {"data": X_test, "raw_label": Y_test}

            Y_raw_copy = Y_raw.copy()
            Y_test_copy = Y_test.copy()
            for i in range(train_data_num):

                if i % slot == 0:
                    index = (Y_raw[i] != 0).argmax(axis=0)  # get the one-hot
                    random = (index + 1) % self.config.output_num
                    Y_raw_copy[i] *= 0
                    Y_raw_copy[i][random] = 1

            train_data_dict["noise_label"] = Y_raw_copy

            np.save("{}{}\{}".format(self.config.dir_noise, self.config.dataset_name,
                                     "train_data_dict" + r"_" + str(ratio) + ".npy"), train_data_dict)

            for i in range(test_data_num):
                if i % slot == 0:
                    index = (Y_test[i] != 0).argmax(axis=0)
                    random = (index + 1) % self.config.output_num
                    Y_test_copy[i] *= 0
                    Y_test_copy[i][random] = 1

            test_data_dict["noise_label"] = Y_test_copy

            np.save("{}{}\{}".format(self.config.dir_noise, self.config.dataset_name,
                                     "validate_data_dict_" + str(ratio) + ".npy"), test_data_dict)
