# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 18:06
# @Author  : JRQ
# @FileName: two_stage_knn_model.py
from models.registry import Model
from models.two_stage_base_model import TwoStageBaseModel
from global_config import GlobalConfig
import numpy as np
import sys

__all__ = ["KNNModel"]


@Model.register("knn")
class KNNModel(TwoStageBaseModel):
    def __init__(self, config: GlobalConfig):
        super(KNNModel, self).__init__(config)
        self.filtered = []  # used for saving the two-stage samples' index
        self.correct = 0  # the number of well recognized samples

    def perform(self):
        self.get_data_feature()
        for i in range(len(self.outlier_marks.shape[0])):
            if int(self.outlier_marks[i, 0]) == 1:
                # check its neighbors
                if self.knn(i):
                    # judge it as a noisy sample
                    self.filtered.append(i)
                    if not (self.raw_labels[i] == self.noise_labels[i]).all():
                        self.correct += 1

    def knn(self, i):
        """
        find the k-th nearest neighbors and decide whether this sample is noisy
        """

        distance = [self.distance(i, j) for j in range(self.data_size)]
        sortedIdx = np.argsort(distance)
        bin = {m: 0 for m in range(self.config.output_num)}  # used for storing the frequency of labels
        for k in range(self.config.k):
            bin[int(self.noise_labels[int(sortedIdx[k])]).nonzero()[0]] += 1
        max_idx = max(bin, key=bin.get)
        if int(max_idx) == int(self.noise_labels[i].nonzero()[0]):
            return False
        return True

    def get_model_precision(self):
        return (self.correct + 0.0) / len(self.filtered)

    def get_model_recall(self):
        return (self.correct + 0.0) / (self.config.noise_ratio * self.data_size)

    def clear_precision(self):
        self.correct = 0
        self.filtered = []

    def distance(self, i, j):
        """
        l2 distance
        """
        if int(self.outlier_marks[i]) == 1:  # if this is a outlier, ignore
            return sys.maxsize
        return (((abs(self.feature_maps[i] - self.feature_maps[j])) ** 2).sum()) ** 0.5
