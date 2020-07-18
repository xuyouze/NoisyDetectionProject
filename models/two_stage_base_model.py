# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 18:05
# @Author  : JRQ
# @FileName: two_stage_base_model.py
from global_config import GlobalConfig
from abc import abstractmethod
import os
import torch
from torch.autograd import Variable
from torch import nn
from datasets import *
from networks import *
import numpy as np

__all__ = ["TwoStageBaseModel"]


class TwoStageBaseModel:
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.net_name = config.network_name
        self.save_path = self.config.dir_checkpoints
        setattr(self, "net_{}".format(config.network_name), create_network_model(config))
        self.outlier_marks = None  # used for marking the outliers : 1 means outlier, 0 means non-outlier
        self.feature_maps = None  # used for saving features
        self.noise_labels = None  # used for saving noisy labels
        self.raw_labels = None  # used for saving raw labels
        self.data_size = None  # the number of training samples

    def setup(self):
        """
        load the pre-trained networks in one-stage
        """
        load_prefix = self.config.last_epoch
        self.load_networks(load_prefix)

    def load_networks(self, prefix):
        load_file = "{}_net_{}.pth".format(prefix, self.net_name)

        load_path = os.path.join(self.save_path, load_file)

        net = getattr(self, "net_" + self.net_name)

        if isinstance(net, nn.DataParallel):
            net = net.module

        print("loading from {}".format(load_path))

        state_dict = torch.load(load_path)

        net.load_state_dict(state_dict)

    @abstractmethod
    def perform(self):
        pass

    @abstractmethod
    def get_model_precision(self):
        pass

    @abstractmethod
    def get_model_recall(self):
        pass

    def get_data_feature(self):
        """
        this method aims to get data features and outliers in one-stage detection
        """
        net = getattr(self, "net_{}".format(self.net_name))
        dataset = create_dataset(self.config)
        self.feature_maps = np.zeros((len(dataset), self.config.feature_dim))
        self.noise_labels = np.zeros((len(dataset), self.config.output_num))
        self.outlier_marks = np.zeros((len(dataset), 1))
        self.raw_labels = np.zeros((len(dataset), self.config.output_num))
        self.data_size = len(dataset)
        for i, data in enumerate(dataset):
            img, nl, rl = data
            img = Variable(img).cuda()
            f, pred = net(img)
            outlier = self.get_outlier_from_batch(pred, nl)
            self.feature_maps[i * self.config.batch_size:(i + 1) * self.config.batch_size, :] = f.cpu().detach().numpy()
            self.noise_labels[i * self.config.batch_size:(i + 1) * self.config.batch_size, :] = nl.cpu().detach().numpy()
            self.outlier_marks[i * self.config.batch_size:(i + 1) * self.config.batch_size, :] = outlier.cpu().detach().numpy()
            self.raw_labels[i * self.config.batch_size:(i + 1) * self.config.batch_size, :] = rl

    def get_outlier_from_batch(self, pred, nl):
        pred = pred.double().cuda()
        nl = nl.double().cuda()
        pred_real_label = torch.sum(pred * nl, dim=1)
        pred_max_label = torch.max(pred, dim=1)[0]
        noise_detection = ((pred_max_label - pred_real_label) > 0.2) + 0
        noise_detection = torch.unsqueeze(noise_detection, 1)
        return noise_detection
