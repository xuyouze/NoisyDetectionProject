# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 18:05
# @Author  : JRQ
# @FileName: two_stage_base_model.py
from global_config import GlobalConfig
import os
import torch
from torch import nn

__all__ = ["TwoStageBaseModel"]


class TwoStageBaseModel():
    def __init__(self, config: GlobalConfig):
        self.config = config

    def setup(self):
        """
        load the pre-trained networks in one-stage
        """
        load_prefix = self.config.last_epoch
        self.load_networks(load_prefix)

    def load_networks(self, prefix):
        load_file = "{}_net_{}.pth".format(prefix, self.net_names)

        load_path = os.path.join(self.save_path, load_file)

        net = getattr(self, "net_" + self.net_names)

        if isinstance(net, nn.DataParallel):
            net = net.module

        print("loading from {}".format(load_path))

        state_dict = torch.load(load_path)

        net.load_state_dict(state_dict)

    def test(self):
        pass

    def get_model_precision(self):
        pass

    def get_model_recall(self):
        pass
