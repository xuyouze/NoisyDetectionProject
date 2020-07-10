# -*- coding: utf-8 -*-
# @Time    : 2020/6/28 10:36
# @Author  : JRQ
# @FileName: base_model.py
from global_config import GlobalConfig
from abc import ABC, abstractmethod
from torch import nn
from networks import *
import os
import torch


class BaseModel(ABC):
    def __init__(self, config: GlobalConfig):
        super(BaseModel, self).__init__()
        self.config = config
        self.net_names = self.config.network_name
        setattr(self, "optimizer_{}".format(self.net_names), None)
        self.scheduler = None
        self.raw_label = None
        self.noisy_label = None
        self.test_size = None
        self.save_path = self.config.dir_checkpoints

        self.correct = 0  # the well recognized noisy samples
        self.output = None
        self.precision = 0
        self.recall = 0
        self.pos_num = 0  # the number of samples which are judged as noisy samples

    @abstractmethod
    def set_input(self, x):
        pass

    @abstractmethod
    def forward(self):
        pass

    def optimize_parameters(self):
        self.forward()
        # self.optimizer.zero_grad()
        getattr(self, "optimizer_{}".format(self.config.network_name)).zero_grad()
        self.backward()
        getattr(self, "optimizer_{}".format(self.config.network_name)).step()
        # self.optimizer.step()

    def backward(self):
        setattr(self, "loss_{}".format(self.net_names),
                getattr(self, "criterion_{}".format(self.net_names))(getattr(self, "output_{}".format(self.config.network_name)),
                                                                     getattr(self,
                                                                             "noise_label_{}".format(
                                                                                 self.net_names))))
        getattr(self, "loss_{}".format(self.net_names)).backward()

    def setup(self):
        if self.config.stage_one_train:
            self.scheduler = get_scheduler(getattr(self, "optimizer_{}".format(self.net_names)), self.config)
        if not self.config.stage_one_train or self.config.continue_train:
            load_prefix = "iter_%d" % self.config.load_iter if self.config.load_iter > 0 else self.config.last_epoch
            self.load_networks(load_prefix)

        self.print_networks()

    def print_networks(self):
        net = getattr(self, "net_" + self.net_names)
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        self.config.logger.info('[Network %s] Total number of parameters : %.3f M' % (self.net_names, num_params / 1e6))

    def save_networks(self, prefix):
        save_filename = "{}_net_{}.pth".format(prefix, self.net_names)
        save_path = os.path.join(self.save_path, save_filename)
        net = getattr(self, "net_{}".format(self.net_names))
        torch.save(net.module.cpu().state_dict(), save_path)
        net.cuda()

    def load_networks(self, prefix):
        load_file = "{}_net_{}.pth".format(prefix, self.net_names)

        load_path = os.path.join(self.save_path, load_file)

        net = getattr(self, "net_" + self.net_names)

        if isinstance(net, nn.DataParallel):
            net = net.module

        print("loading from {}".format(load_path))

        state_dict = torch.load(load_path)

        net.load_state_dict(state_dict)

    def get_learning_rate(self):
        return getattr(self, "optimizer_{}".format(self.net_names)).param_groups[0]["lr"]

    def get_model_precision(self):
        return self.correct / self.pos_num

    def get_model_recall(self):
        return self.correct / (self.test_size * self.config.noise_ratio)

    def train(self):
        net = getattr(self, "net_" + self.net_names)
        net.train()

    def eval(self):
        net = getattr(self, "net_" + self.net_names)
        net.eval()

    def test(self):
        with torch.no_grad:
            self.forward()
            output = getattr(self, "output_{}".format(self.net_names)).cpu()
            noise_label = getattr(self, "noise_label_{}".format(self.net_names))
            raw_label = getattr(self, "raw_label_{}".format(self.net_names))

            # detect the noisy samples and mark the noisy labels
            noise_detection = ((1 - output) * noise_label > self.config.noise_threshold) + 0

            # the number of identified samples
            self.pos_num += noise_detection.sum()

            # if there is a 1, there is a wrong prediction
            accuracy = noise_detection * raw_label

            self.correct += (noise_detection.sum() - accuracy.sum())

    def set_test_size(self, length):
        self.test_size = length

    def clear_precision(self):
        self.correct = 0
        self.pos_num = 0

    def create_network_model(self):
        return create_network_model(config=self.config)

    def get_current_loss(self):
        return getattr(self, "loss_{}".format(self.net_names))

    def update_learning_rate(self):
        self.scheduler.step()