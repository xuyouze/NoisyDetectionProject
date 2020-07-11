# -*- coding: utf-8 -*-
# @Time    : 2020/7/8 21:08
# @Author  : JRQ
# @FileName: dropout_loss.py

from losses.registry import Loss
from torch import nn
from global_config import GlobalConfig
import torch


@Loss.register("dropout")
class DropoutLoss(nn.Module):
    def __init__(self, config: GlobalConfig):
        super(DropoutLoss, self).__init__()
        self.config = config
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        if self.config.current_epoch < 5:
            if self.config.gpu:
                pred = pred.cuda()
                label = label.cuda()
                loss = self.ce(pred, label)
                return loss.mean()
            else:
                loss = self.ce(pred, label)
                return loss.mean()
        else:
            """
                after 5 epochs, starting dropping samples
            """
            label = self.convert_number_to_one_hot(label)
            noise_detection = ((1 - pred) * label < self.config.noise_threshold) + 0
            weight = torch.sum(noise_detection, dim=1)  # to decide which to be dropped
            loss = self.ce(pred, label, weight=weight)
            return loss.mean()

    def convert_number_to_one_hot(self, Y):
        """
        this method converts the labels with real numbers to one-zero array
        """
        data_num = int(Y.shape[0])
        one_hit = torch.zeros(data_num, self.config.output_num)
        for i in range(data_num):
            one_hit[i][int(Y[i])] = 1
        return one_hit


a = torch.tensor([[0.0, 1, 0.0], [3, 0, 2]])

b = torch.tensor([[0.0, 1, 0.0], [1, 0, 3]])

b = a.argmax(dim=1)[0]
print(b)