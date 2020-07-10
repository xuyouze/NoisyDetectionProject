# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 8:06
# @Author  : JRQ
# @FileName: ce_loss.py
from torch import nn
from global_config import GlobalConfig
from .registry import Loss

__all__ = ["CELoss"]


@Loss.register("ce")
class CELoss(nn.Module):
    def __init__(self, config: GlobalConfig):
        super(CELoss, self).__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, y, y_hat):
        y_hat = y_hat.long()

        if self.config.gpu:
            y = y.cuda()
            y_hat = y_hat.cuda()

        loss = self.criterion(y, y_hat)

        return loss.mean()
