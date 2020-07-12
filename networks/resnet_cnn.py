# -*- coding: utf-8 -*-
# @Time    : 2020/7/12 15:53
# @Author  : JRQ
# @FileName: resnet_cnn.py

import torch
from torch import nn
from torchvision.models import resnet34

from networks.registry import Network


@Network.register("resnet")
def define_net(config):
    net_whole = Resnet34(True, config.dataset_config.attribute_num)
    return torch.nn.DataParallel(net_whole).cuda()


class Resnet34(nn.Module):
    def __init__(self, pre_train, output_num=10):
        super().__init__()
        pre_model = resnet34(pretrained=pre_train)
        self.resnet_layer = nn.Sequential(*list(pre_model.children())[:-1])
        self.Linear_layer = nn.Linear(512, output_num, bias=False)
        self.BN_layer = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.resnet_layer(x)
        # nn.Conv2d(kernel_size=3, stride=1, padding=0, bias=False)
        x = self.BN_layer(x)

        x = x.view(x.size(0), -1)

        feature = x.clone()

        x = self.Linear_layer(x)
        return feature, x
