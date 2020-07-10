# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 21:21
# @Author  : JRQ
# @FileName: ordinary_cnn.py

from torch import nn
import torch
from networks.registry import Network

__all__ = ["define_net"]


@Network.register("ordinary")
def define_net(config):
    net = OrdinaryCNN(config.output_num)
    return torch.nn.DataParallel(net).cuda()
    # return net


class OrdinaryCNN(nn.Module):

    def __init__(self, output_num=10):
        super(OrdinaryCNN, self).__init__()

        # the 1st convolution layer
        # input: 32 * 32 * 3
        # output: 16 * 16 * 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # the 1st pooling layer
        # input: 16 * 16 * 64
        # output: 8 * 8 * 64
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # the 2nd convolution layer
        # input: 8 * 8 * 64
        # output: 4 * 4 * 96
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        # the 2nd pooling layer
        # input: 4 * 4 * 96
        # output: 2 * 2 * 96
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # the 3rd convolution layer
        # input: 2 * 2 * 96
        # output: 1 * 1 * 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        # the 4th convolution layer
        # input: 12 * 12 * 128
        # output: 6 * 6 * 128
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )

        # the classifier structure contains 1 FC layer
        self.classifier = nn.Sequential(
            nn.Linear(128, output_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        # out = self.conv2(out)
        out = self.pool1(out)
        out = self.conv2(out)
        # out = self.conv4(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = out.view(out.size(0), -1)
        feature = out
        out = self.classifier(out)
        return feature, out
