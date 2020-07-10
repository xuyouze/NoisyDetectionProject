# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 8:14
# @Author  : JRQ
# @FileName: registry.py

from tool.registry import Registry

Loss = Registry()

#
# import torch
# g = torch.randn(3,3)
# print(g)
# e1 = 0.4
# e2 = 0.6
# idx = (g >= e1) & (g < e2)
# print(idx)
# from torch import nn
# import torch
# a = torch.randn(3, 10)
# b = torch.randn(3, 10)
# loss =nn.BCEWithLogitsLoss(reduction="none")(a, b)
# loss2 = nn.BCEWithLogitsLoss(reduction="none")(a, b).sum(0)
#
# print(loss)
# print(loss2)
import torch
output = torch.Tensor(1, 3)
print(output)
output = output.double()
label = torch.Tensor([0.4, 0.6, 0])
label = label.double()
noise_detection = ((output - 1) * label > 0.7) + 0
print(((1 - label) > 0.5) + 0)
