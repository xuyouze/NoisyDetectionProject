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
# # print(loss2)
# import numpy as np
# a = [123, 124]
# b = np.asarray(a)
# load_file = "{}_net_{}.pth".format(1, 1)
#
#
#
# load_path = os.path.join(self.save_path, load_file)