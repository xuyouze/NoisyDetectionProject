# coding:utf-8
# @Time         : 2019/9/18
# @Author       : xuyouze
# @File Name    : dropout_balance_loss.py


import torch
import torch.nn as nn

from global_config import GlobalConfig
import numpy as np
from .registry import Loss

__all__ = ["DropLoss"]


@Loss.register("drop")
class DropLoss(nn.Module):
	def __init__(self, config: GlobalConfig):
		super(DropLoss, self).__init__()
		self.config = config

	def forward(self, pred, target, *args, **kwargs):
		# 选择最大的预测值
		pred_max_list = np.max(pred, axis=1)

		# 真实标签的预测值
		true_label_list = np.max(target * pred, axis=1)

		# 如果最大的预测值 - 真实标签的预测值 > 0.2
		outlier_list = (pred_max_list - true_label_list) > 0.2

		# 将可能为outlier的权重设为 0，
		weights = np.ones(pred.shape[0]) - outlier_list * 1
		criterion = nn.CrossEntropyLoss()

		loss = criterion(pred, target) * weights
		return loss.mean()
