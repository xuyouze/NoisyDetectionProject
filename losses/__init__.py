# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 10:12
# @Author  : JRQ
# @FileName: __init__.py.py

from .build import build_loss
from global_config import GlobalConfig

__all__ = ["create_loss"]


def create_loss(config: GlobalConfig):
    # loss = find_loss_using_name(config.loss_name)
    loss = build_loss(config.loss_name)
    instance = loss(config)
    config.logger.info("{0} loss has been created".format(config.loss_name))
    return instance
