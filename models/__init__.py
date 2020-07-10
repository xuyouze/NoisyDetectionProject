# coding:utf-8
# @Time         : 2019/5/15
# @Author       : xuyouze
# @File Name    : __init__.py


import importlib

from global_config import GlobalConfig
from models.base_model import BaseModel
from models.build import build_model

__all__ = ["create_model"]


def create_model(config: GlobalConfig):
    # model = find_model_using_name(config.model_name)
    model = build_model(config.model_name)
    instance = model(config)
    config.logger.info("{0} model has been created".format(config.model_name))
    return instance
