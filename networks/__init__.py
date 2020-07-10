# coding:utf-8
# @Time         : 2019/6/27
# @Author       : xuyouze
# @File Name    : __init__.py.py
import importlib

from networks.build import build_network
from .scheduler import get_scheduler, init_net

__all__ = ["create_network_model", "get_scheduler"]


def create_network_model(config):
    network = build_network(config.network_name)

    config.logger.info("{0} network has been created".format(config.network_name))

    return network(config)