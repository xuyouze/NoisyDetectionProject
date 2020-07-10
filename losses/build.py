# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 8:16
# @Author  : JRQ
# @FileName: build.py
import importlib
import os
import glob
from .registry import Loss


def build_loss(loss_name):
    # import all the loss classes
    [importlib.import_module("losses." + os.path.basename(f)[:-3]) for f in
     glob.glob(os.path.join(os.path.dirname(__file__), "*_loss.py"))]

    assert loss_name in Loss, \
        f'loss name {loss_name} is not registered in registry :{Loss.keys()}'
    return Loss[loss_name]
