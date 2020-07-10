# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 9:43
# @Author  : JRQ
# @FileName: build.py

from networks.registry import Network

import importlib
import os
import glob

def build_network(network_name):
    [importlib.import_module("networks." + os.path.basename(f)[:-3]) for f in
     glob.glob(os.path.join(os.path.dirname(__file__), "*_cnn.py"))]

    assert network_name in Network, \
        f'networks name {network_name} is not registered in registry :{Network.keys()}'
    return Network[network_name]