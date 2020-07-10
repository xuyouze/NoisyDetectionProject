# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 10:59
# @Author  : JRQ
# @FileName: registry.py

from tool.registry import Registry

Model = Registry()



import  numpy as np

a = np.array([1, 2, 3])

b = (a > 1) + 0

print(b * a)