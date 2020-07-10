# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 18:06
# @Author  : JRQ
# @FileName: two_stage_knn_model.py
from .registry import Model
from .two_stage_base_model import TwoStageBaseModel
from global_config import GlobalConfig

__all__ = ["KNNModel"]


@Model.register("knn")
class KNNModel(TwoStageBaseModel):
    def __init__(self, config: GlobalConfig):
        super(KNNModel, self).__init__(config)
        setattr(self, "net_{}".format(config.network_name), self.create_network_model())
