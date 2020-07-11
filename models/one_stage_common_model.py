# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 10:56
# @Author  : JRQ
# @FileName: one_stage_common_model.py

from torch import optim
from torch.autograd import Variable

from global_config import GlobalConfig
from losses import *
from .base_model import BaseModel
from .registry import Model

__all__ = ["CommonModel"]


@Model.register("common")
class CommonModel(BaseModel):
    def __init__(self, config: GlobalConfig):
        super(CommonModel, self).__init__(config)
        self.config = config
        print(self.config.stage_one_train)
        self.net_names = config.network_name

        # default name is ordinary
        setattr(self, "net_{}".format(config.network_name), self.create_network_model())

        setattr(self, "img_{}".format(self.config.network_name), None)

        setattr(self, "output_{}".format(self.config.network_name), None)

        setattr(self, "noise_label_{}".format(self.config.network_name), None)

        setattr(self, "raw_label_{}".format(self.config.network_name), None)

        if self.config.stage_one_train:
            setattr(self, "criterion_{}".format(self.config.network_name), create_loss(self.config))

            setattr(self, "optimizer_{}".format(self.config.network_name), optim.Adam(
                getattr(self, "net_{}".format(self.config.network_name)).parameters(), lr=self.config.lr,
                betas=(self.config.beta1, 0.999)
            ))

            setattr(self, "loss_{}".format(self.config.network_name), None)
        else:
            self.correct = 0

    def forward(self):
        setattr(self, "output_{}".format(self.config.network_name),
                getattr(self, "net_{}".format(self.config.network_name)
                        )(getattr(self, "img_{}".format(self.config.network_name)))[1])

    def set_input(self, x):
        if not self.config.stage_one_train:

            img, noise, raw = x

            setattr(self, "img_{}".format(self.config.network_name), Variable(img).cuda())

            setattr(self, "noise_label_{}".format(self.config.network_name), Variable(noise).cuda())

            setattr(self, "raw_label_{}".format(self.config.network_name), Variable(raw).cuda())
        else:


            img, noise = x

            # setattr(self, "img_{}".format(self.net_names), img)
            # setattr(self, "noise_label_{}".format(self.net_names), noise)

            # setattr(self, getattr(self, "img_{}".format(self.config.network_name)), Variable(img).cuda())
            #
            # setattr(self, getattr(self, "noise_label_{}".format(self.config.network_name)), Variable(noise).cuda())

            setattr(self, "img_{}".format(self.config.network_name), Variable(img).cuda())

            setattr(self, "noise_label_{}".format(self.config.network_name), Variable(noise).cuda())
            #
            # setattr(self, getattr(self, "raw_label_{}".format(self.config.network_name)), Variable(raw).cuda())
