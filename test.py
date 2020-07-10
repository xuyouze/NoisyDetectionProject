# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 18:20
# @Author  : JRQ
# @FileName: test.py

import os
from global_config import GlobalConfig
from datasets import *


def one_stage_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = GlobalConfig()

    config.stage_one_train = False

    logger = config.test_logger

    logger.info("{} model was initialized".format(config.model_name))

    dataset = create_dataset(config)

    model = create_dataset(config)

    model.setup()  # load the last epoch model parameters

    model.set_test_size(len(dataset))

    model.clear_precision()

    model.eval()

    total_iter = int(len(dataset) / config.batch_size)
    for i, data in enumerate(dataset):
        model.set_input(data)
        logger.info("[{}/{}]".format(i, total_iter))
        model.test()

    logger.info("The precision is {}, the recall is {}".format(model.get_model_precison(),
                                                               model.get_model_recall()))

def two_stage_test():
    pass