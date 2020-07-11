# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 17:58
# @Author  : JRQ
# @FileName: global_config.py
import logging
from logger_config import config

__all__ = ["GlobalConfig"]


class GlobalConfig:
    """
    GlobalConfig configs all the details in the noise detection project.
    """

    def __init__(self):
        self.dataset_name = "cifar10"
        self.loss_name = "ce"
        self.stage = 1  # specify the stage of noise detecting
        self.network_name = "ordinary"  # "Ordinary" specifies a simple CNN
        self.model_name = "common"

        self.print_freq = 200

        self.save_latest_freq = 256 * 1024

        self.gpu = True

        self.output_num = 10

        self.current_epoch = 0

        self.batch_size = 128

        self.num_threads = 0

        self.epoch_num = 50

        self.dir_noise = "/home/xuyouze/Downloads/cifar10/"

        self.dir_raw_data = r"E:\cifar-10-batches-py"

        self.stage_one_train = True # if true, train the model, else testing the model

        self.stage_one_validate = False

        self.stage_two_train = False  # if true, train the model, else test the model

        # self.noise_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.noise_ratio = 0

        self.dir_checkpoints = "ckp"

        self.last_epoch = "last"

        self.load_iter = 0

        self.noise_threshold = 0.7  # the threshold for identifying noisy samples

        self.lr = 0.01

        self.beta1 = 0.9

        logging.config.dictConfig(config)

        self.logger = logging.getLogger("TrainLogger")

        self.test_logger = logging.getLogger("TestLogger")

        self.start_epoch = 0  # used for continuing training

        self.niter_decay = 100

        self.niter = 0

        self.continue_train = False

        self.lr_policy = "warm_up"

        ####################################################
        """
        the following configurations are about dropout_balance_loss
        """
        ####################################################