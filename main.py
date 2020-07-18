# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 11:45
# @Author  : JRQ
# @FileName: main.py

from global_config import GlobalConfig
from test import test
from tool.cifar10_process import CIFAR10Process
from models import *
from datasets import *
import os
import time


def validate(model):
    # validate the networks
    config = GlobalConfig()
    config.stage_one_train = False
    config.stage_one_validate = True
    model.config.stage_one_train = False
    logger = config.test_logger
    logger.info("--------------------------------------------------------")
    logger.debug("test the model using the validate dataset")
    validate_dataset = create_dataset(config)
    model.clear_precision()
    model.eval()
    model.set_test_size(len(validate_dataset))
    logger.info("validate dataset len: %d " % len(validate_dataset))
    validate_total_iter = int(len(validate_dataset) / config.batch_size)
    for j, valida_data in enumerate(validate_dataset):
        model.set_input(valida_data)
        logger.debug("[%s/%s]" % (j, validate_total_iter))
        model.test()

    logger.info("precision now is {}".format(model.get_model_precision()))
    logger.info("recall now is {}".format(model.get_model_recall()))
    logger.info("accuracy now is {}".format(model.get_model_accuracy()))
    logger.debug("validate mode end")
    logger.info("--------------------------------------------------------")
    model.config.stage_one_train = True


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = GlobalConfig()
    # prc = CIFAR10Process(config)
    # prc.add_noise()

    if config.stage == 1:
        # the one-stage detection
        if config.stage_one_train:
            # train the networks
            logger = config.logger
            logger.info("current batch size is {}".format(config.batch_size))
            dataset = create_dataset(config)
            model = create_model(config)
            model.setup()  # initialize the model components
            config.print_freq = int(len(dataset) / config.batch_size / 10)  # printing frequency
            total_iter = 0
            logger.info("dataset length: {}".format(str(len(dataset))))
            logger.info("----------------------- model build complete ----------------------------")
            for epoch in range(config.start_epoch, config.niter + config.niter_decay + 1):
                epoch_start_time = time.time()
                epoch_iter = 0
                logger.info(
                    "epoch [{}/{}] begin at: {} ,learning rate : {}".format(epoch, config.niter + config.niter_decay,
                                                                            time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                          time.localtime(
                                                                                              epoch_start_time)),
                                                                            model.get_learning_rate()))
                model.config.current_epoch += 1
                for i, data in enumerate(dataset):
                    iter_start_time = time.time()

                    total_iter += config.batch_size
                    epoch_iter += config.batch_size
                    model.set_input(data)
                    model.optimize_parameters()

                    if i % config.print_freq == 0:
                        losses = model.get_current_loss()  # 当前的损失函数值
                        t_comp = (time.time() - iter_start_time) / config.batch_size  # 一次批操作时间
                        logger.debug("epoch[%d/%d], iter[%d/%d],current loss=%s,Time consuming: %s sec" % (
                            epoch, config.niter + config.niter_decay, epoch_iter, len(dataset), losses, t_comp))

                        if total_iter % config.save_latest_freq == 0:
                            logger.debug("saving the last model (epoch %d, total iters %d)" % (epoch, total_iter))
                            model.save_networks(config.last_epoch)
                logger.debug('saving the model at the end of epoch %d, iters %d' % (epoch, total_iter))
                model.save_networks(config.last_epoch)
                model.save_networks("iter_%d" % epoch)
                validate(model)
                model.train()
                logger.info('End of epoch %d / %d \t Time Taken: %d sec' % (
                    epoch, config.niter + config.niter_decay, time.time() - epoch_start_time))
                model.update_learning_rate()

        else:
            # test the ont-stage detection's recall and precision
            # print(123)
            test()

    elif config.stage == 2:
        # the two-stage detection
        logger = config.logger
        logger.info("current dataset is {}".format(config.dataset_name))
        model = create_model(config)
        model.setup()
        logger.info("----------------------- model build complete ----------------------------")
        start_time = time.time()
        model.perform()
        logger.info("time costing: {}".format(time.time() - start_time))
        logger.info("precision now is {}".format(model.get_model_precision()))
        logger.info("recall now is {}".format(model.get_model_recall()))
        logger.debug("model {} ends".format(config.model_name))
        logger.info("-------------------------------------------------------------------------")
