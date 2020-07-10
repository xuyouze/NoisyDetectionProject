# coding:utf-8
# @Time         : 2019/5/15
# @Author       : xuyouze
# @File Name    : __init__.py


import importlib

from torch.utils import data

from global_config import GlobalConfig
from .build import build_dataset

__all__ = ["create_dataset"]


class CustomDatasetDataLoader(object):

    def __init__(self, config: GlobalConfig) -> None:
        super().__init__()
        self.config = config
        # dataset_class = find_dataset_using_name(config.dataset_name)
        # 反射机制 获取类
        dataset_class = build_dataset(config.dataset_name)
        self.dataset = dataset_class(config)
        config.logger.info("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=config.stage_one_train,
            num_workers=int(config.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data


def create_dataset(config):
    data_loader = CustomDatasetDataLoader(config)
    dataset = data_loader.load_data()
    return dataset
