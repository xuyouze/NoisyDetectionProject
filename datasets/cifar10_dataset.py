# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 18:21
# @Author  : JRQ
# @FileName: cifar10_dataset.py
from global_config import GlobalConfig
from torch.utils import data
from torchvision.transforms import transforms
from .registry import Dataset
from PIL import Image
import numpy as np
import torch


@Dataset.register("cifar10")
class CIFAR10(data.Dataset):
    def __init__(self, config: GlobalConfig):
        super(CIFAR10, self).__init__()
        self.config = config
        if config.stage_one_train:
            # load the noisy training data
            dir_data = "{}{}{}{}".format(self.config.dir_noise, self.config.dataset_name,
                                         "\\", "train_data_dict_" + str(self.config.noise_ratio) +
                                         ".npy")
            data_dict = np.load(dir_data, allow_pickle=True).item()

            self.data = data_dict["data"]
            self.noise_label = data_dict["noise_label"]

            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif not self.config.stage_one_validate:
            # load the clean test data (training data)
            dir_data = "{}{}\{}".format(self.config.dir_noise, self.config.dataset_name,
                                        "train_data_dict_" + str(self.config.noise_ratio) +
                                        ".npy")
            data_dict = np.load(dir_data).item()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.data = data_dict["data"]
            self.noise_label = data_dict["noise_label"]
            self.raw_label = data_dict["raw_label"]

        elif self.config.stage_one_validate:
            # load data for validating model
            dir_data = "{}{}{}{}".format(self.config.dir_noise, self.config.dataset_name,
                                         "\\", "validate_data_dict_" + str(self.config.noise_ratio) +
                                         ".npy")
            data_dict = np.load(dir_data).item()
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.data = data_dict["data"]
            self.noise_label = data_dict["noise_label"]
            self.raw_label = data_dict["raw_label"]

    def __getitem__(self, item):
        if self.config.stage_one_train:
            img = self.data[item]
            img = Image.fromarray(np.uint8(img)).convert('RGB')
            img = self.transform(img)
            self.noise_label[item] = torch.from_numpy(self.noise_label[item])
            # print(self.noise_label[item].nonzero()[0])
            return img, int(self.noise_label[item].nonzero()[0])
        else:
            img = self.data[item]
            img = self.transform(img)
            img = Image.fromarray(img)
            return img, self.noise_label[item], self.raw_label[item]

    def __len__(self):
        return len(self.data)
