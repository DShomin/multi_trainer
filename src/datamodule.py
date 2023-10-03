# Python code for defining a PyTorch Lightning data module for image classification

# Import necessary libraries
import pytorch_lightning as pl
import torch
import torchvision

from torch.utils.data import DataLoader

from benchmark_datasets import (
    get_classification_dataset,
    get_segmentation_dataset,
    get_detection_dataset,
)

import torchvision.transforms.transforms as transforms


def get_data_module(task, batch_size, num_workers, train_transform, valid_transform):
    if task == "classification":
        return BaseDataModule(
            batch_size,
            num_workers,
            train_transform,
            valid_transform,
            get_classification_dataset,
        )
    elif task == "segmentation":
        return BaseDataModule(
            batch_size,
            num_workers,
            train_transform,
            valid_transform,
            get_segmentation_dataset,
        )
    elif task == "detection":
        return BaseDataModule(
            batch_size,
            num_workers,
            train_transform,
            valid_transform,
            get_detection_dataset,
        )
    else:
        raise ValueError(f"Unsupported task {task}")


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers,
        train_transform,
        valid_transform,
        get_dataset_function,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.get_dataset_function = get_dataset_function

    def convert_transform_config(self, transform_config):
        transform_list = []
        for transform_config_e in transform_config:
            transform_name = list(transform_config_e.keys())[0]
            transform_name = transform_config_e[transform_name]

            transform_class = getattr(transforms, transform_name)

            # withou _target_ from config
            if len(transform_config_e) == 1:
                transform_list.append(transform_class())
            else:
                # load without _target_ from config
                transfrom_args = {
                    k: v for k, v in transform_config_e.items() if k != "_target_"
                }
                transform_list.append(transform_class(**transfrom_args))

        return torchvision.transforms.Compose(transform_list)

    def get_dataset(self, get_function, transform):
        return get_function(transform)

    def setup(self, stage=None):
        train_transform = self.convert_transform_config(self.train_transform)
        valid_transform = self.convert_transform_config(self.valid_transform)
        test_transform = self.convert_transform_config(self.valid_transform)

        self.train_dataset = self.get_dataset(
            self.get_dataset_function, train_transform
        )
        self.valid_dataset = self.get_dataset(
            self.get_dataset_function, valid_transform
        )
        self.test_dataset = self.get_dataset(
            self.get_dataset_function,
            test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
