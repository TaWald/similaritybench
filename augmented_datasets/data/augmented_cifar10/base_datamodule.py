from __future__ import annotations

import os
from typing import Union

from augmented_datasets.data.augmented_cifar10.grayscale_dataset import BasicAugmentedGrayCifarDataset
from augmented_datasets.utils import AugmentationDatasetInfo
from pytorch_lightning import LightningDataModule
from torch.utils import data


class BasicDataModule(LightningDataModule):
    aug_dir_name: str
    param_key: Union[str, None] = None
    aug_props: list = []

    def __init__(self, augmented_dataset_info: AugmentationDatasetInfo, value: int | float | None):
        super(LightningDataModule, self).__init__()
        self.dataset_name = "AugmentedCIFAR10"

        self.augmentation_dirname = augmented_dataset_info.augmentation_dirname
        self.augmentation_value = str(value)
        self.test_dataset: Union[None, data.Dataset] = None
        self.prepared_already: bool = False
        self.gray_mean_std = (0.47336, (0.2393 / 3.0))

        if "RAW_DATA" in os.environ:
            dataset_path = os.environ["RAW_DATA"]
        elif "data" in os.environ:
            dataset_path = os.environ["data"]
        else:
            raise EnvironmentError

        if value is not None:
            if value not in augmented_dataset_info.values:
                raise ValueError(
                    "Unexpected value passed." "Got{}; Expects: {}".format(value, augmented_dataset_info.values)
                )
            dataset_path = os.path.join(
                dataset_path,
                self.dataset_name,
                "data",  # Loads numpy images
                self.augmentation_dirname,
                "test",
                self.augmentation_value,
            )
        else:
            dataset_path = os.path.join(
                dataset_path,
                self.dataset_name,
                "data",  # Loads numpy images
                self.augmentation_dirname,
                "test",
            )

        test_contents = os.listdir(dataset_path)
        test_set_contents = [os.path.join(dataset_path, test_content) for test_content in test_contents]
        self.test_dataset = BasicAugmentedGrayCifarDataset(test_set_contents, self.gray_mean_std)
        # TrainWoAug, Merge and Test are static across splits

    def test_dataloader(self, **kwargs) -> data.DataLoader:

        if "batch_size" in kwargs.keys():
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = 200

        if "shuffle" in kwargs.keys():
            shuffle = kwargs.get("shuffle")
        else:
            shuffle = False

        if "num_workers" in kwargs.keys():
            num_workers = kwargs.get("num_workers")
        else:
            num_workers = 1

        if "pin_memory" in kwargs.keys():
            pin_memory = kwargs.get("pin_memory")
        else:
            pin_memory = True

        if "drop_last" in kwargs.keys():
            drop_last = kwargs.get("drop_last")
        else:
            drop_last = False

        dataloader = data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return dataloader
