import os
from pathlib import Path

from ke.data.base_datamodule import BaseDataModule
from ke.randaugment.randaugment import CIFAR10Policy
from ke.randaugment.randaugment import Cutout
from ke.util import data_structs as ds
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms as trans
from torchvision.datasets import CIFAR10

# from ke.data import auto_augment

# from ke.randaugment.randaugment  import CIFAR10Policy

# from ke.data.auto_augment import CIFAR10Policy

# from ke.data import cutout_aug


class CIFAR10DataModule(BaseDataModule):
    datamodule_id = ds.Dataset.CIFAR10
    n_train = 50000
    n_test = 10000
    n_classes = 10

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(self, advanced_da: bool):
        """ """
        super().__init__()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        self.image_size = (32, 32)
        if advanced_da:
            self.train_trans = trans.Compose(
                [
                    trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                    trans.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    Cutout(size=16),
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.train_trans = trans.Compose(
                [
                    trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                    trans.RandomHorizontalFlip(),
                    Cutout(size=16),
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        self.val_trans = trans.Compose(
            [
                trans.ToTensor(),
                trans.Normalize(self.mean, self.std),
            ]
        )

        if "RAW_DATA" in os.environ:
            dataset_path_p = Path(os.environ["RAW_DATA"]) / "CIFAR10"
        elif "data" in os.environ:
            dataset_path_p = Path(os.environ["data"]) / "cifar10"
        else:
            raise KeyError(
                "Couldn't find environ variable 'RAW_DATA' or 'data'." "Therefore unable to find CIFAR10 dataset"
            )

        assert dataset_path_p.exists(), f"CIFAR10 dataset not found at {dataset_path_p}"

        dataset_path: str = str(dataset_path_p)
        self.dataset_path = dataset_path

    def train_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a train dataloader"""
        dataset = CIFAR10(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
        )
        train_ids, _ = self.get_train_val_split(split)
        dataset = Subset(dataset, train_ids)
        return DataLoader(dataset=dataset, **kwargs)

    def val_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a validation dataloader"""
        dataset = CIFAR10(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
        )
        _, val_ids = self.get_train_val_split(split)
        dataset = Subset(dataset, val_ids)
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = CIFAR10(
            root=self.dataset_path,
            train=False,
            download=False,
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)
