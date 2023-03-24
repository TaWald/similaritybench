import os
from pathlib import Path

from ke.data.base_datamodule import BaseDataModule
from ke.data.imagenet100_ds import ImageNet100Dataset
from ke.randaugment.randaugment import ImageNetPolicy
from ke.util import data_structs as ds
from torch.utils.data import DataLoader
from torchvision import transforms as trans


# from ke.randaugment.randaugment import ImageNetPolicy

# from RandAugment import RandAugment


class Imagenet100DataModule(BaseDataModule):
    datamodule_id = ds.Dataset.IMAGENET100
    n_train = 130000

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule override
    #   wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(self):
        """ """
        super().__init__()
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.image_size = (160, 160)

        self.train_trans = trans.Compose(
            [
                trans.RandomResizedCrop(list(self.image_size)),
                trans.RandomHorizontalFlip(),
                ImageNetPolicy(),
                trans.ToTensor(),
                trans.Normalize(self.mean, self.std),
            ]
        )
        self.val_trans = trans.Compose(
            [
                trans.Resize(size=list(self.image_size)),
                trans.ToTensor(),
                trans.Normalize(self.mean, self.std),
            ]
        )
        if "RAW_DATA" in os.environ:
            self.dataset_path = Path(os.environ["RAW_DATA"])
        elif "data" in os.environ:
            self.dataset_path = Path(os.environ["data"])
        else:
            raise KeyError("Couldn't find environ variable 'RAW_DATA' or 'data'.")

    def train_dataloader(self, split: int, transform: ds.Augmentation, **kwargs) -> DataLoader:
        """Get a train dataloader"""
        dataset = ImageNet100Dataset(
            root=self.dataset_path,
            split="train",
            kfold_split=split,
            transform=self.get_transforms(transform),
        )
        # INFO: Currently does not differentiate into different folds, as the
        #   Dataset comes with a deliberate validation set.
        return DataLoader(dataset=dataset, **kwargs)

    def val_dataloader(self, split: int, transform: ds.Augmentation, **kwargs) -> DataLoader:
        """Get a validation dataloader"""
        dataset = ImageNet100Dataset(
            root=self.dataset_path,
            split="val",
            kfold_split=split,
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation, **kwargs) -> DataLoader:
        dataset = ImageNet100Dataset(
            root=self.dataset_path,
            split="test",
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)
