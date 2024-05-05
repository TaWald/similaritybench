import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import Compose
from albumentations import GaussNoise
from albumentations import HorizontalFlip
from albumentations import Normalize
from albumentations import PadIfNeeded
from albumentations import RandomCrop
from albumentations import RandomResizedCrop
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from repsim.benchmark.paths import VISION_DATA_PATH
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms as trans
from torchvision.datasets import CIFAR10
from vision.data.base_datamodule import BaseDataModule
from vision.data.imagenet100_dm import Imagenet100DataModule
from vision.data.imagenet100_ds import ImageNet100Dataset
from vision.randaugment.randaugment import CIFAR10Policy
from vision.util import data_structs as ds


# from ke.data import auto_augment

# from ke.randaugment.randaugment  import CIFAR10Policy

# from ke.data.auto_augment import CIFAR10Policy

# from ke.data import cutout_aug


class IN100_AlbuDataset(ImageNet100Dataset):
    def __getitem__(self, index: int):
        img, target = self.samples[index][0], self.samples[index][1]

        img = Image.open(img)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transforms is not None:
            img = self.transforms(image=np.array(img))

        # if self.target_transform is not None:
        #     target = self.target_transform(target=target)

        return img["image"], target


class Gauss_Max_Imagenet100DataModule(Imagenet100DataModule):
    datamodule_id = ds.Dataset.IMAGENET100
    n_train = 50000
    n_test = 10000
    n_classes = 10
    var_limit = (0, 4000)  # (5, 20), (10, 50), (25, 75), (40, 100)
    dot_diameter = 5

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(
        self,
        advanced_da: bool,
    ):
        """ """
        super().__init__(advanced_da)

        self.train_trans = Compose(
            [
                RandomResizedCrop(height=self.image_size[0], width=self.image_size[1], scale=(0.75, 1)),
                HorizontalFlip(),
                GaussNoise(
                    var_limit=0 if self.var_limit is None else self.var_limit,
                    p=0 if self.var_limit is None else 1,
                    always_apply=True,
                ),
                # trans.GaussianBlur(5, sigma=(0.1, 2.0)),
                Normalize(self.mean, self.std),
                ToTensorV2(),
            ]
        )

        self.val_trans = Compose(
            [
                Resize(height=self.image_size[0], width=self.image_size[1]),
                GaussNoise(
                    var_limit=0 if self.var_limit is None else self.var_limit,
                    p=0 if self.var_limit is None else 1,
                    always_apply=True,
                ),
                Normalize(self.mean, self.std),
                ToTensorV2(),
            ]
        )
        self.dataset_path = self.prepare_data()

    def prepare_data(self, **kwargs) -> None:
        if "RAW_DATA" in os.environ:
            # Setting the path for this can also be made optional (It's 170 mb afterall)
            dataset_path = os.environ["RAW_DATA"]
        else:
            # Test that it is as expected
            dataset_path = VISION_DATA_PATH
        return dataset_path

    def train_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a train dataloader"""
        dataset = IN100_AlbuDataset(
            root=self.dataset_path,
            split="train",
            kfold_split=0,
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
        dataset = IN100_AlbuDataset(
            root=self.dataset_path,
            split="train",
            kfold_split=0,
            transform=self.get_transforms(transform),
        )
        _, val_ids = self.get_train_val_split(split)
        dataset = Subset(dataset, val_ids)
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = IN100_AlbuDataset(
            root=self.dataset_path,
            train="val",
            kfold_split=0,
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)


class Gauss_L_Imagenet100DataModule(Gauss_Max_Imagenet100DataModule):
    var_limit = (0, 3000)


class Gauss_M_Imagenet100DataModule(Gauss_Max_Imagenet100DataModule):
    var_limit = (0, 2000)


class Gauss_S_Imagenet100DataModule(Gauss_Max_Imagenet100DataModule):
    var_limit = (0, 1000)


class Gauss_Off_Imagenet100DataModule(Gauss_Max_Imagenet100DataModule):
    var_limit = None


if __name__ == "__main__":
    # Plot the Gauss Max Datamodule and produce some examplary images.
    # Load 20 images for each datamodule and save to disk
    dmoff = Gauss_Off_Imagenet100DataModule(advanced_da=True)
    dms = Gauss_S_Imagenet100DataModule(advanced_da=True)
    dmm = Gauss_M_Imagenet100DataModule(advanced_da=True)
    dml = Gauss_L_Imagenet100DataModule(advanced_da=True)
    dmmax = Gauss_Max_Imagenet100DataModule(advanced_da=True)

    # Load train dataloader for each datamodule
    train_max = dmmax.val_dataloader(split=0, transform=ds.Augmentation.VAL, batch_size=1, shuffle=False)
    train_l = dml.val_dataloader(split=0, transform=ds.Augmentation.VAL, batch_size=1, shuffle=False)
    train_m = dmm.val_dataloader(split=0, transform=ds.Augmentation.VAL, batch_size=1, shuffle=False)
    train_s = dms.val_dataloader(split=0, transform=ds.Augmentation.VAL, batch_size=1, shuffle=False)
    train_off = dmoff.val_dataloader(split=0, transform=ds.Augmentation.VAL, batch_size=1, shuffle=False)

    # Save 20 images from each dataloader to disk
    save_dir = os.path.join(os.path.dirname(__file__), "IN100_GaussExamples")
    os.makedirs(save_dir, exist_ok=True)

    # Save 20 images from each dataloader to disk
    save_dir = os.path.join(os.path.dirname(__file__), "IN100_GaussExamples")
    os.makedirs(save_dir, exist_ok=True)

    for cnt, (name, dl) in enumerate(
        [("off", train_off), ("s", train_s), ("m", train_m), ("l", train_l), ("max", train_max)]
    ):
        for i, batch in enumerate(dl):
            if i >= 20:
                break
            image = torch.squeeze(batch[0])
            save_path = os.path.join(save_dir, f"{cnt}_{name}_image_{i}.png")
            image = torch.squeeze(batch[0]).permute(1, 2, 0).numpy()
            image = (image - image.min()) / (image.max() - image.min())  # Just rescale.
            plt.imsave(save_path, image)
