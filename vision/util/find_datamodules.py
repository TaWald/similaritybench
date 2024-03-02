from vision.data.base_datamodule import BaseDataModule
from vision.data.cifar100_dm import CIFAR100DataModule
from vision.data.cifar10_dm import CIFAR10DataModule
from vision.data.imagenet100_dm import Imagenet100DataModule
from vision.data.imagenet_dm import ImagenetDataModule
from vision.data.medmnist_dm import DermaMNISTDataModule
from vision.data.shortcuts.sc_cifar10_dm import (
    ColorDot_0_CIFAR10DataModule,
    ColorDot_100_CIFAR10DataModule,
    ColorDot_25_CIFAR10DataModule,
    ColorDot_50_CIFAR10DataModule,
    ColorDot_75_CIFAR10DataModule,
)
from vision.data.test_dm import TestDataModule
from vision.data.tiny_imagenet_dm import TinyImagenetDataModule
from vision.util import data_structs as ds


def get_datamodule(dataset: ds.Dataset | str, advanced_da: bool = True) -> BaseDataModule:
    """Returns the datamodule specified by the Dataset and the train/val/test split."""
    if isinstance(dataset, str):
        dataset = ds.Dataset(dataset)
    if dataset == ds.Dataset.CIFAR10:
        return CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.TEST:
        return TestDataModule()
    elif dataset == ds.Dataset.CIFAR100:
        return CIFAR100DataModule(advanced_da)
    elif dataset == ds.Dataset.IMAGENET:
        return ImagenetDataModule(advanced_da)
    elif dataset == ds.Dataset.IMAGENET100:
        return Imagenet100DataModule(advanced_da)
    elif dataset == ds.Dataset.TinyIMAGENET:
        return TinyImagenetDataModule(advanced_da)
    elif dataset == ds.Dataset.DermaMNIST:
        return DermaMNISTDataModule(advanced_da)
    elif dataset == ds.Dataset.SPLITCIFAR100:
        raise NotImplementedError()
    elif dataset == ds.Dataset.CDOT100:
        return ColorDot_100_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.CDOT75:
        return ColorDot_75_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.CDOT50:
        return ColorDot_50_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.CDOT25:
        return ColorDot_25_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.CDOT0:
        return ColorDot_0_CIFAR10DataModule(advanced_da)
        # return split_cifar100_dm.SplitCIFAR100KFoldBaseDataModule(params)
    else:
        raise ValueError


def get_min_max_shortcut_datamodules(
    dataset: ds.Dataset | str, advanced_da: bool = True
) -> tuple[BaseDataModule, BaseDataModule]:
    if isinstance(dataset, str):
        dataset = ds.Dataset(dataset)

    if dataset in [ds.Dataset.CDOT0, ds.Dataset.CDOT75, ds.Dataset.CDOT50, ds.Dataset.CDOT25, ds.Dataset.CDOT100]:
        return get_datamodule(ds.Dataset.CDOT0, advanced_da), get_datamodule(ds.Dataset.CDOT100, advanced_da)
    else:
        raise NotImplementedError("Only implemented for shortcut datasets.")
