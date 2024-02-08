from vision.data.base_datamodule import BaseDataModule
from vision.data.cifar100_dm import CIFAR100DataModule
from vision.data.cifar10_dm import CIFAR10DataModule
from vision.data.imagenet100_dm import Imagenet100DataModule
from vision.data.imagenet_dm import ImagenetDataModule
from vision.data.medmnist_dm import DermaMNISTDataModule
from vision.data.test_dm import TestDataModule
from vision.data.tiny_imagenet_dm import TinyImagenetDataModule
from vision.util import data_structs as ds


def get_datamodule(dataset: ds.Dataset, advanced_da: bool = True) -> BaseDataModule:
    """Returns the datamodule specified by the Dataset and the train/val/test split."""
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
        # return split_cifar100_dm.SplitCIFAR100KFoldBaseDataModule(params)
    else:
        raise ValueError
