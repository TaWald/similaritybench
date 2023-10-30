from ke.data.base_datamodule import BaseDataModule
from ke.data.cifar100_dm import CIFAR100DataModule
from ke.data.cifar10_dm import CIFAR10DataModule
from ke.data.imagenet100_dm import Imagenet100DataModule
from ke.data.imagenet_dm import ImagenetDataModule
from ke.data.medmnist_dm import DermaMNISTDataModule
from ke.data.test_dm import TestDataModule
from ke.data.tiny_imagenet_dm import TinyImagenetDataModule
from ke.util import data_structs as ds


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
