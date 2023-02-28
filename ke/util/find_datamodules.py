from ke.data.base_datamodule import BaseDataModule
from ke.data.cifar100_dm import CIFAR100DataModule
from ke.data.cifar10_dm import CIFAR10DataModule
from ke.data.imagenet_dm import ImagenetDataModule
from ke.data.medmnist_dm import DermaMNISTDataModule
from ke.util import data_structs as ds


def get_datamodule(dataset: ds.Dataset) -> BaseDataModule:
    """Returns the datamodule specified by the Dataset and the train/val/test split."""
    if dataset == ds.Dataset.CIFAR10:
        return CIFAR10DataModule()
    elif dataset == ds.Dataset.CIFAR100:
        return CIFAR100DataModule()
    elif dataset == ds.Dataset.IMAGENET:
        return ImagenetDataModule()
    elif dataset == ds.Dataset.DermaMNIST:
        return DermaMNISTDataModule()
    elif dataset == ds.Dataset.SPLITCIFAR100:
        raise NotImplementedError()
        # return split_cifar100_dm.SplitCIFAR100KFoldBaseDataModule(params)
    else:
        raise ValueError
