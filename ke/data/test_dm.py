import torch
from ke.data.base_datamodule import BaseDataModule
from ke.util import data_structs as ds
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# from ke.data import auto_augment

# from ke.randaugment.randaugment  import CIFAR10Policy

# from ke.data.auto_augment import CIFAR10Policy

# from ke.data import cutout_aug


class TestDataset(Dataset):
    def __init__(self):
        torch.manual_seed(0)  # Make this data deterministic
        self.data = torch.randn(200, 3, 32, 32)
        self.targets = torch.randint(0, 10, (200,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TestDataModule(BaseDataModule):
    datamodule_id = ds.Dataset.TEST
    n_train = 200
    n_test = 200
    n_classes = 10

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(self):
        """ """
        super().__init__()
        self.train_data = TestDataset()
        self.val_data = TestDataset()
        self.test_data = TestDataset()

    def train_dataloader(
        self,
        *args,
        **kwargs,
    ) -> DataLoader:
        """Get a train dataloader"""
        return DataLoader(dataset=self.train_data, batch_size=100, shuffle=False, num_workers=0)

    def val_dataloader(
        self,
        *args,
        **kwargs,
    ) -> DataLoader:
        """Get a dataloader"""
        return DataLoader(dataset=self.val_data, batch_size=100, shuffle=False, num_workers=0)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(dataset=self.test_data, batch_size=100, shuffle=False, num_workers=0)
