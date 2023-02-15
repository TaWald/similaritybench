import os
from typing import List
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BasicAugmentedGrayCifarDataset(Dataset):
    def __init__(self, augmented_cifar_image_paths: List[str], mean_std: Tuple[float, float]):
        self.image_paths = augmented_cifar_image_paths
        self.current_id = 0
        self.mean = mean_std[0]
        self.std = mean_std[1]

    def __getitem__(self, item) -> Tuple[torch.Tensor, int]:
        current_entry_path = self.image_paths[item]
        filename = os.path.basename(current_entry_path)
        _, label = filename[:-4].split("_")
        label = int(label)
        train_image = np.load(current_entry_path)
        train_image = np.expand_dims(train_image, axis=0)

        norm_train_image = (train_image - self.mean) / self.std

        train_image = np.repeat(norm_train_image, repeats=3, axis=0)
        train_image_torch = torch.from_numpy(train_image)

        return train_image_torch, label

    def __next__(self):
        current_out = self.__getitem__(self.current_id)
        self.current_id = self.current_id + 1
        return current_out

    def __len__(self):
        return len(self.image_paths)
