import os
import random
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from vision.util.file_io import load_json


class ImageNet100Dataset(Dataset):
    def __init__(self, root: str | Path, split: str, kfold_split: int, transform: Optional[transforms.Compose]):
        """Creates an instance of the ImageNet Dataset

        :param root: Root folder containing the necessary data & meta files
        :param split: Split indicating if train/val/test images are to be loaded
        :param transform: optional transforms that are to be applied when getting items
        """
        super().__init__()
        assert split in [
            "train",
            "val",
            "test",
        ], "Has to be either 'train', 'val' or test"

        self.transforms: transforms.Compose = transform
        self.samples: list[tuple[Path, int]] = []
        self.root: Path = Path(root) / "Imagenet100"

        self.max_kfold_split: int = 10
        self.kfold_split = kfold_split

        self.sanity_check()

        metafile = load_json(self.root / "Labels.json")
        classes = list(sorted(metafile.keys()))  # Always the same classes
        self.wnid_to_id = {dk: cnt for cnt, dk in enumerate(classes)}

        # Returns all the samples in tuples of (path, label)
        self.gather_samples(split)
        if split in ["train", "val"]:
            self.draw_kfold_subset(split, kfold_split)
        self.samples = list(sorted(self.samples))
        rng = np.random.default_rng(32)
        rng.shuffle(self.samples)
        return

    def draw_kfold_subset(self, split: str, kf_split: int) -> None:
        """Draws a split from the class in deterministic fashion.

        :param split: Split to draw
        :param kf_split: Use the kfold split to train/val
        :return:
        """
        tmp_samples = []
        for wnid in self.wnid_to_id.values():
            current_samples = [sample for sample in self.samples if sample[1] == wnid]
            n_cur_samples = len(current_samples)
            if kf_split == self.max_kfold_split:
                max_id_to_draw = n_cur_samples
            else:
                max_id_to_draw = (n_cur_samples // self.max_kfold_split) * (kf_split + 1)
            min_id_to_draw = (n_cur_samples // self.max_kfold_split) * kf_split
            val_samples = set(current_samples[min_id_to_draw:max_id_to_draw])
            train_samples = set(current_samples) - val_samples
            if split == "val":
                tmp_samples.extend(list(val_samples))
            else:
                tmp_samples.extend(list(train_samples))
        self.samples = tmp_samples

    def sanity_check(self):
        """Validates that the dataset is present and all samples exist.

        :return:
        """
        assert os.path.exists(self.root), f"Dataset not found at path {self.root}"

        for data_dir, n_data in zip(["train", "val"], [1300, 50]):
            train_data = self.root / data_dir
            train_data_class_dirs = list(train_data.iterdir())
            train_data_class_dirs = [d for d in train_data_class_dirs if d.is_dir()]
            n_dirs = len(train_data_class_dirs)
            if n_dirs != 100:
                raise ValueError(f"Expected 100 directories, found {n_dirs}")
            for data_subdir in train_data_class_dirs:
                samples = [s for s in list(data_subdir.iterdir()) if s.endswith(".JPEG")]
                if len(samples) != n_data:
                    raise ValueError(
                        f"Expected {n_data} {data_dir} images! " f"Found {len(samples)} in {data_subdir.name}"
                    )

        return

    def gather_samples(self, split: str):
        """Loads samples into the self.samples list.
        Contains [image_path, class_id].

        :return:
        """
        data_root_dir = self.root
        if split in ["train", "val"]:
            data_dir = data_root_dir / "train"
        elif split == "test":
            data_dir = data_root_dir / "val"
        else:
            raise ValueError(f"Got faulty split: {split} passed.")

        all_samples = []
        for wnid, class_id in self.wnid_to_id.items():
            class_path = data_dir / wnid
            images: list[tuple[Path, int]] = [
                (cp, class_id) for cp in class_path.iterdir() if cp.name.endswith(".JPEG")
            ]
            all_samples.extend(images)
        self.samples = all_samples
        return

    def __getitem__(self, item: int) -> tuple[Any, int]:
        im: Image.Image = Image.open(self.samples[item][0])
        if im.mode != "RGB":
            im = im.convert("RGB")
        trans_im = self.transforms(im)
        lbl = self.samples[item][1]

        return trans_im, lbl

    def __len__(self) -> int:
        return len(self.samples)
