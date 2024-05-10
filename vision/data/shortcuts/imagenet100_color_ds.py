import os
import random
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image
from torchvision import transforms
from vision.data.imagenet100_ds import ImageNet100Dataset
from vision.data.shortcuts.shortcut_transforms import ColorDotShortcut
from vision.util.file_io import load_json


class ColorDotImageNet100Dataset(ImageNet100Dataset):

    def __init__(
        self,
        root: str | Path,
        split: str,
        kfold_split: int,
        transform: Optional[transforms.Compose],
        dot_correlation: int = 100,
        dot_diameter=5,
    ):
        """Creates an instance of the ImageNet Dataset

        :param root: Root folder containing the necessary data & meta files
        :param split: Split indicating if train/val/test images are to be loaded
        :param transform: optional transforms that are to be applied when getting items
        """
        super().__init__(root, split, kfold_split, transform)
        assert split in [
            "train",
            "val",
            "test",
        ], "Has to be either 'train', 'val' or test"

        self.resize_transform = transforms.Resize((224, 224))
        self.transforms: transforms.Compose = transform
        self.samples: list[tuple[Path, int]] = []
        self.root: Path = Path(root) / "Imagenet100"

        self.max_kfold_split: int = 10
        self.kfold_split = kfold_split

        self.sanity_check()

        metafile = load_json(self.root / "Labels.json")
        classes = list(sorted(metafile.keys()))  # Always the same classes
        self.wnid_to_id = {dk: cnt for cnt, dk in enumerate(classes)}

        self.gather_samples(split)
        if split in ["train", "val"]:
            self.draw_kfold_subset(split, kfold_split)
        self.samples = list(sorted(self.samples))
        rng = np.random.default_rng(32)
        rng.shuffle(self.samples)

        # Returns all the samples in tuples of (path, label)
        self._color_sc_gen = ColorDotShortcut(
            n_classes=100,
            n_channels=3,
            image_size=(224, 224),
            dataset_mean=0,
            dataset_std=1,
            correlation_prob=dot_correlation / 100.0,
            dot_diameter=dot_diameter,
        )
        # Save the coordinates and the color for each sample
        self.color_dot_coords = [self._color_sc_gen._get_color_dot_coords(sample[1]) for sample in self.samples]

        return

    def __getitem__(self, item: int) -> tuple[Any, int, int]:
        try:
            im: Image.Image = Image.open(self.samples[item][0])
        except IndexError as e:
            logger.info(f"Item id: {item}")
            logger.info(f"Length of samples: {len(self.samples)}")
            logger.info(f"Shape {len(self.samples[0])}")
            raise e

        if im.mode != "RGB":
            im = im.convert("RGB")
        im_resized = self.resize_transform(im)
        x_center, y_center, color, color_label, cls_lbl = self.color_dot_coords[item]
        sc_mask, color_mask = self._color_sc_gen._color_dot_from_coords(x_center, y_center, color, dtype=np.uint8)
        im_np_resized = self._color_sc_gen.apply_shortcut(np.array(im_resized), sc_mask, color_mask)
        trans_im = self.transforms(Image.fromarray(im_np_resized, mode="RGB"))
        lbl = self.samples[item][1]
        return trans_im, lbl, int(color_label)

    def __len__(self) -> int:
        return len(self.samples)
