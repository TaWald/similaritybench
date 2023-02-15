from __future__ import annotations

from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Any
from typing import Callable

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class AugmentedDataLoader:
    value: int | float | None
    name: str
    dataloader: DataLoader


class InputImageTypes(Enum):
    GRAY = auto()
    LOWCONTRASTGRAY = auto()
    COLOR = auto()


@dataclass
class AugmentationDatasetInfo:
    augmentation_dirname: str
    values: list[int | float] | None
    augmentation: Callable[[np.ndarray, Any], np.ndarray]
    input_image_type: InputImageTypes
