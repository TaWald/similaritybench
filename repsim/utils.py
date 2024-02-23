import logging
from dataclasses import dataclass

import numpy as np
import torch
from repsim.measures.utils import SHAPE_TYPE


@dataclass
class SingleLayerRepresentation:
    layer_id: int
    representation: torch.Tensor | np.ndarray
    shape: SHAPE_TYPE


@dataclass
class ModelRepresentations:
    setting_identifier: str | None
    architecture_name: str
    train_dataset: str
    seed_id: int  # Additional identifier to distinguish between different models with the same name
    representation_dataset: str
    representations: tuple[SingleLayerRepresentation]  # immutable to maintain ordering


log = logging.getLogger(__name__)


def convert_to_path_compatible(s: str) -> str:
    return s.replace("\\", "-").replace("/", "-")
