import logging
from typing import Optional
from dataclasses import dataclass
import numpy as np
import datasets

import torch

from repsim.measures.utils import SHAPE_TYPE


@dataclass
class SingleLayerRepresentation:
    layer_id: int
    representation: torch.Tensor | np.ndarray
    shape: SHAPE_TYPE


@dataclass
class ModelRepresentations:
    architecture_name: str
    model_id: int  # Additional identifier to distinguish between different models with the same name
    representations: tuple[SingleLayerRepresentation]  # immutable to maintain ordering
    dataset: str


log = logging.getLogger(__name__)


def get_dataset(dataset_name: str, config: Optional[str] = None) -> datasets.dataset_dict.DatasetDict:
    ds = datasets.load_dataset(dataset_name, config)
    assert isinstance(ds, datasets.dataset_dict.DatasetDict)
    return ds


def convert_to_path_compatible(s: str) -> str:
    return s.replace("\\", "-").replace("/", "-")
