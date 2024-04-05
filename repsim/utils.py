import os
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import TYPE_CHECKING

from repsim.benchmark.paths import CACHE_PATH
from vision.util.vision_rep_extraction import get_single_layer_vision_representation_on_demand

if TYPE_CHECKING:
    from vision.arch.abstract_acti_extr import AbsActiExtrArch
    from vision.util import data_structs as ds
else:
    AbsActiExtrArch = None
    ds = None
from vision.arch.arch_loading import load_model_from_info_file


import numpy as np
import torch
from loguru import logger
from repsim.measures.utils import SHAPE_TYPE
from vision.util.file_io import get_vision_model_info


@dataclass
class SingleLayerRepresentation:
    layer_id: int
    cache: bool = False
    _representation: torch.Tensor | np.ndarray | None = None
    _shape: SHAPE_TYPE | None = None
    _architecture_name: str | None = field(default=None, init=False)
    _train_dataset: str | None = field(default=None, init=False)
    _seed: int | None = field(default=None, init=False)
    _representation_dataset: str | None = field(default=None, init=False)
    _setting_identifier: str | None = field(default=None, init=False)

    @property
    def representation(self) -> torch.Tensor | np.ndarray:
        """
        Allows omission of setting representations when creating, and creating on demand.
        """
        if self._representation is None:
            assert self._extract_representation is not None, "No extraction function provided."
            unique_id = self.unique_identifier()
            cache_rep_path = os.path.join(CACHE_PATH, unique_id + ".npz")
            if self.cache and os.path.exists(cache_rep_path):
                self._representation = np.load(cache_rep_path)["representation"]
            else:
                self._representation = self._extract_representation()
                if self.cache:
                    np.savez(cache_rep_path, representation=self._representation)
        return self._representation

    @representation.setter
    def representation(self, v: torch.Tensor | np.ndarray) -> None:
        """Allow setting the representation as before"""
        self._representation = v
        return self._representation

    @abstractmethod
    def _extract_representation(self) -> torch.Tensor | np.ndarray:
        raise NotImplementedError

    @property
    def shape(self) -> SHAPE_TYPE:
        if self._shape is None:
            if len(self.representation.shape) == 4:
                shape = "nchw"
            elif len(self.representation.shape) == 3:
                shape = "ntd"
            elif len(self.representation.shape) == 2:
                shape = "nd"
            else:
                raise ValueError(f"Unknown shape of representations: {self.representation.shape}")
        else:
            shape = self._shape
        return shape

    @shape.setter
    def shape(self, v: SHAPE_TYPE) -> None:
        """Allow setting the shape as before"""
        self._shape = v
        return self._shape

    def unique_identifier(self) -> str:
        """
        Generates a unique identifier for the SingleLayerRepresentation object.

        Returns:
            A string representing the unique identifier.
        """
        assert all(
            [
                self._architecture_name is not None,
                self._train_dataset is not None,
                self._seed is not None,
            ],
        ), "SingleLayerRepresentation has not been set with the necessary information."
        return "__".join(
            [  # type:ignore
                self._setting_identifier,
                self._architecture_name,
                self._train_dataset,
                str(self._seed),
                self._representation_dataset,
                str(self.layer_id),
            ]
        )


class SingleLayerVisionRepresentation(SingleLayerRepresentation):
    def _extract_representation(self) -> torch.Tensor | np.ndarray:
        return get_single_layer_vision_representation_on_demand(
            architecture_name=self._architecture_name,
            train_dataset=self._train_dataset,
            seed=self._seed,
            setting_identifier=self._setting_identifier,
            representation_dataset=self._representation_dataset,
            layer_id=self.layer_id,
        )


@dataclass
class ModelRepresentations:
    setting_identifier: str | None
    architecture_name: str
    train_dataset: str
    seed: int  # Additional identifier to distinguish between different models with the same name
    representation_dataset: str
    representations: tuple[SingleLayerRepresentation, ...]  # immutable to maintain ordering

    def __post_init__(self):
        """Automatically adds the ModelRepresentations infos to the SingleLayerRepresentations infos."""
        if self.representations is not None:
            self._set_single_layer_infos()
        else:
            logger.warning(
                """ModelRepresentations has not been set with the necessary information. Make sure to call `_set_single_layer_infos` or saving will fail!"""
            )

    def _set_single_layer_infos(self):
        for rep in self.representations:
            rep._setting_identifier = self.setting_identifier
            rep._architecture_name = self.architecture_name
            rep._train_dataset = self.train_dataset
            rep._seed = self.seed
            rep._representation_dataset = self.representation_dataset
            rep._setting_identifier = self.setting_identifier


def get_vision_representation_on_demand(
    architecture_name: str,
    train_dataset: str,
    seed_id: int,
    setting_identifier: str | None,
    representation_dataset: str,
) -> ModelRepresentations:
    """Creates Model Representations with representations that can be extracted only when needed)"""
    if setting_identifier == "Normal":
        model_info: ds.ModelInfo = get_vision_model_info(
            architecture_name=architecture_name,
            dataset=train_dataset,
            seed_id=seed_id,
        )
    else:
        model_info: ds.ModelInfo = get_vision_model_info(
            architecture_name=architecture_name,
            dataset=train_dataset,
            seed_id=seed_id,
            setting_identifier=setting_identifier,
        )
    model_type: AbsActiExtrArch = load_model_from_info_file(model_info, load_ckpt=True)
    n_layers = len(model_type.hooks)
    # ---------- Create the on-demand-callable functions for each layer ---------- #
    all_single_layer_reps = []
    for i in range(n_layers):
        all_single_layer_reps.append(SingleLayerVisionRepresentation(i))
    model_rep = ModelRepresentations(
        setting_identifier=model_info.setting_identifier,
        architecture_name=model_info.architecture,
        seed=model_info.seed,
        train_dataset=model_info.dataset,
        representation_dataset=representation_dataset,
        representations=tuple(all_single_layer_reps),
    )
    return model_rep


def convert_to_path_compatible(s: str) -> str:
    return s.replace("\\", "-").replace("/", "-")
