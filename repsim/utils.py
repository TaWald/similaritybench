from dataclasses import dataclass
from dataclasses import field

import numpy as np
import torch
from loguru import logger
from repsim.measures.utils import SHAPE_TYPE


@dataclass
class SingleLayerRepresentation:
    layer_id: int
    representation: torch.Tensor | np.ndarray
    shape: SHAPE_TYPE
    _setting_identifier: str | None = field(default=None, init=False)
    _architecture_name: str | None = field(default=None, init=False)
    _train_dataset: str | None = field(default=None, init=False)
    _seed: int | None = field(default=None, init=False)
    _representation_dataset: str | None = field(default=None, init=False)

    def unique_identifier(self) -> str:
        """
        Generates a unique identifier for the SingleLayerRepresentation object.

        Returns:
            A string representing the unique identifier.
        """
        assert all(
            [
                self._setting_identifier is not None,
                self._architecture_name is not None,
                self._train_dataset is not None,
                self._seed is not None,
            ],
        ), "SingleLayerRepresentation has not been set with the necessary information."
        setting = "None" if self._setting_identifier is None else self._setting_identifier
        return "__".join(
            [  # type:ignore
                setting,
                self._architecture_name,
                self._train_dataset,
                str(self._seed),
                self._representation_dataset,
                str(self.layer_id),
            ]
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


def convert_to_path_compatible(s: str) -> str:
    return s.replace("\\", "-").replace("/", "-")
