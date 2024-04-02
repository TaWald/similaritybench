import time
from abc import abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import ExperimentStorer


class AbstractExperiment:
    def __init__(
        self,
        measures: list[Callable],
        representation_dataset: str,
        storage_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Needs the measures to be employed, the dataset to be used and the path where to store the results.
        """
        self.measures: list[Callable] = measures
        self.representation_dataset: str = representation_dataset
        self.storage_path: str = storage_path
        self.kwargs: dict = kwargs

    @abstractmethod
    def eval(self) -> None:
        """Evaluate the results of the experiment"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the experiment storer."""
