from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

from vision.util import name_conventions as nc
from vision.util.status_check import output_json_has_nans

# Static Naming conventions for writing out files and finding the files again.

logger = logging.getLogger(__name__)


def load_json(filepath: str | Path) -> Any:
    """Load the json again

    :param filepath:
    :return:
    """
    with open(str(filepath)) as f:
        ret = json.load(f)
    return ret


@dataclass(frozen=True)
class GroupMetrics:
    all_to_all: list[list[float]] | object
    all_to_all_mean: float
    last_to_others: list[float] | object
    last_to_others_mean: float
    last_to_first: float


class Dataset(Enum):
    """Info which dataset should be used"""

    TEST = "TEST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    SPLITCIFAR100 = "SplitCIFAR100"
    IMAGENET = "ImageNet"
    IMAGENET100 = "ImageNet100"
    DermaMNIST = "DermaMNIST"
    TinyIMAGENET = "TinyImageNet"


class BaseArchitecture(Enum):
    VGG11 = "VGG11"
    VGG16 = "VGG16"
    VGG19 = "VGG19"
    DYNVGG19 = "DYNVGG19"
    RESNET18 = "ResNet18"
    RESNET34 = "ResNet34"
    RESNET50 = "ResNet50"
    RESNET101 = "ResNet101"
    DYNRESNET101 = "DYNResNet101"
    DENSENET121 = "DenseNet121"
    DENSENET161 = "DenseNet161"


class Augmentation(Enum):
    TRAIN = "train"
    VAL = "val"
    NONE = "none"


@dataclass
class ArchitectureInfo:
    arch_type_str: str
    arch_kwargs: dict
    checkpoint: str | Path | None
    hooks: tuple[Hook] | None


@dataclass(frozen=True)
class ModelInfo:
    """
    Most important ModelInfo class. This contains all the parameters,
    paths and other information needed to load a model and access its results.
    """

    dir_name: str
    group_id: int

    # HParams  -- Needed for continuation and reinit
    architecture: str
    dataset: str
    learning_rate: float
    split: int
    weight_decay: float
    batch_size: int

    # Basic paths
    path_data_root: Path
    path_ckpt_root: Path

    path_sequence_dir_path: Path = field(init=False)
    sequence_single_json: Path = field(init=False)
    sequence_ensemble_json: Path = field(init=False)
    sequence_calibrated_ensemble_json: Path = field(init=False)

    robust_sequence_single_json: Path = field(init=False)
    robust_sequence_ensemble_json: Path = field(init=False)
    robust_calib_sequence_ensemble_json: Path = field(init=False)

    path_ckpt: Path = field(init=False)
    path_activations: Path = field(init=False)
    path_predictions_train: Path = field(init=False)
    path_predictions_test: Path = field(init=False)
    path_groundtruths_train: Path = field(init=False)
    path_groundtruths_test: Path = field(init=False)
    path_output_json: Path = field(init=False)
    path_last_metrics_json: Path = field(init=False)
    path_calib_json: Path = field(init=False)
    path_generalization_json: Path = field(init=False)
    path_train_log: Path = field(init=False)
    path_train_info_json: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "path_ckpt", self.path_ckpt_root / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME)
        object.__setattr__(self, "path_activations", self.path_ckpt_root / nc.ACTI_DIR_NAME)
        object.__setattr__(
            self, "path_predictions_train", self.path_ckpt_root / nc.ACTI_DIR_NAME / nc.MODEL_TRAIN_PD_TMPLT
        )
        object.__setattr__(
            self, "path_predictions_test", self.path_ckpt_root / nc.ACTI_DIR_NAME / nc.MODEL_TEST_PD_TMPLT
        )
        object.__setattr__(
            self, "path_groundtruths_train", self.path_ckpt_root / nc.ACTI_DIR_NAME / nc.MODEL_TRAIN_GT_TMPLT
        )
        object.__setattr__(
            self, "path_groundtruths_test", self.path_ckpt_root / nc.ACTI_DIR_NAME / nc.MODEL_TEST_GT_TMPLT
        )
        object.__setattr__(self, "path_output_json", self.path_ckpt_root / nc.OUTPUT_TMPLT)
        object.__setattr__(self, "path_last_metrics_json", self.path_ckpt_root / nc.LAST_METRICS_TMPLT)
        object.__setattr__(self, "path_train_log", self.path_ckpt_root / nc.LOG_DIR)
        object.__setattr__(self, "path_train_info_json", self.path_ckpt_root / nc.KE_INFO_FILE)

    def is_trained(self) -> bool:
        """Checks whether model has been trained by testing if output_json exists."""
        return self.path_output_json.exists()

    def model_converged(self) -> bool:
        if self.is_trained():
            output_json = load_json(self.path_output_json)
            if output_json_has_nans(output_json) or (output_json["val"]["accuracy"] < 0.2):
                return False
            else:
                return True
        else:
            return False

    def training_succeeded(self, unregularized_model: ModelInfo) -> bool:
        """
        Checks that the new model converged to reasonable accuracy,
        relative to its first (unregularized model).
        Should the difference
        be too large, the training is considered a failure and training should abort."""
        if self.is_trained():
            own_accuracy = load_json(self.path_output_json)["val"]["accuracy"]
            first_accuracy = load_json(unregularized_model.path_output_json)["val"]["accuracy"]
            if first_accuracy - own_accuracy > 0.15:
                return False
            else:
                return True

    def has_checkpoint(self):
        """Checks whether model has a checkpoint."""
        return self.path_ckpt.exists()

    def has_predictions(self) -> bool:
        """
        Returns true if model has prediction logits and groundtruths of the test set already.
        """
        preds = self.path_predictions_test
        gts = self.path_groundtruths_test
        return preds.exists() and gts.exists()

    def model_is_finished(self) -> bool:
        """Checks whether model has been trained and checkpoint exists."""
        return self.is_trained() and self.has_checkpoint()


@dataclass
class Params:
    """Dataclass containing all hyperparameters needed for a basic training"""

    num_epochs: int
    batch_size: int
    label_smoothing: bool
    label_smoothing_val: float
    architecture_name: str
    save_last_checkpoint: bool
    momentum: float
    learning_rate: float
    nesterov: bool
    weight_decay: float
    cosine_annealing: bool
    gamma: float
    split: int
    dataset: str
    advanced_da: bool = True
    optimizer: dict[str, Any] | None = None


@dataclass
class Hook:
    architecture_index: int
    name: str
    keys: list[str]
    n_channels: int = 0
    downsampling_steps: int = -1  # == Undefined! - Sometimes dynamic --> Set when initialized the model!
    resolution: tuple[int, int] = (0, 0)  # == Undefined! - Depends on Dataset --> Set when initialized!
    resolution_relative_depth: float = -1