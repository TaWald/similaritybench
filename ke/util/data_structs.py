from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

from ke.util import name_conventions as nc
from ke.util.status_check import output_json_has_nans

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


class LenseArchitecture(Enum):
    S = "small"
    M = "medium"
    L = "large"


class BasicPretrainableArchitectures(Enum):
    TV_VGG11: str = "TvVGG11"
    TV_VGG16: str = "TvVGG16"
    TV_VGG19: str = "TvVGG19"
    TV_RESNET18: str = "TvResNet18"
    TV_RESNET34: str = "TvResNet34"


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
class TransferLayersInfo:
    # KE Trans Params
    trans_hooks: list[int] | None
    trans_depth: int | None
    trans_kernel: int | None


@dataclass(frozen=True)
class ParallelModelInfo:
    dir_name: str
    n_models: int
    group_id: int

    architecture: str
    dataset: str

    path_root: Path
    # Results json
    path_output_json: Path = field(init=False)
    sequence_ensemble_json: Path = field(init=False)
    sequence_calibrated_ensemble_json: Path = field(init=False)
    path_last_metrics_json: Path = field(init=False)
    path_calib_json: Path = field(init=False)
    path_train_log: Path = field(init=False)
    path_train_info_json: Path = field(init=False)

    robust_sequence_ensemble_json: Path = field(init=False)
    robust_calib_sequence_ensemble_json: Path = field(init=False)

    path_ckpt: Path = field(init=False)
    path_activations: Path = field(init=False)
    path_predictions_train: Path = field(init=False)
    path_predictions_test: Path = field(init=False)
    path_groundtruths_train: Path = field(init=False)
    path_groundtruths_test: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "sequence_ensemble_json", self.path_root / nc.ENSEMBLE_RESULTS_FILE)
        object.__setattr__(
            self,
            "sequence_calibrated_ensemble_json",
            self.path_root / nc.CALIBRATED_ENSEMBLE_RESULTS_FILE,
        )
        object.__setattr__(self, "robust_sequence_single_json", self.path_root / nc.ROBUST_SINGLE_RESULTS)
        object.__setattr__(self, "robust_sequence_ensemble_json", self.path_root / nc.ROBUST_ENSEMBLE_RESULTS)
        object.__setattr__(
            self,
            "robust_calib_sequence_ensemble_json",
            self.path_root / nc.ROBUST_CALIB_ENS_RESULTS,
        )

        object.__setattr__(self, "path_ckpt", self.path_root / nc.CKPT_DIR_NAME)
        self.path_ckpt.mkdir(exist_ok=True, parents=True)
        object.__setattr__(self, "path_activations", self.path_root / nc.ACTI_DIR_NAME)
        object.__setattr__(
            self, "path_predictions_train", self.path_root / nc.ACTI_DIR_NAME / nc.MODEL_TRAIN_PD_TMPLT
        )
        object.__setattr__(self, "path_predictions_test", self.path_root / nc.ACTI_DIR_NAME / nc.MODEL_TEST_PD_TMPLT)
        object.__setattr__(
            self, "path_groundtruths_train", self.path_root / nc.ACTI_DIR_NAME / nc.MODEL_TRAIN_GT_TMPLT
        )
        object.__setattr__(self, "path_groundtruths_test", self.path_root / nc.ACTI_DIR_NAME / nc.MODEL_TEST_GT_TMPLT)
        object.__setattr__(self, "path_output_json", self.path_root / nc.OUTPUT_TMPLT)
        object.__setattr__(self, "path_last_metrics_json", self.path_root / nc.LAST_METRICS_TMPLT)
        object.__setattr__(self, "path_calib_json", self.path_root / nc.CALIB_TMPLT)
        object.__setattr__(self, "path_generalization_json", self.path_root / nc.GENERALIZATION_TMPLT)
        object.__setattr__(self, "path_train_log", self.path_root / nc.LOG_DIR)
        object.__setattr__(self, "path_train_info_json", self.path_root / nc.KE_INFO_FILE)

    def get_root_path(self):
        return self.path_root


@dataclass(frozen=True)
class FirstModelInfo:
    """
    Most important ModelInfo class. This contains all the parameters,
    paths and other information needed to load a model and access its results.
    """

    dir_name: str
    model_id: int  # 0 Indicates it has not been regularized.
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
        object.__setattr__(self, "path_sequence_dir_path", self.path_ckpt_root.parent)
        object.__setattr__(self, "sequence_single_json", self.path_sequence_dir_path / nc.SINGLE_RESULTS_FILE)
        object.__setattr__(self, "sequence_ensemble_json", self.path_sequence_dir_path / nc.ENSEMBLE_RESULTS_FILE)
        object.__setattr__(
            self,
            "sequence_calibrated_ensemble_json",
            self.path_sequence_dir_path / nc.CALIBRATED_ENSEMBLE_RESULTS_FILE,
        )
        object.__setattr__(
            self, "robust_sequence_single_json", self.path_sequence_dir_path / nc.ROBUST_SINGLE_RESULTS
        )
        object.__setattr__(
            self, "robust_sequence_ensemble_json", self.path_sequence_dir_path / nc.ROBUST_ENSEMBLE_RESULTS
        )
        object.__setattr__(
            self,
            "robust_calib_sequence_ensemble_json",
            self.path_sequence_dir_path / nc.ROBUST_CALIB_ENS_RESULTS,
        )

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
        object.__setattr__(self, "path_calib_json", self.path_ckpt_root / nc.CALIB_TMPLT)
        object.__setattr__(self, "path_generalization_json", self.path_ckpt_root / nc.GENERALIZATION_TMPLT)
        object.__setattr__(self, "path_train_log", self.path_ckpt_root / nc.LOG_DIR)
        object.__setattr__(self, "path_train_info_json", self.path_ckpt_root / nc.KE_INFO_FILE)

    def sequence_performance_exists(self, single: bool, ensemble: bool, calibrated: bool):
        single_exists = True
        ensemble_exists = True
        calibrated_exists = True
        if single:
            single_exists = self.sequence_single_json.exists()
        if ensemble:
            ensemble_exists = self.sequence_ensemble_json.exists()
        if calibrated:
            calibrated_exists = self.sequence_calibrated_ensemble_json.exists()
        return single_exists and ensemble_exists and calibrated_exists

    def robust_sequence_performance_exists(self, single: bool, ensemble: bool, calibrated: bool):
        single_exists = True
        ensemble_exists = True
        calibrated_exists = True
        if single:
            single_exists = self.robust_sequence_single_json.exists()
        if ensemble:
            ensemble_exists = self.robust_sequence_ensemble_json.exists()
        if calibrated:
            calibrated_exists = self.robust_calib_sequence_ensemble_json.exists()
        return single_exists and ensemble_exists and calibrated_exists

    def is_trained(self) -> bool:
        """Checks whether model has been trained by testing if output_json exists."""
        return self.path_output_json.exists()

    def is_calibrated(self) -> bool:
        """Checks whether model has been calibrated by testing if calib_json exists."""
        return self.path_calib_json.exists()

    def model_converged(self) -> bool:
        if self.is_trained():
            output_json = load_json(self.path_output_json)
            if output_json_has_nans(output_json) or (output_json["val"]["accuracy"] < 0.2):
                return False
            else:
                return True
        else:
            return False

    def training_succeeded(self, unregularized_model: FirstModelInfo) -> bool:
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

    def measured_generalization(self):
        """Checks whether generalization has been measured."""
        return self.path_generalization_json.exists()

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


@dataclass(frozen=True)
class BasicTrainingInfo(FirstModelInfo):
    experiment_name: str
    experiment_description: str


@dataclass(frozen=True)
class PretrainedTrainingInfo(BasicTrainingInfo):
    """
    Scheme where new model gets regularized in its predictions in order to increase diversity.
    """

    pretrained: bool
    warmup_pretrained: bool
    linear_probe_only: bool


@dataclass(frozen=True)
class KEOutputTrainingInfo(BasicTrainingInfo):
    """
    Scheme where new model gets regularized in its predictions in order to increase diversity.
    """

    dissimilarity_loss: str
    crossentropy_loss_weight: float
    dissimilarity_loss_weight: float
    softmax_metrics: bool
    epochs_before_regularization: int
    pc_grad: bool


@dataclass(frozen=True)
class KEAdversarialLenseOutputTrainingInfo(BasicTrainingInfo):
    lense_reconstruction_weight: float
    lense_adversarial_weight: float
    lense_setting: str
    adversarial_loss: str
    path_lense_examples: Path = field(init=False)
    path_lense_checkpoint: Path = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "path_lense_examples", self.path_ckpt_root / nc.LENSE_EXAMPLE_DIR_NAME)
        object.__setattr__(
            self, "path_lense_checkpoint", self.path_ckpt_root / nc.CKPT_DIR_NAME / nc.KE_LENSE_CKPT_NAME
        )


@dataclass(frozen=True)
class KEAlternatingOutputTrainingInfo(KEOutputTrainingInfo):
    """
    Scheme where new model get regularized in its prediction but that loss is only supposed to only influence
    layers above the hook_id!
    """

    hook_id: int


@dataclass(frozen=True)
class KESubTrainingInfo(BasicTrainingInfo, TransferLayersInfo):
    """
    Scheme where old models try to approximate the new ones and just substract it from the original one.
    Similarity needs to be learned Dissimilarity doesn't!
    """

    similarity_loss: str
    similarity_loss_weight: float
    crossentropy_loss_weight: float
    epochs_before_regularization: int


@dataclass(frozen=True)
class KETrainingInfo(KESubTrainingInfo):
    """
    Scheme where old models approximate the new one.
    Adversarial Approach where old models try to approximate and new try to make dissimilar
    """

    # Dissimilarity stuff
    dissimilarity_loss: str
    dissimilarity_loss_weight: float
    aggregate_source_reps: bool
    softmax_metrics: bool


@dataclass(frozen=True)
class KEUnuseableDownstreamTrainingInfo(BasicTrainingInfo, TransferLayersInfo):
    """
    Scheme where new model tries to use partial backbone of old models to predict the samples.
    The CE loss is backpropped and transfer layers try to minimize it while gradient reversal makes previous part
    try and maximize it.
    """

    # Basic shit.
    crossentropy_loss_weight: float
    transfer_loss_weight: float
    gradient_reversal_scale: float
    epochs_before_regularization: int


@dataclass(frozen=True)
class KEAdversarialTrainingInfo(BasicTrainingInfo, TransferLayersInfo):
    """
    Scheme where new model tries to approximate intermediate representations of old models.
    Gradient reversal happens inbetween to make the previous layers bad to predict the representations.
    """

    adversarial_loss: str
    adversarial_loss_weight: float
    gradient_reversal_scale: float
    crossentropy_loss_weight: float
    epochs_before_regularization: int


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
