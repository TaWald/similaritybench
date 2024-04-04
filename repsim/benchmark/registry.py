from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import get_args
from typing import Literal
from typing import Optional

import repsim.benchmark.paths
import repsim.nlp
import repsim.utils
import torch
from graphs.get_reps import get_graph_representations
from repsim.benchmark.types_globals import DOMAIN_TYPE
from repsim.benchmark.types_globals import EXPERIMENT_DICT
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_SEED
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import LAYER_EXPERIMENT_NAME
from repsim.benchmark.types_globals import NLP_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import NLP_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import STANDARD_SETTING
from repsim.benchmark.types_globals import VISION_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import VISION_DATASET_TRAINED_ON
from repsim.utils import get_vision_representation_on_demand


# from repsim.nlp import get_representations


# ---------------------------- SIMILARITY_METRICS --------------------------- #
# Maybe only?


# ToDo:
#   This is currently pretty redundant to the ModelRepresentation class. We should very likely merge it.
@dataclass
class TrainedModel:
    """
    Class that should contain all the infos about the trained models.
    This can be used to filter the models and categorize them into groups.
    """

    domain: DOMAIN_TYPE
    architecture: VISION_ARCHITECTURE_TYPE | NLP_ARCHITECTURE_TYPE | GRAPH_ARCHITECTURE_TYPE
    train_dataset: VISION_DATASET_TRAINED_ON | NLP_DATASET_TRAINED_ON | GRAPH_DATASET_TRAINED_ON
    identifier: SETTING_IDENTIFIER
    seed: int
    additional_kwargs: Optional[dict] = field(
        kw_only=True, default=None
    )  # Maybe one can remove this to make it more general

    def get_representation(
        self, representation_dataset: Optional[str] = None, **kwargs
    ) -> repsim.utils.ModelRepresentations:
        """
        This function should return the representation of the model.
        """

        if representation_dataset is None:
            representation_dataset = self.train_dataset

        if self.domain == "VISION":
            return get_vision_representation_on_demand(
                architecture_name=self.architecture,
                train_dataset=self.train_dataset,
                seed_id=self.seed,
                setting_identifier=self.identifier,
                representation_dataset=representation_dataset,
            )
        elif self.domain == "NLP":
            raise ValueError("NLP Models should exist as HuggingfaceModel instances.")
        if self.domain == "GRAPHS":
            return get_graph_representations(
                architecture_name=self.architecture,
                train_dataset=self.train_dataset,
                seed=self.seed,
                setting_identifier=self.identifier,
                representation_dataset=representation_dataset,
            )
        else:
            raise ValueError("Unknown domain type")

    def _get_unique_model_identifier(self) -> str:
        """
        This function should return a unique identifier for the model.
        """
        return f"{self.domain}_{self.architecture}_{self.train_dataset}_{self.identifier}_{self.seed}_{self.additional_kwargs}"


@dataclass
class NLPDataset:
    name: str  # human-readable identifier
    path: str  # huggingface hub path, e.g., sst2. Will be ignored if `local_path` is given
    config: Optional[str] = None  # huggingface hub dataset config, e.g., mnli for glue.
    local_path: Optional[str] = (
        None  # path to a local dataset directory. If given, the dataset will be loaded from here.
    )
    split: str = "train"  # part of the dataset that was/will be used
    feature_column: Optional[str] = None
    label_column: Optional[str] = "label"

    # Information about shortcuts
    shortcut_rate: Optional[float] = None
    shortcut_seed: Optional[int] = None

    # Information about changed labels
    memorization_rate: Optional[float] = None
    memorization_n_new_labels: Optional[int] = None
    memorization_seed: Optional[int] = None

    # Information about augmentation
    augmentation_type: Optional[str] = None
    augmentation_rate: Optional[float] = None


NLP_TRAIN_DATASETS = {
    "sst2": NLPDataset("sst2", "sst2"),
    "sst2_sc_rate0558": NLPDataset("sst2_sc_rate0558", path="sst2", shortcut_rate=0.558, shortcut_seed=0),
    "sst2_sc_rate0668": NLPDataset("sst2_sc_rate0668", path="sst2", shortcut_rate=0.668, shortcut_seed=0),
    "sst2_sc_rate0779": NLPDataset("sst2_sc_rate0779", path="sst2", shortcut_rate=0.779, shortcut_seed=0),
    "sst2_sc_rate0889": NLPDataset("sst2_sc_rate0889", path="sst2", shortcut_rate=0.889, shortcut_seed=0),
    "sst2_sc_rate10": NLPDataset("sst2_sc_rate10", path="sst2", shortcut_rate=1.0, shortcut_seed=0),
}
NLP_REPRESENTATION_DATASETS = {
    "sst2": NLPDataset("sst2", path="sst2", split="validation"),
    "sst2_sc_rate0": NLPDataset(
        name="sst2_sc_rate0",
        path="sst2",
        # The local version would be useful if the modified tokenizer is saved with the trained models. But it's not,
        # so the shortcuts are added on the fly.
        # local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "shortcut" / "sst2_sc_rate0"),
        split="validation",
        feature_column="sentence",
        shortcut_rate=0,
        shortcut_seed=0,
    ),
    "sst2_sc_rate0558": NLPDataset(
        name="sst2_sc_rate00558",
        path="sst2",
        split="validation",
        feature_column="sentence",
        shortcut_rate=0.558,
        shortcut_seed=0,
    ),
    "sst2_mem_rate0": NLPDataset("sst2", "sst2", split="validation"),
    "sst2_aug_rate0": NLPDataset("sst2", "sst2", split="validation"),
}


@dataclass(kw_only=True)
class NLPModel(TrainedModel):
    domain: DOMAIN_TYPE = "NLP"
    architecture: NLP_ARCHITECTURE_TYPE = "BERT-L"
    path: str
    tokenizer_name: str
    train_dataset: Literal[
        "sst2", "sst2_sc_rate0558", "sst2_sc_rate0668", "sst2_sc_rate0779", "sst2_sc_rate0889", "sst2_sc_rate10"
    ]
    model_type: Literal["sequence-classification"] = "sequence-classification"
    token_pos: Optional[int] = (
        None  # Index of the token relevant for classification. If set, only the representation of this token will be extracted.
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset_obj: NLPDataset = field(init=False)

    def __post_init__(self):
        self.train_dataset_obj = NLP_TRAIN_DATASETS[self.train_dataset]

    def get_representation(self, representation_dataset_id: str) -> repsim.utils.ModelRepresentations:
        if self.domain != "NLP":
            raise ValueError("This class should only be used for NLP models with huggingface.")
        if representation_dataset_id not in NLP_REPRESENTATION_DATASETS.keys():
            raise ValueError(
                f"Dataset must be one of {list(NLP_REPRESENTATION_DATASETS.keys())}, but is {representation_dataset_id}"
            )
        representation_dataset = NLP_REPRESENTATION_DATASETS[representation_dataset_id]
        reps = repsim.nlp.get_representations(
            self.path,
            self.model_type,
            self.tokenizer_name,
            representation_dataset.path,
            representation_dataset.config,
            representation_dataset.local_path,
            representation_dataset.split,
            self.device,
            self.token_pos,
            shortcut_rate=representation_dataset.shortcut_rate,
            shortcut_seed=representation_dataset.shortcut_seed,
            feature_column=representation_dataset.feature_column,
        )
        slrs = tuple([repsim.utils.SingleLayerRepresentation(i, r, "nd") for i, r in enumerate(reps)])
        return repsim.utils.ModelRepresentations(
            self.identifier,
            self.architecture,
            self.train_dataset,
            self.seed,
            representation_dataset_id,
            slrs,
        )


@dataclass
class TrainedModelRep(TrainedModel):
    """
    The same as above, just also has the ID of the layer.
    """

    layer_id: int


def all_trained_vision_models() -> list[TrainedModel]:
    all_trained_vision_models = []
    for i in range(5):
        for arch in ["ResNet18"]:
            for dataset in ["CIFAR10", "CIFAR100"]:
                for identifier in [STANDARD_SETTING]:
                    all_trained_vision_models.append(
                        TrainedModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=identifier,
                            seed=i,
                            additional_kwargs={},
                        )
                    )
    for i in range(5):
        for arch in ["ResNet18"]:
            for dataset in [
                "ColorDot_100_CIFAR10DataModule",
                "ColorDot_75_CIFAR10DataModule",
                "ColorDot_50_CIFAR10DataModule",
                "ColorDot_25_CIFAR10DataModule",
                "ColorDot_0_CIFAR10DataModule",
            ]:
                for identifier in ["Shortcut_ColorDot"]:
                    all_trained_vision_models.append(
                        TrainedModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=identifier,
                            seed=i,
                            additional_kwargs={},
                        )
                    )
    for i in range(2):
        for arch in ["ResNet18"]:
            for dataset in [
                "Gauss_Max_CIFAR10DataModule",
                "Gauss_L_CIFAR10DataModule",
                "Gauss_M_CIFAR10DataModule",
                "Gauss_S_CIFAR10DataModule",
                "ColorDot_Off_CIFAR10DataModule",  # N
            ]:
                for identifier in ["GaussNoise"]:
                    all_trained_vision_models.append(
                        TrainedModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=identifier,
                            seed=i,
                            additional_kwargs={"setting_identifier": identifier},
                        )
                    )

    return all_trained_vision_models


def all_trained_nlp_models() -> Sequence[TrainedModel]:
    base_sst2_models = [
        NLPModel(
            train_dataset="sst2",
            identifier="Normal",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
        )
        for i in range(10)
    ]

    shortcut_sst2_models = []
    for seed in range(10):
        for rate in ["0558", "0668", "0779", "0889", "10"]:
            shortcut_sst2_models.append(
                NLPModel(
                    identifier="Shortcut_0",  # always using the same identifier, because actual sc rate does not match with labels here
                    seed=seed,
                    train_dataset=f"sst2_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH / "shortcut" / f"sst2_pre{seed}_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )

    return base_sst2_models + shortcut_sst2_models


def all_trained_graph_models() -> list[TrainedModel]:
    all_trained_models = []

    for i in get_args(GRAPH_EXPERIMENT_SEED):
        for arch in get_args(GRAPH_ARCHITECTURE_TYPE):
            for dataset in get_args(GRAPH_DATASET_TRAINED_ON):
                for experiment in [LAYER_EXPERIMENT_NAME, LABEL_EXPERIMENT_NAME]:
                    for setting in EXPERIMENT_DICT[experiment]:
                        all_trained_models.append(
                            TrainedModel(
                                domain="GRAPHS",
                                architecture=arch,
                                train_dataset=dataset,
                                identifier=setting,
                                seed=i,
                                additional_kwargs={},
                            )
                        )
    return all_trained_models


ALL_TRAINED_MODELS: list[TrainedModel | NLPModel] = []
ALL_TRAINED_MODELS.extend(all_trained_vision_models())
ALL_TRAINED_MODELS.extend(all_trained_nlp_models())
ALL_TRAINED_MODELS.extend(all_trained_graph_models())
