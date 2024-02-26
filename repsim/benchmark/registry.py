from dataclasses import dataclass
from typing import get_args
from typing import List

from graphs.get_reps import get_graph_representations
from repsim.benchmark.config import DOMAIN_TYPE
from repsim.benchmark.config import EXPERIMENT_DICT
from repsim.benchmark.config import EXPERIMENT_IDENTIFIER
from repsim.benchmark.config import EXPERIMENT_SEED
from repsim.benchmark.config import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.config import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.config import NLP_ARCHITECTURE_TYPE
from repsim.benchmark.config import NLP_DATASET_TRAINED_ON
from repsim.benchmark.config import SETTING_IDENTIFIER
from repsim.benchmark.config import VISION_ARCHITECTURE_TYPE
from repsim.benchmark.config import VISION_DATASET_TRAINED_ON
from repsim.utils import ModelRepresentations
from vision.get_reps import get_vision_representations

# ---------------------------- SIMILARITY_METRICS --------------------------- #
# Maybe only?


@dataclass
class TrainedModel:
    """
    Class that should contain all the infos about the trained models.
    This can be used to filter the models and categorize them into groups.
    """

    domain: DOMAIN_TYPE
    architecture: VISION_ARCHITECTURE_TYPE | NLP_ARCHITECTURE_TYPE | GRAPH_ARCHITECTURE_TYPE
    train_dataset: VISION_DATASET_TRAINED_ON | NLP_DATASET_TRAINED_ON | GRAPH_DATASET_TRAINED_ON
    experiment_identifier: EXPERIMENT_IDENTIFIER
    setting_identifier: SETTING_IDENTIFIER
    additional_kwargs: dict  # Maybe one can remove this to make it more general

    def get_representation(self, representation_dataset: str) -> ModelRepresentations:
        """
        This function should return the representation of the model.
        """
        if self.domain == "VISION":
            return get_vision_representations(
                architecture_name=self.architecture,
                train_dataset=self.train_dataset,
                seed_id=self.additional_kwargs["seed_id"],
                setting_identifier=self.setting_identifier,
                representation_dataset=representation_dataset,
            )
        elif self.domain == "NLP":
            raise NotImplementedError
        elif self.domain == "GRAPHS":
            return get_graph_representations(
                architecture_name=self.architecture,
                train_dataset=self.train_dataset,
                seed_id=self.additional_kwargs["seed_id"],
                experiment_identifier=self.experiment_identifier,
                setting_identifier=self.setting_identifier,
                representation_dataset=representation_dataset,
            )
        else:
            raise ValueError("Unknown domain type")


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
                for identifier in ["Normal"]:
                    all_trained_vision_models.append(
                        TrainedModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            setting_identifier=identifier,
                            additional_kwargs={"seed_id": i, "setting_identifier": None},
                        )
                    )
    return all_trained_vision_models


def all_trained_nlp_models() -> list[TrainedModel]:
    all_trained_nlp_models = []
    # Enter models here
    return all_trained_nlp_models


def all_trained_graph_models() -> list[TrainedModel]:
    all_trained_models = []

    for i in get_args(EXPERIMENT_SEED):
        for arch in get_args(GRAPH_ARCHITECTURE_TYPE):
            for dataset in get_args(GRAPH_DATASET_TRAINED_ON):
                for identifier in ["layer_test"]:
                    for setting in EXPERIMENT_DICT[identifier]:
                        all_trained_models.append(
                            TrainedModel(
                                domain="GRAPHS",
                                architecture=arch,
                                train_dataset=dataset,
                                experiment_identifier=identifier,
                                setting_identifier=setting,
                                additional_kwargs={"seed_id": i, "setting_identifier": None},
                            )
                        )
    return all_trained_models


ALL_TRAINED_MODELS: List[TrainedModel] = []
ALL_TRAINED_MODELS.extend(all_trained_vision_models())
ALL_TRAINED_MODELS.extend(all_trained_nlp_models())
ALL_TRAINED_MODELS.extend(all_trained_graph_models())
