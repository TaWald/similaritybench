from dataclasses import dataclass
from typing import List, Literal
from repsim.utils import ModelRepresentations
from vision.get_reps import get_vision_representations, VisionModelInfo


# -------------------- All categories  trained model that can -------------------- #
DOMAIN_TYPE = Literal["VISION", "NLP", "GRAPHS"]
# ----------------------------- All Architectures ---------------------------- #
VISION_ARCHITECTURE_TYPE = Literal["ResNet18", "ResNet34", "ResNet101", "VGG11", "VGG19", "ViT-b19"]
NLP_ARCHITECTURE_TYPE = Literal["BERT", "LSTM", "GRU"]
GRAPH_ARCHITECTURE_TYPE = Literal["GCN", "GAT", "GraphSAGE"]

# ----------------------------- Datasets trained on ---------------------------- #
VISION_DATASET_TRAINED_ON = Literal["CIFAR10", "CIFAR100", "ImageNet"]
NLP_DATASET_TRAINED_ON = Literal["IMDB"]
GRAPH_DATASET_TRAINED_ON = Literal["Cora", "CiteSeer", "PubMed"]

# ---------------------------- Identifier_settings --------------------------- #
# These are shared across domains and datasets
EXPERIMENT_IDENTIFIER = Literal[
    "Normal",
    "RandomInit",
    "RandomLabels_25",
    "RandomLabels_50",
    "RandomLabels_75",
    "RandomLabels_100",
    "Shortcut_25",
    "Shortcut_50",
    "Shortcut_75",
]

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
    identifier: EXPERIMENT_IDENTIFIER
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
                setting_identifier=self.identifier,
                representation_dataset=representation_dataset,
            )
        elif self.domain == "NLP":
            raise NotImplementedError
        elif self.domain == "GRAPHS":
            raise NotImplementedError
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
                            identifier=identifier,
                            additional_kwargs={"seed_id": i, "setting_identifier": None},
                        )
                    )
    return all_trained_vision_models


def all_trained_nlp_models() -> list[TrainedModel]:
    all_trained_nlp_models = []
    # Enter models here
    return all_trained_nlp_models


def all_trained_graph_models() -> list[TrainedModel]:
    all_trained_graph_models = []
    # Enter models here
    return all_trained_graph_models


ALL_TRAINED_MODELS: List[TrainedModel] = []
ALL_TRAINED_MODELS.extend(all_trained_vision_models())
ALL_TRAINED_MODELS.extend(all_trained_nlp_models())
ALL_TRAINED_MODELS.extend(all_trained_graph_models())
