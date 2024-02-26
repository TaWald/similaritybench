from dataclasses import dataclass
from typing import Literal

import repsim.nlp
from repsim.utils import ModelRepresentations
from repsim.utils import SingleLayerRepresentation
from vision.get_reps import get_vision_representations

# -------------------- All categories  trained model that can -------------------- #
DOMAIN_TYPE = Literal["VISION", "NLP", "GRAPHS"]
# ----------------------------- All Architectures ---------------------------- #
VISION_ARCHITECTURE_TYPE = Literal["ResNet18", "ResNet34", "ResNet101", "VGG11", "VGG19", "ViT-b19"]
NLP_ARCHITECTURE_TYPE = Literal["BERT"]
GRAPH_ARCHITECTURE_TYPE = Literal["GCN", "GAT", "GraphSAGE"]

# ----------------------------- Datasets trained on ---------------------------- #
VISION_DATASET_TRAINED_ON = Literal["CIFAR10", "CIFAR100", "ImageNet"]
NLP_DATASET_TRAINED_ON = Literal["MNLI", "SST2"]
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

    def get_representation(self, representation_dataset: str, **kwargs) -> ModelRepresentations:
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
            # TODO: this requires so many additional arguments. We should likely have some specialized classes for the
            #  different domains
            additional_required_model_args = ["tokenizer_name", "model_type", "model_path"]
            if not all((key in self.additional_kwargs for key in additional_required_model_args)):
                raise ValueError(f"Unable to load model. One or more of {additional_required_model_args} missing.")

            kwargs["dataset_path"] = kwargs["dataset_path"] if kwargs["dataset_path"] is not None else ""
            kwargs["dataset_config"] = kwargs["dataset_config"] if kwargs["dataset_config"] is not None else ""
            kwargs["dataset_split"] = kwargs["dataset_split"] if kwargs["dataset_split"] is not None else ""

            reps = repsim.nlp.get_representations(
                self.additional_kwargs["model_path"],
                self.additional_kwargs["model_type"],
                self.additional_kwargs["tokenizer_name"],
                kwargs["dataset_path"],
                kwargs["dataset_config"],
                kwargs["dataset_split"],
                kwargs["device"],
                kwargs["token_pos"],
            )
            return ModelRepresentations(
                self.identifier,
                self.architecture,
                self.train_dataset,
                None,
                kwargs["dataset_path"] + kwargs["dataset_config"] + kwargs["dataset_split"],
                tuple(SingleLayerRepresentation(i, r, "nd") for i, r in enumerate(reps)),
            )
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
    return [
        TrainedModel(
            domain="NLP",
            architecture="BERT",
            train_dataset="SST2",
            identifier="Normal",
            additional_kwargs={
                "human_name": "multibert-0-sst2",
                "model_path": "/root/LLM-comparison/outputs/2024-01-31/13-12-49",
                "model_type": "sequence-classification",
                "tokenizer_name": "google/multiberts-seed_0",
            },
        )
    ]


def all_trained_graph_models() -> list[TrainedModel]:
    all_trained_graph_models = []
    # Enter models here
    return all_trained_graph_models


ALL_TRAINED_MODELS: list[TrainedModel] = []
ALL_TRAINED_MODELS.extend(all_trained_vision_models())
ALL_TRAINED_MODELS.extend(all_trained_nlp_models())
ALL_TRAINED_MODELS.extend(all_trained_graph_models())
