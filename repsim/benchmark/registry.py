from dataclasses import dataclass
from typing import get_args

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
from repsim.utils import ModelRepresentations

# from vision.get_reps import get_vision_representations

# from repsim.nlp import get_representations


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
    identifier: SETTING_IDENTIFIER
    additional_kwargs: dict  # Maybe one can remove this to make it more general
    seed: int

    def get_representation(self, representation_dataset: str = None, **kwargs) -> ModelRepresentations:
        """
        This function should return the representation of the model.
        """

        if representation_dataset is None:
            representation_dataset = self.train_dataset

        # if self.domain == "VISION":
        #     return get_vision_representations(
        #         architecture_name=self.architecture,
        #         train_dataset=self.train_dataset,
        #         seed_id=self.additional_kwargs["seed_id"],
        #         setting_identifier=self.identifier,
        #         representation_dataset=representation_dataset,
        #     )
        # elif self.domain == "NLP":
        #     # TODO: this requires so many additional arguments. We should likely have some specialized classes for the
        #     #  different domains
        #     additional_required_model_args = ["tokenizer_name", "model_type", "model_path"]
        #     if not all((key in self.additional_kwargs for key in additional_required_model_args)):
        #         raise ValueError(f"Unable to load model. One or more of {additional_required_model_args} missing.")
        #
        #     kwargs["dataset_path"] = kwargs["dataset_path"] if kwargs["dataset_path"] is not None else ""
        #     kwargs["dataset_config"] = kwargs["dataset_config"] if kwargs["dataset_config"] is not None else ""
        #     kwargs["dataset_split"] = kwargs["dataset_split"] if kwargs["dataset_split"] is not None else ""
        #
        #     reps = get_representations(
        #         self.additional_kwargs["model_path"],
        #         self.additional_kwargs["model_type"],
        #         self.additional_kwargs["tokenizer_name"],
        #         kwargs["dataset_path"],
        #         kwargs["dataset_config"],
        #         kwargs["dataset_split"],
        #         kwargs["device"],
        #         kwargs["token_pos"],
        #     )
        #     return ModelRepresentations(
        #         setting_identifier=self.identifier,
        #         architecture_name=self.architecture,
        #         train_dataset=self.train_dataset,
        #         seed_id=None,
        #         representation_dataset=kwargs["dataset_path"] + kwargs["dataset_config"] + kwargs["dataset_split"],
        #         representations=tuple(SingleLayerRepresentation(i, r, "nd") for i, r in enumerate(reps)),
        #     )
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
        return f"{self.domain}_{self.architecture}_{self.train_dataset}_{self.identifier}_{self.additional_kwargs}"


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
                            additional_kwargs={"seed_id": i, "setting_identifier": None},
                        )
                    )
    for i in range(2):
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
                            additional_kwargs={"seed_id": i, "setting_identifier": identifier},
                        )
                    )

    return all_trained_vision_models


def all_trained_nlp_models() -> list[TrainedModel]:
    return [
        TrainedModel(
            domain="NLP",
            architecture="BERT",
            train_dataset="SST2",
            identifier=STANDARD_SETTING,
            additional_kwargs={
                "human_name": "multibert-0-sst2",
                "model_path": "/root/LLM-comparison/outputs/2024-01-31/13-12-49",
                "model_type": "sequence-classification",
                "tokenizer_name": "google/multiberts-seed_0",
            },
        )
    ]


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


ALL_TRAINED_MODELS: list[TrainedModel] = []
# ALL_TRAINED_MODELS.extend(all_trained_vision_models())
# ALL_TRAINED_MODELS.extend(all_trained_nlp_models())
ALL_TRAINED_MODELS.extend(all_trained_graph_models())
