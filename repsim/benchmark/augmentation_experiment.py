import itertools
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Optional

import numpy as np
import repsim.benchmark.paths as paths
import repsim.measures
import repsim.utils
from loguru import logger
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import Result


@dataclass
class Augmentation:
    name: str
    strength: float
    strength_variable: str
    additional_info: dict = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    path: str
    config: Optional[str] = None
    split: str = "train"


@dataclass
class MyModel(TrainedModel):
    augmentation: Augmentation
    train_dataset: Dataset


AUGMENTATIONS = {
    "None": Augmentation(name="No augmentation", strength=0.0, strength_variable=""),
    "EasyDataAugment_05": Augmentation(
        name="EasyDataAugmentation",
        strength=0.5,
        strength_variable="pct_words_to_swap",
        additional_info=dict(transformations_per_example=1),
    ),
    "EasyDataAugment_08": Augmentation(
        name="EasyDataAugmentation",
        strength=0.8,
        strength_variable="pct_words_to_swap",
        additional_info=dict(transformations_per_example=1),
    ),
    "EasyDataAugment_10": Augmentation(
        name="EasyDataAugmentation",
        strength=1.0,
        strength_variable="pct_words_to_swap",
        additional_info=dict(transformations_per_example=1),
    ),
}

TRAIN_DATASETS = {
    "sst2": Dataset(name="sst2", path="sst2"),
    "sst2_eda_05": Dataset(name="sst2_eda_05", path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_05")),
    "sst2_eda_08": Dataset(name="sst2_eda_08", path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_08")),
    "sst2_eda_10": Dataset(name="sst2_eda_10", path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_10")),
}

REPRESENTATION_DATASETS = {
    "sst2": Dataset(name="sst2", path="sst2", split="validation"),
    "sst2_eda_05": Dataset(
        name="sst2_eda_05", path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_05"), split="validation"
    ),
    "sst2_eda_08": Dataset(
        name="sst2_eda_08", path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_08"), split="validation"
    ),
    "sst2_eda_10": Dataset(
        name="sst2_eda_10", path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_10"), split="validation"
    ),
}

MODELS = {
    "bert_sst2_clean": MyModel(
        domain="NLP",
        architecture="BERT",
        train_dataset=TRAIN_DATASETS["sst2"],
        identifier="Normal",
        additional_kwargs={
            "human_name": "multibert-0-sst2",
            "model_path": "/root/LLM-comparison/outputs/2024-01-31/13-12-49",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
            "percent_words_changed": 0.0,
        },
        augmentation=AUGMENTATIONS["None"],
    ),
    "bert_sst2_eda_05": MyModel(
        domain="NLP",
        architecture="BERT",
        train_dataset=TRAIN_DATASETS["sst2_eda_05"],
        identifier="augmented",
        additional_kwargs={
            "human_name": "multibert-0-sst2_eda_05",
            "model_path": paths.NLP_MODEL_PATH / "robustness/sst2_augmented_eda_05",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
        },
        augmentation=AUGMENTATIONS["EasyDataAugment_05"],
    ),
    "bert_sst2_eda_08": MyModel(
        domain="NLP",
        architecture="BERT",
        train_dataset=TRAIN_DATASETS["sst2_eda_08"],
        identifier="augmented",
        additional_kwargs={
            "human_name": "multibert-0-sst2_eda_08",
            "model_path": paths.NLP_MODEL_PATH / "robustness/sst2_augmented_eda_08",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
        },
        augmentation=AUGMENTATIONS["EasyDataAugment_08"],
    ),
    "bert_sst2_eda_10": MyModel(
        domain="NLP",
        architecture="BERT",
        train_dataset=TRAIN_DATASETS["sst2_eda_10"],
        identifier="augmented",
        additional_kwargs={
            "human_name": "multibert-0-sst2_eda_10",
            "model_path": paths.NLP_MODEL_PATH / "robustness/sst2_augmented_eda_10",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
        },
        augmentation=AUGMENTATIONS["EasyDataAugment_10"],
    ),
}


CONFIGS = {
    "nlp": [
        {
            "representation_dataset": REPRESENTATION_DATASETS["sst2"],
            "models": [
                MODELS["bert_sst2_clean"],
                MODELS["bert_sst2_eda_05"],
                MODELS["bert_sst2_eda_08"],
                MODELS["bert_sst2_eda_10"],
            ],
        },
        {
            "representation_dataset": REPRESENTATION_DATASETS["sst2_eda_10"],
            "models": [
                MODELS["bert_sst2_clean"],
                MODELS["bert_sst2_eda_05"],
                MODELS["bert_sst2_eda_08"],
                MODELS["bert_sst2_eda_10"],
            ],
        },
        # then other datasets
        {
            "representation_dataset": "MNLI-test_matched",
            "models": [...],
        },
    ],
    "vision": [],
    "graph": [],
}


class FeatureTest:
    def __init__(
        self,
        ground_truth_attribute: str,
        models: list[MyModel],
        measures: list[Callable],
        representation_dataset_id: str,
    ) -> None:
        self.models = models
        self.measures = measures
        self.representation_dataset_id = representation_dataset_id

        self.results = Result("augmentation")
        logger.add(self.results.basedir / "{time}.log")

    def _final_layer_representation(self, model: MyModel) -> repsim.utils.SingleLayerRepresentation:
        start_time = time.perf_counter()
        reps = model.get_representation(
            self.representation_dataset_id
        )  # TODO: make this work with Dataset and MyModel
        logger.info(
            f"Representation extraction for '{str(model)}' completed in {time.perf_counter() - start_time:.1f} seconds."
        )
        return reps.representations[-1]

    def _score(self, vals):
        # something based on self.models und self.ground_truth_attribute
        pass

    def run(self):
        for (i, model_i), (j, model_j) in itertools.combinations(enumerate(self.models), r=2):
            # optimization possible by using combinations on representations instead of models, but we need the model
            # info to correctly store results
            rep_i = self._final_layer_representation(model_i)
            rep_j = self._final_layer_representation(model_j)
            for measure in self.measures:
                logger.info(f"Starting with '{measure.__name__}'.")
                start_time = time.perf_counter()

                vals = np.full(
                    (len(self.models), len(self.models)),
                    fill_value=np.nan,
                    dtype=np.float32,
                )
                # All metrics should be symmetric
                try:
                    ret = measure(rep_i, rep_j, rep_i.shape)
                    vals[i, j] = ret
                    vals[j, i] = vals[i, j]
                except ValueError as e:
                    logger.exception(e)

                logger.info(
                    f"Comparisons for '{measure.__name__}' completed in {time.perf_counter() - start_time:.1f} seconds."
                )
                # TODO: replace with solution from Tassilo?
                self.results.add(
                    numpy_vals=vals,
                    model_i=model_i,
                    model_j=model_j,
                    quality_score=self._score(vals),
                    measure=measure.__name__,
                )
                self.results.save()


if __name__ == "__main__":
    test = FeatureTest(
        "percent_words_changed",
        CONFIGS["nlp"][0]["models"],
        repsim.measures.SYMMETRIC_MEASURES,
        "SST2-val-max-augment",
    )
