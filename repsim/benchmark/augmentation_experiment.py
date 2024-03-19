import itertools
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable
from typing import Optional

import numpy as np
import repsim.benchmark.paths as paths
import repsim.measures
import repsim.nlp
import repsim.utils
from loguru import logger
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import ExperimentStorer
from repsim.benchmark.utils import name_of_measure
from repsim.utils import ModelRepresentations


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
    token_pos: int | None = None

    def get_representation(self, representation_dataset: str, device: str) -> ModelRepresentations:
        if self.domain == "NLP":
            # TODO: this requires so many additional arguments. We should likely have some specialized classes for the
            #  different domains
            additional_required_model_args = ["tokenizer_name", "model_type", "model_path"]
            if not all((key in self.additional_kwargs for key in additional_required_model_args)):
                raise ValueError(f"Unable to load model. One or more of {additional_required_model_args} missing.")

            reps = repsim.nlp.get_representations(
                self.additional_kwargs["model_path"],
                self.additional_kwargs["model_type"],
                self.additional_kwargs["tokenizer_name"],
                REPRESENTATION_DATASETS[representation_dataset].path,
                REPRESENTATION_DATASETS[representation_dataset].config,
                REPRESENTATION_DATASETS[representation_dataset].split,
                device,
                self.token_pos,
            )
            return ModelRepresentations(
                self.identifier,
                self.architecture,
                str(self.train_dataset),
                -1,
                representation_dataset,
                tuple(repsim.utils.SingleLayerRepresentation(i, r, "nd") for i, r in enumerate(reps)),
            )
        else:
            raise NotImplementedError(f"{self.domain=}")


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
        },
        augmentation=AUGMENTATIONS["None"],
        token_pos=0,
    ),
    "bert_sst2_eda_05": MyModel(
        domain="NLP",
        architecture="BERT",
        train_dataset=TRAIN_DATASETS["sst2_eda_05"],
        identifier="augmented_05",
        additional_kwargs={
            "human_name": "multibert-0-sst2_eda_05",
            "model_path": paths.NLP_MODEL_PATH / "robustness" / "sst2_augmented_eda_05",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
        },
        augmentation=AUGMENTATIONS["EasyDataAugment_05"],
        token_pos=0,
    ),
    "bert_sst2_eda_08": MyModel(
        domain="NLP",
        architecture="BERT",
        train_dataset=TRAIN_DATASETS["sst2_eda_08"],
        identifier="augmented_08",
        additional_kwargs={
            "human_name": "multibert-0-sst2_eda_08",
            "model_path": paths.NLP_MODEL_PATH / "robustness" / "sst2_augmented_eda_08",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
        },
        augmentation=AUGMENTATIONS["EasyDataAugment_08"],
        token_pos=0,
    ),
    "bert_sst2_eda_10": MyModel(
        domain="NLP",
        architecture="BERT",
        train_dataset=TRAIN_DATASETS["sst2_eda_10"],
        identifier="augmented_10",
        additional_kwargs={
            "human_name": "multibert-0-sst2_eda_10",
            "model_path": paths.NLP_MODEL_PATH / "robustness" / "sst2_augmented_eda_10",
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
        storage_path: Path | str | None,
        device: str,
    ) -> None:
        self.models = models
        self.measures = measures
        self.representation_dataset_id = representation_dataset_id
        self.storage_path = str(storage_path)
        self.device = device

    def _final_layer_representation(self, model: MyModel) -> repsim.utils.SingleLayerRepresentation:
        start_time = time.perf_counter()
        reps = model.get_representation(
            self.representation_dataset_id, self.device
        )  # TODO: make this work with Dataset and MyModel
        logger.info(
            f"Representation extraction for '{str(model)}' completed in {time.perf_counter() - start_time:.1f} seconds."
        )
        return reps.representations[-1]

    def run(self):
        all_sims = np.full(
            (len(self.models), len(self.models), len(self.measures)),
            fill_value=np.nan,
            dtype=np.float32,
        )
        with ExperimentStorer(self.storage_path) as storer:
            for (i, model_i), (j, model_j) in itertools.combinations(enumerate(self.models), r=2):
                # optimization possible by using combinations on representations instead of models, but we need the model
                # info to correctly store results
                rep_i = self._final_layer_representation(model_i)
                rep_j = self._final_layer_representation(model_j)
                for measure_idx, measure in enumerate(self.measures):
                    measure_name = name_of_measure(measure)
                    if storer.comparison_exists(rep_i, rep_j, measure_name):
                        # ---------------------------- Just read from file --------------------------- #
                        logger.info(f"Found existing scores for {measure_name}. Skipping computation.")
                        sim = storer.get_comp_result(rep_i, rep_j, measure_name)
                    else:
                        logger.info(f"Starting with '{measure_name}'.")
                        start_time = time.perf_counter()

                        try:
                            sim = measure(rep_i.representation, rep_j.representation, rep_i.shape)
                            runtime = time.perf_counter() - start_time
                            storer.add_results(rep_i, rep_j, measure_name, sim, runtime)
                            logger.info(
                                f"Similarity '{sim:.02f}', measure '{measure_name}' comparison for '{str(model_i)}' and"
                                + f" '{str(model_j)}' completed in {time.perf_counter() - start_time:.1f} seconds."
                            )
                        except ValueError as e:
                            sim = np.nan
                            logger.error(
                                f"'{measure.__name__}' comparison for '{str(model_i)}' and '{str(model_j)}' failed."
                            )
                            logger.error(e)
                            continue

                    # All metrics should be symmetric
                    all_sims[i, j, measure_idx] = sim
                    all_sims[j, i, measure_idx] = sim


if __name__ == "__main__":
    logger.add(str(paths.EXPERIMENT_RESULTS_PATH / "augmentation" / "{time}.log"))

    test = FeatureTest(
        "percent_words_changed",
        CONFIGS["nlp"][0]["models"],
        [m() for m in repsim.measures.CLASSES if m().is_symmetric],
        "sst2",
        # paths.EXPERIMENT_RESULTS_PATH / "augmentation",
        None,
        "cuda:0",
    )
    test.run()
