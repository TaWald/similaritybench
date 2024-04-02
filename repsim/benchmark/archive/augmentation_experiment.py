import itertools
import time
from dataclasses import dataclass
from dataclasses import field
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
from repsim.measures.utils import SimilarityMeasure
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
    augmentation: Optional[Augmentation] = None


@dataclass
class NLPModel(TrainedModel):
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
                self.additional_kwargs.get("finetuning_seed", "None"),
                representation_dataset,
                tuple(repsim.utils.SingleLayerRepresentation(i, r, "nd") for i, r in enumerate(reps)),
            )
        else:
            raise NotImplementedError(f"{self.domain=}")


# ---------------------------------- Augmentation Configs -------------------------------------------
AUGMENTATIONS = {"None": Augmentation(name="No augmentation", strength=0.0, strength_variable="")}
AUGMENTATIONS = AUGMENTATIONS | {
    f"EasyDataAugment_{str(strength).replace('.','')}": Augmentation(
        name="EasyDataAugmentation",
        strength=strength,
        strength_variable="pct_words_to_swap",
        additional_info=dict(transformations_per_example=1),
    )
    for strength in [0.25, 0.5, 0.75, 0.8, 1.0]
}

# ---------------------------------- Training Dataset Configs -------------------------------------------
TRAIN_DATASETS = {
    "sst2": Dataset(name="sst2", path="sst2"),
    "sst2_eda_05_v1": Dataset(
        name="sst2_eda_05_v1",
        path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_05"),
        augmentation=AUGMENTATIONS["EasyDataAugment_05"],
    ),
    "sst2_eda_08_v1": Dataset(
        name="sst2_eda_08_v1",
        path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_08"),
        augmentation=AUGMENTATIONS["EasyDataAugment_08"],
    ),
    "sst2_eda_10_v1": Dataset(
        name="sst2_eda_10_v1",
        path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_10"),
        augmentation=AUGMENTATIONS["EasyDataAugment_10"],
    ),
}
TRAIN_DATASETS = TRAIN_DATASETS | {
    f"sst2_eda_strength{strength}": Dataset(
        name=f"sst2_eda_strength{strength}",
        path=str(paths.NLP_DATA_PATH / "robustness" / f"sst2_eda_strength{strength}"),
        augmentation=AUGMENTATIONS[f"EasyDataAugment_{strength}"],
    )
    for strength in ["025", "05", "075", "10"]
}

# ---------------------------------- Representation Dataset Configs -------------------------------------------
REPRESENTATION_DATASETS = {
    "sst2": Dataset(name="sst2", path="sst2", split="validation"),
    "sst2_eda_05_v1": Dataset(
        name="sst2_eda_05_v1",
        path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_05"),
        split="validation",
        augmentation=AUGMENTATIONS["EasyDataAugment_05"],
    ),
    "sst2_eda_08_v1": Dataset(
        name="sst2_eda_08_v1",
        path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_08"),
        split="validation",
        augmentation=AUGMENTATIONS["EasyDataAugment_08"],
    ),
    "sst2_eda_10_v1": Dataset(
        name="sst2_eda_10_v1",
        path=str(paths.NLP_DATA_PATH / "robustness/sst2_augmented_eda_10"),
        split="validation",
        augmentation=AUGMENTATIONS["EasyDataAugment_10"],
    ),
}
REPRESENTATION_DATASETS = REPRESENTATION_DATASETS | {
    key: Dataset(ds.name, ds.path, split="validation", augmentation=ds.augmentation)
    for key, ds in TRAIN_DATASETS.items()
    if "strength" in key
}

# ---------------------------------- Model Configs -------------------------------------------
MODELS = {
    "bert_sst2_clean": NLPModel(
        domain="NLP",
        architecture="BERT-L",
        train_dataset=TRAIN_DATASETS["sst2"],
        identifier="Normal",
        seed=0,
        additional_kwargs={
            "human_name": "multibert-0-sst2",
            "model_path": "/root/LLM-comparison/outputs/2024-01-31/13-12-49",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
        },
        token_pos=0,
    ),
    "bert_sst2_eda_05_v1": NLPModel(
        domain="NLP",
        architecture="BERT-L",
        train_dataset=TRAIN_DATASETS["sst2_eda_05_v1"],
        identifier="augmented_05",
        seed=0,
        additional_kwargs={
            "human_name": "multibert-0-sst2_eda_05_v1",
            "model_path": paths.NLP_MODEL_PATH / "robustness" / "sst2_augmented_eda_05",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
        },
        token_pos=0,
    ),
    "bert_sst2_eda_08_v1": NLPModel(
        domain="NLP",
        architecture="BERT-L",
        train_dataset=TRAIN_DATASETS["sst2_eda_08_v1"],
        identifier="augmented_08",
        seed=0,
        additional_kwargs={
            "human_name": "multibert-0-sst2_eda_08_v1",
            "model_path": paths.NLP_MODEL_PATH / "robustness" / "sst2_augmented_eda_08",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
        },
        token_pos=0,
    ),
    "bert_sst2_eda_10_v1": NLPModel(
        domain="NLP",
        architecture="BERT-L",
        train_dataset=TRAIN_DATASETS["sst2_eda_10_v1"],
        identifier="augmented_10",
        seed=0,
        additional_kwargs={
            "human_name": "multibert-0-sst2_eda_10_v1",
            "model_path": paths.NLP_MODEL_PATH / "robustness" / "sst2_augmented_eda_10",
            "model_type": "sequence-classification",
            "tokenizer_name": "google/multiberts-seed_0",
        },
        token_pos=0,
    ),
}
MODELS = MODELS | {
    f"bert_sst2_clean_pre{i}_ft{i}": NLPModel(
        domain="NLP",
        architecture="BERT-L",
        train_dataset=TRAIN_DATASETS["sst2"],
        identifier="Normal",
        seed=i,
        additional_kwargs={
            "human_name": f"multibert-{i}-sst2",
            "model_path": f"/root/similaritybench/experiments/models/nlp/standard/sst2_pretrain{i}_finetune{i}",
            "model_type": "sequence-classification",
            "tokenizer_name": f"google/multiberts-seed_{i}",
            "pretraining_seed": i,
            "finetuning_seed": i,
        },
        token_pos=0,
    )
    for i in range(10)
}
MODELS = MODELS | {
    f"bert_sst2_pre{i}_ft{i}_eda_strength{strength}": NLPModel(
        domain="NLP",
        architecture="BERT-L",
        train_dataset=TRAIN_DATASETS[f"sst2_eda_strength{strength}"],
        identifier=f"augmented_{strength}",
        seed=i,
        additional_kwargs={
            "human_name": f"multibert-{i}-sst2-eda-{strength}",
            "model_path": f"/root/similaritybench/experiments/models/nlp/augmentation/sst2_pre{i}_ft{i}_eda_strength{strength}",
            "model_type": "sequence-classification",
            "tokenizer_name": f"google/multiberts-seed_{i}",
            "pretraining_seed": i,
            "finetuning_seed": i,
        },
        token_pos=0,
    )
    for i, strength in itertools.product(range(10), ["025", "05", "075", "10"])
}


# ---------------------------------- Experiment Configs -------------------------------------------
CONFIGS = {
    "nlp": [
        {
            "representation_dataset": REPRESENTATION_DATASETS["sst2"],
            "models": [
                MODELS["bert_sst2_clean"],
                MODELS["bert_sst2_eda_05_v1"],
                MODELS["bert_sst2_eda_08_v1"],
                MODELS["bert_sst2_eda_10_v1"],
            ],
        },
        {
            "representation_dataset": REPRESENTATION_DATASETS["sst2_eda_10_v1"],
            "models": [
                MODELS["bert_sst2_clean"],
                MODELS["bert_sst2_eda_05_v1"],
                MODELS["bert_sst2_eda_08_v1"],
                MODELS["bert_sst2_eda_10_v1"],
            ],
        },
        {
            "representation_dataset": REPRESENTATION_DATASETS["sst2_eda_strength10"],
            "models": [MODELS[k] for k in MODELS.keys() if "pre" in k],
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


class AugmentationTest:
    def __init__(
        self,
        models: list[NLPModel],
        measures: list[SimilarityMeasure],
        representation_dataset_id: str,
        storage_path: str | None,
        device: str,
    ) -> None:
        self.models = models
        self.measures = measures
        self.representation_dataset_id = representation_dataset_id
        self.storage_path = storage_path
        self.device = device

    def _final_layer_representation(self, model: NLPModel) -> repsim.utils.SingleLayerRepresentation:
        start_time = time.perf_counter()
        reps = model.get_representation(self.representation_dataset_id, self.device)
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
                                f"'{measure_name}' comparison for '{str(model_i)}' and '{str(model_j)}' failed."
                            )
                            logger.error(e)
                            continue

                    # All metrics should be symmetric
                    all_sims[i, j, measure_idx] = sim
                    all_sims[j, i, measure_idx] = sim


if __name__ == "__main__":
    logger.add(str(paths.EXPERIMENT_RESULTS_PATH / "augmentation" / "{time}.log"))

    from repsim.measures import IMDScore, GeometryScore

    config_id = 2
    cfg = CONFIGS["nlp"][config_id]
    test = AugmentationTest(
        cfg["models"],
        [m() for m in repsim.measures.CLASSES if m().is_symmetric and not isinstance(m(), (IMDScore, GeometryScore))],
        # "sst2",
        representation_dataset_id="sst2_eda_strength10",
        storage_path=str(paths.EXPERIMENT_RESULTS_PATH / "augmentation" / "results.parquet"),
        device="cuda:0",
    )
    test.run()
