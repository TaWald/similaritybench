import itertools
import time
from typing import Callable

import numpy as np
import repsim.measures
import repsim.utils
from loguru import logger
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import Result

AUGMENTATION_MODELS = {
    "nlp": [
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
                "augmentation": None,
                "percent_words_changed": 0.0,
            },
        ),
        TrainedModel,
    ]
}

AUGMENTATION_DATASETS = {"nlp": []}

CONFIGS = {
    "nlp": [
        {
            "representation_dataset": {
                "id": "SST2-val",
                "path": "sst2",
                "split": "validation",
            },
            "models": AUGMENTATION_MODELS["nlp"],
        },
        {
            "representation_dataset": {
                "id": "SST2-val-max-augment",
                "path": "/root/similaritybench/multirun/2024-03-07/14-41-50/2",
                "split": "validation",
            },
            "models": AUGMENTATION_MODELS["nlp"],
        },
        # then other datasets
        {
            "representation_dataset": "MNLI-test_matched",
            "models": [
                TrainedModel,
                TrainedModel,
            ],
        },
    ],
    "vision": [],
    "graph": [],
}


class FeatureTest:
    def __init__(
        self,
        ground_truth_attribute: str,
        models: list[TrainedModel],
        measures: list[Callable],
        representation_dataset: str,
    ) -> None:
        self.models = models
        self.measures = measures
        self.representation_dataset = representation_dataset

        self.results = Result("augmentation")
        logger.add(self.results.basedir / "{time}.log")

    def _final_layer_representation(self, model: TrainedModel) -> repsim.utils.SingleLayerRepresentation:
        start_time = time.perf_counter()
        reps = model.get_representation(self.representation_dataset)
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
