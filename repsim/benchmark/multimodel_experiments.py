import argparse
import itertools
import time
from typing import Callable
from typing import Dict
from typing import get_args

import numpy as np
import repsim.utils
from loguru import logger
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.types_globals import BENCHMARK_DATASET
from repsim.benchmark.types_globals import BENCHMARK_DATASETS_LIST
from repsim.benchmark.types_globals import BENCHMARK_NN_ARCHITECTURES
from repsim.benchmark.types_globals import EXPERIMENT_DICT
from repsim.benchmark.types_globals import MULTIMODEL_EXPERIMENT_IDENTIFIER
from repsim.benchmark.utils import ExperimentStorer
from repsim.benchmark.utils import name_of_measure
from repsim.measures import CLASSES
from repsim.measures.utils import SimilarityMeasure


class MultiModelExperiment:
    def __init__(
        self,
        experiment_name: MULTIMODEL_EXPERIMENT_IDENTIFIER,
        models: list[TrainedModel],
        measures: list[SimilarityMeasure],
        device: str,
        storage_path: str | None = None,
        representation_dataset: BENCHMARK_DATASET = None,
    ) -> None:

        self.experiment_name = experiment_name
        self.settings = EXPERIMENT_DICT[self.experiment_name]

        self.models = [m for m in models if (m.identifier in self.settings)]
        self.measures = measures

        self.seeds = np.unique([m.seed for m in self.models])
        self._seed_map = dict(
            {f"{seed1}-{seed2}": i for i, (seed1, seed2) in enumerate(itertools.product(self.seeds, self.seeds))}
        )

        self.similarities = np.full(
            (len(self.settings), len(self.settings), len(self.seeds) ** 2, len(self.measures)),
            fill_value=np.nan,
            dtype=np.float32,
        )

        self.storage_path = storage_path
        self.device = device

        # TODO: find consensus on whether or not this attribute is required, same for potential device attribute
        self.representation_dataset = representation_dataset

    def get_seed_index(self, s1, s2):
        return self._seed_map[f"{s1}-{s2}"]

    def _final_layer_representation(self, model: TrainedModel) -> repsim.utils.SingleLayerRepresentation:
        start_time = time.perf_counter()
        reps = model.get_representation(representation_dataset=self.representation_dataset)
        logger.info(
            f"Representation extraction for '{str(model)}' completed in {time.perf_counter() - start_time:.1f} seconds."
        )
        return reps.representations[-1]

    def run(self) -> None:
        """Run the experiment."""

        with ExperimentStorer() as storer:

            setting_map = {setting: i for (i, setting) in enumerate(self.settings)}

            for model_a, model_b in itertools.combinations(self.models, r=2):
                sngl_rep_a = self._final_layer_representation(model_a)
                sngl_rep_b = self._final_layer_representation(model_b)
                i = setting_map[model_a.identifier]
                j = setting_map[model_b.identifier]
                seed_index = self.get_seed_index(model_a.seed, model_b.seed)

                for cnt_m, measure in enumerate(self.measures):

                    measure_name = name_of_measure(measure)

                    if storer.comparison_exists(sngl_rep_a, sngl_rep_b, measure_name):
                        # ---------------------------- Just read from file --------------------------- #
                        logger.info(f"Found {measure_name} loaded rep.")

                        sim = storer.get_comp_result(sngl_rep_a, sngl_rep_b, measure_name)
                    else:
                        try:
                            start_time = time.perf_counter()
                            sim = measure(sngl_rep_a.representation, sngl_rep_b.representation, sngl_rep_a.shape)
                            runtime = time.perf_counter() - start_time
                            storer.add_results(sngl_rep_a, sngl_rep_b, measure_name, sim, runtime)
                            logger.info(
                                f"Similarity '{sim:.02f}', measure '{measure_name}' comparison for '{str(model_a)}' and"
                                + f" '{str(model_b)}' completed in {time.perf_counter() - start_time:.1f} seconds."
                            )

                        except Exception as e:
                            sim = np.nan
                            logger.error(
                                f"'{measure_name}' comparison for '{str(model_a)}' and '{str(model_b)}' failed."
                            )
                            logger.error(e)

                    self.similarities[i, j, seed_index, cnt_m] = sim

    def eval_measures(self, meta_measure: Callable) -> Dict[str, float]:

        results = dict()

        for i_m, measure in enumerate(self.measures):

            vals = self.similarities[:, :, :, i_m]

            results[name_of_measure(measure)] = meta_measure(vals)

        return results


# TODO: this could likely be moved somewhere else
MEASURE_DICT = {m.__name__: m() for m in CLASSES}
MEASURE_LIST = list(MEASURE_DICT.keys())


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-e",
        "--experiments",
        type=str,
        nargs="*",
        default=list(get_args(MULTIMODEL_EXPERIMENT_IDENTIFIER)),
        choices=list(get_args(MULTIMODEL_EXPERIMENT_IDENTIFIER)),
        help="Experiments to run.",
    )
    # TODO: consider argument for measures, but that would require additional name index for these.
    #  Might not be necessary here due to experiment storer, but sometimes it might be desirable to
    #  just apply single individual measures
    parser.add_argument(
        "-m",
        "--measures",
        type=str,
        nargs="*",
        default=MEASURE_LIST,
        choices=MEASURE_LIST,
        help="Tests to run.",
    )
    # TODO: consider whether domain argument may be desirable for easier filtering of models
    # parser.add_argument(
    #     "-dom",
    #     "--domains",
    #     type=DOMAIN_TYPE,
    #     default=list(get_args(DOMAIN_TYPE)),
    #     help="Tests to run.",
    # )
    parser.add_argument(
        "-a",
        "--architectures",
        nargs="*",
        type=str,
        default=BENCHMARK_NN_ARCHITECTURES,
        choices=BENCHMARK_NN_ARCHITECTURES,
        help="NN architectures that should be compared",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        type=str,
        default=BENCHMARK_DATASETS_LIST,
        choices=BENCHMARK_DATASETS_LIST,
        help="Datasets to be used in evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    measures = [MEASURE_DICT[m] for m in args.measures]

    for curr_experiment in args.experiments:

        for architecture, dataset in itertools.product(args.architectures, args.datasets):

            curr_models = [
                m for m in ALL_TRAINED_MODELS if (m.architecture == architecture) and (m.train_dataset == dataset)
            ]

            # curr_models can only contain actual models if there is no domain mismatch
            if len(curr_models) > 0:

                experiment = MultiModelExperiment(
                    experiment_name=curr_experiment, models=curr_models, measures=measures, device="cuda:0"
                )
                experiment.run()
