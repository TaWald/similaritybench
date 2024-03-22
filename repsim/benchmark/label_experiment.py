import time
from typing import Callable
from typing import get_args

import numpy as np
import numpy.typing as npt
from loguru import logger
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.types_globals import ARXIV_DATASET
from repsim.benchmark.types_globals import BENCHMARK_DATASET
from repsim.benchmark.types_globals import DOMAIN_TYPE
from repsim.benchmark.types_globals import EXPERIMENT_DICT
from repsim.benchmark.types_globals import GRAPH_DOMAIN
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_SEED
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import NN_ARCHITECTURE_TYPE
from repsim.benchmark.utils import ExperimentStorer
from repsim.benchmark.utils import Result
from repsim.benchmark.utils import SingleLayerRepresentation
from repsim.measures import CLASSES
from repsim.measures import SYMMETRIC_MEASURES
from repsim.measures.utils import SimilarityMeasure
from scipy.stats import spearmanr


class LabelExperiment:
    def __init__(
        self,
        domain: DOMAIN_TYPE,
        architecture_type: NN_ARCHITECTURE_TYPE,
        dataset: BENCHMARK_DATASET,
        representation_dataset: BENCHMARK_DATASET = None,
        measures: list[SimilarityMeasure] = CLASSES,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        self.measures = measures
        self.domain = domain
        self.dataset = dataset

        # TODO: figure out whether this variable/distinction is really necessary
        if representation_dataset is None:
            self.representation_dataset = dataset
        self.architecture_type = architecture_type
        self.kwargs = kwargs

        self.experiment_name = LABEL_EXPERIMENT_NAME
        self.settings = EXPERIMENT_DICT[self.experiment_name]

        self.models = [
            m
            for m in ALL_TRAINED_MODELS
            if (m.domain == self.domain)
            and (m.identifier in self.settings)
            and (m.architecture == self.architecture_type)
            and (m.train_dataset == self.dataset)
        ]

        self.similarities = np.full(
            (len(self.settings), len(self.settings), len(self.measures)),
            fill_value=np.nan,
            dtype=np.float32,
        )

        self.results = Result(LABEL_EXPERIMENT_NAME)

    def _layerwise_forward_sim(self, sim: np.ndarray) -> float:
        """Calculate the spearman rank correlation of the similarity to the layers"""
        aranged_1 = np.arange(sim.shape[0])[:, None]
        aranged_2 = np.arange(sim.shape[0])[None, :]
        dist = np.abs(aranged_1 - aranged_2)

        forward_corrs = []
        backward_corrs = []
        for i in range(sim.shape[0]):
            current_line_sim = sim[i]
            current_line_dist = dist[i]
            forward_sims = current_line_sim[i:]
            backward_sims = current_line_sim[:i]
            forward_dists = current_line_dist[i:]
            backward_dists = current_line_dist[:i]
            if len(forward_sims) > 1:
                corr, _ = spearmanr(forward_sims, forward_dists)
                forward_corrs.append(corr)
            if len(backward_sims) > 1:
                corr, _ = spearmanr(backward_sims, backward_dists)
                backward_corrs.append(corr)

        return np.nanmean(forward_corrs + backward_corrs)

    def _meta_accuracy(self, sim: np.ndarray) -> float:
        """Calculate the spearman rank correlation of the similarity to the layers"""

        n_rows, n_cols = sim.shape

        n_violations = 0
        n_comb_count = 0

        for i in range(n_rows):
            for j in range(i + 1, n_cols):

                for k in range(i, j):
                    for l in range(k + 1, j + 1):
                        n_comb_count += 1
                        if sim[i, j] < sim[k, l]:
                            n_violations += 1

        return 1 - n_violations / n_comb_count

    # def run_old(self) -> None:
    #     """Run the experiment. Results can be accessed afterwards via the .results attribute"""
    #
    #     experiment_settings = EXPERIMENT_DICT[LABEL_EXPERIMENT_NAME]
    #     setting_map = {i: setting for (i, setting) in zip(range(len(experiment_settings)), experiment_settings)}
    #
    #     for seed in get_args(GRAPH_EXPERIMENT_SEED):
    #
    #         curr_models = [model for model in self.models if model.additional_kwargs["seed_id"] == seed]
    #         reps = dict()
    #         for model in curr_models:
    #             curr_setting = model.identifier
    #             reps[curr_setting] = model.get_representation(representation_dataset=self.dataset)
    #
    #             start_time = time.perf_counter()
    #             logger.info(
    #                 f"Representation extraction for '{str(model)}' completed in {time.perf_counter() - start_time:.1f} seconds."
    #             )
    #
    #         for measure in self.measures:
    #
    #             vals = np.full(
    #                 shape=(len(setting_map), len(setting_map)),
    #                 fill_value=np.nan,
    #                 dtype=np.float32,
    #             )
    #
    #             start_time = time.perf_counter()
    #             for i in range(len(setting_map)):
    #                 for j in range(i + 1, len(setting_map)):
    #
    #                     # TODO: check consensus on how which layers are extracted/considered here
    #                     first = reps[setting_map[i]].representations[-2]
    #                     second = reps[setting_map[j]].representations[-2]
    #
    #                     ret = measure(first.representation, second.representation, first.shape)
    #                     vals[i, j] = ret
    #                     vals[j, i] = vals[i, j]
    #
    #             logger.info(
    #                 f"Comparisons for '{measure.__name__}' completed in {time.perf_counter() - start_time:.1f} seconds."
    #             )
    #             self.results.add(
    #                 numpy_vals=vals,
    #                 domain=self.domain,
    #                 model=self.architecture_type,
    #                 dataset=self.dataset,
    #                 spearman_rank_corr=self._layerwise_forward_sim(vals),
    #                 meta_accuracy=self._meta_accuracy(vals),
    #                 seed_id=seed,
    #                 measure=measure.__name__,
    #             )
    #     self.results.save()

    def run_measures(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""

        with ExperimentStorer() as storer:

            setting_map = {setting: i for (i, setting) in enumerate(self.settings)}

            for cnt_a, model_a in enumerate(self.models):
                model_reps_a = model_a.get_representation(self.representation_dataset, **self.kwargs)
                sngl_rep_a: SingleLayerRepresentation = model_reps_a.representations[-1]

                for cnt_b, model_b in enumerate(self.models):
                    if cnt_a > cnt_b:
                        continue
                    model_reps_b = model_b.get_representation(self.representation_dataset, **self.kwargs)
                    sngl_rep_b = model_reps_b.representations[-1]

                    for cnt_m, measure in enumerate(self.measures):
                        curr_measure = measure()
                        if storer.comparison_exists(sngl_rep_a, sngl_rep_b, measure.__name__):
                            # ---------------------------- Just read from file --------------------------- #
                            logger.info(f"Found {measure.__name__} loaded rep.")

                            sim = storer.get_comp_result(sngl_rep_a, sngl_rep_b, measure.__name__)
                        else:
                            try:
                                start_time = time.perf_counter()
                                sim = curr_measure(
                                    sngl_rep_a.representation, sngl_rep_b.representation, sngl_rep_a.shape
                                )
                                runtime = time.perf_counter() - start_time
                                storer.add_results(sngl_rep_a, sngl_rep_b, measure.__name__, sim, runtime)
                                logger.info(
                                    f"Similarity '{sim:.02f}', measure '{measure.__name__}' comparison for '{str(model_a)}' and"
                                    + f" '{str(model_b)}' completed in {time.perf_counter() - start_time:.1f} seconds."
                                )

                            except Exception as e:
                                sim = np.nan
                                logger.error(
                                    f"'{measure.__name__}' comparison for '{str(model_a)}' and '{str(model_b)}' failed."
                                )
                                logger.error(e)

                        i = setting_map[model_a.identifier]
                        j = setting_map[model_b.identifier]
                        self.similarities[i, j, cnt_m] = sim

                        # TODO: consider asymmetric measures, and what to do with diagonal values
                        if curr_measure.is_symmetric:
                            self.similarities[j, i, cnt_m] = sim

    def eval_measures(self) -> None:

        for i_m, measure in enumerate(self.measures):

            vals = self.similarities[:, :, i_m]

            if not np.isnan(vals).any():

                self.results.add(
                    domain=self.domain,
                    model=self.architecture_type,
                    dataset=self.dataset,
                    measure=measure.__name__,
                    spearman_rank_corr=self._layerwise_forward_sim(vals),
                    meta_accuracy=self._meta_accuracy(vals),
                    # seed_id=seed,
                )
            self.results.save()


if __name__ == "__main__":
    # subset_of_vision_models = [
    #     m
    #     for m in ALL_TRAINED_MODELS
    #     if (m.domain == "VISION")
    #     and (m.architecture == "ResNet18")
    #     and (m.train_dataset == "CIFAR10")
    #     and (m.additional_kwargs["seed_id"] <= 1)
    # ]
    # subset_of_nlp_models = [
    #     m
    #     for m in ALL_TRAINED_MODELS
    #     if (m.domain == "NLP") and (m.architecture == "BERT") and (m.train_dataset == "SST2")
    # ]
    # subset_of_graph_models = [m
    #     for m in ALL_TRAINED_MODELS
    #     if (m.domain == GRAPH_DOMAIN)
    #     and (m.setting_identifier in EXPERIMENT_DICT[LABEL_EXP_NAME])
    # ]

    for architecture in ["GCN", "GraphSAGE"]:

        experiment = LabelExperiment(
            domain=GRAPH_DOMAIN,
            measures=CLASSES,
            dataset=ARXIV_DATASET,
            architecture_type="GraphSAGE",
        )

    # experiment = SameLayerExperiment(subset_of_graph_models, [centered_kernel_alignment], "CIFAR10")
    experiment.run_measures()
    experiment.eval_measures()

    print(experiment.results)
    print(0)
