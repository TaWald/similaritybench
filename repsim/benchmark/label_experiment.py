import time
from typing import Callable
from typing import get_args

import numpy as np
from loguru import logger
from repsim.benchmark.config import DOMAIN_TYPE
from repsim.benchmark.config import EXPERIMENT_DICT
from repsim.benchmark.config import EXPERIMENT_SEED
from repsim.benchmark.config import GRAPH_DOMAIN
from repsim.benchmark.config import LABEL_TEST_NAME
from repsim.benchmark.config import NN_ARCHITECTURE_TYPE
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.utils import Result
from repsim.measures.cka import centered_kernel_alignment
from scipy.stats import spearmanr


class LabelExperiment:
    def __init__(
        self,
        domain: DOMAIN_TYPE,
        architecture_type: NN_ARCHITECTURE_TYPE,
        dataset: str,
        measures: list[Callable],
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        self.measures = measures
        self.domain = domain
        self.dataset = dataset
        self.architecture_type = architecture_type
        self.kwargs = kwargs
        self.results = Result(LABEL_TEST_NAME)
        logger.add(self.results.basedir / "{time}.log")  # Not sure where this needs to go...

        self.models = [
            m
            for m in ALL_TRAINED_MODELS
            if (m.domain == self.domain)
            and (m.experiment_identifier == LABEL_TEST_NAME)
            and (m.setting_identifier in EXPERIMENT_DICT[LABEL_TEST_NAME])
            and (m.architecture == self.architecture_type)
            and (m.train_dataset == self.dataset)
        ]

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

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""

        label_test_settings = EXPERIMENT_DICT[LABEL_TEST_NAME]
        setting_map = {i: setting for (i, setting) in zip(range(len(label_test_settings)), label_test_settings)}

        for seed in get_args(EXPERIMENT_SEED):

            curr_models = [model for model in self.models if model.seed_id == seed]
            print("number of curr models is ", len(curr_models))
            reps = dict()
            for model in curr_models:
                curr_setting = model.setting_identifier
                print(curr_setting)
                reps[curr_setting] = model.get_representation(representation_dataset=self.dataset)

                start_time = time.perf_counter()
                logger.info(
                    f"Representation extraction for '{str(model)}' completed in {time.perf_counter() - start_time:.1f} seconds."
                )

            for measure in self.measures:

                vals = np.full(
                    shape=(len(setting_map), len(setting_map)),
                    fill_value=np.nan,
                    dtype=np.float32,
                )

                start_time = time.perf_counter()
                for i in range(len(setting_map)):
                    for j in range(i + 1, len(setting_map)):
                        ret = measure(reps[setting_map[i]], reps[setting_map[j]])
                        vals[i, j] = ret
                        vals[j, i] = vals[i, j]

                logger.info(
                    f"Comparisons for '{measure.__name__}' completed in {time.perf_counter() - start_time:.1f} seconds."
                )
                self.results.add(
                    numpy_vals=vals,
                    model=model,
                    spearman_rank_corr=self._layerwise_forward_sim(vals),
                    meta_accuracy=self._meta_accuracy(vals),
                    seed_id=seed,
                    measure=measure.__name__,
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
    #     and (m.setting_identifier in EXPERIMENT_DICT[LABEL_TEST_NAME])
    # ]
    experiment = LabelExperiment(
        domain=GRAPH_DOMAIN, measures=[centered_kernel_alignment], dataset="ogbn-arxiv", architecture_type="GCN"
    )

    # experiment = SameLayerExperiment(subset_of_graph_models, [centered_kernel_alignment], "CIFAR10")
    result = experiment.run()
    print(result)
    print(0)
