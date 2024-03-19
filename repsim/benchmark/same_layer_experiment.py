import time
from typing import Callable

import numpy as np
from loguru import logger
from registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import Result
from repsim.measures.cka import centered_kernel_alignment
from scipy.stats import spearmanr


class SameLayerExperiment:
    def __init__(
        self, models: list[TrainedModel], measures: list[Callable], representation_dataset: str, **kwargs
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        self.models: list[TrainedModel] = models
        self.measures = measures
        self.representation_dataset = representation_dataset
        self.kwargs = kwargs
        self.results = Result("monotonicity")
        logger.add(self.results.basedir / "{time}.log")  # Not sure where this needs to go...

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
        """Calculate the rate at which similarity is lower for layers further apart"""

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
        for model in self.models:
            start_time = time.perf_counter()
            reps = model.get_representation(self.representation_dataset, **self.kwargs)
            logger.info(
                f"Representation extraction for '{str(model)}' completed in {time.perf_counter() - start_time:.1f} seconds."
            )

            for measure in self.measures:
                vals = np.full(
                    (len(reps.representations), len(reps.representations)),
                    fill_value=np.nan,
                    dtype=np.float32,
                )
                start_time = time.perf_counter()
                for first in reps.representations:
                    for second in reps.representations:
                        i = first.layer_id
                        j = second.layer_id

                        # All metrics should be symmetric
                        if j > i:
                            continue

                        ret = measure(first.representation, second.representation, first.shape)
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
                    measure=measure.__name__,
                )
        self.results.save()


def monotonicity_test():
    """
    Contains all the different clusters of SameLayerExperiment that one wants to conduct.
    """
    pass


if __name__ == "__main__":
    subset_of_vision_models = [
        m
        for m in ALL_TRAINED_MODELS
        if (m.domain == "VISION")
        and (m.architecture == "ResNet18")
        and (m.train_dataset == "CIFAR10")
        and (m.additional_kwargs["seed_id"] <= 1)
    ]
    subset_of_nlp_models = [
        m
        for m in ALL_TRAINED_MODELS
        if (m.domain == "NLP") and (m.architecture == "BERT") and (m.train_dataset == "SST2")
    ]
    subset_of_graph_models = []
    # experiment = SameLayerExperiment(subset_of_vision_models, [centered_kernel_alignment], "CIFAR10")
    experiment = SameLayerExperiment(
        subset_of_nlp_models,
        [centered_kernel_alignment],
        "SST2",
        dataset_path="sst2",
        dataset_config=None,
        dataset_split="test",
        device="cuda:0",
        token_pos=None,
    )
    # experiment = SameLayerExperiment(subset_of_graph_models, [centered_kernel_alignment], "CIFAR10")
    result = experiment.run()
    print(result)
    print(0)
