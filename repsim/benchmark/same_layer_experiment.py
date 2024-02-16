from repsim.benchmark.registry import TrainedModel
import numpy as np
from scipy.stats import spearmanr
from registry import ALL_TRAINED_MODELS
from repsim.measures.cka import centered_kernel_alignment


class SameLayerExperiment:
    def __init__(self, models: list[TrainedModel], measures: list[callable], representation_dataset: str) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        self.models: list[TrainedModel] = models
        self.measures = measures
        self.representation_dataset = representation_dataset

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

    def run(self) -> dict[str, dict]:
        """Run the experiment"""
        model_wise_results = []
        for model in self.models:
            measure_wise_result = {}
            for measure in self.measures:
                reps = model.get_representation(self.representation_dataset)
                vals = np.full(
                    (len(reps.representations), len(reps.representations)),
                    fill_value=np.nan,
                    dtype=np.float32,
                )
                for first in reps.representations:
                    for second in reps.representations:
                        # All metrics should be symmetric
                        if j > i:
                            continue

                        i = first.layer_id
                        j = second.layer_id
                        ret = measure(first.representation, second.representation, first.shape)
                        vals[i, j] = ret
                        vals[j, i] = vals[i, j]
                measure_wise_result[measure.__name__] = {
                    "models": self.models,
                    "raw_values": vals,
                    "spearman_rank_correlation": self._layerwise_forward_sim(vals),
                }
            model_wise_results.append(measure_wise_result)
        return model_wise_results  # TODO: Aggregate the results somehow?


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
    subset_of_nlp_models = []
    subset_of_graph_models = []
    experiment = SameLayerExperiment(subset_of_vision_models, [centered_kernel_alignment], "CIFAR10")
    # experiment = SameLayerExperiment(subset_of_nlp_models, [centered_kernel_alignment], "CIFAR10")
    # experiment = SameLayerExperiment(subset_of_graph_models, [centered_kernel_alignment], "CIFAR10")
    result = experiment.run()
    print(result)
    print(0)
