import time
from typing import Callable

import numpy as np
from loguru import logger
from repsim.benchmark.measure_quality_metrics import auprc
from repsim.benchmark.measure_quality_metrics import violation_rate
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import ExperimentStorer
from repsim.benchmark.utils import get_in_group_cross_group_sims
from repsim.benchmark.utils import get_ingroup_outgroup_SLRs
from repsim.measures import distance_correlation
from repsim.measures.cca import pwcca
from repsim.measures.cca import svcca
from repsim.measures.cka import centered_kernel_alignment
from repsim.measures.eigenspace_overlap import eigenspace_overlap_score
from repsim.measures.gulp import gulp
from repsim.measures.procrustes import orthogonal_procrustes
from repsim.measures.procrustes import permutation_procrustes
from repsim.measures.rsa import representational_similarity_analysis
from repsim.utils import SingleLayerRepresentation
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from vision.util.file_io import save_json


def flatten(xss):
    return [x for xs in xss for x in xs]


class OrdinalGroupSeparationExperiment:
    def __init__(
        self,
        experiment_identifier: str,
        models: list[TrainedModel],
        group_splitting_func: Callable,
        measures: list[Callable],
        representation_dataset: str,
        storage_path: str | None = None,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        self.groups_of_models: tuple[list[TrainedModel]] = group_splitting_func(
            models
        )  # Expects lists of models ordered by expected ordinality
        self.measures = measures
        self.representation_dataset = representation_dataset
        self.kwargs = kwargs
        self.storage_path = storage_path

    def measure_violation_rate(self, measure: Callable) -> float:
        n_groups = len(self.groups_of_models)
        group_violations = []
        with ExperimentStorer(self.storage_path) as storer:
            for i in range(n_groups):
                in_group_slrs, out_group_slrs = get_ingroup_outgroup_SLRs(
                    self.groups_of_models,
                    i,
                    rep_layer_id=-1,
                    representation_dataset=self.representation_dataset,
                )

                in_group_sims, cross_group_sims = get_in_group_cross_group_sims(
                    in_group_slrs,
                    out_group_slrs,
                    measure.__name__,
                    storer,
                )
                # Calculate the violations, i.e. the number of times the in-group similarity is lower than the cross-group similarity
                group_violations.append(violation_rate(in_group_sims, cross_group_sims))

        return float(np.mean(group_violations))

    def auprc(self, measure: Callable) -> float:
        """Calculate the mean auprc for the in-group and cross-group similarities"""
        n_groups = len(self.groups_of_models)
        group_auprcs = []
        with ExperimentStorer(self.storage_path) as storer:
            for i in range(n_groups):
                in_group_slrs, out_group_slrs = get_ingroup_outgroup_SLRs(
                    self.groups_of_models,
                    i,
                    rep_layer_id=-1,
                    representation_dataset=self.representation_dataset,
                )
                in_group_sims, cross_group_sims = get_in_group_cross_group_sims(
                    in_group_slrs,
                    out_group_slrs,
                    measure.__name__,
                    storer,
                )
                # Calculate the area under the precision-recall curve for the in-group and cross-group similarities
                group_auprcs.append(auprc(in_group_sims, cross_group_sims))

        return float(np.mean(group_auprcs))

    def eval(self) -> dict:
        """Evaluate the results of the experiment"""
        measure_wise_results = {}
        for measure in self.measures:
            violation_rate = self.measure_violation_rate(measure)
            auprc = self.auprc(measure)
            measure_wise_results[measure.__name__] = {"violation_rate": violation_rate, "auprc": auprc}
        # Still gotta debate what to do with these values actually. Very preference dependent!
        return measure_wise_results

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        flat_models = flatten(self.groups_of_models)
        all_sims = np.full(
            (len(flat_models), len(flat_models), len(self.measures)),
            fill_value=np.nan,
            dtype=np.float32,
        )

        with ExperimentStorer(self.storage_path) as storer:
            for cnt_a, model_a in enumerate(flat_models):
                model_reps_a = model_a.get_representation(self.representation_dataset, **self.kwargs)
                sngl_rep_a: SingleLayerRepresentation = model_reps_a.representations[-1]

                for cnt_b, model_b in enumerate(flat_models):
                    if cnt_a > cnt_b:
                        continue
                    model_reps_b = model_b.get_representation(self.representation_dataset, **self.kwargs)
                    sngl_rep_b = model_reps_b.representations[-1]

                    for cnt_m, measure in enumerate(self.measures):
                        if storer.comparison_exists(sngl_rep_a, sngl_rep_b, measure.__name__):
                            # ---------------------------- Just read from file --------------------------- #
                            logger.info(f"Found {measure.__name__} loaded rep.")

                            sim = storer.get_comp_result(sngl_rep_a, sngl_rep_b, measure.__name__)
                        else:
                            try:
                                start_time = time.perf_counter()
                                sim = measure(sngl_rep_a.representation, sngl_rep_b.representation, sngl_rep_a.shape)
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

                        all_sims[cnt_a, cnt_b, cnt_m] = sim
                        all_sims[cnt_b, cnt_a, cnt_m] = sim
        return all_sims


"""
SimCKA for shortcut of seed 0 only.
array([[1.        , 0.88063586, 0.8452915 , 0.78969693, 0.29663917],
       [0.88063586, 1.        , 0.874399  , 0.83381474, 0.3271054 ],
       [0.8452915 , 0.874399  , 1.        , 0.87857467, 0.37550303],
       [0.78969693, 0.83381474, 0.87857467, 1.        , 0.44088718],
       [0.29663917, 0.3271054 , 0.37550303, 0.44088718, 1.        ]],
      dtype=float32)

array([[1.        , 0.902106  , 0.88063586, 0.8806002 , 0.8452915 , 0.8495685 , 0.78969693, 0.79049397, 0.29663917, 0.29089352],
       [0.902106  , 1.        , 0.88480216, 0.884095  , 0.84894454, 0.85240364, 0.7934904 , 0.79633653, 0.29962784, 0.2929612 ],
       [0.88063586, 0.88480216, 1.        , 0.9024318 , 0.874399  , 0.87824595, 0.83381474, 0.8330886 , 0.3271054 , 0.320484  ],
       [0.8806002 , 0.884095  , 0.9024318 , 1.        , 0.87760025, 0.88188475, 0.8338022 , 0.833005  , 0.32659015, 0.31945845],
       [0.8452915 , 0.84894454, 0.874399  , 0.87760025, 1.        , 0.9094963 , 0.87857467, 0.8773353 , 0.37550303, 0.36777332],
       [0.8495685 , 0.85240364, 0.87824595, 0.88188475, 0.9094963 , 1.        , 0.8756632 , 0.87609726, 0.37522557, 0.36792675],
       [0.78969693, 0.7934904 , 0.83381474, 0.8338022 , 0.87857467, 0.8756632 , 1.        , 0.9111501 , 0.44088718, 0.43294594],
       [0.79049397, 0.79633653, 0.8330886 , 0.833005  , 0.8773353 , 0.87609726, 0.9111501 , 1.        , 0.4439877 , 0.4365481 ],
       [0.29663917, 0.29962784, 0.3271054 , 0.32659015, 0.37550303, 0.37522557, 0.44088718, 0.4439877 , 1.        , 0.94804287],
       [0.29089352, 0.2929612 , 0.320484  , 0.31945845, 0.36777332, 0.36792675, 0.43294594, 0.4365481 , 0.94804287, 1.        ]],
      dtype=float32)


"""

if __name__ == "__main__":

    # ToDo: Issues during development
    # - Choice of shortcut (how many different groups, how many seeds)
    # - Saving and loading of representations <--  Saving intermediate representations would make sense?
    # - Saving and loading of results!
    #

    subset_of_vision_models = [
        m
        for m in ALL_TRAINED_MODELS
        if (m.domain == "VISION")
        and (m.architecture == "ResNet18")
        and (
            m.train_dataset
            in [
                "ColorDot_100_CIFAR10DataModule",
                "ColorDot_75_CIFAR10DataModule",
                "ColorDot_50_CIFAR10DataModule",
                "ColorDot_25_CIFAR10DataModule",
                "ColorDot_0_CIFAR10DataModule",
            ]
        )
        and (m.additional_kwargs["seed_id"] <= 1)
    ]
    subset_of_nlp_models = [
        m
        for m in ALL_TRAINED_MODELS
        if (m.domain == "NLP") and (m.architecture == "BERT") and (m.train_dataset == "SST2")
    ]
    subset_of_graph_models = []

    # experiment = SameLayerExperiment(subset_of_vision_models, [centered_kernel_alignment], "CIFAR10")
    def vision_group_split_func(models: list[TrainedModel]) -> tuple[list[TrainedModel]]:
        """Split the models into groups based on the dataset they were trained on."""
        model_train_ds = sorted(list(set([m.train_dataset for m in models])), key=lambda x: int(x.split("_")[1]))
        model_groups = []
        for tr_ds in model_train_ds:
            group = [m for m in models if m.train_dataset == tr_ds]
            model_groups.append(group)
        return tuple(model_groups)

    experiment = OrdinalGroupSeparationExperiment(
        experiment_identifier="Vision_ColorDot_Separation",
        models=subset_of_vision_models,
        group_splitting_func=vision_group_split_func,
        measures=[
            centered_kernel_alignment,  # 2.4 seconds
            # orthogonal_procrustes,  # 77.2 seconds
            permutation_procrustes,  # 16.2 seconds
            # eigenspace_overlap_score,  # 245 seconds for one comp! -- 4 minutes
            # gulp,  # failed
            # svcca,  #  157.5/129.7 seconds
            # pwcca,  # failed?
            # representational_similarity_analysis, # 94.5 seconds
            # distance_correlation,  # 75.2
        ],
        representation_dataset="ColorDot_0_CIFAR10DataModule",
    )
    # result = experiment.run()
    eval = experiment.eval()
    # print(result)
    print(0)
