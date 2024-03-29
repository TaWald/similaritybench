import time
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from repsim.benchmark.measure_quality_metrics import auprc
from repsim.benchmark.measure_quality_metrics import violation_rate
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import create_pivot_excel_table
from repsim.benchmark.utils import ExperimentStorer
from repsim.benchmark.utils import get_in_group_cross_group_sims
from repsim.benchmark.utils import get_ingroup_outgroup_SLRs
from repsim.measures.cca import PWCCA
from repsim.measures.cca import SVCCA
from repsim.measures.cka import centered_kernel_alignment
from repsim.measures.cka import CKA
from repsim.measures.correlation_match import hard_correlation_match
from repsim.measures.correlation_match import HardCorrelationMatch
from repsim.measures.correlation_match import soft_correlation_match
from repsim.measures.correlation_match import SoftCorrelationMatch
from repsim.measures.distance_correlation import DistanceCorrelation
from repsim.measures.eigenspace_overlap import eigenspace_overlap_score
from repsim.measures.eigenspace_overlap import EigenspaceOverlapScore
from repsim.measures.geometry_score import geometry_score
from repsim.measures.geometry_score import GeometryScore
from repsim.measures.gulp import Gulp
from repsim.measures.gulp import gulp
from repsim.measures.linear_regression import linear_reg
from repsim.measures.linear_regression import LinearRegression
from repsim.measures.multiscale_intrinsic_distance import imd_score
from repsim.measures.multiscale_intrinsic_distance import IMDScore
from repsim.measures.nearest_neighbor import jaccard_similarity
from repsim.measures.nearest_neighbor import JaccardSimilarity
from repsim.measures.nearest_neighbor import joint_rank_jaccard_similarity
from repsim.measures.nearest_neighbor import NearestNeighborSimilarityFunction
from repsim.measures.nearest_neighbor import rank_similarity
from repsim.measures.nearest_neighbor import RankSimilarity
from repsim.measures.nearest_neighbor import second_order_cosine_similarity
from repsim.measures.nearest_neighbor import SecondOrderCosineSimilarity
from repsim.measures.procrustes import aligned_cossim
from repsim.measures.procrustes import AlignedCosineSimilarity
from repsim.measures.procrustes import orthogonal_angular_shape_metric
from repsim.measures.procrustes import orthogonal_angular_shape_metric_centered
from repsim.measures.procrustes import orthogonal_procrustes
from repsim.measures.procrustes import orthogonal_procrustes_centered_and_normalized
from repsim.measures.procrustes import OrthogonalAngularShapeMetricCentered
from repsim.measures.procrustes import OrthogonalProcrustesCenteredAndNormalized
from repsim.measures.procrustes import permutation_aligned_cossim
from repsim.measures.procrustes import permutation_angular_shape_metric
from repsim.measures.procrustes import permutation_procrustes
from repsim.measures.procrustes import PermutationProcrustes
from repsim.measures.procrustes import procrustes_size_and_shape_distance
from repsim.measures.procrustes import ProcrustesSizeAndShapeDistance
from repsim.measures.rsa import RSA
from repsim.measures.rsm_norm_difference import RSMNormDifference
from repsim.measures.statistics import concentricity_nrmse
from repsim.measures.statistics import ConcentricityDifference
from repsim.measures.statistics import magnitude_nrmse
from repsim.measures.statistics import MagnitudeDifference
from repsim.measures.statistics import UniformityDifference
from repsim.measures.utils import flatten
from repsim.measures.utils import SimilarityMeasure
from repsim.utils import SingleLayerRepresentation
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from vision.util.file_io import save_json


def flatten_nested_list(xss):
    return [x for xs in xss for x in xs]


class OrdinalGroupSeparationExperiment:
    def __init__(
        self,
        experiment_identifier: str,
        models: list[TrainedModel],
        group_splitting_func: Callable,
        measures: list[SimilarityMeasure],
        representation_dataset: str,
        storage_path: str | None = None,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        self.experiment_identifier: str = experiment_identifier
        self.groups_of_models: tuple[list[TrainedModel]] = group_splitting_func(
            models
        )  # Expects lists of models ordered by expected ordinality
        self.measures = measures
        self.representation_dataset = representation_dataset
        self.kwargs = kwargs
        self.storage_path = storage_path

    def measure_violation_rate(self, measure: SimilarityMeasure) -> float:
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
                group_violations.append(
                    violation_rate(
                        in_group_sims,
                        cross_group_sims,
                    )
                )

        return float(np.mean(group_violations))

    def auprc(self, measure: SimilarityMeasure) -> float:
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

    def eval(self) -> pd.DataFrame:
        """Evaluate the results of the experiment"""
        measure_wise_results: list[dict] = []
        examplary_model = self.groups_of_models[0][0]
        meta_data = {
            "domain": examplary_model.domain,
            "architecture": examplary_model.architecture,
            "representation_dataset": self.representation_dataset,
            "identifier": examplary_model.identifier,
            "experiment_identifier": self.experiment_identifier,
        }
        for measure in tqdm(self.measures, desc=f"Evaluating quality of measures"):
            violation_rate = self.measure_violation_rate(measure)
            measure_wise_results.append(
                {
                    "similarity_measure": measure.__name__,
                    "quality_measure": "violation_rate",
                    "value": violation_rate,
                    **meta_data,
                }
            )
            auprc = self.auprc(measure)
            measure_wise_results.append(
                {
                    "similarity_measure": measure.__name__,
                    "quality_measure": "AUPRC",
                    "value": auprc,
                    **meta_data,
                }
            )
        pd_df = pd.DataFrame(measure_wise_results)
        create_pivot_excel_table(
            pd_df,
            row_index="similarity_measure",
            columns=["quality_measure", "architecture"],
            value_key="value",
            file_path="results.xlsx",
            sheet_name="shortcut_results",
        )
        return

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        flat_models = flatten_nested_list(self.groups_of_models)
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
                            logger.info(f"Found previous {measure.__name__} comparison.")

                            sim = storer.get_comp_result(sngl_rep_a, sngl_rep_b, measure.__name__)
                        else:
                            try:
                                reps_a = sngl_rep_a.representation
                                reps_b = sngl_rep_b.representation
                                shape = sngl_rep_a.shape
                                # reps_a, reps_b = flatten(reps_a, reps_b, shape=shape)
                                logger.info(f"'{measure.__name__}' calculation starting ...")
                                start_time = time.perf_counter()
                                sim = measure(reps_a, reps_b, shape)
                                runtime = time.perf_counter() - start_time
                                storer.add_results(sngl_rep_a, sngl_rep_b, measure.__name__, sim, runtime)
                                logger.info(
                                    f"Similarity '{sim:.02f}' in {time.perf_counter() - start_time:.1f}s for '{str(model_a)}' and"
                                    + f" '{str(model_b)}'."
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
            # cka,
            CKA,
            # correlation_match,
            HardCorrelationMatch,
            SoftCorrelationMatch,
            # distance_correlation
            DistanceCorrelation,
            # eigenspace_overlap
            EigenspaceOverlapScore,
            # geometry_score
            GeometryScore,
            # gulp
            Gulp,
            # linear regression
            LinearRegression,
            # multiscale_intrinsic_distance
            IMDScore,
            # nearest_neighbor
            NearestNeighborSimilarityFunction,
            JaccardSimilarity,
            SecondOrderCosineSimilarity,
            RankSimilarity,
            # Procrustes
            # procrustes_size_and_shape_distance,
            ProcrustesSizeAndShapeDistance,
            # orthogonal_procrustes_centered_and_normalized,
            OrthogonalProcrustesCenteredAndNormalized,
            # permutation_procrustes,
            PermutationProcrustes,
            #  permutation_angular_shape_metric, <-- No Class?
            # orthogonal_angular_shape_metric_centered,
            OrthogonalAngularShapeMetricCentered,
            # aligned_cossim,
            AlignedCosineSimilarity,
            # permutation_aligned_cossim, <-- No Class?
            # rsa
            RSA,
            # representational_similarity_analysis,
            # rsm_norm_diff
            RSMNormDifference,
            # statistics
            MagnitudeDifference,
            # magnitude_nrmse,
            UniformityDifference,
            ConcentricityDifference,
            # concentricity_nrmse,
            # cca
            SVCCA,
            PWCCA,
        ],
        representation_dataset="ColorDot_0_CIFAR10DataModule",
    )
    result = experiment.run()
    eval = experiment.eval()
    # print(result)
    print(0)
