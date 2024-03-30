import os
import time
from collections.abc import Sequence
from contextlib import contextmanager
from contextlib import redirect_stdout
from itertools import product

import numpy as np
import pandas as pd
from loguru import logger
from repsim.benchmark.abstract_experiment import AbstractExperiment
from repsim.benchmark.measure_quality_metrics import auprc
from repsim.benchmark.measure_quality_metrics import violation_rate
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import create_pivot_excel_table
from repsim.benchmark.utils import ExperimentStorer
from repsim.benchmark.utils import get_in_group_cross_group_sims
from repsim.benchmark.utils import get_ingroup_outgroup_SLRs
from repsim.measures.utils import SimilarityMeasure
from repsim.utils import SingleLayerRepresentation
from tqdm import tqdm


@contextmanager
def suppress():
    with open(os.devnull, "w") as null:
        with redirect_stdout(null):
            yield


def flatten_nested_list(xss):
    return [x for xs in xss for x in xs]


class GroupSeparationExperiment(AbstractExperiment):
    def __init__(
        self,
        grouped_models: list[Sequence[TrainedModel]],
        measures: list[SimilarityMeasure],
        representation_dataset: str,
        storage_path: str | None = None,
        meta_data: dict | None = None,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        self.meta_data: dict = meta_data
        self.groups_of_models: tuple[list[TrainedModel]] = grouped_models
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

    def eval(self) -> list[dict]:
        """Evaluate the results of the experiment"""
        measure_wise_results: list[dict] = []
        examplary_model = self.groups_of_models[0][0]
        meta_data = {
            "domain": examplary_model.domain,
            "architecture": examplary_model.architecture,
            "representation_dataset": self.representation_dataset,
            "identifier": examplary_model.identifier,
        }
        if self.meta_data is not None:
            meta_data.update(self.meta_data)

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
        # Temporarily remove the saving of results to excel. -- Move this outside.
        # pd_df = pd.DataFrame(measure_wise_results)
        # create_pivot_excel_table(
        #     pd_df,
        #     row_index="similarity_measure",
        #     columns=["quality_measure", "architecture"],
        #     value_key="value",
        #     file_path="results.xlsx",
        #     sheet_name="shortcut_results",
        # )
        return measure_wise_results

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        flat_models = flatten_nested_list(self.groups_of_models)
        combos = product(flat_models, flat_models)  # Necessary for non-symmetric values

        with ExperimentStorer(self.storage_path) as storer:
            for model_src, model_tgt in combos:
                if model_src == model_tgt:
                    continue  # Skip self-comparisons
                model_reps_src = model_src.get_representation(self.representation_dataset, **self.kwargs)
                sngl_rep_src: SingleLayerRepresentation = model_reps_src.representations[-1]
                model_reps_tgt = model_tgt.get_representation(self.representation_dataset, **self.kwargs)
                sngl_rep_tgt: SingleLayerRepresentation = model_reps_tgt.representations[-1]

                for measure in self.measures:
                    if storer.comparison_exists(sngl_rep_src, sngl_rep_tgt, measure):
                        # ---------------------------- Just read from file --------------------------- #
                        logger.info(f"Found previous {measure.name} comparison.")
                        sim = storer.get_comp_result(sngl_rep_src, sngl_rep_tgt, measure)
                    else:
                        try:
                            reps_a = sngl_rep_src.representation
                            reps_b = sngl_rep_tgt.representation
                            shape = sngl_rep_src.shape
                            # reps_a, reps_b = flatten(reps_a, reps_b, shape=shape)

                            logger.info(f"'{measure.name}' calculation starting ...")
                            start_time = time.perf_counter()
                            with suppress():  # Mute printouts of the measures
                                sim = measure(reps_a, reps_b, shape)
                            runtime = time.perf_counter() - start_time
                            storer.add_results(sngl_rep_src, sngl_rep_tgt, measure, sim, runtime)
                            logger.info(f"Similarity '{sim:.02f}' in {time.perf_counter() - start_time:.1f}s.")

                        except Exception as e:
                            sim = np.nan
                            logger.error(f"'{measure.name}' comparison failed.")
                            logger.error(e)
        return
