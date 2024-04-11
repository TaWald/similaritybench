import multiprocessing
import os
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from contextlib import redirect_stdout
from itertools import chain
from itertools import product
from multiprocessing import Manager
from multiprocessing.managers import BaseManager

import numpy as np
import pandas as pd
from loguru import logger
from repsim.benchmark.abstract_experiment import AbstractExperiment
from repsim.benchmark.measure_quality_metrics import auprc
from repsim.benchmark.measure_quality_metrics import violation_rate
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
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


def compare_single_measure(rep_a, rep_b, measure: SimilarityMeasure, shape):
    """Compare a single measure between two representations."""
    logger.info(f"Starting {measure.name}.")
    try:
        start_time = time.perf_counter()
        with suppress():  # Mute printouts of the measures
            sim = measure(rep_a, rep_b, shape)
            runtime = time.perf_counter() - start_time
    except Exception as e:
        sim = np.nan
        runtime = np.nan
        logger.error(f"'{measure.name}' comparison failed.")
        logger.error(e)

    return {
        "metric": measure,
        "metric_value": sim,
        "runtime": runtime,
    }


def gather_representations(sngl_rep_src, sngl_rep_tgt, lock):
    """Get the representations without trying to access the GPU simultaneously."""
    try:

        lock.acquire()
        # logger.debug("Acquired Lock, starting Rep extraction ...")
        with suppress():
            rep_a = sngl_rep_src.representation
            rep_b = sngl_rep_tgt.representation
            shape = sngl_rep_src.shape
    finally:
        lock.release()
    return rep_a, rep_b, shape


def compare(
    comps: list[tuple[SingleLayerRepresentation, SingleLayerRepresentation, SimilarityMeasure]],
    rep_lock: multiprocessing.Lock,
    storage_lock: multiprocessing.Lock,
    storer: ExperimentStorer,
) -> list[dict]:
    """
    Multithreaded comparison function with GPU blocking support.
    Does all comparisons for a single model in series, to minimize redundant representation loading.
    """
    # --------------------------- Start extracting reps -------------------------- #
    sngl_rep_src, sngl_rep_tgt, _ = comps[0]
    measures = [c[2] for c in comps]

    rep_a, rep_b, shape = gather_representations(sngl_rep_src, sngl_rep_tgt, rep_lock)
    # ----------------------------- Start metric calculation ----------------------------- #
    results: list[dict] = []
    for measure in measures:
        res = compare_single_measure(rep_a, rep_b, measure, shape)
        res["sngl_rep_src"] = sngl_rep_src
        res["sngl_rep_tgt"] = sngl_rep_tgt
        try:
            storage_lock.acquire()
            logger.info("Saved result to file")
            storer.add_results(**res)
            storer.save_to_file()
        finally:
            storage_lock.release()
    return results


class GroupSeparationExperiment(AbstractExperiment):
    def __init__(
        self,
        grouped_models: list[Sequence[TrainedModel]],
        measures: list[SimilarityMeasure],
        representation_dataset: str,
        storage_path: str | None = None,
        meta_data: dict | None = None,
        threads: int = 1,
        cache: bool = False,
        only_extract_reps: bool = False,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        super().__init__(measures, representation_dataset, storage_path, threads, cache, only_extract_reps)
        self.groups_of_models = grouped_models
        self.meta_data = meta_data
        self.kwargs = kwargs

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
                    measure,
                    storer,
                )

                # We remove NaNs and return Nones if stuff failed, so if the metric has these, we skip it! the similarity lists
                if len(in_group_sims) == 0 or len(cross_group_sims) == 0:
                    continue

                # Calculate the violations, i.e. the number of times the in-group similarity is lower than the cross-group similarity
                group_violations.append(
                    violation_rate(
                        in_group_sims, cross_group_sims, larger_is_more_similar=measure.larger_is_more_similar
                    )
                )
        if len(group_violations) == 0:
            return float(np.nan)

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
                    measure,
                    storer,
                )
                if len(in_group_sims) == 0 or len(cross_group_sims) == 0:
                    continue
                # Calculate the area under the precision-recall curve for the in-group and cross-group similarities
                group_auprcs.append(auprc(in_group_sims, cross_group_sims, measure.larger_is_more_similar))

        if len(group_auprcs) == 0:
            return float(np.nan)

        return float(np.mean(group_auprcs))

    def eval(self) -> list[dict]:
        """Evaluate the results of the experiment"""
        measure_wise_results: list[dict] = []
        examplary_model = self.groups_of_models[0][0]
        # This here currently assumes that both models are of the same architecture (which may not always remain true)
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
                    "similarity_measure": measure.name,
                    "quality_measure": "violation_rate",
                    "value": violation_rate,
                    **meta_data,
                }
            )
            auprc = self.auprc(measure)
            measure_wise_results.append(
                {
                    "similarity_measure": measure.name,
                    "quality_measure": "AUPRC",
                    "value": auprc,
                    **meta_data,
                }
            )
        return measure_wise_results

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards."""
        if self.threads == 1:
            self._run_single_threaded()
        else:
            self._run_multiprocessed()

    def _get_todo_combos(
        self, combos, storer: ExperimentStorer
    ) -> tuple[list[tuple[SingleLayerRepresentation, SingleLayerRepresentation, list[SimilarityMeasure]]], int]:
        comparisons_todo = []
        n_total = 0
        for model_src, model_tgt in combos:
            if model_src == model_tgt:
                continue  # Skip self-comparisons
            model_reps_src = model_src.get_representation(self.representation_dataset, **self.kwargs)
            sngl_rep_src: SingleLayerRepresentation = model_reps_src.representations[-1]
            model_reps_tgt = model_tgt.get_representation(self.representation_dataset, **self.kwargs)
            sngl_rep_tgt: SingleLayerRepresentation = model_reps_tgt.representations[-1]

            # Need to fix this. It's a tuple of Single_reps
            todo_by_measure = []

            for measure in self.measures:
                if storer.comparison_exists(sngl_rep_src, sngl_rep_tgt, measure):
                    pass
                else:
                    todo_by_measure.append(measure)
                    n_total += 1
            if len(todo_by_measure) > 0:
                comparisons_todo.append((sngl_rep_src, sngl_rep_tgt, todo_by_measure))
        return comparisons_todo, n_total

    def _run_single_threaded(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        flat_models = flatten_nested_list(self.groups_of_models)
        combos = product(flat_models, flat_models)  # Necessary for non-symmetric values

        logger.info(f"")
        with ExperimentStorer(self.storage_path) as storer:
            todo_combos, n_total = self._get_todo_combos(combos, storer)
            with tqdm(total=n_total, desc="Comparing representations") as pbar:
                for sngl_rep_src, sngl_rep_tgt, measures in todo_combos:
                    sngl_rep_src: SingleLayerRepresentation
                    sngl_rep_tgt: SingleLayerRepresentation
                    sngl_rep_src.cache = self.cache
                    sngl_rep_tgt.cache = self.cache
                    measures: list[SimilarityMeasure]
                    for measure in measures:
                        if storer.comparison_exists(sngl_rep_src, sngl_rep_tgt, measure):
                            # We still need to check during execution, as symmetry not accounted in the `_get_todo_combos` call!
                            continue
                        try:
                            reps_a = sngl_rep_src.representation
                            reps_b = sngl_rep_tgt.representation
                            if self.only_extract_reps:
                                logger.info("Only extracting representations. Skipping comparison.")
                                # Break as all measures use the same rep.
                                pbar.update(len(measures))
                                break  # Skip the actual comparison and prepare all reps for e.g. a CPU only machine.
                            shape = sngl_rep_src.shape
                            # reps_a, reps_b = flatten(reps_a, reps_b, shape=shape)
                            start_time = time.perf_counter()
                            with suppress():  # Mute printouts of the measures
                                sim = measure(reps_a, reps_b, shape)
                            runtime = time.perf_counter() - start_time
                            storer.add_results(sngl_rep_src, sngl_rep_tgt, measure, sim, runtime)
                            logger.debug(f"Similarity '{sim:.02f}' in {time.perf_counter() - start_time:.1f}s.")

                        except Exception as e:
                            storer.add_results(
                                sngl_rep_src, sngl_rep_tgt, measure, metric_value=np.nan, runtime=np.nan
                            )

                            logger.error(f"'{measure.name}' comparison failed.")
                            logger.error(e)

                        if measure.is_symmetric:
                            pbar.update(1)
                        pbar.update(1)
                    sngl_rep_src.representation = None  # Clear memory
                    sngl_rep_tgt.representation = None  # Clear memory

        return

    @DeprecationWarning
    def _run_multiprocessed(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        flat_models = flatten_nested_list(self.groups_of_models)
        combos = product(flat_models, flat_models)  # Necessary for non-symmetric values

        BaseManager.register("ExperimentStorer", ExperimentStorer)
        BaseManager.register("Lock", multiprocessing.Lock)

        total_comps = 0
        with BaseManager() as manager:
            # Create a lock using the Manager
            rep_lock = manager.Lock()
            storage_lock = manager.Lock()
            storer = manager.ExperimentStorer(self.storage_path)

            comparisons_to_do = []
            for model_src, model_tgt in combos:
                if model_src == model_tgt:
                    continue  # Skip self-comparisons
                model_reps_src = model_src.get_representation(self.representation_dataset, **self.kwargs)
                sngl_rep_src: SingleLayerRepresentation = model_reps_src.representations[-1]
                model_reps_tgt = model_tgt.get_representation(self.representation_dataset, **self.kwargs)
                sngl_rep_tgt: SingleLayerRepresentation = model_reps_tgt.representations[-1]

                todo_by_measure = []
                for measure in self.measures:
                    if storer.comparison_exists(sngl_rep_src, sngl_rep_tgt, measure):
                        pass
                    else:
                        todo_by_measure.append((sngl_rep_src, sngl_rep_tgt, measure))
                        total_comps += 1
                comparisons_to_do.append((todo_by_measure, rep_lock, storage_lock, storer))
            n_comparisons = len(comparisons_to_do)
            logger.info(f"{n_comparisons} model comparisons remaining. {total_comps} singular comparisons.")

            with tqdm(total=n_comparisons, desc="Comparing representations") as pbar:
                with multiprocessing.get_context("spawn").Pool(
                    self.n_threads, initargs=(rep_lock, storage_lock, storer)
                ) as p:
                    results = []
                    for comp in comparisons_to_do:
                        results.append(p.apply_async(compare, comp, callback=lambda _: pbar.update(1)))
                        # Add mutex to args
                    for res in results:
                        res.wait()

        return
