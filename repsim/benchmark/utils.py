import os
from collections.abc import Sequence
from dataclasses import asdict
from itertools import chain
from itertools import combinations
from itertools import product

import git
import pandas as pd
from loguru import logger
from repsim.benchmark.paths import EXPERIMENT_RESULTS_PATH
from repsim.benchmark.registry import TrainedModel
from repsim.measures.utils import SimilarityMeasure
from repsim.utils import SingleLayerRepresentation


class ExperimentStorer:

    def __init__(self, path_to_store: str | None = None) -> None:
        if path_to_store is None:
            path_to_store = os.path.join(EXPERIMENT_RESULTS_PATH, "experiments.parquet")
        self.path_to_store = path_to_store
        self.experiments = (
            pd.read_parquet(self.path_to_store) if os.path.exists(self.path_to_store) else pd.DataFrame()
        )

    def add_results(
        self,
        src_single_rep: SingleLayerRepresentation,
        tgt_single_rep: SingleLayerRepresentation,
        metric: SimilarityMeasure,
        metric_value: float,
        runtime: float | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add a comparison result of the experiments to disk.
        Serializes all the information into a unique identifier and stores the results in a pandas dataframe.
        """
        if self.comparison_exists(src_single_rep, tgt_single_rep, metric) and not overwrite:
            logger.info("Comparison already exists and Overwrite is False. Skipping.")

        if metric.is_symmetric:
            reps = [(src_single_rep, tgt_single_rep), (tgt_single_rep, src_single_rep)]
        else:
            reps = [(src_single_rep, tgt_single_rep)]

        for source_rep, target_rep in reps:
            comp_id = self._get_comparison_id(
                src_single_rep=source_rep,
                tgt_single_rep=target_rep,
                metric_name=metric.name,
            )

            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha

            ids_of_interest = [
                "layer_id",
                "_architecture_name",
                "_train_dataset",
                "_representation_dataset",
                "_seed",
                "_setting_identifier",
            ]
            content = {"source_" + k: v for k, v in asdict(source_rep).items() if k in ids_of_interest}
            content.update({"target_" + k: v for k, v in asdict(target_rep).items() if k in ids_of_interest})
            content.update(
                {
                    "metric": metric.name,
                    "metric_value": metric_value,
                    "runtime": runtime,
                    "id": comp_id,
                    "is_symmetric": metric.is_symmetric,
                    "larger_is_more_similar": metric.larger_is_more_similar,
                    "is_metric": metric.is_metric,
                    "invariant_to_affine": metric.invariant_to_affine,
                    "invariant_to_invertible_linear": metric.invariant_to_invertible_linear,
                    "invariant_to_ortho": metric.invariant_to_ortho,
                    "invariant_to_permutation": metric.invariant_to_permutation,
                    "invariant_to_isotropic_scaling": metric.invariant_to_isotropic_scaling,
                    "invariant_to_translation": metric.invariant_to_translation,
                }
            )
            content.update({"hash": sha, "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")})
            content_df = pd.DataFrame(content, index=[comp_id])
            self.experiments = self.experiments._append(content_df, ignore_index=False)

    def _sort_models(
        self,
        single_rep_a: SingleLayerRepresentation,
        single_rep_b: SingleLayerRepresentation,
    ) -> tuple[SingleLayerRepresentation, SingleLayerRepresentation]:
        """Return the SingeLayerRepresentations in a sorted order, to avoid permutation issues."""
        id_a, id_b = single_rep_a.unique_identifier(), single_rep_b.unique_identifier()
        if id_a < id_b:
            return single_rep_a, single_rep_b
        return single_rep_b, single_rep_a

    def _get_comparison_id(
        self,
        src_single_rep: SingleLayerRepresentation,
        tgt_single_rep: SingleLayerRepresentation,
        metric_name: str,
    ) -> str:
        """
        Serialize the experiment setting into a unique identifier that can be used to index if it already exists in the dataframe.

        Args:
            single_rep_a (SingleLayerRepresentation): The representation of a single layer in model A.
            single_rep_b (SingleLayerRepresentation): The representation of a single layer in model B.
            metric_name (str): The name of the metric used for comparison.

        Returns:
            str: A unique identifier that represents the experiment setting.
        """
        src_id_reps = src_single_rep.unique_identifier()
        tgt_id_reps = tgt_single_rep.unique_identifier()

        first_id, second_id = [src_id_reps, tgt_id_reps]
        joint_id = "___".join([first_id, second_id, metric_name])
        return joint_id

    def get_comp_result(
        self,
        src_single_rep: SingleLayerRepresentation,
        tgt_single_rep: SingleLayerRepresentation,
        metric: SimilarityMeasure,
    ) -> float:
        """
        Return the result of the comparison
        Arsg:
            src_single_rep: SingleLayerRepresentation
            tgt_single_rep: SingleLayerRepresentation
            metric: SimilarityMeasure
        Returns:
            float: The result of the comparison
        Raises:
            ValueError: If the comparison does not exist in the dataframe.
        """
        comp_id = self._get_comparison_id(src_single_rep, tgt_single_rep, metric.name)
        if comp_id not in self.experiments.index:
            if metric.is_symmetric:
                comp_id = self._get_comparison_id(tgt_single_rep, src_single_rep, metric.name)

        if comp_id not in self.experiments.index:
            raise ValueError(f"Comparison {comp_id} does not exist in the dataframe.")

        res = self.experiments.loc[comp_id]
        sim_value = res["metric_value"]
        return sim_value

    def comparison_exists(
        self,
        src_single_rep: SingleLayerRepresentation,
        tgt_single_rep: SingleLayerRepresentation,
        metric: SimilarityMeasure,
    ) -> bool:
        """
        Check if the comparison (or if symmetrict the inverse) already exists in the dataframe.
        Args:
            src_single_rep: SingleLayerRepresentation  # Represents the source model that in non-symmetric cases
            tgt_single_rep: SingleLayerRepresentation  # represents the target in non-symmetric cases
            metric: SimilarityMeasure
        Returns:
            bool: True if the comparison exists, False otherwise.
        """
        comp_id = self._get_comparison_id(src_single_rep, tgt_single_rep, metric.name)
        # If it exists we just continue, otherwise we check for the symmetric one
        if comp_id not in self.experiments.index:
            if metric.is_symmetric:
                comp_id = self._get_comparison_id(tgt_single_rep, src_single_rep, metric.name)
        return comp_id in self.experiments.index

    def save_to_file(self) -> None:
        """Save the results of the experiment to disk"""
        self.experiments.to_parquet(self.path_to_store)

    def __enter__(self):
        """When entering a context, load the experiments from disk"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """When exiting a context, save the experiments to disk"""
        self.save_to_file()
        self.experiments = None
        return False


def get_in_group_cross_group_sims(
    in_group_slrs, out_group_slrs, measure: SimilarityMeasure, storer: ExperimentStorer
):
    """
    Get the in-group and cross-group similarities for a given measure.
    Args:
        in_group_slrs: List of SingleLayerRepresentations of the in-group models.
        out_group_slrs: List of SingleLayerRepresentations of the out-group models.
        measure_name: Name of the measure to be used.
        storer: ExperimentStorer object to store and retrieve the results.
        Returns:
        in_group_sims: List of in-group similarities.
        cross_group_sims: List of cross-group similarities.
    """
    in_group_comps = combinations(in_group_slrs, 2)
    cross_group_comps = product(in_group_slrs, out_group_slrs)
    assert all(
        [storer.comparison_exists(slr1, slr2, measure) for slr1, slr2 in in_group_comps]
    ), "Not all in-group comparisons exist."
    assert all(
        [storer.comparison_exists(slr1, slr2, measure) for slr1, slr2 in cross_group_comps]
    ), "Not all cross-group comparisons exist."
    # Redo to not have empty iterable
    in_group_comps = combinations(in_group_slrs, 2)
    cross_group_comps = product(in_group_slrs, out_group_slrs)
    in_group_sims = [storer.get_comp_result(slr1, slr2, measure) for slr1, slr2 in in_group_comps]
    cross_group_sims = [storer.get_comp_result(slr1, slr2, measure) for slr1, slr2 in cross_group_comps]
    return in_group_sims, cross_group_sims


def get_ingroup_outgroup_SLRs(
    groups_of_models: tuple[list[TrainedModel]], in_group_id: int, rep_layer_id: int, representation_dataset: str
) -> tuple[list[SingleLayerRepresentation], list[SingleLayerRepresentation]]:
    n_groups = set(range(len(groups_of_models)))

    out_group_ids = n_groups - {in_group_id}
    in_group_models = [
        m.get_representation(representation_dataset).representations[rep_layer_id]
        for m in groups_of_models[in_group_id]
    ]
    out_group_models = [
        m.get_representation(representation_dataset).representations[rep_layer_id]
        for m in chain(*[groups_of_models[out_id] for out_id in out_group_ids])
    ]
    return in_group_models, out_group_models


def create_pivot_excel_table(
    eval_result: pd.DataFrame,
    row_index: str | Sequence[str],
    columns: str | Sequence[str],
    value_key: str,
    filename: str,
    sheet_name: str = "Sheet1",
) -> None:
    """
    Convert the evaluation result to a pandas dataframe
    Args:
        eval_result: Dictionary of evaluation results.
    Returns:
        None, but writes out a table to disk.
    """
    pivoted_result = eval_result.pivot(index=row_index, columns=columns, values=value_key)
    file_path = os.path.join(EXPERIMENT_RESULTS_PATH, filename)
    if filename.endswith(".xlsx"):
        with pd.ExcelWriter(file_path) as writer:
            pivoted_result.to_excel(writer, sheet_name=sheet_name)
    elif filename.endswith(".csv"):
        pivoted_result.to_csv(file_path)
    elif filename.endswith(".tex"):
        pivoted_result.to_latex(file_path)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def name_of_measure(obj):
    if isinstance(obj, SimilarityMeasure):
        return name_of_measure(obj.sim_func)
    elif hasattr(obj, "__name__"):
        # repsim.measures.utils.Pipeline
        return obj.__name__
    elif hasattr(obj, "func"):
        # functools.partial
        if hasattr(obj.func, "__name__"):
            # pure function
            return obj.func.__name__
        else:
            # TODO: Not sure this still works. Remove?
            # on a callable class instance
            return str(obj.func)
    else:
        return str(obj)
