import os
from dataclasses import asdict
from typing import Callable

import git
import numpy as np
import pandas as pd
from loguru import logger
from repsim.benchmark.paths import EXPERIMENT_RESULTS_PATH
from repsim.benchmark.registry import TrainedModelRep
from repsim.measures.utils import SimilarityMeasure
from repsim.utils import ModelRepresentations
from repsim.utils import SingleLayerRepresentation


class ExperimentStorer:

    def __init__(self, path_to_store: str | None = None) -> None:
        if path_to_store is None:
            path_to_store = os.path.join(EXPERIMENT_RESULTS_PATH, "experiments.parquet")
        self.path_to_store = path_to_store
        self.experiments: pd.DataFrame | None = None

    def add_results(
        self,
        single_rep_a: SingleLayerRepresentation,
        single_rep_b: SingleLayerRepresentation,
        metric_name: str,
        metric_value: float,
        runtime: float | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add a comparison result of the experiments to disk.
        Serializes all the information into a unique identifier and stores the results in a pandas dataframe.
        """
        if self.comparison_exists(single_rep_a, single_rep_b, metric_name) and not overwrite:
            logger.info("Comparison already exists and Overwrite is False. Skipping.")
        first_rep, second_rep = self._sort_models(single_rep_a, single_rep_b)
        comp_id = self._get_comparison_id(first_rep, second_rep, metric_name)

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        content = {"first_" + k: v for k, v in asdict(first_rep).items() if k != "representation"}
        content.update({"second_" + k: v for k, v in asdict(second_rep).items() if k != "representation"})
        content.update({"metric": metric_name, "metric_value": metric_value, "runtime": runtime, "id": comp_id})
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
        single_rep_a: SingleLayerRepresentation,
        single_rep_b: SingleLayerRepresentation,
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
        id_reps_a = single_rep_a.unique_identifier()
        id_reps_b = single_rep_b.unique_identifier()

        first_id, second_id = list(sorted([id_reps_a, id_reps_b]))
        joint_id = "___".join([first_id, second_id, metric_name])
        return joint_id

    def get_comp_result(
        self, single_rep_a: SingleLayerRepresentation, single_rep_b: SingleLayerRepresentation, metric_name: str
    ) -> dict:
        """Return the result of the comparison"""
        comp_id = self._get_comparison_id(single_rep_a, single_rep_b, metric_name)
        res = self.experiments.loc[comp_id]
        sim_value = res["metric_value"]
        return sim_value

    def comparison_exists(
        self,
        single_rep_a: SingleLayerRepresentation,
        single_rep_b: SingleLayerRepresentation,
        metric_name: str,
    ) -> bool:
        """Check if the comparison already exists in the dataframe"""
        comp_id = self._get_comparison_id(single_rep_a, single_rep_b, metric_name)
        return comp_id in self.experiments.index

    def save_to_file(self) -> None:
        """Save the results of the experiment to disk"""
        self.experiments.to_parquet(self.path_to_store)

    def __enter__(self):
        """When entering a context, load the experiments from disk"""
        if os.path.exists(self.path_to_store):  # If it exists, load.
            self.experiments = pd.read_parquet(self.path_to_store)
        else:
            self.experiments = pd.DataFrame()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """When exiting a context, save the experiments to disk"""
        self.save_to_file()
        self.experiments = None
        return False


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
