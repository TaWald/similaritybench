from argparse import ArgumentParser
from collections.abc import Sequence
from typing import Type

import yaml
from loguru import logger
from repsim.benchmark.abstract_experiment import AbstractExperiment
from repsim.benchmark.group_separation_experiment import GroupSeparationExperiment
from repsim.benchmark.model_selection import get_grouped_models
from repsim.benchmark.monotonicity_experiment import MonotonicityExperiment
from repsim.benchmark.multimodel_experiments import MultiModelExperiment
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import create_pivot_excel_table
from repsim.measures import ALL_MEASURES
from repsim.measures.utils import SimilarityMeasure


def get_experiment_from_name(name: str) -> Type[AbstractExperiment]:
    if name == "GroupSeparationExperiment":
        return GroupSeparationExperiment
    elif name == "MonotonicityExperiment":
        return MonotonicityExperiment
    elif name == "MultiModelExperiment":
        return MultiModelExperiment  # type: ignore
    else:
        raise ValueError(f"Invalid experiment name: {name}")


def read_yaml_config(config_path: str) -> dict:
    """
    Read a yaml file and return the dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # ------------ If no measures provided we default to all measures ------------ #
    return config


def get_measures(config: dict) -> list[SimilarityMeasure]:
    include_key = "included_measures"
    exclude_key = "excluded_measures"
    assert not (
        include_key in config and exclude_key in config
    ), "Cannot include and exclude measures at the same time"
    used_measures = None
    if include_key in config:
        if config[include_key] == "all":
            used_measures = list(ALL_MEASURES.values())
        else:
            if isinstance(config[include_key], str):
                used_measures = [ALL_MEASURES[config[include_key]]]
            else:
                used_measures = [ALL_MEASURES[measurename] for measurename in config[include_key]]
    elif exclude_key in config:
        if isinstance(config[exclude_key], str):
            excluded_measures = [config[exclude_key]]
        else:
            excluded_measures = config[exclude_key]
        used_measures = [measure for name, measure in ALL_MEASURES.items() if name not in excluded_measures]
    else:
        used_measures = list(ALL_MEASURES.values())

    logger.info(f"Using measures: {[m.name for m in used_measures]}")
    return used_measures


def verify_config(config: dict) -> None:
    """
    Raise error if config is not valid
    """
    if "excluded_measures" in config:
        assert isinstance(
            config["excluded_measures"], (str, list)
        ), "Excluded measures should be a string or a list of strings"
        assert "included_measures" not in config, "Cannot include and exclude measures at the same time"
        assert all(
            measure in ALL_MEASURES for measure in config["excluded_measures"]
        ), "Some measures in excluded_measures are not valid"
    if "included_measures" in config:
        assert isinstance(
            config["included_measures"], (str, list)
        ), "Included measures should be a string or a list of strings"
        assert "excluded_measures" not in config, "Cannot include and exclude measures at the same time"
        assert all(
            measure in ALL_MEASURES for measure in config["included_measures"]
        ), "Some measures in included_measures are not valid"
    if "excluded_measures" not in config and "included_measures" not in config:
        logger.info(
            "Not specifying which measures to compute. Defaulting to all. Specify through 'included_measures' or 'excluded_measures'."
        )
    # Make sure either one or the other is in the config
    assert not (
        ("excluded_measures" in config) and ("included_measures" in config)
    ), "Cannot include and exclude measures at the same time"

    assert "experiments" in config, "No experiments in config."
    # for exp in config["experiments"]:
    #     filter_key_vals = config.get("filter_key_vals", None)
    #     if filter_key_vals:
    #         for key, val in filter_key_vals.items():
    #             assert isinstance(key, str), "Key should be a string"
    #             assert isinstance(val, str) or isinstance(val, list), "Value should be a string or a list of strings"

    #     differentiation_keys = config.get("differentiation_keys", None)
    #     if differentiation_keys:
    #         for key in differentiation_keys:
    #             assert isinstance(key, str), "Differentiation key should be a string"


def create_table(config: dict):
    if config.get("table_creation", None) is not None:
        return True
    return False


def run(config_path: str):
    config = read_yaml_config(config_path)
    verify_config(config)
    logger.debug("Config is valid")
    measures = get_measures(config)
    threads = config.get("threads", 1)

    logger.info(f"Running with {threads} Threads! Reduce if running OOM or set to 1 for single threaded execution.")

    all_experiments = []
    for experiment in config["experiments"]:
        if experiment["type"] == "GroupSeparationExperiment":
            filter_key_vals = experiment.get("filter_key_vals", None)
            grouping_keys = experiment.get("grouping_keys", None)
            separation_keys = experiment.get("separation_keys", None)

            # Not quite sure how to best make this work with the entire setup.
            #   Some form of hierarchy would need to be added, but that doesn't exist atm.
            #   Maybe create a hierarchical nested structure and make this a recursive function? to auto-aggregate?
            grouped_models: list[list[Sequence[TrainedModel]]] = get_grouped_models(
                models=ALL_TRAINED_MODELS,
                filter_key_vals=filter_key_vals,
                separation_keys=separation_keys,
                grouping_keys=grouping_keys,
            )
            for group in grouped_models:
                exp = GroupSeparationExperiment(
                    grouped_models=group,
                    measures=measures,
                    representation_dataset=experiment["representation_dataset"],
                    threads=threads,
                )
                all_experiments.append(exp)

    # -------------------- Now compare/eval the grouped models ------------------- #
    exp_results = []
    for ex in all_experiments:
        ex.run()
        exp_results.extend(ex.eval())

    if create_table:
        create_pivot_excel_table(
            exp_results,
            **config["table_creation"],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Domain selection to run experiments on.",
    )
    args = parser.parse_args()
    logger.debug("Parsing config")
    config_path = args.config
    # config_path = os.path.join(os.path.dirname(__file__), "configs", "hierarchical_vision_shortcuts.yaml")
    run(config_path)
