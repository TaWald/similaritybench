import os
from argparse import ArgumentParser
from collections.abc import Sequence
from typing import get_args

import yaml
from loguru import logger
from repsim.benchmark.archive.shortcut_experiment import GroupSeparationExperiment
from repsim.benchmark.model_selection import get_grouped_trained_models
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import DOMAIN_TYPE
from repsim.benchmark.registry import TrainedModel
from repsim.measures.measure_index import ALL_MEASURES
from tqdm import tqdm


def read_yaml_config(config_path: str) -> dict:
    """
    Read a yaml file and return the dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # ------------ If no measures provided we default to all measures ------------ #
    if "measures" not in config:
        logger.info("No measures provided in config. Defaulting to all measures")
        config["measures"] = list(ALL_MEASURES.keys())
    return config


def verify_config(config: dict) -> None:
    """
    Raise error if config is not valid
    """
    assert config["domain"] in get_args(DOMAIN_TYPE), "Invalid domain, choose from: {}".format(DOMAIN_TYPE)
    # assert config["experiment"] in EXPERIMENT_IDENTIFIER, "Invalid experiment, choose from: {}".format(
    #     EXPERIMENT_IDENTIFIER
    # )
    assert all(
        [measurename in ALL_MEASURES for measurename in config["measures"]]
    ), f"Invalid measures. Got {[f'{c}' for c in config['measures']]} choose from: {list(ALL_MEASURES.keys())}"
    assert "representation_dataset" in config, "No representation dataset provided"

    filter_key_vals = config.get("filter_key_vals", None)
    if filter_key_vals:
        for key, val in filter_key_vals.items():
            assert isinstance(key, str), "Key should be a string"
            assert isinstance(val, str) or isinstance(val, list), "Value should be a string or a list of strings"

    differentiation_keys = config.get("differentiation_keys", None)
    if differentiation_keys:
        for key in differentiation_keys:
            assert isinstance(key, str), "Differentiation key should be a string"


def run(config_path: str):
    config = read_yaml_config(config_path)
    verify_config(config)
    logger.debug("Config is valid")
    filter_key_vals = config.get("filter_key_vals", None)
    differentiation_keys = config.get("differentiation_keys", None)
    # Not quite sure how to best make this work with the entire setup.
    #   Some form of hierarchy would need to be added, but that doesn't exist atm.
    #   Maybe create a hierarchical nested structure and make this a recursive function? to auto-aggregate?
    grouped_models: list[Sequence[TrainedModel]] = get_grouped_trained_models(
        ALL_TRAINED_MODELS,
        filter_key_vals,
        differentiation_keys,
    )
    measures = [ALL_MEASURES[measurename] for measurename in config["measures"]]
    # -------------------- Now compare/eval the grouped models ------------------- #
    exp = GroupSeparationExperiment(
        grouped_models=grouped_models,
        measures=measures,  # We should likely introduce a "TAG" system to select measures
        representation_dataset=config["representation_dataset"],
    )
    exp.run()
    exp.eval()


if __name__ == "__main__":
    # parser = ArgumentParser()
    #     parser.add_argument(
    #         "-config",
    #         type=str,
    #         help="Domain selection to run experiments on.",
    #     )
    #     args = parser.parse_args()
    #     logger.debug("Parsing config"
    config_path = os.path.join(os.path.dirname(__file__), "configs", "vision_shortcuts.yaml")
    run(config_path)
