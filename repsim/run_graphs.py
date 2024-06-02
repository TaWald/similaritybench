import argparse
import os
from typing import get_args
from typing import List

import yaml
from repsim.benchmark.types_globals import AUGMENTATION_EXPERIMENT_NAME
from repsim.benchmark.types_globals import BENCHMARK_EXPERIMENTS_LIST
from repsim.benchmark.types_globals import DEFAULT_SEEDS
from repsim.benchmark.types_globals import EXPERIMENT_DICT
from repsim.benchmark.types_globals import EXPERIMENT_IDENTIFIER
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GROUP_SEPARATION_EXPERIMENT
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import LAYER_EXPERIMENT_NAME
from repsim.benchmark.types_globals import MONOTONICITY_EXPERIMENT
from repsim.benchmark.types_globals import OUTPUT_CORRELATION_EXPERIMENT
from repsim.benchmark.types_globals import OUTPUT_CORRELATION_EXPERIMENT_NAME
from repsim.benchmark.types_globals import SHORTCUT_EXPERIMENT_NAME
from repsim.benchmark.types_globals import THREE_GROUP_EXPERIMENT_DICT
from repsim.benchmark.types_globals import TWO_GROUP_EXPERIMENT_DICT
from repsim.measures import ALL_MEASURES
from repsim.run import run

CONFIG_INCLUDED_MEASURES_KEY = "included_measures"
CONFIG_EXCLUDED_MEASURES_KEY = "excluded_measures"
CONFIG_THREADS_KEY = "threads"
CONFIG_CACHE_DISK_KEY = "cache_to_disk"
CONFIG_CACHE_MEMORY_KEY = "cache_to_mem"
CONFIG_RERUN_NANS_KEY = "rerun_nans"
CONFIG_EXTRACT_REPS_ONLY_KEY = "only_extract_reps"
CONFIG_REPRESENTATION_DATASET_KEY = "representation_dataset"

CONFIG_EXPERIMENTS_KEY = "experiments"
CONFIG_EXPERIMENTS_NAME_SUBKEY = "name"
CONFIG_EXPERIMENTS_TYPE_SUBKEY = "type"
CONFIG_EXPERIMENTS_FILTER_SUBKEY = "filter_key_vals"
CONFIG_EXPERIMENTS_TRAIN_DATA_SUBKEY = "train_dataset"
CONFIG_EXPERIMENTS_SEEDS_SUBKEY = "seed"
CONFIG_EXPERIMENTS_IDENTIFIER_SUBKEY = "identifier"
CONFIG_EXPERIMENTS_GROUPING_SUBKEY = "grouping_keys"
CONFIG_EXPERIMENTS_SEPARATION_SUBKEY = "separation_keys"
CONFIG_EXPERIMENTS_REPRESENTATION_DATA_SUBKEY = "representation_dataset"
CONFIG_EXPERIMENTS_DOMAIN_SUBKEY = "domain"

CONFIG_RAW_RESULTS_FILENAME_KEY = "raw_results_filename"

CONFIG_RES_TABLE_CREATION_KEY = "table_creation"
CONFIG_RES_TABLE_SAVE_SUBKEY = "save_full_df"
CONFIG_RES_TABLE_FILENAME_SUBKEY = "full_df_filename"
CONFIG_AGG_TABLE_SAVE_SUBKEY = "save_aggregated_df"
CONFIG_AGG_TABLE_INDEX_SUBKEY = "row_index"
CONFIG_AGG_TABLE_COLUMNS_SUBKEY = "columns"
CONFIG_AGG_TABLE_VALUE_SUBKEY = "value_key"
CONFIG_AGG_TABLE_FILENAME_SUBKEY = "filename"

EXPERIMENT_TYPE_DICT = {
    OUTPUT_CORRELATION_EXPERIMENT_NAME: OUTPUT_CORRELATION_EXPERIMENT,
    LAYER_EXPERIMENT_NAME: MONOTONICITY_EXPERIMENT,
    SHORTCUT_EXPERIMENT_NAME: GROUP_SEPARATION_EXPERIMENT,
    AUGMENTATION_EXPERIMENT_NAME: GROUP_SEPARATION_EXPERIMENT,
    LABEL_EXPERIMENT_NAME: GROUP_SEPARATION_EXPERIMENT,
}


def PARQUET_FILE_NAME(experiment, dataset):
    return f"results_{experiment}_{dataset}.parquet"


def FULL_DF_FILE_NAME(experiment, dataset, groups=5):
    if groups != 3 and experiment != OUTPUT_CORRELATION_EXPERIMENT:
        return f"{experiment}_{dataset}_{groups}groups_full.csv"
    return f"{experiment}_{dataset}_full.csv"


def AGG_DF_FILE_NAME(experiment, dataset, groups=3):
    if groups != 3 and experiment != OUTPUT_CORRELATION_EXPERIMENT:
        return f"{experiment}_{dataset}_{groups}groups.csv"
    return f"{experiment}_{dataset}.csv"


def YAML_CONFIG_FILE_NAME(experiment, dataset, groups=3):
    if groups != 3 and experiment != OUTPUT_CORRELATION_EXPERIMENT:
        return f"{experiment}_{dataset}_{groups}groups.yaml"
    return f"{experiment}_{dataset}.yaml"


def build_graph_config(
    experiment: EXPERIMENT_IDENTIFIER,
    dataset: GRAPH_DATASET_TRAINED_ON,
    measures: List = None,
    save_to_memory=True,
    save_to_disk=False,
    groups: int = 5,
):
    if groups == 5:
        experiment_settings = EXPERIMENT_DICT[experiment]
    elif groups == 3:
        experiment_settings = THREE_GROUP_EXPERIMENT_DICT[experiment]
    else:
        experiment_settings = TWO_GROUP_EXPERIMENT_DICT[experiment]

    experiment_type = EXPERIMENT_TYPE_DICT[experiment]

    save_agg_table = True if experiment_type != OUTPUT_CORRELATION_EXPERIMENT else False
    yaml_dict = {
        CONFIG_THREADS_KEY: 1,
        CONFIG_CACHE_MEMORY_KEY: save_to_memory,
        CONFIG_CACHE_DISK_KEY: save_to_disk,
        CONFIG_EXTRACT_REPS_ONLY_KEY: False,
        CONFIG_EXPERIMENTS_KEY: [
            {
                CONFIG_EXPERIMENTS_NAME_SUBKEY: f"{experiment} {dataset}",
                CONFIG_EXPERIMENTS_TYPE_SUBKEY: experiment_type,
                CONFIG_REPRESENTATION_DATASET_KEY: dataset,
                CONFIG_EXPERIMENTS_FILTER_SUBKEY: {
                    CONFIG_EXPERIMENTS_IDENTIFIER_SUBKEY: experiment_settings,
                    CONFIG_EXPERIMENTS_TRAIN_DATA_SUBKEY: [dataset],
                    CONFIG_EXPERIMENTS_SEEDS_SUBKEY: DEFAULT_SEEDS,
                    CONFIG_EXPERIMENTS_DOMAIN_SUBKEY: "GRAPHS",
                },
                CONFIG_EXPERIMENTS_GROUPING_SUBKEY: ["identifier"],
                CONFIG_EXPERIMENTS_SEPARATION_SUBKEY: ["architecture"],
            },
        ],
        CONFIG_RAW_RESULTS_FILENAME_KEY: PARQUET_FILE_NAME(experiment, dataset),
        #
        CONFIG_RES_TABLE_CREATION_KEY: {
            CONFIG_RES_TABLE_SAVE_SUBKEY: True,
            CONFIG_RES_TABLE_FILENAME_SUBKEY: FULL_DF_FILE_NAME(experiment, dataset, groups),
            CONFIG_AGG_TABLE_SAVE_SUBKEY: save_agg_table,
            CONFIG_AGG_TABLE_INDEX_SUBKEY: "similarity_measure",
            CONFIG_AGG_TABLE_COLUMNS_SUBKEY: ["quality_measure", "architecture"],
            CONFIG_AGG_TABLE_VALUE_SUBKEY: "value",
            CONFIG_AGG_TABLE_FILENAME_SUBKEY: AGG_DF_FILE_NAME(experiment, dataset, groups),
        },
    }
    if measures is None:
        yaml_dict[CONFIG_EXCLUDED_MEASURES_KEY] = ["RSMNormDifference", "IMDScore", "GeometryScore", "PWCCA"]
    else:
        yaml_dict[CONFIG_INCLUDED_MEASURES_KEY] = measures

    return yaml_dict


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=list(get_args(GRAPH_DATASET_TRAINED_ON)),
        help="Datasets used in evaluation.",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        choices=BENCHMARK_EXPERIMENTS_LIST,
        help="Test to run.",
    )
    parser.add_argument(
        "-m",
        "--measures",
        type=str,
        nargs="*",
        choices=list(ALL_MEASURES.keys()),
        default=None,
        help="Test to run.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        choices=[2, 3, 5],
        default=5,
        help="Number of groups to separate per experiment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n_groups = args.groups

    yaml_config = build_graph_config(
        experiment=args.experiment,
        dataset=args.dataset,
        measures=args.measures,
        groups=n_groups,
    )

    config_path = os.path.join("repsim", "configs", YAML_CONFIG_FILE_NAME(args.experiment, args.dataset, args.groups))
    with open(config_path, "w") as file:
        yaml.dump(yaml_config, file)

    run(config_path=config_path)
