import argparse
import os
from typing import get_args

import yaml
from repsim.benchmark.types_globals import BENCHMARK_EXPERIMENTS_LIST
from repsim.benchmark.types_globals import DEFAULT_SEEDS
from repsim.benchmark.types_globals import EXPERIMENT_COMPARISON_TYPE
from repsim.benchmark.types_globals import EXPERIMENT_DICT
from repsim.benchmark.types_globals import EXPERIMENT_IDENTIFIER
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GROUP_SEPARATION_EXPERIMENT
from repsim.benchmark.types_globals import OUTPUT_CORRELATION_EXPERIMENT
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


def build_graph_config(
    experiment: EXPERIMENT_IDENTIFIER,
    comparison_type: EXPERIMENT_COMPARISON_TYPE,
    dataset: GRAPH_DATASET_TRAINED_ON,
    filename_prefix: str,
    save_to_memory=False,
    save_to_disk=True,
):
    yaml_dict = {
        CONFIG_EXCLUDED_MEASURES_KEY: ["RSMNormDifference", "IMDScore", "GeometryScore", "PWCCA"],
        CONFIG_THREADS_KEY: 1,
        CONFIG_CACHE_MEMORY_KEY: save_to_memory,
        CONFIG_CACHE_DISK_KEY: save_to_disk,
        CONFIG_EXTRACT_REPS_ONLY_KEY: False,
        CONFIG_EXPERIMENTS_KEY: [
            {
                CONFIG_EXPERIMENTS_NAME_SUBKEY: f"{experiment} {comparison_type} {dataset}",
                CONFIG_EXPERIMENTS_TYPE_SUBKEY: comparison_type,
                CONFIG_REPRESENTATION_DATASET_KEY: dataset,
                CONFIG_EXPERIMENTS_FILTER_SUBKEY: {
                    CONFIG_EXPERIMENTS_IDENTIFIER_SUBKEY: EXPERIMENT_DICT[experiment],
                    CONFIG_EXPERIMENTS_TRAIN_DATA_SUBKEY: [dataset],
                    CONFIG_EXPERIMENTS_SEEDS_SUBKEY: DEFAULT_SEEDS,
                    CONFIG_EXPERIMENTS_DOMAIN_SUBKEY: "GRAPHS",
                },
                CONFIG_EXPERIMENTS_GROUPING_SUBKEY: ["identifier"],
                CONFIG_EXPERIMENTS_SEPARATION_SUBKEY: ["architecture"],
            },
        ],
        CONFIG_RAW_RESULTS_FILENAME_KEY: f"{filename_prefix}.parquet",
        #
        CONFIG_RES_TABLE_CREATION_KEY: {
            CONFIG_RES_TABLE_SAVE_SUBKEY: True,
            CONFIG_RES_TABLE_FILENAME_SUBKEY: f"{filename_prefix}_full.csv",
            CONFIG_AGG_TABLE_SAVE_SUBKEY: True,
            CONFIG_AGG_TABLE_INDEX_SUBKEY: "similarity_measure",
            CONFIG_AGG_TABLE_COLUMNS_SUBKEY: ["quality_measures", "architecture"],
            CONFIG_AGG_TABLE_VALUE_SUBKEY: "value",
            CONFIG_AGG_TABLE_FILENAME_SUBKEY: f"{filename_prefix}.csv",
        },
    }
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
        "--output_corr",
        action="store_true",
        help="Whether to retrain existing models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    exp_type = OUTPUT_CORRELATION_EXPERIMENT if args.output_corr else GROUP_SEPARATION_EXPERIMENT
    exp_type_str = "output_correlation" if args.output_corr else "group_separation"
    file_prefix = f"{args.experiment}_{exp_type_str}_{args.dataset}"
    yaml_config = build_graph_config(
        experiment=args.experiment, comparison_type=exp_type, dataset=args.dataset, filename_prefix=file_prefix
    )
    yaml_filename = os.path.join("repsim", "configs", f"{file_prefix}.yaml")
    with open(yaml_filename, "w") as file:
        yaml.dump(yaml_config, file)

    run(config_path=yaml_filename)
