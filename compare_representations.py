import itertools
import logging
import os
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import llmcomp.measures
import llmcomp.measures.utils
import llmcomp.representations
import llmcomp.utils

log = logging.getLogger(__name__)


def compare_pair(
    dir1: Path,
    dir2: Path,
    strategy: llmcomp.representations.Strategy,
    measures: List[Callable],
    modelname1: str,
    modelname2: str,
    datasetname: str,
    splitname: str,
    pair_results_path: Optional[Path] = None,
):
    rep1 = llmcomp.representations.load_representations(dir1)
    rep2 = llmcomp.representations.load_representations(dir2)
    results_pair = defaultdict(list)

    log.info(f"Comparing with strategy {strategy.strat_id}")
    for sim_func in measures:
        start = time.perf_counter()
        results = strategy(rep1, rep2, sim_func)
        llmcomp.utils.extend_dict_of_lists(results_pair, results)
        # TODO: can we specify these keys in a more modifiable way?
        n_times_to_add = len(results_pair["score"]) - len(results_pair["model1"])
        results_pair["model1"].extend([modelname1] * n_times_to_add)
        results_pair["model2"].extend([modelname2] * n_times_to_add)
        results_pair["dataset"].extend([datasetname] * n_times_to_add)
        results_pair["split"].extend([splitname] * n_times_to_add)

        if hasattr(sim_func, "__name__"):
            measure_name = sim_func.__name__
        elif hasattr(sim_func, "func"):
            measure_name = sim_func.func.__name__
        else:
            measure_name = str(sim_func)

        results_pair["measure"].extend([measure_name] * n_times_to_add)
        results_pair["strategy"].extend([strategy.strat_id] * n_times_to_add)

        log.info(
            f"{measure_name} completed in {time.perf_counter() - start:.1f} seconds"  # noqa: E501
        )
    pd.DataFrame.from_dict(results_pair).to_parquet(pair_results_path)
    return results_pair


def compare_all_models_over_one_dataset(
    basedir: Path,
    results_dir: Path,
    must_contain_all: List[str],
    must_contain_any: List[str],
    must_not_contain: List[str],
    one_must_contain: List[str],
    recompute: bool,
    strategy: llmcomp.representations.Strategy,
    measures: List[Callable],
):
    results = {}

    representation_dirs = list(sorted(basedir.iterdir()))
    results_dir.mkdir(exist_ok=True)

    combinations = list(itertools.combinations(representation_dirs, 2))
    combinations = llmcomp.utils.filter_combinations(
        combinations,
        must_contain_all,
        must_contain_any,
        must_not_contain,
        one_must_contain,
    )

    progress_bar = tqdm(total=len(combinations), desc="Model pairs")
    for dir1, dir2 in combinations:
        info1 = llmcomp.utils.extract_info(dir1)
        info2 = llmcomp.utils.extract_info(dir2)
        partial_results_path = Path(
            results_dir,
            f"similarity_{info1['model']}_{info2['model']}_{info1['dataset']}_{info1['split']}.parquet",  # noqa: E501
        )

        if not llmcomp.utils.pair_should_be_compared(
            info1, info2, partial_results_path, recompute
        ):
            progress_bar.update(1)
            continue
        log.info(
            f"Comparing {info1['model']} and {info2['model']} on {info1['dataset']}"
        )

        results_pair = compare_pair(
            dir1,
            dir2,
            strategy,
            measures,
            info1["model"],
            info2["model"],
            info1["dataset"],
            info1["split"],
            pair_results_path=partial_results_path,
        )
        llmcomp.utils.extend_dict_of_lists(results, results_pair)

        progress_bar.update(1)

    pd.DataFrame.from_dict(results).to_parquet(Path(results_dir, "similarity.parquet"))


def is_single_dataset_dir(directory: Path) -> bool:
    start_depth = len(directory.parents)

    def depth(path: Path) -> int:
        return len(path.parents)

    def to_dirpath(os_walk_output_elem: Tuple) -> Path:
        return Path(os_walk_output_elem[0])

    max_depth = max(map(depth, map(to_dirpath, os.walk(directory))))
    remaining_depth_from_directory = max_depth - start_depth
    if remaining_depth_from_directory == 2:
        return False
    elif remaining_depth_from_directory == 1:
        return True
    else:
        raise ValueError(
            f"Directory ({directory}) does not seem to be valid. Remaining depth of "
            f"directory tree is > 2 ({remaining_depth_from_directory})"
        )


@hydra.main(config_path="config", config_name="compare", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))

    strategy = hydra.utils.instantiate(cfg.strategy)
    measures = hydra.utils.instantiate(cfg.measures)
    compare_fn = partial(
        compare_all_models_over_one_dataset,
        must_contain_all=cfg.filter.must_contain_all,
        must_contain_any=cfg.filter.must_contain_any,
        must_not_contain=cfg.filter.must_not_contain,
        one_must_contain=cfg.filter.one_must_contain,
        recompute=cfg.recompute,
        strategy=strategy,
        measures=measures,
    )

    results_dir = Path(
        cfg.storage.root_dir, cfg.storage.results_subdir
    )  # TODO: this path should somehow automatically relate to the dataset

    basedir = Path(cfg.storage.root_dir, cfg.storage.reps_subdir)
    if is_single_dataset_dir(basedir):
        compare_fn(basedir=basedir, results_dir=results_dir)
    elif cfg.multi_dataset.allowed and not cfg.multi_dataset.combine_representations:
        for dataset_dir in basedir.iterdir():
            log.info("Comparing representations from %s", str(dataset_dir))
            results_subdir = results_dir / dataset_dir.name
            compare_fn(basedir=dataset_dir, results_dir=results_subdir)
    elif cfg.multi_dataset.allowed and cfg.multi_dataset.combine_representations:
        # TODO: per model pair, where both models have reps for all datasets (except those filtered out by cfg.filter):
        # 1) load all reps per model and each create one concatted rep matrix (along dim 0)
        # 2) run pair comparison as normal
        pass


if __name__ == "__main__":
    main()
