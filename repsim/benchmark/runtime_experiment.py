import argparse
import itertools
import time
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import repsim.benchmark.paths
import repsim.measures
import torch
from loguru import logger
from repsim.measures.utils import SHAPE_TYPE
from tqdm import tqdm


def wakeup_device(device: torch.device) -> None:
    """Do some matrix multiplication to have HW switch to performance mode and stop
    power saving.

    Args:
        device (torch.device): device to wake up
    """
    MATRIX_SIZE: int = 3000
    N_REPEAT: int = 5

    x = torch.randn(size=(MATRIX_SIZE, MATRIX_SIZE), device=device)
    for _ in range(N_REPEAT):
        x @ x  # type:ignore


def _get_func_name(func: Callable) -> str:
    if hasattr(func, "__name__"):  # repsim.measures.utils.Pipeline
        name = func.__name__
    elif hasattr(func, "func"):  # functools.partial
        if hasattr(func.func, "__name__"):  # on a pure function
            name = func.func.__name__
        else:  # on a callable class instance
            name = str(func.func)
    else:
        name = str(func)
    return name


def bench_cpu(
    rng: np.random.Generator,
    matrix_sizes: list[tuple[int, int]],
    funcs: list[Callable[[torch.Tensor | npt.NDArray, torch.Tensor | npt.NDArray, SHAPE_TYPE], float]],
    n_repeat: int,
) -> list[tuple[str, int, int, float]]:
    wakeup_device(torch.device("cpu"))
    times = []
    for n_instances, n_features in tqdm(matrix_sizes):
        logger.info(f"Using matrix size of {n_instances, n_features}.")
        x = rng.random((n_instances, n_features))
        y = rng.random((n_instances, n_features))
        for func in funcs:
            func_name = _get_func_name(func)
            logger.info(f"Testing {func_name}.")
            try:
                for _ in range(n_repeat):
                    start = time.perf_counter()
                    func(x, y, "nd")  # we create the matrices ourselves, so shape is fixed to nd
                    end = time.perf_counter()
                    times.append((func_name, n_instances, n_features, end - start))  # seconds
            except Exception as e:
                logger.info(f"Test failed with {str(e)}. Continuing with next function.")
                times.append((func_name, n_instances, n_features, np.nan))  # seconds
    return times


def bench_cuda(
    rng: np.random.Generator,
    matrix_sizes: list[tuple[int, int]],
    funcs: tuple[Callable],
    cuda_device: torch.device,
    n_repeat: int,
) -> list[tuple[str, float]]:
    MS_TO_SEC: float = 1 / 1000
    wakeup_device(cuda_device)
    times = []
    for n_instances, n_features in tqdm(matrix_sizes):
        x = rng.random((n_instances, n_features))
        for func in funcs:
            func_name = _get_func_name(func)
            for _ in range(n_repeat):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record(torch.cuda.current_stream(cuda_device))
                func(x, x)
                end.record(torch.cuda.current_stream(cuda_device))

                torch.cuda.synchronize()
                times.append(
                    (
                        func_name,
                        n_instances,
                        n_features,
                        start.elapsed_time(end) * MS_TO_SEC,
                    )
                )  # seconds
    return times


class RuntimeBenchmark:

    def __init__(
        self,
        save_dir: Optional[Path] = None,
        seed: int = 100,
        overwrite_columns: Optional[list[str]] = None,
        n_repeat: int = 1,
    ) -> None:
        self.save_dir = repsim.benchmark.paths.EXPERIMENT_RESULTS_PATH / "runtime" if save_dir is None else save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_repeat = n_repeat

        self.columns = (
            ["Function", "N_instances", "N_features", "Time"] if overwrite_columns is None else overwrite_columns
        )

    def run(
        self,
        n_instances_range: tuple[int, int],
        n_features_range: tuple[int, int],
        n_instances_steps: int,
        n_features_steps: int,
        functions_to_bench: list[Callable],
        n_jobs: Optional[tuple[int, ...]] = None,
        fname: str = "benchmark_results.csv",
    ):
        logger.add(self.save_dir / "{time}.log")

        matrix_sizes = list(
            itertools.product(
                np.linspace(*n_instances_range, dtype=int, num=n_instances_steps),
                np.linspace(*n_features_range, dtype=int, num=n_features_steps),
            )
        )
        logger.info(f"Benchmark will use matrix sizes: {matrix_sizes}")

        if n_jobs is None:
            records = bench_cpu(self.rng, matrix_sizes, functions_to_bench, self.n_repeat)
            self.df = pd.DataFrame.from_records(records, columns=self.columns)
        else:
            self.columns.append("N_Jobs")
            self.df = pd.DataFrame(columns=self.columns)
            for job_count in n_jobs:
                logger.info(f"Running with {job_count} jobs")
                funcs = [
                    partial(f, n_jobs=job_count) for f in functions_to_bench
                ]  # assumes n_jobs is accepted by function
                records = bench_cpu(self.rng, matrix_sizes, funcs, self.n_repeat)
                records = [(*record, job_count) for record in records]
                self.df = pd.concat((self.df, pd.DataFrame.from_records(records, columns=self.columns)), axis=0)

        self.df["Device"] = "CPU"
        self.df.to_csv(self.save_dir / fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", metavar="N", type=int, default=0, help="index of cuda device to use")
    parser.add_argument(
        "--n-repeat",
        metavar="N",
        type=int,
        default=1,
        help="number of repetitions of operations",
    )
    parser.add_argument("--max-inputs", type=int, default=20000)
    parser.add_argument("--min-inputs", type=int, default=4000)
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--min-features", type=int, default=1000)
    parser.add_argument("--inputs-steps", type=int, default=5)
    parser.add_argument("--features-steps", type=int, default=1)
    parser.add_argument("--jobs", type=int, nargs="+", default=None)
    parser.add_argument("--fname", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    cuda_device = torch.device(f"cuda:{args.cuda}")
    n_repeat = args.n_repeat
    n_instances_range = (args.min_inputs, args.max_inputs)
    n_features_range = (args.min_features, args.max_features)

    # functions_to_bench_cpu = repsim.measures.SYMMETRIC_MEASURES
    # lib = repsim.measures
    lib = repsim.measures_sklearn
    functions_to_bench_cpu = [
        # lib.magnitude_difference,
        # lib.concentricity_difference,
        # lib.uniformity_difference,
        # lib.rsm_norm_diff,
        # lib.eigenspace_overlap_score,
        # lib.aligned_cossim,
        # lib.procrustes_size_and_shape_distance,
        # lib.orthogonal_procrustes_centered_and_normalized,
        # lib.permutation_procrustes,
        # lib.representational_similarity_analysis,
        # lib.centered_kernel_alignment,
        # lib.hard_correlation_match,
        # lib.soft_correlation_match,
        lib.DistanceCorrelation(),
        lib.HardCorrelationMatch(),
        # lib.orthogonal_angular_shape_metric_centered,
        # lib.jaccard_similarity,
        # lib.second_order_cosine_similarity,
        # lib.rank_similarity,
        # # repsim.measures.geometry_score,
        # # repsim.measures.imd_score,
        # lib.gulp,
        # lib.svcca,
    ]
    benchmark = RuntimeBenchmark(n_repeat=args.n_repeat)
    benchmark.run(
        functions_to_bench=functions_to_bench_cpu,
        n_instances_range=n_instances_range,
        n_features_range=n_features_range,
        n_instances_steps=args.inputs_steps,
        n_features_steps=args.features_steps,
        n_jobs=args.jobs,
        fname=args.fname,
    )
