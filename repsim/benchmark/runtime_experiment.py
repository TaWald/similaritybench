import argparse
import itertools
import time
from typing import Callable

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", metavar="N", type=int, default=0, help="index of cuda device to use")
    parser.add_argument(
        "--n-repeat",
        metavar="N",
        type=int,
        default=1,
        help="number of repetitions of operations",
    )
    parser.add_argument("--max_inputs", type=int, default=1000)
    parser.add_argument("--min_inputs", type=int, default=100)
    parser.add_argument("--max_features", type=int, default=1000)
    parser.add_argument("--min_features", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=5)

    return parser.parse_args()


def _get_func_name(func: Callable) -> str:
    if hasattr(func, "__name__"):  # repsim.measures.utils.Pipeline
        name = func.__name__
    elif hasattr(func, "func"):  # functools.partial
        name = func.func.__name__
    else:
        name = str(func)
    return name


def bench_cpu(
    rng: np.random.Generator,
    matrix_sizes: list[tuple[int, int]],
    funcs: list[Callable[[torch.Tensor | npt.NDArray, torch.Tensor | npt.NDArray, SHAPE_TYPE], float]],
    n_repeat: int,
) -> list[tuple[str, float]]:
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


if __name__ == "__main__":
    args = parse_args()
    cuda_device = torch.device(f"cuda:{args.cuda}")
    n_repeat = args.n_repeat

    save_dir = repsim.benchmark.paths.EXPERIMENT_RESULTS_PATH / "runtime"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.add(save_dir / "{time}.log")

    functions_to_bench_cpu = repsim.measures.SYMMETRIC_MEASURES
    # functions_to_bench_cuda = (partial(centered_kernel_alignment_cuda, device=cuda_device),)

    seed: int = 100
    matrix_sizes = list(
        itertools.product(
            np.linspace(args.min_inputs, args.max_inputs, dtype=int, num=args.num_steps),
            np.linspace(args.min_features, args.max_features, dtype=int, num=args.num_steps),
        )
    )
    logger.info(f"Benchmark will use matrix sizes: {matrix_sizes}")
    rng = np.random.default_rng(seed)

    cpu_times = bench_cpu(rng, matrix_sizes, functions_to_bench_cpu, n_repeat)
    # cuda_times = bench_cuda(rng, matrix_sizes, functions_to_bench_cuda, cuda_device, n_repeat)

    columns = ["Function", "N_instances", "N_features", "Time"]
    df_cpu = pd.DataFrame.from_records(cpu_times, columns=columns)
    # df_cuda = pd.DataFrame.from_records(cuda_times, columns=columns)
    df_cpu["Device"] = "CPU"
    # df_cuda["Device"] = "CUDA"
    # df = pd.concat((df_cpu, df_cuda))
    df = df_cpu
    print(df.groupby(["Function", "N_instances", "N_features", "Device"]).agg([np.mean, np.std]))
    df.to_csv(save_dir / "benchmark_results.csv")
