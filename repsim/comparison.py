import itertools
import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy.typing as npt
import pandas as pd
import torch
from repsim.measures.utils import SHAPE_TYPE
from tqdm import tqdm

log = logging.getLogger(__name__)


def name_of_similarity_function(
    sim_func: Callable[
        [
            Union[npt.NDArray, torch.Tensor],
            Union[npt.NDArray, torch.Tensor],
            SHAPE_TYPE,
        ],
        float,
    ]
) -> str:
    """Depending on what kind of object the similarity function (normal function, Pipeline object, partial object) is,
    we need to get the name of that function in different ways.

    Args:
        sim_func (Callable[[ Union[npt.NDArray, torch.Tensor], Union[npt.NDArray, torch.Tensor], SHAPE_TYPE, ], float]):
            measures similarity/distance between representations

    Returns:
        str: name of the similarity function
    """
    if hasattr(sim_func, "__name__"):  # repsim.measures.utils.Pipeline
        measure_name = sim_func.__name__
    elif hasattr(sim_func, "func"):  # functools.partial
        measure_name = sim_func.func.__name__
    else:
        measure_name = str(sim_func)
    return measure_name


def compare_representations(
    rep1: Sequence[Union[npt.NDArray, torch.Tensor]],
    rep2: Sequence[Union[npt.NDArray, torch.Tensor]],
    shape: SHAPE_TYPE,  # assuming rep1 and rep2 have the same shape
    measures: List[
        Callable[
            [
                Union[npt.NDArray, torch.Tensor],
                Union[npt.NDArray, torch.Tensor],
                SHAPE_TYPE,
            ],
            float,
        ]
    ],
    modelname1: str,
    modelname2: str,
    datasetname: str,
    splitname: str,
    results_path: Optional[Path] = None,
    **metadata,
) -> pd.DataFrame:
    """Compute pairwise similarities between two sequences of representations.

    Args:
        rep1 (Sequence[Union[npt.NDArray, torch.Tensor]]): Sequence of representations, each element of the sequence
            will be compared with all items of rep2
        rep2 (Sequence[Union[npt.NDArray, torch.Tensor]]): See rep1
        shape (SHAPE_TYPE): The shape each representation has, so that the similarity measures may reshape meaningfully
        measures (List[Callable]): each callable is a similarity/distance function between representations
        modelname1 (str): name of model 1. Metadata that will be added to results.
        modelname2 (str): name of model 2
        datasetname (str): name of dataset inputs are taken from
        splitname (str): subset of the dataset used as inputs
        results_path (Optional[Path], optional): If given, results will be stored as parquet file under this path.
            Defaults to None.
        **metadata: additional data that will be added as constant columns to the resulting dataframe

    Raises:
        NotImplementedError: if the number of layers of rep1 and rep2 do not match

    Returns:
        pd.DataFrame: contains all pairwise similarity/distance scores. Has columns
            "layer1,layer2,score,model1,model2,dataset,split,measure," and potentially further metadata columns
    """
    results = defaultdict(list)
    n_layers1 = len(rep1)
    n_layers2 = len(rep2)

    if n_layers1 != n_layers2:
        # TODO
        raise NotImplementedError(
            "Current implementation assumes representations of models with identical architecture as inputs to make a"
            "few small optimizations. We can turn those off when the number of layers is different."
        )

    for sim_func in measures:
        log.info("Assuming symmetric similarity measures, so skipping upper triangle of score matrix.")

        start = time.perf_counter()
        measure_name = name_of_similarity_function(sim_func)
        first_layer_to_compare1 = 0
        first_layer_to_compare2 = 0

        # We first compute the similarity scores for the lower triangle of the square score matrix, which should contain
        # pairwise similarity scores between layers
        scores = torch.zeros(n_layers1, n_layers2, dtype=torch.double)
        combinations = torch.tril_indices(n_layers1, n_layers2).transpose(1, 0)
        for rep1_layer_idx, rep2_layer_idx in tqdm(combinations, total=combinations.size(0)):
            rep1_layer_idx, rep2_layer_idx = int(rep1_layer_idx), int(rep2_layer_idx)
            log.debug("Comparing layers: %d, %d", rep1_layer_idx, rep2_layer_idx)
            score = sim_func(rep1[rep1_layer_idx], rep2[rep2_layer_idx], shape)
            scores[rep1_layer_idx, rep2_layer_idx] = score

        # Then we iterate over __all__ pairwise comparisons to populate our dataframe of results. Because we assume
        # symmetry of the similarity scores, we can use the score of, e.g., (A, B) for the request of (B, A).
        for rep1_layer_idx, rep2_layer_idx in itertools.product(
            range(first_layer_to_compare1, n_layers1),
            range(first_layer_to_compare2, n_layers2),
        ):
            if rep2_layer_idx > rep1_layer_idx:
                score = scores[rep2_layer_idx, rep1_layer_idx].item()
            else:
                score = scores[rep1_layer_idx, rep2_layer_idx].item()
            results["layer1"].append(rep1_layer_idx)
            results["layer2"].append(rep2_layer_idx)
            results["score"].append(score)

        # Finally add some more metadata to the dataframe
        n_times_to_add = len(results["score"])
        results["model1"].extend([modelname1] * n_times_to_add)
        results["model2"].extend([modelname2] * n_times_to_add)
        results["dataset"].extend([datasetname] * n_times_to_add)
        results["split"].extend([splitname] * n_times_to_add)
        results["measure"].extend([measure_name] * n_times_to_add)
        for key, value in metadata.items():
            results[key].extend([value] * n_times_to_add)

        log.info(f"{measure_name} completed in {time.perf_counter() - start:.1f} seconds")  # noqa: E501

    df = pd.DataFrame.from_dict(results)
    if results_path:
        df.to_parquet(results_path)
    return df
