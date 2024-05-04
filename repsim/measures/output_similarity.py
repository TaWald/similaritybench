from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.spatial.distance
import scipy.special
import torch
from repsim.measures.utils import FunctionalSimilarityMeasure
from repsim.measures.utils import to_numpy_if_needed


def check_has_two_axes(x: npt.NDArray | torch.Tensor):
    if len(x.shape) != 2:
        raise ValueError(f"Matrix must have two dimensions, but has {len(x)}")


class JSD(FunctionalSimilarityMeasure):
    def __init__(self):
        super().__init__(larger_is_more_similar=False, is_symmetric=True)

    def __call__(self, output_a: torch.Tensor | npt.NDArray, output_b: torch.Tensor | npt.NDArray) -> Any:
        check_has_two_axes(output_a)
        check_has_two_axes(output_b)

        output_a = scipy.special.softmax(output_a, axis=1)
        output_b = scipy.special.softmax(output_b, axis=1)
        return np.nanmean(
            [
                scipy.spatial.distance.jensenshannon(output_a_i, output_b_i)
                for output_a_i, output_b_i in zip(output_a, output_b)
            ]
        )


class Disagreement(FunctionalSimilarityMeasure):
    def __init__(self):
        super().__init__(larger_is_more_similar=False, is_symmetric=True)

    def __call__(self, output_a: torch.Tensor | npt.NDArray, output_b: torch.Tensor | npt.NDArray) -> Any:
        check_has_two_axes(output_a)
        check_has_two_axes(output_b)

        output_a, output_b = to_numpy_if_needed(output_a, output_b)
        return (output_a.argmax(axis=1) != output_b.argmax(axis=1)).sum() / len(output_a)
