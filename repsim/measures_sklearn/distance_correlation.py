from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
import sklearn.metrics
import torch
from repsim.measures.utils import double_center
from repsim.measures.utils import flatten
from repsim.measures.utils import NUM_CPU_CORES
from repsim.measures.utils import SHAPE_TYPE
from repsim.measures.utils import SimilarityMeasure
from repsim.measures.utils import to_numpy_if_needed


class DistanceCorrelation(SimilarityMeasure):
    def __init__(
        self,
    ):
        super().__init__(
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=True,
        )

    @staticmethod
    def estimate_good_number_of_jobs(R, Rp):
        # TODO: this is something we have to compute for all RSM-based measures, so need to change inheritance to avoid duplication
        # RSMs in this measure are NxN, so the number of jobs should roughly scale quadratically with increase in N
        base_N = 1000
        jobs_at_base_N = 1
        actual_N = R.shape[0]
        return min(max(1, jobs_at_base_N * (actual_N / base_N) ** 2), NUM_CPU_CORES)

    @classmethod
    def __call__(
        cls,
        R: torch.Tensor | npt.NDArray,
        Rp: torch.Tensor | npt.NDArray,
        shape: SHAPE_TYPE,
        n_jobs: Optional[int] = None,
    ) -> float:

        if n_jobs is None:
            n_jobs = cls.estimate_good_number_of_jobs(R, Rp)
        return distance_correlation(R, Rp, shape, n_jobs=n_jobs)


def distance_correlation(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    n_jobs: Optional[int] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    S = sklearn.metrics.pairwise_distances(R, metric="euclidean", n_jobs=n_jobs)
    Sp = sklearn.metrics.pairwise_distances(Rp, metric="euclidean", n_jobs=n_jobs)

    S = double_center(S)
    Sp = double_center(Sp)

    def dCov2(x: npt.NDArray, y: npt.NDArray) -> np.floating:
        return np.multiply(x, y).mean()

    return float(np.sqrt(dCov2(S, Sp) / np.sqrt(dCov2(S, S) * dCov2(Sp, Sp))))
