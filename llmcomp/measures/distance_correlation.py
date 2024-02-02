from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.spatial.distance
import torch

from llmcomp.measures.utils import double_center, to_numpy_if_needed


def distance_correlation(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)

    S = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(R, metric="euclidean")
    )
    Sp = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(Rp, metric="euclidean")
    )

    S = double_center(S)
    Sp = double_center(Sp)

    def dCov2(x: npt.NDArray, y: npt.NDArray) -> np.floating:
        return np.multiply(x, y).mean()

    return float(np.sqrt(dCov2(S, Sp) / np.sqrt(dCov2(S, S) * dCov2(Sp, Sp))))
