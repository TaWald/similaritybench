from typing import Union

import numpy.typing as npt
import scipy.optimize
import scipy.spatial.distance
import torch

from llmcomp.measures.utils import center_columns, to_numpy_if_needed


def correlation_match(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    mode: str = "hard",
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = center_columns(R), center_columns(Rp)

    # cdist computes the correlation _distance_ matrix (1 - corr)
    corr_matrix = 1 - scipy.spatial.distance.cdist(R.T, Rp.T, metric="correlation")
    if mode == "hard":
        # Let D = R.shape[1], Dp = Rp.shape[1] (the number of neurons).
        # Wlog, let D < Dp. Then all neurons of R get matched to a neuron of Rp, but
        # not all neurons of Rp are considered for the final score. The aligned matrices
        # will both have D neurons, the rest is discarded.
        # PR indexes the best-matched neurons of R, PRp are the corresponding matches in
        # Rp.
        PR, PRp = scipy.optimize.linear_sum_assignment(corr_matrix, maximize=True)
        return corr_matrix[PR, PRp].mean()
    elif mode == "soft":
        # Different to "hard" mode, all neurons of the first representation R are
        # considered, i.e., they will have a match with some neuron of Rp, but not
        # necessarily do all Rp neurons have a match. This means that this raw score is
        # asymmetric. We report the average between both directions to symmetrize it.
        score1 = corr_matrix.max(axis=1).mean()
        score2 = corr_matrix.max(axis=0).mean()
        return (score1 + score2) / 2
    else:
        raise ValueError(
            f"Unknown matching mode: {mode}. Must be one of 'hard' or 'soft'."
        )
