from typing import Union

import numpy as np
import numpy.typing as npt
import torch

from repsim.measures.utils import to_numpy_if_needed


def eigenspace_overlap_score(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    u, _, _ = np.linalg.svd(R)
    v, _, _ = np.linalg.svd(Rp)
    u = u[:, : np.linalg.matrix_rank(R)]
    v = v[:, : np.linalg.matrix_rank(Rp)]
    return (
        1
        / np.max([R.shape[1], Rp.shape[1]])
        * (np.linalg.norm(u.T @ v, ord="fro") ** 2)
    )
