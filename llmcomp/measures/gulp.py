from typing import Union

import numpy as np
import numpy.typing as npt
import torch

from llmcomp.measures.utils import to_numpy_if_needed


# Copied from https://github.com/sgstepaniants/GULP/blob/main/distance_functions.py
def predictor_dist(
    A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, lmbda=0
):
    """
    Computes distance bewteen best linear predictors on representations A and B
    """
    k, n = A.shape
    l, _ = B.shape
    assert k <= n
    assert l <= n

    if evals_a is None or evecs_a is None:
        evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    if evals_b is None or evecs_b is None:
        evals_b, evecs_b = np.linalg.eigh(B @ B.T)

    evals_a = (evals_a + np.abs(evals_a)) / (2 * n)
    if lmbda > 0:
        inv_a_lmbda = np.array(
            [1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_a]
        )
    else:
        inv_a_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_a])

    evals_b = (evals_b + np.abs(evals_b)) / (2 * n)
    if lmbda > 0:
        inv_b_lmbda = np.array(
            [1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_b]
        )
    else:
        inv_b_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_b])

    T1 = np.sum(np.square(evals_a * inv_a_lmbda))
    T2 = np.sum(np.square(evals_b * inv_b_lmbda))

    cov_ab = A @ B.T / n
    T3 = np.trace(
        (np.diag(np.sqrt(inv_a_lmbda)) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ np.diag(inv_b_lmbda) @ evecs_b.T)
        @ cov_ab.T
        @ (evecs_a @ np.diag(np.sqrt(inv_a_lmbda)))
    )

    return T1 + T2 - 2 * T3


# End of copy


def gulp(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    lmbda: float = 0,
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)

    # The GULP paper assumes DxN matrices; we have NxD matrices.
    return predictor_dist(R.T, Rp.T, lmbda=lmbda)  # type:ignore
