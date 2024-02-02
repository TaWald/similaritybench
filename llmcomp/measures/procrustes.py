from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.optimize
import torch

from llmcomp.measures.utils import (
    adjust_dimensionality,
    normalize_matrix_norm,
    to_numpy_if_needed,
)


def orthogonal_procrustes(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    nucnorm = scipy.linalg.orthogonal_procrustes(R, Rp)[1]
    return np.sqrt(
        -2 * nucnorm
        + np.linalg.norm(R, ord="fro") ** 2
        + np.linalg.norm(Rp, ord="fro") ** 2
    )


def permutation_procrustes(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    optimal_permutation_alignment: Optional[Tuple[npt.NDArray, npt.NDArray]] = None,
) -> float:
    # ) -> Dict[str, Any]:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)

    if not optimal_permutation_alignment:
        PR, PRp = scipy.optimize.linear_sum_assignment(
            R.T @ Rp, maximize=True
        )  # returns column assignments
        optimal_permutation_alignment = (PR, PRp)
    PR, PRp = optimal_permutation_alignment
    return float(np.linalg.norm(R[:, PR] - Rp[:, PRp], ord="fro"))
    # return {
    #     "score": float(np.linalg.norm(R[:, PR] - Rp[:, PRp], ord="fro")),
    #     "optimal_permutation_alignment": optimal_permutation_alignment,
    # }


def permutation_angular_shape_metric(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    optimal_permutation_alignment: Optional[Tuple[npt.NDArray, npt.NDArray]] = None,
) -> Dict[str, Any]:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    R, Rp = normalize_matrix_norm(R), normalize_matrix_norm(Rp)

    PR, PRp = scipy.optimize.linear_sum_assignment(
        R.T @ Rp, maximize=True
    )  # returns column assignments

    aligned_R = R[:, PR]
    aligned_Rp = Rp[:, PRp]

    # matrices are already normalized so no division necessary
    corr = np.trace(aligned_R.T @ aligned_Rp)

    # From https://github.com/ahwillia/netrep/blob/0f3d825aad58c6d998b44eb0d490c0c5c6251fc9/netrep/utils.py#L107  # noqa: E501
    # numerical precision issues require us to clip inputs to arccos
    return {
        "score": np.arccos(np.clip(corr, -1.0, 1.0)),
        "optimal_permutation_alignment": optimal_permutation_alignment,
    }


def orthogonal_angular_shape_metric(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    R, Rp = normalize_matrix_norm(R), normalize_matrix_norm(Rp)

    Qstar, nucnorm = scipy.linalg.orthogonal_procrustes(R, Rp)
    # matrices are already normalized so no division necessary
    corr = np.trace(Qstar.T @ R.T @ Rp)  # = \langle RQ, R' \rangle

    # From https://github.com/ahwillia/netrep/blob/0f3d825aad58c6d998b44eb0d490c0c5c6251fc9/netrep/utils.py#L107  # noqa: E501
    # numerical precision issues require us to clip inputs to arccos
    return float(np.arccos(np.clip(corr, -1.0, 1.0)))


def aligned_cossim(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    align, _ = scipy.linalg.orthogonal_procrustes(R, Rp)

    R_aligned = R @ align
    sum_cossim = 0
    for r, rp in zip(R_aligned, Rp):
        sum_cossim += r.dot(rp) / (np.linalg.norm(r) * np.linalg.norm(rp))
    return sum_cossim / R.shape[0]


def permutation_aligned_cossim(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)

    PR, PRp = scipy.optimize.linear_sum_assignment(
        R.T @ Rp, maximize=True
    )  # returns column assignments
    R_aligned = R[:, PR]
    Rp_aligned = Rp[:, PRp]

    sum_cossim = 0
    for r, rp in zip(R_aligned, Rp_aligned):
        sum_cossim += r.dot(rp) / (np.linalg.norm(r) * np.linalg.norm(rp))
    return sum_cossim / R.shape[0]
