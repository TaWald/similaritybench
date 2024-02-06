from typing import List, Set, Union

import numpy as np
import numpy.typing as npt
import scipy.spatial.distance
import sklearn.neighbors
import torch

from repsim.measures.utils import SHAPE_TYPE, flatten, to_numpy_if_needed


def _jac_sim_i(idx_R: Set[int], idx_Rp: Set[int]) -> float:
    return len(idx_R.intersection(idx_Rp)) / len(idx_R.union(idx_Rp))


def jaccard_similarity(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    k: int = 10,
    inner: str = "cosine",
    n_jobs: int = 8,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    indices_R = nn_array_to_setlist(top_k_neighbors(R, k, inner, n_jobs))
    indices_Rp = nn_array_to_setlist(top_k_neighbors(Rp, k, inner, n_jobs))

    return float(
        np.mean(
            [_jac_sim_i(idx_R, idx_Rp) for idx_R, idx_Rp in zip(indices_R, indices_Rp)]
        )
    )


def top_k_neighbors(
    R: npt.NDArray,
    k: int,
    inner: str,
    n_jobs: int,
) -> npt.NDArray:
    # k+1 nearest neighbors, because we pass in all the data, which means that a point
    # will be the nearest neighbor to itself. We remove this point from the results and
    # report only the k nearest neighbors distinct from the point itself.
    nns = sklearn.neighbors.NearestNeighbors(
        n_neighbors=k + 1, metric=inner, n_jobs=n_jobs
    )
    nns.fit(R)
    _, nns = nns.kneighbors(R)
    return nns[:, 1:]


def nn_array_to_setlist(nn: npt.NDArray) -> List[Set[int]]:
    return [set(idx) for idx in nn]


def second_order_cosine_similarity(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    k: int = 10,
    n_jobs: int = 8,
) -> float:
    inner = "cosine"
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    nns_R = top_k_neighbors(R, k, inner, n_jobs)
    nns_Rp = top_k_neighbors(Rp, k, inner, n_jobs)

    union_nns = [
        list(set(nns_Ri).union(set(nns_Rpi))) for nns_Ri, nns_Rpi in zip(nns_R, nns_Rp)
    ]

    dists_R = scipy.spatial.distance.pdist(R, inner)
    dists_Rp = scipy.spatial.distance.pdist(Rp, inner)

    return float(
        np.mean(
            [
                1
                - scipy.spatial.distance.cosine(
                    dists_R[union_nns_i], dists_Rp[union_nns_i]
                )
                for union_nns_i in union_nns
            ]
        )
    )


def _rank_sim_i(
    joint_nns_i: List[int], nns_Ri: npt.NDArray, nns_Rpi: npt.NDArray
) -> float:
    if not joint_nns_i:
        return 0

    normalization = sum((1 / k for k in range(1, len(joint_nns_i) + 1)))
    score = 0
    for j in joint_nns_i:
        rankj_R = np.argwhere(nns_Ri == j)[0][0] + 1
        rankj_Rp = np.argwhere(nns_Rpi == j)[0][0] + 1
        ranksim_score = 1 + abs(rankj_R - rankj_Rp)
        score_weight = rankj_R + rankj_Rp
        score += 2 / (ranksim_score * score_weight)
    return score / normalization


def rank_similarity(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    k: int = 10,
    inner: str = "cosine",
    n_jobs: int = 8,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    nns_R = top_k_neighbors(R, k, inner, n_jobs)
    nns_Rp = top_k_neighbors(Rp, k, inner, n_jobs)

    nns_of_both = [
        list(set(nns_Ri).intersection(set(nns_Rpi)))
        for nns_Ri, nns_Rpi in zip(nns_R, nns_Rp)
    ]

    scores = np.zeros(R.shape[0])
    for i, nns_i in enumerate(nns_of_both):
        scores[i] = _rank_sim_i(nns_i, nns_R[i], nns_Rp[i])
    return float(np.mean(scores))


def joint_rank_jaccard_similarity(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    k: int = 10,
    inner: str = "cosine",
    n_jobs: int = 8,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    nns_R = top_k_neighbors(R, k, inner, n_jobs)
    nns_Rp = top_k_neighbors(Rp, k, inner, n_jobs)

    indices_R = nn_array_to_setlist(nns_R)
    indices_Rp = nn_array_to_setlist(nns_Rp)

    nns_of_both = [
        list(set(nns_Ri).intersection(set(nns_Rpi)))
        for nns_Ri, nns_Rpi in zip(nns_R, nns_Rp)
    ]

    return float(
        np.mean(
            [
                _jac_sim_i(idx_R, idx_Rp) * _rank_sim_i(nns, nns_R[i], nns_Rp[i])
                for i, (idx_R, idx_Rp, nns) in enumerate(
                    zip(indices_R, indices_Rp, nns_of_both)
                )
            ]
        )
    )
