from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.spatial.distance
import torch

from repsim.measures.utils import to_numpy_if_needed


def magnitude_difference(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)

    return abs(float(np.linalg.norm(R.mean(axis=0)) - np.linalg.norm(Rp.mean(axis=0))))


def magnitude_nrmse(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)

    di_bar = np.hstack(
        [
            np.linalg.norm(R, axis=1, ord=2, keepdims=True),
            np.linalg.norm(Rp, axis=1, ord=2, keepdims=True),
        ]
    ).mean(axis=1)
    rmse = np.sqrt(
        1
        / 2
        * (
            (np.linalg.norm(R, axis=1, ord=2) - di_bar) ** 2
            + (np.linalg.norm(Rp, axis=1, ord=2) - di_bar) ** 2
        )
    )
    normalization = np.abs(
        np.linalg.norm(R, axis=1, ord=2) - np.linalg.norm(Rp, axis=1, ord=2)
    )
    # this might create nans as normalization can theoretically be zero, but we fix this
    # by setting the nan values to zero (If there is no difference in the norm of the
    # instance in both representations, then the RMSE term will also be zero. We then
    # say that 0/0 = 0 variance.).
    per_instance_nrmse = rmse / normalization
    per_instance_nrmse[np.isnan(per_instance_nrmse)] = 0
    return float(per_instance_nrmse.mean())


def uniformity_difference(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    def uniformity(x, t=2):
        pdist = scipy.spatial.distance.pdist(x, metric="sqeuclidean")
        pdist = scipy.spatial.distance.squareform(pdist)
        return np.log(np.exp(-t * pdist).sum() / x.shape[0] ** 2)

    return float(abs(uniformity(R) - uniformity(Rp)))


def concentricity(x):
    return 1 - scipy.spatial.distance.cdist(
        x, x.mean(axis=0, keepdims=True), metric="cosine"
    )


def concentricity_difference(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    return float(abs(concentricity(R).mean() - concentricity(Rp).mean()))


def concentricity_nrmse(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    alphai_bar = np.hstack(
        [
            concentricity(R),
            concentricity(Rp),
        ]
    ).mean(axis=1, keepdims=True)
    rmse = np.sqrt(
        ((concentricity(R) - alphai_bar) ** 2 + (concentricity(Rp) - alphai_bar) ** 2)
        / 2
    )
    normalization = np.abs(concentricity(R) - concentricity(Rp))

    # this might create nans as normalization can theoretically be zero, but we fix this
    # by setting the nan values to zero (If there is no difference in the norm of the
    # instance in both representations, then the RMSE term will also be zero. We then
    # say that 0/0 = 0 variance.).
    per_instance_nrmse = rmse / normalization
    per_instance_nrmse[np.isnan(per_instance_nrmse)] = 0
    return float(per_instance_nrmse.mean())
