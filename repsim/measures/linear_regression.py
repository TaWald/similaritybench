from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.linalg
import torch

from repsim.measures.utils import (
    SHAPE_TYPE,
    center_columns,
    flatten,
    to_numpy_if_needed,
)


def linear_reg(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R = center_columns(R)
    Rp = center_columns(Rp)
    Rp_orthonormal_base = Rp @ scipy.linalg.inv(  # type:ignore
        scipy.linalg.sqrtm(Rp.T @ Rp)  # type:ignore
    )
    return float(
        (np.linalg.norm(Rp_orthonormal_base.T @ R, ord="fro") ** 2)
        / (np.linalg.norm(R, ord="fro") ** 2)
    )