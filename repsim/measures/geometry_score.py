from typing import Union

import numpy as np
import numpy.typing as npt
import torch

from repsim.measures.utils import SHAPE_TYPE, flatten, to_numpy_if_needed


def geometry_score(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    **kwargs
) -> float:
    try:
        import gs
    except ImportError as e:
        print(
            "Install the geometry score from"
            "https://github.com/KhrulkovV/geometry-score."
            "Clone and cd into directory, then `pip install .`"
        )
        raise e

    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    rlt_R = gs.rlts(R, **kwargs)
    mrlt_R = np.mean(rlt_R, axis=0)

    rlt_Rp = gs.rlts(Rp, **kwargs)
    mrlt_Rp = np.mean(rlt_Rp, axis=0)

    return float(np.sum((mrlt_R - mrlt_Rp) ** 2))
