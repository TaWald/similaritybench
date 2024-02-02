from typing import Union

import numpy as np
import numpy.typing as npt
import torch

from llmcomp.measures.utils import to_numpy_if_needed


def geometry_score(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray], **kwargs
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

    R, Rp = to_numpy_if_needed(R, Rp)

    rlt_R = gs.rlts(R, **kwargs)
    mrlt_R = np.mean(rlt_R, axis=0)

    rlt_Rp = gs.rlts(Rp, **kwargs)
    mrlt_Rp = np.mean(rlt_Rp, axis=0)

    return float(np.sum((mrlt_R - mrlt_Rp) ** 2))
