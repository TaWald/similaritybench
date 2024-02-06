from typing import Union

import numpy as np
import numpy.typing as npt
import torch

from repsim.measures.utils import to_numpy_if_needed


def imd_score(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    approximation_steps: int = 8000,
    n_repeat: int = 5,
) -> float:
    try:
        import msid
    except ImportError as e:
        print(
            "Install IMD from"
            "https://github.com/xgfs/imd.git."
            "Clone and cd into directory, then `pip install .`"
        )
        raise e

    R, Rp = to_numpy_if_needed(R, Rp)

    # We use much higher defaults for the heat kernel approximation steps as the results
    # have very high variance otherwise. We also repeat the estimation to get an idea
    # about the variance of the score (TODO: report variance)
    scores = [
        msid.msid_score(R, Rp, niters=approximation_steps) for _ in range(n_repeat)
    ]
    return float(np.mean(scores))
