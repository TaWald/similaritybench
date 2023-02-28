from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActivationResult:
    values: np.ndarray
    layer: int
    samples: int
    error: np.ndarray | None = None
