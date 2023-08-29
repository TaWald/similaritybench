from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np


@dataclass
class ModelToModelComparison:
    g_id_a: int | None
    g_id_b: int | None
    m_id_a: int | None
    m_id_b: int | None
    layerwise_cka: list[float]
    accuracy_orig: float
    accuracy_reg: float
    cohens_kappa: float
    jensen_shannon_div: float
    ensemble_acc: float

    cka_off_diagonal: list[list[float]] | None = None


@dataclass
class SeedResult:
    """Contains the HParams of a seed (a trained sequence) and the path to the models as dict of model_id -> path.
    Additionally it contains the path to the checkpoints as dict of model_id -> path for convenience.
    IMPORTANT: Per default models and checkpoints are not set and need to be filled post-init!
    """

    hparams: dict
    models: dict[int, Path] = field(init=False)
    checkpoints: dict[int, Path] = field(init=False)


@dataclass
class OutputEnsembleResults:
    n_models: int
    new_model_accuracy: float
    mean_single_accuracy: float
    ensemble_accuracy: float
    relative_ensemble_performance: float
    cohens_kappa: float
    jensen_shannon_div: float
    error_ratio: float
    regularization_metric: str = field(init=False)
    regularization_position: int = field(init=False)


@dataclass
class BatchCKAResult:
    lk: float
    ll: float
    kk: float
    negative: bool


@dataclass
class ActivationResult:
    values: np.ndarray
    layer: int
    samples: int
    error: np.ndarray | None = None
