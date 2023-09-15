from dataclasses import dataclass
from dataclasses import field
from logging import warn
from pathlib import Path

import numpy as np
from ke.util import name_conventions as nc
from ke.util.file_io import load_json


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

    def remove_non_converged_entries(self) -> "SeedResult":
        first_model_acc = load_json(self.models[0] / nc.OUTPUT_TMPLT)["val"]["accuracy"]
        one_failed = False

        for k, v in self.checkpoints.items():
            if one_failed:
                self.models.pop(k)
                self.checkpoints.pop(k)
                continue

            if not v.exists():
                self.models.pop(k)
                self.checkpoints.pop(k)
                one_failed = True
                warn(f"Model {k} did not converge!")
            else:
                new_model_acc = load_json(self.models[k] / nc.OUTPUT_TMPLT)["val"]["accuracy"]
                if new_model_acc < first_model_acc - 0.15:
                    self.models.pop(k)
                    self.checkpoints.pop(k)
                    one_failed = True
                    warn(f"Model {k} did not converge!")
        return self


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
