from dataclasses import dataclass
from dataclasses import field
from logging import warn
from pathlib import Path

import numpy as np
from simbench.util import name_conventions as nc
from simbench.util.file_io import load_json


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

    def remove_non_converged_entries(self):
        first_model_acc = load_json(self.models[0] / nc.OUTPUT_TMPLT)["val"]["accuracy"]
        keys_to_pop = []

        for k, v in self.checkpoints.items():
            if len(keys_to_pop) > 0:
                keys_to_pop.append(k)
                continue

            if not v.exists():
                keys_to_pop.append(k)
                warn(f"Model {k} did not converge!")
            else:
                new_model_acc = load_json(self.models[k] / nc.OUTPUT_TMPLT)["val"]["accuracy"]
                if new_model_acc < first_model_acc - 0.15:
                    keys_to_pop.append(k)
        if len(keys_to_pop) > 0:
            warn(f"Model {keys_to_pop[0]} did not converge, popping models {keys_to_pop}!")
            [self.models.pop(k) for k in keys_to_pop]
            [self.checkpoints.pop(k) for k in keys_to_pop]
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
