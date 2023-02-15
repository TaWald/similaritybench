from __future__ import annotations

import pathlib
from dataclasses import dataclass
from functools import cache
from typing import Optional

import numpy as np
import torch as t
from numpy import genfromtxt
from rep_trans.metrics.cohens_kappa import binary_cohens_kappa
from rep_trans.losses.utils import celu_explained_variance
from rep_trans.losses.utils import centered_kernel_alignment
from rep_trans.losses.utils import correlation
from rep_trans.losses.utils import cosine_similarity
from rep_trans.losses.utils import euclidean_distance_csim
from rep_trans.metrics.aurc import aurc
from rep_trans.metrics.error_ratios import error_ratios
from rep_trans.util import data_structs as ds
from torch.nn import functional as F
from torchmetrics.functional import calibration_error

# Read once and keep it in memory:


@cache
def get_csv_lookup(
    dataset: ds.Dataset, architecture: ds.BaseArchitecture
) -> np.ndarray:
    if dataset == ds.Dataset.CIFAR10 and architecture == ds.BaseArchitecture.RESNET34:
        lookup_csv = pathlib.Path(__file__).parent / "cifar10_cohens_kappa_lookup.csv"
        csv_lookup = genfromtxt(
            lookup_csv, delimiter=",", skip_header=1, usecols=[1, 2, 3, 4]
        )
        return csv_lookup
    else:
        raise NotImplementedError


@cache
def look_up_baseline_cohens_kappa(
    accuracy: float, dataset: ds.Dataset, arch: ds.BaseArchitecture
) -> float:
    try:
        cohens_kappa_csv = get_csv_lookup(dataset, arch)
        row_id = np.argmin(np.abs(cohens_kappa_csv[:, 0] - accuracy))
        cohens_kappa = cohens_kappa_csv[row_id, 1]
        return cohens_kappa
    except NotImplementedError:
        return np.NAN


@dataclass
class SingleOutMetrics:
    accuracy: float
    ece: float
    aurc: float


@dataclass
class MultiOutMetrics(SingleOutMetrics):
    mean_old_acc: float
    ensemble_acc: float
    ensemble_aurc: float
    ece: float
    cohens_kappa: float
    error_ratio: float
    relative_cohens_kappa: float


@dataclass
class RepresentationMetrics:
    celu_r2: float | None = None
    corr: float | None = None
    cka: float | None = None
    rel_rep_cosine_similarity: float | None = None


def single_output_metrics(
    new_output: t.Tensor,
    groundtruth: t.Tensor,
) -> SingleOutMetrics:

    new_prob = F.softmax(new_output, dim=-1)
    new_y_hat_class_id = t.argmax(new_prob, dim=-1)
    acc = float(t.mean(new_y_hat_class_id == groundtruth, dtype=t.float).detach().cpu())
    ece = calibration_error(new_prob, groundtruth, n_bins=15, norm="l1").detach().cpu()
    ece = float(ece.detach().cpu())

    confidence = t.max(new_prob, dim=-1)  # Softmax max probability
    residual = new_y_hat_class_id == groundtruth

    aurc_val = aurc(residual, confidence)

    return SingleOutMetrics(accuracy=acc, ece=ece, aurc=aurc_val)


def multi_output_metrics(
    new_output: t.Tensor,
    old_outputs: t.Tensor,
    groundtruth: t.Tensor,
    dataset=ds.Dataset.CIFAR10,
    architecture=ds.BaseArchitecture.RESNET34,
) -> MultiOutMetrics:
    """
    Calculates a variety of metrics that are based on multiple output predictions being present.
    """
    with t.no_grad():
        # Calculation of probabilties and predicted classes.
        old_probs = F.softmax(old_outputs, dim=-1)
        old_y_hat_class_ids = t.argmax(old_probs, dim=-1)

        new_prob = F.softmax(new_output, dim=-1)
        new_y_hat_class_id = t.argmax(new_prob, dim=-1)

        ensemble_probs = t.mean(
            t.concat([old_probs, new_prob[None, ...]], dim=0), dim=0
        )
        ensemble_y_hat = t.argmax(ensemble_probs, dim=1)
        # ----------- Metric calculation -----------
        # ---- New model accuracy
        acc = t.mean(new_y_hat_class_id == groundtruth, dtype=t.float)
        acc = float(acc.detach().cpu())
        new_aurc = aurc(new_y_hat_class_id == groundtruth, t.max(new_prob, dim=-1))
        # ---- Old mean accuracy
        old_acc = t.mean(
            t.stack(
                [
                    t.mean(tmp_y == groundtruth, dtype=t.float)
                    for tmp_y in old_y_hat_class_ids
                ]
            )
        )
        mean_old_acc = float(old_acc.detach().cpu())

        # ---- Error ratio
        all_error_ratios: list[float] = [
            error_ratios(new_y_hat_class_id, old_pred_cls, groundtruth)
            for old_pred_cls in old_y_hat_class_ids
        ]
        mean_error_ratio = float(np.mean(all_error_ratios))

        # ---- Ensemble Accuracy
        ensemble_acc = t.mean(ensemble_y_hat == groundtruth, dtype=t.float)
        ensemble_acc = float(ensemble_acc.detach().cpu())
        ensemble_aurc = aurc(
            ensemble_y_hat == groundtruth, t.max(ensemble_probs, dim=-1)
        )

        # ---- Calibration Error
        ece = (
            calibration_error(new_prob, groundtruth, n_bins=15, norm="l1")
            .detach()
            .cpu()
        )
        ece = float(ece.detach().cpu())
        # ensemble_entropy = t.sum((-ensemble_probs * t.log(ensemble_probs)), dim=1)
        # ap = average_precision(preds=ensemble_probs, target=gt)
        # true_entropy = t.mean(ensemble_entropy[true_false])
        # false_entropy = t.mean(ensemble_entropy[~true_false])
        # outputs["ensemble_uncertainty"] = (false_entropy - true_entropy).detach()
        # ---- Cohens Kappa
        cohens_kappas = [
            binary_cohens_kappa(new_y_hat_class_id, y, groundtruth)
            for y in old_y_hat_class_ids
        ]
        cohens_kappa = float(t.mean(t.stack(cohens_kappas)).detach().cpu())

        # ---- Relative Cohens Kappa
        baseline_cc = look_up_baseline_cohens_kappa(acc, dataset, architecture)
        relative_cohens_kappa = cohens_kappa - baseline_cc

    return MultiOutMetrics(
        accuracy=acc,
        aurc=new_aurc,
        mean_old_acc=mean_old_acc,
        ensemble_acc=ensemble_acc,
        ensemble_aurc=ensemble_aurc,
        ece=ece,
        error_ratio=mean_error_ratio,
        cohens_kappa=cohens_kappa,
        relative_cohens_kappa=relative_cohens_kappa,
    )


def representation_metrics(
    new_intermediate_reps: Optional[list[t.Tensor]] = None,
    old_intermediate_reps: Optional[list[t.Tensor]] = None,
    sample_size: int = 128,
    calc_r2: bool = True,
    calc_corr: bool = True,
    calc_cka: bool = False,
    calc_cos: bool = False,
) -> RepresentationMetrics:
    """Calculates metrics that are based on intermediate representations."""

    with t.no_grad():
        metrics = RepresentationMetrics()
        # ----------- Metric calculation -----------
        if calc_r2:
            celu_r2 = celu_explained_variance(
                new_intermediate_reps, old_intermediate_reps
            )
            metrics.celu_r2 = float(
                t.mean(t.stack([t.mean(cev) for cev in celu_r2])).detach().cpu()
            )
        if calc_corr:
            corr, _ = correlation(new_intermediate_reps, old_intermediate_reps)
            metrics.corr = float(
                t.mean(t.stack([t.mean(t.abs(c)) for c in corr])).detach().cpu()
            )
        if calc_cka:
            cka = centered_kernel_alignment(
                new_intermediate_reps, old_intermediate_reps
            )
            metrics.cka = float(
                t.mean(t.stack([t.stack(c) for c in cka])).detach().cpu()
            )
        # Basically always makes sense to calculate it.
        if calc_cos:
            if new_intermediate_reps[0].shape[1] > sample_size:
                new_intermediate_reps = [
                    tr[:, :sample_size] for tr in new_intermediate_reps
                ]
                old_intermediate_reps = [
                    a[:, :sample_size] for a in old_intermediate_reps
                ]
            with t.no_grad():
                cos_sim_tr = cosine_similarity(new_intermediate_reps)
                cos_sim_apx = cosine_similarity(old_intermediate_reps)
                euclidean_dist = euclidean_distance_csim(zip(cos_sim_tr, cos_sim_apx))
            metrics.rel_rep_cosine_similarity = float(euclidean_dist.detach().cpu())
    return metrics


def main():
    pseudo_accuracies = [0.8, 0.6, 0.412, 0.632]
    dataset = ds.Dataset.CIFAR10
    architecture = ds.BaseArchitecture.RESNET34
    for psa in pseudo_accuracies:
        cohens_kappa = look_up_baseline_cohens_kappa(psa, dataset, architecture)
        print(cohens_kappa)


if __name__ == "__main__":
    main()
