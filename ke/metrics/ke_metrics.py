from __future__ import annotations

import math
import pathlib
from dataclasses import asdict
from dataclasses import dataclass
from functools import cache
from typing import Optional

import numpy as np
import torch as t
from ke.losses.utils import celu_explained_variance
from ke.losses.utils import centered_kernel_alignment
from ke.losses.utils import correlation
from ke.losses.utils import cosine_similarity
from ke.losses.utils import euclidean_distance_csim
from ke.metrics.aurc import aurc
from ke.metrics.cohens_kappa import calculate_cohens_kappas
from ke.metrics.error_ratios import calculate_error_ratios
from ke.metrics.jensen_shannon_distance import jensen_shannon_divergences
from ke.metrics.relative_ensemble_performance import relative_ensemble_performance
from ke.util import data_structs as ds
from numpy import genfromtxt
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torchmetrics.functional import calibration_error


@cache
def get_csv_lookup(dataset: ds.Dataset | str, architecture: ds.BaseArchitecture) -> np.ndarray:
    if isinstance(dataset, str):
        dataset = ds.Dataset(dataset)
    if isinstance(architecture, str):
        architecture = ds.BaseArchitecture(architecture)
    if dataset == ds.Dataset.CIFAR10 and architecture == ds.BaseArchitecture.RESNET34:
        lookup_csv = (
            pathlib.Path(__file__).parent.parent
            / "training"
            / f"{dataset.value}_{architecture.value}_cohens_kappa_lookup.csv"
        )
        csv_lookup = genfromtxt(lookup_csv, delimiter=",", skip_header=1, usecols=[1, 2, 3, 4])
        return csv_lookup
    else:
        raise ValueError


@cache
def look_up_baseline_cohens_kappa(accuracy: float, dataset: ds.Dataset, arch: ds.BaseArchitecture) -> float:
    try:
        cohens_kappa_csv = get_csv_lookup(dataset, arch)
        row_id = np.argmin(np.abs(cohens_kappa_csv[:, 0] - accuracy))
        cohens_kappa = cohens_kappa_csv[row_id, 1]
        return cohens_kappa
    except ValueError:
        return np.NAN


@dataclass
class SingleOutMetrics:
    accuracy: float
    ce: float
    ece: float
    max_softmax_aurc: float
    mutual_info_aurc: float


@dataclass
class MultiOutMetrics(SingleOutMetrics):
    mean_old_acc: float

    # Ensembled Metrics
    ensemble_accuracy: float
    ensemble_max_softmax_aurc: float
    ensemble_mutual_info_aurc: float
    rel_ensemble_performance: float  # Ensemble Acc / Mean Single Accs
    ensemble_ece: float  # Ensemble estimate calibration error

    # Cohens Kappa
    cohens_kappa: ds.GroupMetrics
    relative_cohens_kappa: float
    error_ratio: ds.GroupMetrics
    jensen_shannon_div: ds.GroupMetrics
    n_models: int


@dataclass
class RepresentationMetrics:
    celu_r2: float | None = None
    corr: float | None = None
    cka: float | None = None
    rel_rep_cosine_similarity: float | None = None


def pred_entropy(logits) -> t.Tensor:
    """Get the mean entropy of multiple logits."""
    k = logits.shape[1]
    out = F.log_softmax(logits, dim=2)  # BxkxD
    out = t.logsumexp(out, dim=1) - math.log(k)  # BxkxD --> BxD
    ent = t.sum(-t.exp(out) * out, dim=1)  # B
    return ent


def exp_entropy(logits) -> t.Tensor:
    out = F.log_softmax(logits, dim=2)  # BxkxD
    out = t.sum(-t.exp(out) * out, dim=2)  # Bxk
    out = t.mean(out, dim=1)
    return out


def mutual_bald(logits) -> t.Tensor:
    """
    Calculation how uncertain the model is for respective samples.
    Also wie viel information dein Modell glaubt zu gainen, wenn dieses bekannt wäre.
    https://arxiv.org/pdf/1803.08533.pdf

    Here logits are expected to have [N_Models x NSamples x NClasses]
    """
    tp_logits = t.transpose(logits, dim0=1, dim1=0)

    return pred_entropy(tp_logits) - exp_entropy(tp_logits)


def single_output_metrics(
    new_output: t.Tensor,
    groundtruth: t.Tensor,
) -> SingleOutMetrics:
    """
    Calculates a variety of metrics for a single model.

    :param new_output: Logits of the new model [N_Samples x N_CLS]
    :param groundtruth: Class int values (not one-hot)  [NSamples]
    """

    num_classes = new_output.shape[-1]
    new_prob = F.softmax(new_output, dim=-1)
    new_y_hat_class_id = t.argmax(new_prob, dim=-1)
    acc: float = float(t.mean(new_y_hat_class_id == groundtruth, dtype=t.float).detach().cpu())
    ece = calibration_error(new_prob, groundtruth, task="multiclass", n_bins=15, num_classes=num_classes)
    ece = float(ece.detach().cpu())

    ce = float(cross_entropy(new_output, groundtruth).detach().cpu())
    max_softmax_confidence = (t.max(new_prob, dim=-1).values).to(t.float64)  # Softmax max probability
    residual = (new_y_hat_class_id == groundtruth).to(t.float64)
    mi_confidence = mutual_bald(new_output[None, ...])

    aurc_sm = aurc(residual, max_softmax_confidence)
    aurc_mi = aurc(residual, mi_confidence)

    return SingleOutMetrics(accuracy=acc, ce=ce, ece=ece, max_softmax_aurc=aurc_sm, mutual_info_aurc=aurc_mi)


def multi_output_metrics(
    new_output: t.Tensor,
    old_outputs: t.Tensor,
    groundtruth: t.Tensor,
    dataset: ds.Dataset | str,
    architecture: ds.Dataset | str,
) -> MultiOutMetrics:
    """
    Calculates a variety of metrics that are based on multiple output predictions being present.
    """
    num_classes = new_output.shape[-1]

    with t.no_grad():
        # Calculation of probabilties and predicted classes.
        all_logits = t.concat([old_outputs, new_output[None, ...]], dim=0)

        # Only existing model stuff
        old_probs = F.softmax(old_outputs, dim=-1)
        old_y_hat_class_ids = t.argmax(old_probs, dim=-1)
        # New model stuff:
        new_prob = F.softmax(new_output, dim=-1)
        new_y_hat_class_id = t.argmax(new_prob, dim=-1)

        joint_probablities = t.concat([old_probs, new_prob[None, ...]], dim=0)
        n_models = joint_probablities.shape[0]
        joint_yhats = t.concat([old_y_hat_class_ids, new_y_hat_class_id[None, ...]], dim=0)

        ensemble_probs = t.mean(joint_probablities, dim=0)
        ensemble_y_hat = t.argmax(ensemble_probs, dim=1)

        # ---- New model accuracy
        single_metrics = single_output_metrics(new_output, groundtruth)

        # ----------- Ensemble Uncertainty: -----------
        max_softmax_confidence = t.max(ensemble_probs.to(t.float64), dim=-1).values
        mi_confidence = mutual_bald(all_logits.to(t.float64))

        residual = (ensemble_y_hat == groundtruth).to(t.float64)
        ensemble_ms_aurc = aurc(residual, max_softmax_confidence)
        ensemble_mi_aurc = aurc(residual, mi_confidence)

        # ----------- Ensemble Calibration: -----------
        ensemble_ece = calibration_error(
            ensemble_probs, groundtruth, task="multiclass", n_bins=15, num_classes=num_classes
        )
        ensemble_ece = float(ensemble_ece.detach().cpu())

        # ---- Ensemble Accuracy
        ensemble_acc = t.mean(ensemble_y_hat == groundtruth, dtype=t.float)
        ensemble_acc = float(ensemble_acc.detach().cpu())

        # ---- Relative Ensemble Performance
        rel_ens_performance = float(relative_ensemble_performance(joint_yhats, ensemble_y_hat, groundtruth))

        # ---- Old mean accuracy
        old_acc = t.mean(t.stack([t.mean(tmp_y == groundtruth, dtype=t.float) for tmp_y in old_y_hat_class_ids]))
        mean_old_acc = float(old_acc.detach().cpu())

        # ---- Error ratio

        # ---- Cohens Kappa
        unbound_probs: list[t.Tensor] = t.unbind(joint_probablities, dim=0)
        unbound_yhats: list[t.Tensor] = t.unbind(joint_yhats, dim=0)

        cks = calculate_cohens_kappas(unbound_yhats, groundtruth)
        jsds = jensen_shannon_divergences(unbound_probs)
        err = calculate_error_ratios(unbound_yhats, groundtruth)

        # ---- Relative Cohens Kappa
        baseline_cc = look_up_baseline_cohens_kappa(single_metrics.accuracy, dataset, architecture)
        relative_cohens_kappa = float(cks.last_to_first - baseline_cc)

    return MultiOutMetrics(
        **asdict(single_metrics),
        mean_old_acc=mean_old_acc,
        ensemble_accuracy=ensemble_acc,
        ensemble_ece=ensemble_ece,
        ensemble_max_softmax_aurc=ensemble_ms_aurc,
        ensemble_mutual_info_aurc=ensemble_mi_aurc,
        rel_ensemble_performance=rel_ens_performance,
        cohens_kappa=cks,
        jensen_shannon_div=jsds,
        error_ratio=err,
        relative_cohens_kappa=relative_cohens_kappa,
        n_models=n_models,
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
            celu_r2 = celu_explained_variance(new_intermediate_reps, old_intermediate_reps)
            metrics.celu_r2 = float(t.mean(t.stack([t.mean(cev) for cev in celu_r2])).detach().cpu())
        if calc_corr:
            corr, _ = correlation(new_intermediate_reps, old_intermediate_reps)
            metrics.corr = float(t.mean(t.stack([t.mean(t.abs(c)) for c in corr])).detach().cpu())
        if calc_cka:
            cka = centered_kernel_alignment(new_intermediate_reps, old_intermediate_reps)
            metrics.cka = float(t.mean(t.stack([t.stack(c) for c in cka])).detach().cpu())
        # Basically always makes sense to calculate it.
        if calc_cos:
            if new_intermediate_reps[0].shape[1] > sample_size:
                new_intermediate_reps = [tr[:, :sample_size] for tr in new_intermediate_reps]
                old_intermediate_reps = [a[:, :sample_size] for a in old_intermediate_reps]
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
