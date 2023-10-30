from pathlib import Path

import numpy as np
import torch
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.manual_introspection.comparison_helper import BatchCKAResult
from ke.manual_introspection.comparison_helper import ModelToModelComparison
from ke.manual_introspection.comparison_helper import OutputEnsembleResults
from ke.metrics.cohens_kappa import calculate_cohens_kappas
from ke.metrics.error_ratios import calculate_error_ratios
from ke.metrics.jensen_shannon_distance import jensen_shannon_divergences
from ke.metrics.ke_metrics import multi_output_metrics
from ke.util import data_structs as ds
from ke.util import find_architectures
from ke.util import find_datamodules
from ke.util.default_params import get_default_arch_params
from ke.util.file_io import strip_state_dict_of_keys
from torch.functional import F


def _consolidate_cka_batch_results(batch_results: list[BatchCKAResult]):
    non_neg_results = [br for br in batch_results if not br.negative]
    lk = np.mean([br.lk for br in non_neg_results])
    kk = np.mean([br.kk for br in non_neg_results])
    ll = np.mean([br.ll for br in non_neg_results])

    cka = lk / (np.sqrt(kk * ll))

    return cka


def unbiased_hsic(K: torch.Tensor, L: torch.Tensor):
    """Calculates the unbiased HSIC estimate between two variables X and Y.
    Shape of the input should be (N, N) (already calculated)

    implementation of HSIC_1 from https://arxiv.org/pdf/2010.15327.pdf (Eq 3)
    """

    N = K.shape[0]

    # Center the activations
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)

    ones = torch.ones((N, 1), device=K.device, dtype=K.dtype)

    first = torch.trace(K @ L)
    second = (ones.T @ K @ ones @ ones.T @ L @ ones) / ((N - 1) * (N - 2))
    third = (ones.T @ K @ L @ ones) * (2 / (N - 2))
    factor = 1 / (N * (N - 3))

    hsic = factor * (first + second - third) + 1e-12
    return torch.squeeze(hsic)


def _batch_cka(K: torch.Tensor, L: torch.Tensor) -> BatchCKAResult:
    """Compares the activations of both networks and outputs.
    Expects the activations to be in format: B x p with p being the number of neurons."""

    K = K.cuda()
    L = L.cuda()

    kl = float(unbiased_hsic(K, L).cpu().numpy())
    kk = float(unbiased_hsic(K, K).cpu().numpy())
    ll = float(unbiased_hsic(L, L).cpu().numpy())

    if kl < 0 or kk < 0 or ll < 0:
        return BatchCKAResult(kl, kk, ll, True)
    else:
        return BatchCKAResult(kl, kk, ll, False)


def final_multi_output_metrics(
    outputs: list[np.ndarray],
    groundtruth: np.ndarray,
) -> OutputEnsembleResults:
    """
    Calculates a variety of metrics that are based on multiple output predictions being present.
    """
    # num_classes = new_output.shape[-1]
    groundtruth = torch.from_numpy(groundtruth)

    last_models_outputs = torch.from_numpy(outputs[-1])
    last_models_probs = F.softmax(last_models_outputs, dim=-1)
    last_models_y_hat = torch.argmax(last_models_probs, dim=-1)

    # ---- New model accuracy
    last_models_accuracy = float(torch.mean(last_models_y_hat == groundtruth, dtype=torch.float).cpu())

    all_logits = torch.from_numpy(np.stack(outputs, axis=0))  # N_Models x N_Batches x N_Classes
    all_probs = F.softmax(all_logits, dim=-1)  # N_Models x N_Batches x N_Classes
    all_y_hats = torch.argmax(all_probs, dim=-1)  # N_Models x N_Batches
    # Only existing model stuff

    ensemble_probs = torch.mean(all_probs, dim=0)
    ensemble_y_hat = torch.argmax(ensemble_probs, dim=1)

    # ---- New model accuracy
    all_models_accuracies = torch.mean(all_y_hats == (groundtruth[None, ...]), dtype=torch.float, dim=1)
    mean_single_model_accuracy = float(torch.mean(all_models_accuracies).detach().cpu().numpy())

    # ---- Ensemble Accuracy
    ensemble_acc = torch.mean(ensemble_y_hat == groundtruth, dtype=torch.float)
    ensemble_acc = float(ensemble_acc.detach().cpu())

    # ---- Relative Ensemble Performance
    rel_ens_performance = float(ensemble_acc / mean_single_model_accuracy)

    # ---- Cohens Kappa
    unbound_probs: list[torch.Tensor] = torch.unbind(all_probs, dim=0)
    unbound_yhats: list[torch.Tensor] = torch.unbind(all_y_hats, dim=0)

    cks = calculate_cohens_kappas(unbound_yhats, groundtruth).all_to_all_mean

    jsds = jensen_shannon_divergences(unbound_probs).all_to_all_mean
    err = calculate_error_ratios(unbound_yhats, groundtruth).all_to_all_mean

    return OutputEnsembleResults(
        n_models=len(outputs),
        new_model_accuracy=last_models_accuracy,
        ensemble_accuracy=ensemble_acc,
        mean_single_accuracy=mean_single_model_accuracy,
        relative_ensemble_performance=rel_ens_performance,
        cohens_kappa=cks,
        jensen_shannon_div=jsds,
        error_ratio=err,
    )


def compare_models_parallel(model_a: Path, model_b: Path, hparams: dict) -> ModelToModelComparison:
    arch_a: AbsActiExtrArch = find_architectures.get_base_arch(ds.BaseArchitecture(hparams["architecture"]))(
        n_cls=10 if hparams["dataset"] == "CIFAR10" else 100
    )
    arch_b: AbsActiExtrArch = find_architectures.get_base_arch(ds.BaseArchitecture(hparams["architecture"]))(
        n_cls=10 if hparams["dataset"] == "CIFAR10" else 100
    )

    ckpt_a: dict = torch.load(str(model_a))
    ckpt_b: dict = torch.load(str(model_b))
    try:
        arch_a.load_state_dict(ckpt_a)
    except RuntimeError as _:  # noqa
        stripped_a = strip_state_dict_of_keys(ckpt_a)
        try:
            arch_a.load_state_dict(stripped_a)
        except RuntimeError as e:
            raise e

    try:
        arch_b.load_state_dict(ckpt_b)
    except RuntimeError as _:  # noqa
        stripped_b = strip_state_dict_of_keys(ckpt_b)
        try:
            arch_b.load_state_dict(stripped_b)
        except RuntimeError as e:
            raise e

    datamodule = find_datamodules.get_datamodule(ds.Dataset(hparams["dataset"]))
    val_dataloader = datamodule.val_dataloader(
        0,
        transform=ds.Augmentation.VAL,
        **{
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "batch_size": 250,
            "num_workers": 0,
            "persistent_workers": False,
        },
    )

    arch_a = arch_a.cuda()
    arch_b = arch_b.cuda()

    gt = []
    logit_a = []
    logit_b = []

    all_handles_a = []
    all_handles_b = []

    all_activations_a = {n.name: [0] for n in arch_a.hooks}
    all_activations_b = {n.name: [0] for n in arch_b.hooks}

    # create 2d array of combinations of all_activations_a and all_activations_b
    all_batch_cka_results: dict[str, dict[str, list[BatchCKAResult]]] = {}
    for a in all_activations_a.keys():
        all_batch_cka_results[a] = {}
        for b in all_activations_b.keys():
            all_batch_cka_results[a][b]: list[BatchCKAResult] = []

    # Register hooks
    for cnt, h in enumerate(arch_a.hooks):
        all_handles_a.append(arch_a.register_parallel_batch_cka_hooks(h, all_activations_a[h.name]))
    for cnt, h in enumerate(arch_b.hooks):
        all_handles_b.append(arch_b.register_parallel_batch_cka_hooks(h, all_activations_b[h.name]))
    with torch.no_grad():
        for batch in val_dataloader:
            x, y = batch
            x = x.cuda()
            gt.append(y.detach().cpu().numpy())
            logit_a.append(arch_a(x).detach().cpu().numpy())
            logit_b.append(arch_b(x).detach().cpu().numpy())

            all_K = {}
            all_L = {}

            for (ka, va), (kb, vb) in zip(all_activations_a.items(), all_activations_b.items()):
                all_K[ka] = all_activations_a[ka][0]
                all_activations_a[ka][0] = 0
                all_L[kb] = all_activations_b[ka][0]
                all_activations_b[ka][0] = 0

            for ka, K in all_K.items():
                for kb, L in all_L.items():
                    all_batch_cka_results[ka][kb].append(_batch_cka(K, L))

    [h.remove() for h in all_handles_a]
    [h.remove() for h in all_handles_b]

    ckas: np.ndarray = np.zeros((len(all_activations_a), len(all_activations_b)))

    for cnt_a, (ka, v) in enumerate(all_batch_cka_results.items()):
        for cnt_b, (kb, cka) in enumerate(v.items()):
            ckas[cnt_a, cnt_b] = float(_consolidate_cka_batch_results(cka))

    diag_ckas: list[float] = list(np.diag(ckas))
    arr_ckas: list[list[float]] = ckas.tolist()

    gt = torch.from_numpy(np.concatenate(gt, axis=0))
    logit_a = torch.from_numpy(np.concatenate(logit_a, axis=0))[None, ...]  # Expand first dim. (is expected
    logit_b = torch.from_numpy(np.concatenate(logit_b, axis=0))

    metrics = multi_output_metrics(
        logit_b, logit_a, gt, hparams["dataset"], hparams["architecture"], datamodule.n_classes
    )

    res = ModelToModelComparison(
        g_id_a=None,
        g_id_b=None,
        m_id_a=None,
        m_id_b=None,
        layerwise_cka=diag_ckas,
        accuracy_orig=metrics.mean_old_acc,
        accuracy_reg=metrics.accuracy,
        cohens_kappa=metrics.cohens_kappa.all_to_all_mean,
        jensen_shannon_div=metrics.jensen_shannon_div.all_to_all_mean,
        ensemble_acc=metrics.ensemble_accuracy,
        cka_off_diagonal=arr_ckas,
    )

    return res


def compare_models_functional(checkpoint_paths: list[Path], hparams: dict) -> list[OutputEnsembleResults]:
    """
    Loads the models and calculates the performance of the growing ensemble of models.
    So returns a list of results of len(models) - 1 (one for each possible stopping point 2, 3, 4, ..., n models)
    """

    arch_params = get_default_arch_params(hparams["dataset"])
    archs = [
        find_architectures.get_base_arch(ds.BaseArchitecture(hparams["architecture"]))(**arch_params)
        for _ in checkpoint_paths
    ]
    for arch, checkpoint_p in zip(archs, checkpoint_paths):
        ckpt: dict = torch.load(str(checkpoint_p))
        try:
            arch.load_state_dict(ckpt)
        except RuntimeError as _:  # noqa
            try:
                stripped = strip_state_dict_of_keys(ckpt)
                arch.load_state_dict(stripped)
            except RuntimeError as e:
                raise e

    datamodule = find_datamodules.get_datamodule(ds.Dataset(hparams["dataset"]))
    if hparams["dataset"] in ["CIFAR10", "CIFAR100"]:
        dataloader = datamodule.test_dataloader(
            transform=ds.Augmentation.VAL,
            **{
                "shuffle": False,
                "drop_last": False,
                "pin_memory": True,
                "batch_size": 128,
                "num_workers": 0,
                "persistent_workers": False,
            },
        )
    elif hparams["dataset"] == "ImageNet":
        dataloader = datamodule.val_dataloader(
            0,
            transform=ds.Augmentation.VAL,
            **{
                "shuffle": False,
                "drop_last": False,
                "pin_memory": True,
                "batch_size": 128,
                "num_workers": 0,
                "persistent_workers": False,
            },
        )
    else:
        raise NotImplementedError(f"Dataset {hparams['dataset']} not implemented yet.")

    archs = [arch.cuda() for arch in archs]

    gt: list[torch.Tensor] = []
    logits: list[list[torch.Tensor]] = [[] for _ in archs]

    for arch in archs:
        arch.eval()

    # create 2d array of combinations of all_activations_a and all_activations_b
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.cuda()
            gt.append(y.detach().cpu())
            for cnt, arch in enumerate(archs):
                logits[cnt].append(arch(x).detach().cpu())

    gt_cat: torch.Tensor = torch.cat(gt, axis=0)
    logits_cat: list[torch.Tensor] = [torch.cat(logit, axis=0) for logit in logits]

    # Calculate the metrics for n_models = 1, 2, 3, ..., n
    result: list[OutputEnsembleResults] = []
    # ke_metrics: list[MultiOutMetrics] = []
    for i in range(2, len(archs) + 1):
        out_metrics = final_multi_output_metrics([l.numpy() for l in logits_cat[:i]], gt_cat.numpy())
        if out_metrics.new_model_accuracy > 0.5:
            result.append(out_metrics)  # Only append if the new model converged!
            # ke_metrics.append(
            #     multi_output_metrics(
            #         logits_cat[i - 1],
            #         torch.stack(logits_cat[: i - 1]),
            #         gt_cat,
            #         hparams["dataset"],
            #         hparams["architecture"],
            #         datamodule.n_classes,
            #     )
            # )

    for r in result:
        r.regularization_metric = hparams["dis_loss"] if "dis_loss" in hparams.keys() else "None"
        r.regularization_position = hparams["hooks"][0]

    return result
