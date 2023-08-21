from __future__ import annotations

import itertools
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.manual_introspection.scripts import grouped_model_results as grm
from ke.metrics.ke_metrics import multi_output_metrics
from ke.util import data_structs as ds
from ke.util import file_io as io
from ke.util import find_architectures
from ke.util import find_datamodules
from ke.util import name_conventions as nc
from ke.util.file_io import load_json
from ke.util.file_io import save_json
from ke.util.file_io import strip_state_dict_of_keys
from tqdm import tqdm

json_results_path = Path(__file__).parent / "representation_comp_results"
output_plots = Path("/home/tassilowald/Data/Results/ke_cifars")
ckpt_results = Path("/mnt/cluster-data/results/knowledge_extension_cifars")


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


"""
def compare_models(model_a: Path, model_b: Path, hparams: dict) -> dict[int, float]:
    arch_a: AbsActiExtrArch = find_architectures.get_base_arch(ds.BaseArchitecture(hparams["architecture"]))()
    arch_b: AbsActiExtrArch = find_architectures.get_base_arch(ds.BaseArchitecture(hparams["architecture"]))()

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
            "batch_size": 128,
            "num_workers": 0,
            "persistent_workers": False,
        },
    )

    layerwise_cka: dict[int, float] = {}
    arch_a = arch_a.cuda()
    arch_b = arch_b.cuda()
    for cnt, h in enumerate(arch_a.hooks):
        arch_a.register_rep_hook(h)
        arch_b.register_rep_hook(h)

        for batch in val_dataloader:
            x, _ = batch
            x = x.cuda()
            arch_a(x)
            arch_b(x)
            # if h.resolution == (32, 32):
            #     if len(arch_a.activations) * arch_a.activations[0].shape[0] > 5000:
            #         break

        arch_a.remove_forward_hook()
        arch_b.remove_forward_hook()

        actis_a = torch.from_numpy((np.concatenate(arch_a.activations, axis=0))[None, ...])
        actis_b = torch.from_numpy((np.concatenate(arch_b.activations, axis=0))[None, ...])

        del arch_a.activations, arch_b.activations
        # # cka = centered_kernel_alignment([actis_a], [actis_b])[0][0]
        #
        # split_actis_a = [torch.from_numpy(nd) for nd in np.split(actis_a, 100, axis=1)]
        # del actis_a
        # split_actis_b = [torch.from_numpy(nd) for nd in np.split(actis_b, 100, axis=1)]
        # del actis_b
        # partial_ckas = [
        #     centered_kernel_alignment([partial_a], [partial_b])
        #     for partial_a, partial_b in zip(split_actis_a, split_actis_b)
        # ]
        # del split_actis_a, split_actis_b
        cka = centered_kernel_alignment([actis_a], [actis_b])[0][0]
        layerwise_cka[cnt] = float(cka)

    return layerwise_cka

"""


def unbiased_hsic(K: torch.Tensor, L: torch.Tensor):
    """Calculates the unbiased HSIC estimate between two variables X and Y.
    Shape of the input should be (batch_size, batch_size (already calced)

    implementation of HSIC_1 from https://arxiv.org/pdf/2010.15327.pdf (Eq 3)
    """

    batch_size = K.shape[0]

    # Center the activations
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)

    ones = torch.ones((batch_size, 1), device=K.device, dtype=K.dtype)

    first = torch.trace(K @ L)
    second = (ones.T @ K @ ones @ ones.T @ L @ ones) / ((batch_size - 1) * (batch_size - 2))
    third = (ones.T @ K @ L @ ones) * (2 / (batch_size - 2))
    factor = 1 / (batch_size * (batch_size - 3))

    hsic = factor * (first + second - third)

    return torch.squeeze(hsic)


@dataclass
class BatchCKAResult:
    lk: float
    ll: float
    kk: float


def _consolidate_cka_batch_results(batch_results: list[BatchCKAResult]):
    lk = np.nanmean([br.lk for br in batch_results])
    kk = np.nanmean([br.kk for br in batch_results])
    ll = np.nanmean([br.ll for br in batch_results])

    cka = lk / (np.sqrt(kk * ll))

    return cka


def _batch_cka(K: torch.Tensor, L: torch.Tensor) -> BatchCKAResult:
    """Compares the activations of both networks and outputs.
    Expects the activations to be in format: B x p with p being the number of neurons."""

    K = K.cuda()
    L = L.cuda()

    kl = unbiased_hsic(K, L).cpu()
    kk = unbiased_hsic(K, K).cpu()
    ll = unbiased_hsic(L, L).cpu()

    return BatchCKAResult(kl, kk, ll)


def reshape(acti: torch.Tensor):
    acti = torch.flatten(acti, start_dim=1)
    return acti


def compare_models_parallel(model_a: Path, model_b: Path, hparams: dict) -> ModelToModelComparison:
    arch_a: AbsActiExtrArch = find_architectures.get_base_arch(ds.BaseArchitecture(hparams["architecture"]))()
    arch_b: AbsActiExtrArch = find_architectures.get_base_arch(ds.BaseArchitecture(hparams["architecture"]))()

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
            "batch_size": 128,
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

    metrics = multi_output_metrics(logit_b, logit_a, gt, "CIFAR10", "ResNet34")

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


def get_matching_model_dirs_of_ke_ensembles(ke_src_path: Path, wanted_hparams: dict) -> list[tuple[Path, dict]]:
    matching_dirs: list[tuple[Path, dict]] = []
    ke_src_paths = list(ke_src_path.iterdir())
    for ke_p in ke_src_paths:
        if ke_p.name.startswith("FIRST"):
            continue
        decodes = io.KENameEncoder.decode(ke_p)
        (
            exp_description,
            dataset,
            architecture,
            hooks,
            tdepth_i,
            kwidth_i,
            group_id_i,
            sim_loss,
            sim_loss_weight,
            dis_loss,
            dis_loss_weight,
            ce_loss_weight,
            agg,
            sm,
            ebr,
        ) = decodes

        decoded_params = {
            "exp_name": exp_description,
            "dataset": dataset,
            "architecture": architecture,
            "hooks": hooks,
            "trans_depth": tdepth_i,
            "kernel_width": kwidth_i,
            "group_id_i": group_id_i,
            "sim_loss": sim_loss,
            "sim_loss_weight": sim_loss_weight,
            "dis_loss": dis_loss,
            "dis_loss_weight": dis_loss_weight,
            "ce_loss_weight": ce_loss_weight,
            "aggregate_reps": agg,
            "softmax": sm,
            "epochs_before_regularization": ebr,
        }
        matches = True

        for k, v in wanted_hparams.items():
            if not matches:
                continue
            val = decoded_params[k]

            if val != v:
                matches = False
        if matches:
            matching_dirs.append((ke_p, decoded_params))
    return matching_dirs


@dataclass
class SeedResult:
    hparams: dict
    models: dict[int, Path] = field(init=False)
    checkpoints: dict[int, Path] = field(init=False)


def get_models_with_ids_from_dir_and_first_model(
    model_paths: list[tuple[Path, dict]], model_ids: list[int]
) -> list[SeedResult]:
    """
    Extracts all models from the path.
    If model_ids list contains the 0 it also grabs the first unregularized models. ([0,...]).
    besides that saves all models of all group_ids in the

    Returns a list of all model_ids (different group_ids) with path to ckpt source.

    """
    all_models_of_ensemble: list[SeedResult] = []

    for mp, hparams in model_paths:
        group_id = hparams["group_id_i"]
        dataset = hparams["dataset"]
        architecture = hparams["architecture"]
        tmp_model_paths: SeedResult = SeedResult(hparams=hparams)
        tmp_model_paths.checkpoints = {}
        tmp_model_paths.models = {}
        if 0 in model_ids:
            base_path = (
                mp.parent
                / nc.KE_FIRST_MODEL_DIR.format(dataset, architecture)
                / nc.KE_FIRST_MODEL_GROUP_ID_DIR.format(group_id)
            )
            tmp_model_paths.models[0] = base_path
        for single_model_dir in mp.iterdir():
            if single_model_dir.name.startswith("model"):
                model_id = int(single_model_dir.name.split("_")[-1])
                if model_id not in model_ids:
                    continue
                else:
                    tmp_model_paths.models[model_id] = single_model_dir
        all_models_of_ensemble.append(tmp_model_paths)
    return all_models_of_ensemble


def get_ckpts_from_paths(seed_result: SeedResult) -> SeedResult:
    """
    Returns the checkpoints of the paths.
    """

    seed_result.checkpoints = {}
    for k, p in seed_result.models.items():
        seed_result.checkpoints[k] = p / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME
    return seed_result


def add_description(results: list[dict], description: str) -> list[dict]:
    all_results = []
    for res in results:
        for k, v in res.items():
            all_results.append({"layer": int(k), "cka": float(v), "regularization": description})
    return all_results


def flatten_result(res: dict, hue_key: str = "Dis. Weight") -> list[dict]:
    hue_key_val = res[hue_key]
    all_results = []
    for k, v in res["whole_cka"].items():
        all_results.append({"layer": int(k), "cka": float(v), hue_key: hue_key_val})
    return all_results


def plot_layer_9_tdepth_9_expvar_DIFF():
    all_res = []
    v: ModelToModelComparison
    res = load_json(json_results_path / "baselines.json")
    baseline = [{"whole_cka": r["layerwise_cka"], "Dis. Weight": int(0.0)} for r in res]
    inter_res = [flatten_result(b, "Dis. Weight") for b in baseline]
    all_res.extend(itertools.chain.from_iterable(inter_res))
    for name, hparam in grm.layer_9_tdepth_9_expvar_DIFF.items():
        res = load_json(json_results_path / f"{name}.json")
        res_json = [{"whole_cka": r["layerwise_cka"], "Dis. Weight": int(hparam["dis_loss_weight"])} for r in res]
        inter_res = [flatten_result(r, "Dis. Weight") for r in res_json]
        all_res.extend(itertools.chain.from_iterable(inter_res))

    full_df = pd.DataFrame(all_res)

    sns.set_theme(style="darkgrid")
    ax: plt.Axes
    _, ax = plt.subplots()
    ax.set_xlim(0, 16)
    # cmap = sns.color_palette(colors)
    sns.lineplot(
        data=full_df, x="layer", y="cka", hue="Dis. Weight", palette=sns.color_palette("rocket", n_colors=8), ax=ax
    )
    plt.savefig(output_plots / "layer_9_tdepth_9_expvar_DIFF.png", dpi=600)


def plot_layer_DIFF_tdepth_1_expvar_1():
    all_res = []

    res = load_json(json_results_path / "baselines.json")
    baseline = [{"whole_cka": r["layerwise_cka"], "Regularization Layer": "None"} for r in res]
    inter_res = [flatten_result(b, "Regularization Layer") for b in baseline]
    all_res.extend(itertools.chain.from_iterable(inter_res))

    for name, hparam in grm.layer_DIFF_tdepth_1_expvar_1.items():
        res = load_json(json_results_path / f"{name}.json")
        res_json = [{"whole_cka": r["layerwise_cka"], "Regularization Layer": int(hparam["hooks"][0])} for r in res]
        inter_res = [flatten_result(r, "Regularization Layer") for r in res_json]
        all_res.extend(itertools.chain.from_iterable(inter_res))

    full_df = pd.DataFrame(all_res)

    sns.set_theme(style="darkgrid")
    ax: plt.Axes
    _, ax = plt.subplots()
    ax.set_xlim(0, 16)
    # cmap = sns.color_palette(colors)
    sns.lineplot(
        data=full_df,
        x="layer",
        y="cka",
        hue="Regularization Layer",
        palette=sns.color_palette("rocket", n_colors=9),
        ax=ax,
    )
    plt.savefig(output_plots / "layer_DIFF_tdepth_1_expvar_1.png", dpi=600)


def plot_layer_SPARSEDIFF_tdepth_1_expvar_1():
    all_res = []

    res = load_json(json_results_path / "baselines.json")
    baseline = [{"whole_cka": r["layerwise_cka"], "Regularization Layer": "None"} for r in res]
    inter_res = [flatten_result(b, "Regularization Layer") for b in baseline]
    all_res.extend(itertools.chain.from_iterable(inter_res))

    for (
        name,
        hparam,
    ) in (
        grm.layer_SPARSEDIFF_tdepth_1_expvar_1.items()
    ):  # comps_single_diff_layer_same_depth_same_dis_loss_strengths_sparse.items():
        res = load_json(json_results_path / f"{name}.json")
        res_json = [{"whole_cka": r["layerwise_cka"], "Regularization Layer": int(hparam["hooks"][0])} for r in res]
        inter_res = [flatten_result(r, "Regularization Layer") for r in res_json]
        all_res.extend(itertools.chain.from_iterable(inter_res))

    full_df = pd.DataFrame(all_res)

    sns.set_theme(style="darkgrid")
    ax: plt.Axes
    _, ax = plt.subplots()
    ax.set_xlim(0, 16)
    # cmap = sns.color_palette(colors)
    sns.lineplot(
        data=full_df,
        x="layer",
        y="cka",
        hue="Regularization Layer",
        palette=sns.color_palette("rocket", n_colors=6),
        ax=ax,
    )
    plt.savefig(output_plots / "layer_SPARSEDIFF_tdepth_1_expvar_1.png", dpi=600)


def plot_layer_9_tdepth_9_lincka_DIFF():
    all_res = []
    v: ModelToModelComparison
    res = load_json(json_results_path / "baselines.json")
    baseline = [{"whole_cka": r["layerwise_cka"], "Dis. Weight": int(0.0)} for r in res]
    inter_res = [flatten_result(b, "Dis. Weight") for b in baseline]
    all_res.extend(itertools.chain.from_iterable(inter_res))
    for name, hparam in grm.layer_9_tdepth_9_lincka_DIFF.items():
        res = load_json(json_results_path / f"{name}.json")
        res_json = [{"whole_cka": r["layerwise_cka"], "Dis. Weight": int(hparam["dis_loss_weight"])} for r in res]
        inter_res = [flatten_result(r, "Dis. Weight") for r in res_json]
        all_res.extend(itertools.chain.from_iterable(inter_res))

    full_df = pd.DataFrame(all_res)

    sns.set_theme(style="darkgrid")
    ax: plt.Axes
    _, ax = plt.subplots()
    ax.set_xlim(0, 16)
    # cmap = sns.color_palette(colors)
    sns.lineplot(
        data=full_df, x="layer", y="cka", hue="Dis. Weight", palette=sns.color_palette("rocket", n_colors=8), ax=ax
    )
    plt.savefig(output_plots / "layer_9_tdepth_9_lincka_DIFF.png", dpi=600)


def create_multi_layer_plot():
    baseline_values = load_json(json_results_path / "baselines.json")
    reg_9 = load_json(json_results_path / "layer_9_tdepth_9_ExpVar_1.json")
    reg_7to11 = load_json(json_results_path / "layer_7to11_ExpVar_1,00.json")
    reg_5to13 = load_json(json_results_path / "layer_5to13_ExpVar_1,00.json")
    reg_3to15 = load_json(json_results_path / "cifar_10_resnet34_hooks_3_to_15.json")
    reg_all = load_json(json_results_path / "cifar_10_resnet34_hooks_all_dis_loss_1,00.json")

    baseline_pd = pd.DataFrame(add_description(baseline_values, "unregularized"))
    reg_9_pd = pd.DataFrame(add_description(reg_9, "regularized_9"))
    reg_7to11_pd = pd.DataFrame(add_description(reg_7to11, "regularized_7_to_11"))
    reg_5to13_pd = pd.DataFrame(add_description(reg_5to13, "regularized_5_to_13"))
    reg_3to15_pd = pd.DataFrame(add_description(reg_3to15, "regularized_3_to_15"))
    reg_all_pd = pd.DataFrame(add_description(reg_all, "regularized_all"))

    results = pd.concat(
        [baseline_pd, reg_9_pd, reg_7to11_pd, reg_5to13_pd, reg_3to15_pd, reg_all_pd], ignore_index=True
    )
    sns.set_theme(style="darkgrid")
    ax: plt.Axes
    _, ax = plt.subplots()
    colors = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65"]
    ax.axhline(y=-0.01, xmin=8.5 / 16, xmax=9.5 / 16, color="#7eb0d5", linewidth=2)
    ax.axhline(y=-0.02, xmin=7 / 16, xmax=11 / 16, color="#b2e061", linewidth=2)
    ax.axhline(y=-0.03, xmin=5 / 16, xmax=13 / 16, color="#bd7ebe", linewidth=2)
    ax.axhline(y=-0.04, xmin=3 / 16, xmax=15 / 16, color="#ffb55a", linewidth=2)
    ax.axhline(y=-0.05, xmin=0 / 16, xmax=16 / 16, color="#ffee65", linewidth=2)
    ax.set_xlim(0, 16)
    # cmap = sns.color_palette(colors)
    sns.lineplot(data=results, x="layer", y="cka", hue="regularization", palette=colors, ax=ax)
    plt.savefig(output_plots / "all_layer_regularization_effect.png", dpi=600)
    print("What what")


def create_single_layer_depth_plot():
    baseline_results_path = Path(
        "/home/tassilowald/Code/FeatureComparisonV2/manual_introspection/representation_comp_results"
    )

    output_plots = Path("/home/tassilowald/Data/Results/knolwedge_extension_pics/layerwise_effects_of_regularization")

    baseline_values = load_json(baseline_results_path / "baselines.json")
    layer_0_td_1 = load_json(baseline_results_path / "layer_9_tdepth_1_expvar_1.json")
    layer_0_td_3 = load_json(baseline_results_path / "layer_9_tdepth_3_expvar_1.json")
    layer_0_td_5 = load_json(baseline_results_path / "layer_9_tdepth_5_expvar_1.json")
    layer_0_td_7 = load_json(baseline_results_path / "layer_9_tdepth_7_expvar_1.json")
    layer_0_td_9 = load_json(baseline_results_path / "layer_9_tdepth_9_ExpVar_1.json")

    baseline_pd = pd.DataFrame(add_description(baseline_values, "unregularized"))
    layer_0_td_1_pd = pd.DataFrame(add_description(layer_0_td_1, "Layer 9 Depth 1"))
    layer_0_td_3_pd = pd.DataFrame(add_description(layer_0_td_3, "Layer 9 Depth 3"))
    layer_0_td_5_pd = pd.DataFrame(add_description(layer_0_td_5, "Layer 9 Depth 5"))
    layer_0_td_7_pd = pd.DataFrame(add_description(layer_0_td_7, "Layer 9 Depth 7"))
    layer_0_td_9_pd = pd.DataFrame(add_description(layer_0_td_9, "Layer 9 Depth 9"))

    results = pd.concat(
        [baseline_pd, layer_0_td_1_pd, layer_0_td_3_pd, layer_0_td_5_pd, layer_0_td_7_pd, layer_0_td_9_pd],
        ignore_index=True,
    )
    sns.set_theme(style="darkgrid")
    ax: plt.Axes
    _, ax = plt.subplots()
    ax.axhline(y=-0.01, xmin=8.5 / 16, xmax=9.5 / 16, color="k", linewidth=2)
    ax.set_xlim(0, 16)
    cmap = sns.color_palette("viridis")
    sns.lineplot(data=results, x="layer", y="cka", hue="regularization", palette=cmap, ax=ax)
    plt.savefig(output_plots / "layer_9_increasing_transfer_depth.png", dpi=600)
    print("What what")


def create_single_layer_increasing_weight_plot():
    baseline_results_path = Path(
        "/home/tassilowald/Code/FeatureComparisonV2/manual_introspection/representation_comp_results"
    )

    output_plots = Path("/home/tassilowald/Data/Results/knolwedge_extension_pics/layerwise_effects_of_regularization")

    baseline_values = load_json(baseline_results_path / "baselines.json")
    layer_0_td_1 = load_json(baseline_results_path / "layer_9_tdepth_1_expvar_1.json")
    layer_0_td_3 = load_json(baseline_results_path / "layer_9_tdepth_3_expvar_1.json")
    layer_0_td_5 = load_json(baseline_results_path / "layer_9_tdepth_5_expvar_1.json")
    layer_0_td_7 = load_json(baseline_results_path / "layer_9_tdepth_7_expvar_1.json")
    layer_0_td_9 = load_json(baseline_results_path / "layer_9_tdepth_9_ExpVar_1.json")

    baseline_pd = pd.DataFrame(add_description(baseline_values, "unregularized"))
    layer_0_td_1_pd = pd.DataFrame(add_description(layer_0_td_1, "Layer 9 Depth 1"))
    layer_0_td_3_pd = pd.DataFrame(add_description(layer_0_td_3, "Layer 9 Depth 3"))
    layer_0_td_5_pd = pd.DataFrame(add_description(layer_0_td_5, "Layer 9 Depth 5"))
    layer_0_td_7_pd = pd.DataFrame(add_description(layer_0_td_7, "Layer 9 Depth 7"))
    layer_0_td_9_pd = pd.DataFrame(add_description(layer_0_td_9, "Layer 9 Depth 9"))

    results = pd.concat(
        [baseline_pd, layer_0_td_1_pd, layer_0_td_3_pd, layer_0_td_5_pd, layer_0_td_7_pd, layer_0_td_9_pd],
        ignore_index=True,
    )
    sns.set_theme(style="darkgrid")
    ax: plt.Axes
    _, ax = plt.subplots()
    ax.axhline(y=-0.01, xmin=8.5 / 16, xmax=9.5 / 16, color="k", linewidth=2)
    ax.set_xlim(0, 16)
    cmap = sns.color_palette("viridis")
    sns.lineplot(data=results, x="layer", y="cka", hue="regularization", palette=cmap, ax=ax)
    plt.savefig(output_plots / "layer_9_increasing_transfer_depth.png", dpi=600)
    print("What what")


def create_baseline_comparisons(hparam: dict, overwrite=False):
    for wanted_hparams_name, hparams_dict in hparam.items():
        this_output_file = json_results_path / f"{wanted_hparams_name}.json"
        if (not overwrite) and this_output_file.exists():
            continue

        models = get_matching_model_dirs_of_ke_ensembles(ckpt_results, hparams_dict)
        model_paths: list[SeedResult] = get_models_with_ids_from_dir_and_first_model(models, [0, 1])
        model_ckpt_paths: list[SeedResult] = [get_ckpts_from_paths(mp) for mp in model_paths]

        layer_results: list[ModelToModelComparison] = []
        all_ckpts = list(set([str(mcp.checkpoints[0]) for mcp in model_ckpt_paths]))  # only first models

        seed_result: SeedResult
        for ckpt_a, ckpt_b in tqdm(list(itertools.combinations(all_ckpts, 2))):
            res = compare_models_parallel(ckpt_a, ckpt_b, hparams=hparams_dict)
            layer_results.append(res)
        save_json(
            [{**asdict(lr), **hparams_dict} for lr in layer_results],
            json_results_path / f"{wanted_hparams_name}.json",
        )


def create_same_seed_comparisons(hparam: dict, overwrite=False):
    for wanted_hparams_name, hparams_dict in hparam.items():
        models = get_matching_model_dirs_of_ke_ensembles(ckpt_results, hparams_dict)

        model_paths: list[SeedResult] = get_models_with_ids_from_dir_and_first_model(models, [0, 1])
        model_ckpt_paths: list[SeedResult] = [get_ckpts_from_paths(mp) for mp in model_paths]

        this_output_file = json_results_path / f"{wanted_hparams_name}.json"
        if (not overwrite) and this_output_file.exists():
            continue

        layer_results: list[ModelToModelComparison] = []
        seed_result: SeedResult
        for seed_result in tqdm(model_ckpt_paths[:20]):
            combis = itertools.combinations(seed_result.checkpoints.values(), r=2)
            for a, b in combis:
                res = compare_models_parallel(model_a=a, model_b=b, hparams=hparams_dict)
                layer_results.append(res)
        if len(layer_results) == 0:
            warn("Nothing to save. skipping file creation!")
        else:
            save_json(
                [{**asdict(lr), **hparams_dict} for lr in layer_results],
                json_results_path / f"{wanted_hparams_name}.json",
            )
    return


def main():
    create_same_seed_comparisons(grm.layer_DIFF_tdepth_9_expvar_1)
    # create_single_diff_layer_single_depth_single_dis_loss_weight_plot_sparse()

    # create_single_diff_layer_single_depth_single_dis_loss_weight_plot
    # create_single_layer_multi_dis_loss_weight_plot_lincka()
    # create_single_layer_multi_dis_loss_weight_plot()
    # create_multi_layer_plot()
    # create_single_layer_depth_plot()


if __name__ == "__main__":
    main()
