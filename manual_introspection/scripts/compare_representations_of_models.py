import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rep_trans.arch.abstract_acti_extr import AbsActiExtrArch
from rep_trans.losses.utils import centered_kernel_alignment
from rep_trans.util import data_structs as ds
from rep_trans.util import file_io as io
from rep_trans.util import find_architectures
from rep_trans.util import find_datamodules
from rep_trans.util import name_conventions as nc
from rep_trans.util.file_io import load_json
from rep_trans.util.file_io import save_json
from rep_trans.util.file_io import strip_state_dict_of_keys
from tqdm import tqdm


# ToDo:
#   1. Load two models of different regularization types
#       a. Load models of the same ensemble
#   2. Register the same hook for both
#   3. Extract the activations (at the layers)
#   4. Pass through comparators
#   5. Save values for layers


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


def get_models_of_ke_ensembles(ke_src_path: Path, wanted_hparams: dict) -> list[tuple[Path, dict]]:
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


def get_models_with_ids_from_dir_and_first_model(
    model_paths: list[tuple[Path, dict]], model_ids: list[int]
) -> list[dict[int, Path]]:
    """
    Extracts models

    """
    all_models_of_ensemble: list[dict[int, Path]] = []

    for mp, hparams in model_paths:
        group_id = hparams["group_id_i"]
        dataset = hparams["dataset"]
        architecture = hparams["architecture"]
        model_paths: dict[int, Path] = {}
        if 0 in model_ids:
            base_path = (
                mp.parent
                / nc.KE_FIRST_MODEL_DIR.format(dataset, architecture)
                / nc.KE_FIRST_MODEL_GROUP_ID_DIR.format(group_id)
            )
            model_paths[0] = base_path
        for single_model_dir in mp.iterdir():
            model_id = int(single_model_dir.name.split("_")[-1])
            if model_id not in model_ids:
                continue
            else:
                model_paths[model_id] = single_model_dir
        all_models_of_ensemble.append(model_paths)
    return all_models_of_ensemble


def get_ckpts_from_paths(paths: dict[int, Path]):
    """
    Returns the checkpoints of the paths.
    """

    ckpt_paths = {}
    for k, p in paths.items():
        ckpt_paths[k] = p / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME
    return ckpt_paths


layer_9_regularization_hparams = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [9],
    "trans_depth": 9,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": 1.00,
    "dis_loss": "ExpVar",
    "dis_loss_weight": 1.00,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
}

layer_7_to_11_regularization_hparams = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [7, 8, 9, 10, 11],
    "trans_depth": 9,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": 1.00,
    "dis_loss": "ExpVar",
    "dis_loss_weight": 1.00,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
}

layer_5_to_13_regularization_hparams = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [5, 6, 7, 8, 9, 10, 11, 12, 13],
    "trans_depth": 9,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": 1.00,
    "dis_loss": "ExpVar",
    "dis_loss_weight": 1.00,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
}

layer_3_to_15_regularization_hparams = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "trans_depth": 9,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": 1.00,
    "dis_loss": "ExpVar",
    "dis_loss_weight": 1.00,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
}

all_layer_regularization_hparams = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "trans_depth": 9,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": 1.00,
    "dis_loss": "ExpVar",
    "ce_loss_weight": 1.00,
    "dis_loss_weight": 1.00,
}

layer_9_tdepth_1_expvar1 = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [9],
    "trans_depth": 1,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": 1.00,
    "dis_loss": "ExpVar",
    "dis_loss_weight": 1.00,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
}

layer_9_tdepth_3_expvar1 = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [9],
    "trans_depth": 3,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": 1.00,
    "dis_loss": "ExpVar",
    "dis_loss_weight": 1.00,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
}

layer_9_tdepth_5_expvar1 = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [9],
    "trans_depth": 3,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": 1.00,
    "dis_loss": "ExpVar",
    "dis_loss_weight": 1.00,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
}

layer_9_tdepth_7_expvar1 = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [9],
    "trans_depth": 3,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": 1.00,
    "dis_loss": "ExpVar",
    "dis_loss_weight": 1.00,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
}


baseline_unregularized = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    #  "hooks": "15",
    #  "trans_depth": 9,  Can be 1,3,5,7,9
    #  "kernel_width": 1,
    # "sim_loss": "ExpVar",
    # "sim_loss_weight": "1.00",
    "dis_loss": "None",
    #  "dis_loss_weight": "1.00",
    #  "ce_loss_weight": "1.00",
    #  "aggregate_reps": True,
    #  "softmax": True,
    #  "epochs_before_regularization": 0,
}

layer_15_regularization = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [15],
    #  "trans_depth": 9,  Can be 1,3,5,7,9
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "sim_loss_weight": "1.00",
    #  "dis_loss": "ExpVar",  Can be ExpVar or None
    #  "dis_loss_weight": "1.00",
    "ce_loss_weight": "1.00",
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
}


def create_json_of_hparams(hparams_dict: dict, output_name: str):
    baseline_results_path = Path(
        "/home/tassilowald/Code/FeatureComparisonV2/manual_introspection/representation_comp_results"
    )
    ckpt_results = Path("/mnt/cluster-checkpoint/results/knowledge_extension")
    models = get_models_of_ke_ensembles(ckpt_results, hparams_dict)
    model_paths: list[dict[int, Path]] = get_models_with_ids_from_dir_and_first_model(models, [0, 1])
    model_ckpt_paths: list[dict[int, Path]] = [get_ckpts_from_paths(mp) for mp in model_paths]

    layer_results: list[dict[int, float]] = []
    for model in tqdm(model_ckpt_paths[:2]):
        combis = itertools.combinations(model.values(), r=2)
        for a, b in combis:
            res = compare_models(model_a=a, model_b=b, hparams=hparams_dict)
            layer_results.append(res)
    save_json(layer_results, baseline_results_path / f"{output_name}.json")
    return


def add_description(results: list[dict], description: str) -> list[dict]:
    all_results = []
    for res in results:
        for k, v in res.items():
            all_results.append({"layer": int(k), "cka": float(v), "regularization": description})
    return all_results


def main():
    wanted_hparams_one = layer_9_tdepth_7_expvar1
    wanted_hparams_name: str = "layer_9_tdepth_7_ExpVar_1"
    if wanted_hparams_one is not None:
        create_json_of_hparams(wanted_hparams_one, wanted_hparams_name)
        sys.exit()

    baseline_results_path = Path(
        "/home/tassilowald/Code/FeatureComparisonV2/manual_introspection/representation_comp_results"
    )

    output_plots = Path("/home/tassilowald/Data/Results/knolwedge_extension_pics/layerwise_effects_of_regularization")

    baseline_values = load_json(baseline_results_path / "baselines.json")
    reg_9 = load_json(baseline_results_path / "layer_9_tdepth_9_ExpVar_1.json")
    reg_7to11 = load_json(baseline_results_path / "layer_7to11_ExpVar_1,00.json")
    reg_5to13 = load_json(baseline_results_path / "layer_5to13_ExpVar_1,00.json")
    reg_3to15 = load_json(baseline_results_path / "cifar_10_resnet34_hooks_3_to_15.json")
    reg_all = load_json(baseline_results_path / "cifar_10_resnet34_hooks_all_dis_loss_1,00.json")

    baseline_pd = pd.DataFrame(add_description(baseline_values, "unregularized"))
    reg_9_pd = pd.DataFrame(add_description(reg_9, "regularized_9"))
    reg_7to11_pd = pd.DataFrame(add_description(reg_7to11, "regularized_7_to_11"))
    reg_5to13_pd = pd.DataFrame(add_description(reg_5to13, "regularized_5_to_13"))
    reg_3to15_pd = pd.DataFrame(add_description(reg_3to15, "regularized_3_to_15"))
    reg_all_pd = pd.DataFrame(add_description(reg_all, "regularized_all"))

    results = pd.concat(
        [baseline_pd, reg_9_pd, reg_7to11_pd, reg_5to13_pd, reg_3to15_pd, reg_all_pd], ignore_index=True
    )
    sns.lineplot(data=results, x="layer", y="cka", hue="regularization")
    plt.savefig(output_plots / "all_layer_regularization_effect.png")
    print("What what")


if __name__ == "__main__":
    main()
