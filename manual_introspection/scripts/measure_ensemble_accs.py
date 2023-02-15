from __future__ import annotations
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from rep_trans.metrics.ke_metrics import look_up_baseline_cohens_kappa
from rep_trans.util import file_io as io
from rep_trans.util import name_conventions as nc
from rep_trans.util.file_io import load_json
from rep_trans.util import data_structs as ds

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


def get_model_with_id_from_dir(
        model_paths: list[tuple[Path, dict]], wanted_model_id: int
        ) -> list[Path]:
    """
    Extracts models

    """
    all_models_of_ensemble: list[Path] = []
    
    for mp, hparams in model_paths:
        group_id = hparams["group_id_i"]
        dataset = hparams["dataset"]
        architecture = hparams["architecture"]
        wanted_model_path: Path | None = None
        if 0 == wanted_model_id:
            wanted_model_path = (
                    mp.parent
                    / nc.KE_FIRST_MODEL_DIR.format(dataset, architecture)
                    / nc.KE_FIRST_MODEL_GROUP_ID_DIR.format(group_id)
            )
        for single_model_dir in mp.iterdir():
            model_id = int(single_model_dir.name.split("_")[-1])
            if model_id != wanted_model_id:
                continue
            else:
                wanted_model_path = single_model_dir
        all_models_of_ensemble.append(wanted_model_path)
    all_models_of_ensemble = [am for am in all_models_of_ensemble if am is not None]
    return all_models_of_ensemble


def get_output_and_info_json(model_path: Path) -> dict | None:
    """
    Returns the checkpoints of the paths.
    """
    if ((model_path / nc.OUTPUT_TMPLT).exists()) and ((model_path / nc.KE_INFO_FILE).exists()):
        output_json = load_json(model_path / nc.OUTPUT_TMPLT)
        info_json = load_json(model_path /nc.KE_INFO_FILE)
        output_json.update(info_json)
        return output_json
    else:
        return None
    
def get_info_json_path(model_path: Path) -> dict | None:
    """
    Returns the checkpoints of the paths.
    """
    if (model_path / nc.OUTPUT_TMPLT).exists():
        info_json = load_json(model_path / nc.KE_INFO_FILE)
        return info_json
    else:
        return None


layer_9_tdepth_9_expvar1 = {
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

all_layers_tdepth_9_all_sim_and_dis_losses = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    "trans_depth": 9,
    "kernel_width": 1,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
    }

layer_7_tdepth_9_expvar_all = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [7],
    "trans_depth": 9,
    "kernel_width": 1,
    "sim_loss": "ExpVar",
    "dis_loss": "ExpVar",
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
    }


layer_16_tdepth_9_expvar_all = {
    "dataset": "CIFAR10",
    "architecture": "ResNet34",
    "hooks": [16],
    "trans_depth": 9,
    "kernel_width": 1,
    "ce_loss_weight": 1.00,
    "aggregate_reps": True,
    "softmax": True,
    "epochs_before_regularization": 0,
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


def gather_results(hparams_dict: dict, model_id: int = 1, ckpt_path: Path = Path("/mnt/cluster-data/results/knowledge_extension")):
    models = get_models_of_ke_ensembles(ckpt_path, hparams_dict)
    model_paths: list[Path] = get_model_with_id_from_dir(models, model_id)
    model_outputs: list[dict | None] = [get_output_and_info_json(mp) for mp in model_paths]
    model_outputs: list[dict] = [m for m in model_outputs if m is not None]

    return model_outputs


def add_description(results: list[dict], description: str) -> list[dict]:
    all_results = []
    for res in results:
        for k, v in res.items():
            all_results.append({"layer": int(k), "cka": float(v), "regularization": description})
    return all_results


def visualize_impact_of_increasing_depth():
    hparams = [baseline_unregularized, layer_9_tdepth_1_expvar1, layer_9_tdepth_3_expvar1, layer_9_tdepth_5_expvar1, layer_9_tdepth_7_expvar1, layer_9_tdepth_9_expvar1]
    names = ["unregularized", "tdepth_1", "tdepth_3", "tdepth_5", "tdepth_7", "tdepth_9"]
    title_ens_acc = "Effect of increasing depth to ensemble accuracy -- ExpVar 1.0"
    title_acc = "Effect of increasing depth to new model accuracy -- ExpVar 1.0"
    title_coka = "Effect of increasing depth to baseline cohens_kappa -- ExpVar 1.0"
    
    all_results: list[dict] = []
    for hparam, name in zip(hparams, names):
        res: list[dict] = gather_results(hparam)
        cleaned_res = []
        for r in res:
            r["name"] = name
            r.update(r["val"])
            r.pop("val")
            r.pop("test")
            # accuracy: float, dataset: ds.Dataset, arch: ds.BaseArchitecture)
            base_cc = look_up_baseline_cohens_kappa(accuracy = r["acc"], dataset= ds.Dataset("CIFAR10"), arch=ds.BaseArchitecture("ResNet34"))
            r["baseline_cohens_kappa"] = base_cc
            r["relative_cohens_kappa"] = r["cohens_kappa"] - base_cc
            cleaned_res.append(r)
        all_results.extend(cleaned_res)
    
    all_results_df = pd.DataFrame(all_results)
    mean_ensemble_acc = all_results_df[all_results_df["name"] == "unregularized"]["ensemble_acc"].mean()
    std_ensemble_acc = all_results_df[all_results_df["name"] == "unregularized"]["ensemble_acc"].std()
    
    baseline_single_acc = all_results_df[all_results_df["name"] == "unregularized"]["acc"].mean()
    std_single_acc = all_results_df[all_results_df["name"] == "unregularized"]["acc"].std()
    
    baseline_cohens_kappa = all_results_df[all_results_df["name"] == "unregularized"]["cohens_kappa"].mean()
    std_cohens_kappa = all_results_df[all_results_df["name"] == "unregularized"]["cohens_kappa"].std()
    
    regularized_results = all_results_df[all_results_df["name"] != "unregularized"]

    output_plots = Path("/home/tassilowald/Data/Results/knolwedge_extension_pics/measure_ensemble_accs")
    plt.figure(figsize=(16,9))
    g = sns.lineplot(data=regularized_results, x="trans_depth", y="ensemble_acc")
    plt.hlines(y=mean_ensemble_acc, xmin=1, xmax=9, colors="r", linestyles="solid")
    plt.hlines(y=mean_ensemble_acc + std_ensemble_acc, xmin=1, xmax=9, colors="r", linestyles="dashed")
    plt.hlines(y=mean_ensemble_acc - std_ensemble_acc, xmin=1, xmax=9, colors="r", linestyles="dashed")
    plt.title(title_ens_acc)
    plt.savefig(output_plots / "transfer_depth_to_ensemble_accuracy_layer_9.png")
    plt.close()
    
    plt.figure(figsize=(16,9))
    g = sns.lineplot(data=regularized_results, x="trans_depth", y="acc")
    plt.hlines(y=baseline_single_acc, xmin=1, xmax=9, colors="r", linestyles="solid")
    plt.hlines(y=baseline_single_acc + std_single_acc, xmin=1, xmax=9, colors="r", linestyles="dashed")
    plt.hlines(y=baseline_single_acc - std_single_acc, xmin=1, xmax=9, colors="r", linestyles="dashed")
    plt.title(title_acc)
    plt.savefig(output_plots / "transfer_depth_to_single_accuracy_layer_9.png")
    plt.close()
    
    plt.figure(figsize=(16,9))
    g = sns.lineplot(data=regularized_results, x="trans_depth", y="cohens_kappa")
    sns.lineplot(data=regularized_results, x="trans_depth", y="baseline_cohens_kappa", color="r")
    plt.title(title_coka)
    plt.savefig(output_plots / "transfer_depth_to_cohens_kappa_layer_9.png")
    plt.close()
    
    print("What what")


def visualize_impact_of_increasing_dis_weight():
    hparams = [baseline_unregularized, all_layers_tdepth_9_all_sim_and_dis_losses]
    names = ["unregularized", "regularized"]
    title_ens_acc = "Effect of increasing dissimilarity weight to ensemble accuracy -- All Layers"
    title_acc = "Effect of increasing dissimilarity to single accuracy -- All Layers"
    title_coka = "Effect of increasing dissimilarity to cohens kappa -- All Layers"
    
    all_results: list[dict] = []
    for hparam, name in zip(hparams, names):
        res: list[dict] = gather_results(hparam)
        cleaned_res = []
        for r in res:
            r["name"] = name
            r.update(r["val"])
            r.pop("val")
            r.pop("test")
            # accuracy: float, dataset: ds.Dataset, arch: ds.BaseArchitecture)
            base_cc = look_up_baseline_cohens_kappa(
                accuracy=r["acc"], dataset=ds.Dataset("CIFAR10"), arch=ds.BaseArchitecture("ResNet34")
                )
            r["baseline_cohens_kappa"] = base_cc
            r["relative_cohens_kappa"] = r["cohens_kappa"] - base_cc
            cleaned_res.append(r)
        all_results.extend(cleaned_res)
    
    all_results_df = pd.DataFrame(all_results)
    all_results_df = all_results_df.sort_values(by=["dissimilarity_loss_weight"])
    all_results_df["dissimilarity_loss_weight"] = all_results_df["dissimilarity_loss_weight"].round(2).apply(str)
    all_results_df = all_results_df
    mean_ensemble_acc = all_results_df[all_results_df["name"] == "unregularized"]["ensemble_acc"].mean()
    std_ensemble_acc = all_results_df[all_results_df["name"] == "unregularized"]["ensemble_acc"].std()
    
    baseline_single_acc = all_results_df[all_results_df["name"] == "unregularized"]["acc"].mean()
    std_single_acc = all_results_df[all_results_df["name"] == "unregularized"]["acc"].std()
    
    regularized_results = all_results_df[all_results_df["name"] != "unregularized"]
    
    output_plots = Path("/home/tassilowald/Data/Results/knolwedge_extension_pics/measure_ensemble_accs")
    plt.figure(figsize=(16, 9))
    g = sns.lineplot(data=regularized_results, x="dissimilarity_loss_weight", y="ensemble_acc", hue="dissimilarity_loss")
    plt.hlines(y=mean_ensemble_acc, xmin=0, xmax=7, colors="r", linestyles="solid")
    plt.hlines(y=mean_ensemble_acc + std_ensemble_acc, xmin=0, xmax=7, colors="r", linestyles="dashed")
    plt.hlines(y=mean_ensemble_acc - std_ensemble_acc, xmin=0, xmax=7, colors="r", linestyles="dashed")
    plt.title(title_ens_acc)
    plt.savefig(output_plots / "increasing_disweight_ensemble_acc_all_layers.png")
    plt.close()
    
    plt.figure(figsize=(16, 9))
    g = sns.lineplot(data=regularized_results, x="dissimilarity_loss_weight", y="acc", hue="dissimilarity_loss")
    plt.hlines(y=baseline_single_acc, xmin=0, xmax=7, colors="r", linestyles="solid")
    plt.hlines(y=baseline_single_acc + std_single_acc, xmin=0, xmax=7, colors="r", linestyles="dashed")
    plt.hlines(y=baseline_single_acc - std_single_acc, xmin=0, xmax=7, colors="r", linestyles="dashed")
    plt.title(title_acc)
    plt.savefig(output_plots / "increasing_disweight_single_acc_all_layers.png")
    plt.close()
    
    plt.figure(figsize=(16, 9))
    g = sns.lineplot(data=regularized_results, x="dissimilarity_loss_weight", y="cohens_kappa", hue="dissimilarity_loss")
    sns.lineplot(data=regularized_results, x="dissimilarity_loss_weight", y="baseline_cohens_kappa", hue="dissimilarity_loss", linestyle="dashed", ax=g)
    plt.title(title_coka)
    plt.savefig(output_plots / "increasing_disweight_cohens_kappa_all_layers.png")
    plt.close()
    
    print("What what")


def visualize_effect_of_additional_models():
    hparams = [baseline_unregularized, layer_7_tdepth_9_expvar_all]
    names = ["unregularized", "regularized"]
    title_ens_acc = "Effect of increasing dissimilarity weight to ensemble accuracy -- All Layers"
    title_acc = "Effect of increasing dissimilarity to single accuracy -- All Layers"
    
    all_results: list[dict] = []
    for model_id in [1,2,3,4,5,6,7,8,9]:
        for hparam, name in zip(hparams, names):
            res: list[dict] = gather_results(hparam, model_id)
            cleaned_res = []
            for r in res:
                if name == "unregularized":
                    r["dissimilarity_loss_weight"] = 0.00
                r["name"] = name
                r["Ensemble Size"] = model_id + 1
                r.update(r["val"])
                r.pop("val")
                r.pop("test")
                # accuracy: float, dataset: ds.Dataset, arch: ds.BaseArchitecture)
                base_cc = look_up_baseline_cohens_kappa(
                    accuracy=r["acc"], dataset=ds.Dataset("CIFAR10"), arch=ds.BaseArchitecture("ResNet34")
                    )
                r["baseline_cohens_kappa"] = base_cc
                r["relative_cohens_kappa"] = r["cohens_kappa"] - base_cc
                cleaned_res.append(r)
            all_results.extend(cleaned_res)
    
    all_results_df = pd.DataFrame(all_results)
    all_results_df = all_results_df.sort_values(by=["dissimilarity_loss_weight"])
    all_results_df["dissimilarity_loss_weight"] = all_results_df["dissimilarity_loss_weight"].round(2).apply(str)
    un_regularized_df = all_results_df[all_results_df["name"] == "unregularized"]
    regularized_df = all_results_df[all_results_df["name"] != "unregularized"]
    n_colors = len(regularized_df["dissimilarity_loss_weight"].unique())
    
    output_plots = Path("/home/tassilowald/Data/Results/knolwedge_extension_pics/measure_ensemble_accs")
    plt.figure(figsize=(16, 9))
    g: plt.Axes
    g = sns.lineplot(
        data=regularized_df, x="Ensemble Size", y="ensemble_acc", hue="dissimilarity_loss_weight", palette=sns.color_palette("viridis", n_colors)
        )
    sns.lineplot(
        data=un_regularized_df, x="Ensemble Size", y="ensemble_acc", color="k", ax=g
        )
    g.set_ylim(bottom=0.965, top=0.98)
    plt.title(title_ens_acc)
    plt.savefig(output_plots / "layer_7_regularization_to_ensemble_acc.png")
    plt.close()
    
    plt.figure(figsize=(16, 9))
    g = sns.lineplot(data=regularized_df,x="Ensemble Size", y="acc", hue="dissimilarity_loss_weight", palette=sns.color_palette("viridis", n_colors))
    sns.lineplot(
        data=un_regularized_df, x="Ensemble Size", y="acc", color="k", ax=g
        )
    g.set_ylim(bottom=0.94, top=0.98)
    plt.title(title_acc)
    plt.savefig(output_plots / "layer_7_regularization_to_single_acc.png")
    plt.close()


def visualize_last_layer_results():
    hparams = [baseline_unregularized, layer_16_tdepth_9_expvar_all]
    names = ["unregularized", "regularized"]
    title_ens_acc = "Effect of increasing dissimilarity weight to ensemble accuracy -- All Layers"
    title_acc = "Effect of increasing dissimilarity to single accuracy -- All Layers"
    title_coka = "Effect of increasing dissimilarity to cohens kappa -- All Layers"

    all_results: list[dict] = []
    for hparam, name in zip(hparams, names):
        res: list[dict] = gather_results(hparam, model_id=1)
        cleaned_res = []
        for r in res:
            r["name"] = name
            r.update(r["val"])
            r.pop("val")
            r.pop("test")
            # accuracy: float, dataset: ds.Dataset, arch: ds.BaseArchitecture)
            base_cc = look_up_baseline_cohens_kappa(
                accuracy=r["acc"], dataset=ds.Dataset("CIFAR10"), arch=ds.BaseArchitecture("ResNet34")
                )
            r["baseline_cohens_kappa"] = base_cc
            r["relative_cohens_kappa"] = r["cohens_kappa"] - base_cc
            cleaned_res.append(r)
        all_results.extend(cleaned_res)
    
    all_results_df = pd.DataFrame(all_results)
    all_results_df = all_results_df[all_results_df["acc"] > 0.9]
    all_results_df = all_results_df.sort_values(by=["dissimilarity_loss_weight"])
    all_results_df["dissimilarity_loss_weight"] = all_results_df["dissimilarity_loss_weight"].round(2).apply(str)
    all_results_df = all_results_df
    mean_ensemble_acc = all_results_df[all_results_df["name"] == "unregularized"]["ensemble_acc"].mean()
    std_ensemble_acc = all_results_df[all_results_df["name"] == "unregularized"]["ensemble_acc"].std()

    baseline_single_acc = all_results_df[all_results_df["name"] == "unregularized"]["acc"].mean()
    std_single_acc = all_results_df[all_results_df["name"] == "unregularized"]["acc"].std()

    regularized_results = all_results_df[all_results_df["name"] != "unregularized"]

    output_plots = Path("/home/tassilowald/Data/Results/knolwedge_extension_pics/measure_ensemble_accs")
    plt.figure(figsize=(16, 9))
    g = sns.lineplot(
        data=regularized_results, x="dissimilarity_loss_weight", y="ensemble_acc", hue="dissimilarity_loss"
        )
    plt.hlines(y=mean_ensemble_acc, xmin=0, xmax=7, colors="r", linestyles="solid")
    plt.hlines(y=mean_ensemble_acc + std_ensemble_acc, xmin=0, xmax=7, colors="r", linestyles="dashed")
    plt.hlines(y=mean_ensemble_acc - std_ensemble_acc, xmin=0, xmax=7, colors="r", linestyles="dashed")
    plt.title(title_ens_acc)
    plt.savefig(output_plots / "increasing_disweight_ensemble_acc_last_layers.png")
    plt.close()

    plt.figure(figsize=(16, 9))
    g = sns.lineplot(data=regularized_results, x="dissimilarity_loss_weight", y="acc", hue="dissimilarity_loss")
    plt.hlines(y=baseline_single_acc, xmin=0, xmax=7, colors="r", linestyles="solid")
    plt.hlines(y=baseline_single_acc + std_single_acc, xmin=0, xmax=7, colors="r", linestyles="dashed")
    plt.hlines(y=baseline_single_acc - std_single_acc, xmin=0, xmax=7, colors="r", linestyles="dashed")
    plt.title(title_acc)
    plt.savefig(output_plots / "increasing_disweight_single_acc_last_layers.png")
    plt.close()

    plt.figure(figsize=(16, 9))
    g = sns.lineplot(
        data=regularized_results, x="dissimilarity_loss_weight", y="cohens_kappa", hue="dissimilarity_loss"
        )
    sns.lineplot(
        data=regularized_results, x="dissimilarity_loss_weight", y="baseline_cohens_kappa", hue="dissimilarity_loss",
        linestyle="dashed", ax=g
        )
    plt.title(title_coka)
    plt.savefig(output_plots / "increasing_disweight_cohens_kappa_last_layers.png")
    plt.close()

    print("What what")


def main():
    a = "/mnt/E132-Projekte/Projects/2022_Wald_Knowledge_Extension/single_layer_dis_loss_weight_influence_lin_cka/knowledge_extension"
    # visualize_impact_of_increasing_depth()
    # visualize_impact_of_increasing_dis_weight()
    visualize_last_layer_results()

if __name__ == "__main__":
    main()
