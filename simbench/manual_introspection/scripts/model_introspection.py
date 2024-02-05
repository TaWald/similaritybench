import random
import sys
from pathlib import Path
from typing import Optional
from warnings import warn

import numpy as np
import pandas as pd
import plotly.express as px
import torch.utils.data
from simbench.arch.abstract_acti_extr import AbsActiExtrArch
from simbench.arch.arch_loading import load_kemodule_model
from simbench.arch.arch_loading import load_model
from simbench.arch.ke_architectures.feature_approximation_gradient_reversal import (
    FAGradientReversalArch,
)
from simbench.data.base_datamodule import BaseDataModule
from simbench.manual_introspection.introspection_tools.histogram_introspection import create_dataframe_from_np_arrays
from simbench.util import data_structs as ds
from simbench.util.load_own_objects import load_datamodule
from simbench.util.status_check import model_is_finished
from plotly import graph_objects as go
from torch.utils.data import DataLoader


def load_model_and_data(base_model: (Path, Path)) -> (AbsActiExtrArch, BaseDataModule):
    """
    Expects Model of Interest/Reference paths and loads the models and the corresponding data module.
    The paths are expected to point to (data, ckpt) directory.
    """
    if not (model_is_finished(*base_model)):
        raise ValueError
    model_of_interest: AbsActiExtrArch = load_model(*base_model)
    datamodule = load_datamodule(base_model[0])
    return model_of_interest, datamodule


def extract_activations(
    model: AbsActiExtrArch, hook_index: int, dataloader: DataLoader, n_samples: int
) -> np.ndarray:
    """
    Registers the hook at the wanted location, does one loop over the dataloader
    """
    model.register_rep_hook(model.hooks[hook_index])
    model = model.cuda()
    with torch.no_grad():
        torch.manual_seed(123456)  # Just like my password - Torch
        np.random.seed(123456)  # Numpy
        random.seed(123456)  # Python
        for batch in dataloader:
            x, y = batch
            x = x.cuda()
            model(x)
            if model.activations[0].shape[0] * len(model.activations) > n_samples:
                break
    activations = np.concatenate(model.activations, axis=0)
    model.remove_forward_hook()
    return activations


def extract_activations_from_ke(
    model: FAGradientReversalArch, dataloader: DataLoader, n_samples: int, try_realign_channels: bool
) -> dict[str, np.ndarray]:
    """
    Registers the hook at the wanted location, does one loop over the dataloader
    """
    approx_intermediate_act = None  # N at each hook position
    true_intermediate_act = None  # One at each hook position
    approx_outs = None  # N outputs
    true_out = None  # 1 output

    model = model.cuda()
    with torch.no_grad():
        torch.manual_seed(123456)  # Just like my password - Torch
        np.random.seed(123456)  # Numpy
        random.seed(123456)  # Python
        for batch in dataloader:
            x, y = batch
            x = x.cuda()
            list_approx_inter, list_true_inter, a_o, t_o = model(x)
            list_approx_inter = [we.detach().cpu() for we in list_approx_inter]
            list_true_inter = [we.detach().cpu() for we in list_true_inter]
            a_o = a_o.detach().cpu()
            t_o = t_o.detach().cpu()

            if true_out is None:
                true_out = t_o
                approx_outs = a_o
                true_intermediate_act = list_true_inter
                approx_intermediate_act = list_approx_inter
            else:
                true_out = torch.concat((true_out, t_o), dim=0)
                approx_outs = [torch.concat((o, new_o), dim=0) for o, new_o in zip(approx_outs, a_o)]
                true_intermediate_act = [
                    torch.concat((ti, new_ti), dim=0) for ti, new_ti in zip(true_intermediate_act, list_true_inter)
                ]
                approx_intermediate_act = [
                    torch.concat((ai, new_ai), dim=1)
                    for ai, new_ai in zip(approx_intermediate_act, list_approx_inter)
                ]

            if true_out.shape[0] > n_samples:
                break

    c_approx_intermediate_act = [a - torch.mean(a, dim=(0, 1, 3, 4), keepdim=True) for a in approx_intermediate_act]
    c_true_intermediate_act = [tr - torch.mean(tr, dim=(0, 1, 3, 4), keepdim=True) for tr in true_intermediate_act]

    if try_realign_channels:
        channel_aligned_aprx_act = []
        for ap, tr in zip(c_approx_intermediate_act, c_true_intermediate_act):
            corr_tendency = torch.mean(ap * tr, dim=(0, 1, 3, 4), keepdim=True) / (
                torch.std(ap, dim=(0, 1, 3, 4), keepdim=True) * torch.std(tr, dim=(0, 1, 3, 4), keepdim=True)
            )
            align = torch.where(
                corr_tendency >= 0,
                torch.ones_like(ap, dtype=torch.float),
                -torch.ones_like(ap, dtype=torch.float),
            )
            channel_aligned_aprx_act.append(align * ap)
        c_approx_intermediate_act = channel_aligned_aprx_act

    approx_intermediate_act = [np.squeeze(aia[:, :n_samples, ...].numpy()) for aia in c_approx_intermediate_act]
    true_intermediate_act = [np.squeeze(tia[:, :n_samples, ...].numpy()) for tia in c_true_intermediate_act]
    approx_outs = [ao[:n_samples].numpy() for ao in approx_outs]
    true_out = true_out[:n_samples].numpy()

    return {
        "true_output": true_out,
        "other_model_outputs": approx_outs,
        "true_intermediate_representations": true_intermediate_act,
        "approximated_intermediate_representations": approx_intermediate_act,
    }


def get_approximation_error(activations: np.ndarray, approximation: np.ndarray):
    error = activations.squeeze(axis=0) - approximation.squeeze(axis=0)
    return error


def get_examples_paths(
    layers: list[int], sim_dis_loss: list[tuple[str, str]], dis_loss_weight: list[str]
) -> list[dict]:
    data_abb = Path("/mnt/cluster-data/results/knowledge_extension")
    ckpt_abb = Path("/mnt/cluster-checkpoint/results/knowledge_extension")
    name = (
        "introspection_models__CIFAR10__ResNet34__GroupID_0__Hooks_{}__TDepth_9__KWidth_3__"
        + "Sim_{}_1.00__Dis_{}_{}_1.00__ar_True__sm_False__ebr_-1/model_0001"
    )
    examples = []
    for layer in layers:
        for sl, dl in sim_dis_loss:
            for dlw in dis_loss_weight:
                data_path = Path(data_abb / name.format(layer, sl, dl, dlw))
                ckpt_path = Path(ckpt_abb / name.format(layer, sl, dl, dlw))
                if data_path.exists() & ckpt_path.exists():
                    examples.append(
                        {
                            "name": f"layer_{layer}__Sim_{sl}__Dis_{dl}_{dlw}",
                            "data": data_path,
                            "ckpt": ckpt_path,
                            "meta_info": {"layer": layer, "sim_loss": sl, "dis_loss": dl, "dis_loss_weight": dlw},
                        }
                    )
                else:
                    warn(f"Path does not exists for {name.format(layer, sl, dl, dlw)}")
    return examples


def get_adversarial_examples_paths(
    layers: list[int], adv_loss: list[str], adv_weights: list[tuple[str]]
) -> list[dict]:
    data_abb = Path("/mnt/cluster-data/results/knowledge_adversarial_extension")
    ckpt_abb = Path("/mnt/cluster-checkpoint/results/knowledge_adversarial_extension")
    name = (
        "test_adversarial__CIFAR10__ResNet34__GroupID_1__Hooks_{}__TDepth_9__KWidth_3__"
        + "Adv_{}_{}_1.00_1.00__ebr_-1}"
    )
    examples = []
    for layer in layers:
        for adv in adv_loss:
            for adv_weight, ce_weight, grs_weight in adv_weights:
                data_path = Path(data_abb / name.format(layer, adv, adv_weight))
                ckpt_path = Path(ckpt_abb / name.format(layer, adv, adv_weight))
                if data_path.exists() & ckpt_path.exists():
                    for model_path in data_path.iterdir():
                        model_id = int(model_path.name.split("_")[-1])
                        examples.append(
                            {
                                "name": f"layer_{layer}__Adv_{adv}_{adv_weight}",
                                "data": data_path,
                                "ckpt": ckpt_path,
                                "meta_info": {
                                    "layer": layer,
                                    "adv_loss": adv,
                                    "adv_weight": adv_weight,
                                    "model_id": model_id,
                                },
                            }
                        )
                else:
                    warn(f"Path does not exists for {name.format(layer, adv_weight, grs_weight)}")
    return examples


def get_error_df(ke_module, dataloader, n_samples: int, meta_info: dict) -> pd.DataFrame:
    try_realign = False
    if meta_info["sim_loss"] in ["L2Corr"]:
        try_realign = True
    actis = extract_activations_from_ke(ke_module, dataloader, n_samples, try_realign)
    baseline_error = get_approximation_error(
        actis["true_intermediate_representations"][0], actis["approximated_intermediate_representations"][0]
    )
    true_vals = actis["true_intermediate_representations"][0]
    apxs_vals = actis["approximated_intermediate_representations"][0]

    normalized_true = (true_vals - np.mean(true_vals, axis=(0, 2, 3), keepdims=True)) / (
        np.std(true_vals, axis=(0, 2, 3), keepdims=True) + 1e-9
    )
    normalized_apxs = (apxs_vals - np.mean(np.squeeze(apxs_vals, 0), axis=(0, 2, 3), keepdims=True)) / (
        np.std(np.squeeze(apxs_vals, 0), axis=(0, 2, 3), keepdims=True) + 1e-9
    )

    indices = np.zeros_like(normalized_true)
    for i in np.arange(normalized_true.shape[0]):
        indices[i, ...] = i

    error_df = create_dataframe_from_np_arrays(
        {
            "true_values": actis["true_intermediate_representations"][0],
            "normalized_true_values": normalized_true,
            "error": baseline_error,
            "approximated_values": actis["approximated_intermediate_representations"][0],
            "normalized_approximated_values": normalized_apxs,
            "normalized_error": normalized_true - normalized_apxs,
            "index": indices,
        },
        meta_info,
    )
    return error_df


def scatter_value_to_approximations_of_key(
    df: pd.DataFrame, color_key: str, symbol_key: Optional[str], title: str, out_path: Path, filename: str
):
    # Scatterplot of only Layer x
    plot_title = f"{title}. Values to approximation zero-meaned."
    fig: go.Figure
    fig = px.scatter(
        df,
        x="true_values",
        y="approximated_values",
        color=color_key,
        symbol=symbol_key,
        title=plot_title,
        trendline="ols",
    )
    fig.write_html(out_path / f"{filename}.html")
    del fig


def scatter_normalized_value_to_approximations_of_key(
    df: pd.DataFrame, color_key: str, symbol_key: Optional[str], title: str, out_path: Path, filename: str
):
    # Scatterplot of only Layer x
    plot_title = f"{title}. Values to approximation normalized."
    fig: go.Figure
    fig = px.scatter(
        df,
        x="normalized_true_values",
        y="normalized_approximated_values",
        color=color_key,
        symbol=symbol_key,
        title=plot_title,
        trendline="ols",
    )
    fig.write_html(out_path / f"{filename}.html")
    del fig


def create_plots(
    output_path: Path,
    dir_name: str,
    hooks: list[int],
    sim_dis_loss: list[tuple[str, str]],
    dis_loss_weight: list[str],
):
    """
    This shows plots of models that have not been regularized and only try to approximate.
    Maybe gives an idea what the different similarity losses actually try to optimize.

    """
    examples = get_examples_paths(hooks, sim_dis_loss, dis_loss_weight)
    if len(examples) == 0:
        raise FileNotFoundError("No examples were found! Is the path correct and the disk mounted?")
    _, datamodule = load_model_and_data((examples[0]["data"], examples[0]["ckpt"]))
    # WARNING: The MODELS BELOW CAN ORIGINATE FROM DIFFERENT MODELS!!

    all_dfs = {}
    correlations = {}
    n_samples = 100
    n_plotting = 2
    print("Reading examples...")
    for ex in examples:
        ke_module = load_kemodule_model(ex["data"], ex["ckpt"])
        val_dl = datamodule.val_dataloader(0, ds.Augmentation.VAL, **{"batch_size": 2000, "num_workers": 0})
        df = get_error_df(ke_module, val_dl, n_samples, ex["meta_info"])
        correlations[ex["name"]] = df["true_values"].corr(df["approximated_values"])
        df["layer"] = df["layer"].astype("category")
        all_dfs[ex["name"]] = df[df["index"] < n_plotting]
        del val_dl
    print("Reading examples - finished.")
    # Plot value to error.
    joint_df = pd.concat(list(all_dfs.values()))
    del all_dfs
    op = output_path / dir_name
    op.mkdir(exist_ok=True)

    # Scatterplot of all Layers and all methods
    single_loss = False
    if len(joint_df["sim_loss"].unique()) == 1:
        single_loss = True
    print("Creating plots ...")
    for h in hooks:
        tmp_df = joint_df[joint_df["layer"] == h]
        # filename_zc = f"zero-centered_value_to_approximation_layer_{h}_dlw_all"
        filename_n = f"normalized_value_to_approximation_layer_{h}_dlw_all"
        title = f"Layer {h} with various disloss weight"
        # scatter_value_to_approximations_of_key(
        #     tmp_df, color_key="sim_loss", symbol_key="dis_loss_weight", title=title, out_path=op, filename=filename_zc
        # )
        if single_loss:
            scatter_normalized_value_to_approximations_of_key(
                tmp_df, color_key="dis_loss_weight", symbol_key=None, title=title, out_path=op, filename=filename_n
            )
        else:
            scatter_normalized_value_to_approximations_of_key(
                tmp_df,
                color_key="sim_loss",
                symbol_key="dis_loss_weight",
                title=title,
                out_path=op,
                filename=filename_n,
            )

    for dlw in dis_loss_weight:
        tmp_df = joint_df[joint_df["dis_loss_weight"] == dlw]
        title = f"Various Layers with {dlw} disloss weight"
        # filename_zc = f"zero-centered_value_to_approximation_layer_all_dlw_{dlw}"
        filename_n = f"normalized_value_to_approximation_layer_all_dlw_{dlw}"
        # scatter_value_to_approximations_of_key(
        #     tmp_df, color_key="sim_loss", symbol_key="dis_loss_weight", title=title, out_path=op, filename=filename_zc
        # )
        if single_loss:
            scatter_normalized_value_to_approximations_of_key(
                tmp_df, color_key="dis_loss_weight", symbol_key=None, title=title, out_path=op, filename=filename_n
            )
        else:
            scatter_normalized_value_to_approximations_of_key(
                tmp_df,
                color_key="sim_loss",
                symbol_key="dis_loss_weight",
                title=title,
                out_path=op,
                filename=filename_n,
            )
        for h in hooks:
            sub_df = tmp_df[tmp_df["layer"] == h]
            title = f"Layer {h} with {dlw} disloss weight"
            # filename_zc = f"zero-centered_value_to_approximation_layer_{h}_dlw_{dlw}"
            filename_n = f"normalized_value_to_approximation_layer_{h}_dlw_{dlw}"
            # scatter_value_to_approximations_of_key(
            #     sub_df, color_key="sim_loss", symbol_key=None, title=title, out_path=op, filename=filename_zc
            # )
            scatter_normalized_value_to_approximations_of_key(
                sub_df, color_key="sim_loss", symbol_key=None, title=title, out_path=op, filename=filename_n
            )
    print("Creating plots - Finished.")


def main():
    moi: AbsActiExtrArch
    im: AbsActiExtrArch
    datamodule: BaseDataModule

    output_path = Path("/home/tassilowald/Data/Results/knolwedge_extension_pics/introspection_output_pics")
    create_plots(
        output_path,
        dir_name="unregularized",
        hooks=[0, 8, 16],
        sim_dis_loss=[("ExpVar", "None"), ("TopkExpVar", "None"), ("L2Corr", "None"), ("TopkL2Corr", "None")],
        dis_loss_weight=["0.00"],
    )
    create_plots(
        output_path,
        dir_name="regularized",
        hooks=[0, 8, 16],
        sim_dis_loss=[
            ("ExpVar", "ExpVar"),
            ("TopkExpVar", "TopkExpVar"),
            ("L2Corr", "L2Corr"),
            ("TopkL2Corr", "TopkL2Corr"),
        ],
        dis_loss_weight=["0.10", "1.00", "8.00"],
    )
    create_plots(
        output_path,
        dir_name="only_CeluExpVar",
        hooks=[0, 8, 16],
        sim_dis_loss=[("ExpVar", "ExpVar"), ("ExpVar", "None")],
        dis_loss_weight=["0.00", "0.10", "1.00", "8.00"],
    )
    create_plots(
        output_path,
        dir_name="only_TopkCeluExpVar",
        hooks=[0, 8, 16],
        sim_dis_loss=[("TopkExpVar", "TopkExpVar"), ("TopkExpVar", "None")],
        dis_loss_weight=["0.00", "0.10", "1.00", "8.00"],
    )
    create_plots(
        output_path,
        dir_name="only_L2Corr",
        hooks=[0, 8, 16],
        sim_dis_loss=[("L2Corr", "L2Corr"), ("L2Corr", "None")],
        dis_loss_weight=["0.00", "0.10", "1.00", "8.00"],
    )
    create_plots(
        output_path,
        dir_name="only_TopkL2Corr",
        hooks=[0, 8, 16],
        sim_dis_loss=[("TopkL2Corr", "TopkL2Corr"), ("TopkL2Corr", "None")],
        dis_loss_weight=["0.00", "0.10", "1.00", "8.00"],
    )

    # Model of interest is a regularized model.
    #   Model of refernece is a "normal" == unregularized model!


if __name__ == "__main__":
    main()
    sys.exit(0)
