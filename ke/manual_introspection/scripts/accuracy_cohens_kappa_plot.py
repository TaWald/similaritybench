from pathlib import Path

import pandas as pd
import seaborn as sns
from ke.manual_introspection.scripts import grouped_model_results as gmr
from ke.util.file_io import load_json
from matplotlib import pyplot as plt

output_dir = Path(
    "/home/tassilowald/Data/Results/knolwedge_extension_pics/layerwise_effects_of_regularization/accu_ck"
)


def plot_accuracy(values: pd.DataFrame, baseline: pd.DataFrame, hue_key: str, palette_name: str = None):
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    _ = sns.scatterplot(data=baseline, x="Acc.", y="Cohens Kappa", color="k", marker="X", ax=ax)
    _ = sns.scatterplot(
        data=values,
        x="Acc.",
        y="Cohens Kappa",
        hue=hue_key,
        palette=palette_name if palette_name is not None else "viridis",
        ax=ax,
    )


def plot_ensemble_accuracy(values: pd.DataFrame, baseline: pd.DataFrame, hue_key: str, palette_name: str = None):
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    _ = sns.scatterplot(data=baseline, x="Ens. Acc.", y="Cohens Kappa", color="k", marker="*", ax=ax)
    _ = sns.scatterplot(
        data=values,
        x="Ens. Acc.",
        y="Cohens Kappa",
        hue=hue_key,
        style="Layer",
        palette=palette_name if palette_name is not None else "viridis",
        ax=ax,
    )


def load_flat_results(grouped_model_result: dict) -> list[dict]:
    res = Path(__file__)
    result_dir = res.parent.parent / "representation_comp_results"
    flat = []
    for name in grouped_model_result.keys():
        potential_file_of_interest = result_dir / f"{name}.json"
        if potential_file_of_interest.exists():
            flat.extend(load_json(potential_file_of_interest))
    return flat


def plot_accuracy_over_cohens_kappa_hue_layer_layer_DIFF_tdepth_1_expvar_1():
    all_res = load_flat_results(gmr.layer_DIFF_tdepth_1_expvar_1)
    reg_df = pd.DataFrame(
        [{"Layer": int(r["hooks"][0]), "Acc.": r["accuracy_reg"], "Cohens Kappa": r["cohens_kappa"]} for r in all_res]
    )

    baseline = load_flat_results(gmr.baseline_unregularized)
    base_df = pd.DataFrame([{"Acc.": r["accuracy_reg"], "Cohens Kappa": r["cohens_kappa"]} for r in baseline])

    plot_accuracy(reg_df, base_df)
    plt.savefig(output_dir / "layer_DIFF_tdepth_1_expvar_1.png", dpi=600)


def plot_ensemble_accuracy_over_cohens_kappa_hue_layer_layer_DIFF_tdepth_1_expvar_1():
    all_res = load_flat_results(gmr.layer_DIFF_tdepth_1_expvar_1)
    reg_df = pd.DataFrame(
        [
            {"Layer": int(r["hooks"][0]), "Ens. Acc.": r["ensemble_acc"], "Cohens Kappa": r["cohens_kappa"]}
            for r in all_res
        ]
    )

    baseline = load_flat_results(gmr.baseline_unregularized)
    base_df = pd.DataFrame([{"Ens. Acc.": r["ensemble_acc"], "Cohens Kappa": r["cohens_kappa"]} for r in baseline])

    plot_ensemble_accuracy(reg_df, base_df)
    plt.savefig(output_dir / "ensemble_layer_DIFF_tdepth_1_expvar_1.png", dpi=600)


def plot_accuracy_over_cohens_kappa_hue_layer_layer_DIFF_tdepth_9_expvar_1():
    all_res = load_flat_results(gmr.layer_DIFF_tdepth_9_expvar_1)
    df = pd.DataFrame(
        [{"Layer": int(r["hooks"][0]), "Acc.": r["accuracy_reg"], "Cohens Kappa": r["cohens_kappa"]} for r in all_res]
    )

    sns.set_theme(style="darkgrid")
    sns.scatterplot(data=df, x="Acc.", y="Cohens Kappa", hue="Layer", palette="viridis")
    plt.savefig(output_dir / "layer_DIFF_tdepth_9_expvar_1.png", dpi=600)


def plot_ensemble_accuracy_over_cohens_kappa_hue_layer_layer_DIFF_tdepth_9_expvar_1():
    all_res = load_flat_results(gmr.layer_DIFF_tdepth_9_expvar_1)
    df = pd.DataFrame(
        [{"Layer": int(r["hooks"][0]), "Acc.": r["accuracy_reg"], "Cohens Kappa": r["cohens_kappa"]} for r in all_res]
    )
    baseline = load_flat_results(gmr.baseline_unregularized)
    base_df = pd.DataFrame([{"Ens. Acc.": r["ensemble_acc"], "Cohens Kappa": r["cohens_kappa"]} for r in baseline])

    plot_ensemble_accuracy(df, base_df, hue_key="Layer")
    plt.savefig(output_dir / "layer_DIFF_tdepth_9_expvar_1.png", dpi=600)


def plot_accuracy_over_cohens_kappa_hue_layer_MULTI_tdepth_9_expvar_1():
    all_res = load_flat_results(gmr.layer_MULTI_tdepth_9_expvar_1)
    df = pd.DataFrame(
        [
            {
                "Layer": f"{r['hooks'][0]} to {r['hooks'][-1]}",
                "Acc.": r["accuracy_reg"],
                "Cohens Kappa": r["cohens_kappa"],
            }
            for r in all_res
        ]
    )

    baseline = load_flat_results(gmr.baseline_unregularized)
    base_df = pd.DataFrame([{"Acc.": r["ensemble_acc"], "Cohens Kappa": r["cohens_kappa"]} for r in baseline])

    plot_accuracy(df, base_df, hue_key="Layer")
    plt.savefig(output_dir / "layer_MULTI_tdepth_9_expvar_1.png", dpi=600)


def plot_ensemble_accuracy_over_cohens_kappa_hue_layer_MULTI_tdepth_9_expvar_1():
    all_res = load_flat_results(gmr.layer_MULTI_tdepth_9_expvar_1)
    df = pd.DataFrame(
        [
            {
                "Layer": f"{r['hooks'][0]} to {r['hooks'][-1]}",
                "Ens. Acc.": r["accuracy_reg"],
                "Cohens Kappa": r["cohens_kappa"],
            }
            for r in all_res
        ]
    )
    baseline = load_flat_results(gmr.baseline_unregularized)
    base_df = pd.DataFrame([{"Ens. Acc.": r["ensemble_acc"], "Cohens Kappa": r["cohens_kappa"]} for r in baseline])

    plot_ensemble_accuracy(df, base_df, hue_key="Layer")
    plt.savefig(output_dir / "ensemble_layer_MULTI_tdepth_9_expvar_1.png", dpi=600)


def main():
    plot_accuracy_over_cohens_kappa_hue_layer_MULTI_tdepth_9_expvar_1()
    # plot_ensemble_accuracy_over_cohens_kappa_hue_layer_MULTI_tdepth_9_expvar_1()


if __name__ == "__main__":
    main()
