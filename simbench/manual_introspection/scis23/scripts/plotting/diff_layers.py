from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from simbench.manual_introspection.scis23.scripts.plot_cka_sim import load_json_from_scis23_scripts_file
from simbench.manual_introspection.scis23.scripts.plot_cka_sim import plot_cka_sim


def plot_different_layers_diagonal_cka(json_names: list[str], line_names: list[str], output_filepath: Path):
    all_values: list[dict] = []
    for json_name, line_name in zip(json_names, line_names):
        res, path_out = load_json_from_scis23_scripts_file(json_name)
        for r in res:
            layerwise_cka = r["layerwise_cka"]
            for cnt, cka in enumerate(layerwise_cka):
                all_values.append({"Layer": cnt, "LinCKA": cka, "Regularization": line_name})

    results = pd.DataFrame(all_values)

    # Plotting happens here
    sns.set_theme(style="whitegrid")
    ax: plt.Axes
    _, ax = plt.subplots()

    sns.lineplot(data=results, x="Layer", y="LinCKA", hue="Regularization", err_style=None, ax=ax)
    plt.savefig(output_filepath)
    plt.close()

    print("What what")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--json_name", type=str, required=True, help="Name of the json file located in representation_comp_results."
    )
    jsons = [
        "layer_1_tdepth_1_expvar_1.json",
        "layer_3_tdepth_1_expvar_1.json",
        "layer_5_tdepth_1_expvar_1.json",
        "layer_7_tdepth_1_expvar_1.json",
        "layer_9_tdepth_1_expvar_1.json",
        "layer_11_tdepth_1_expvar_1.json",
        "layer_13_tdepth_1_expvar_1.json",
        "layer_15_tdepth_1_expvar_1.json",
    ]
    for name in jsons:
        plot_cka_sim(name)
    jsons = ["baseline_cifar_10.json"] + jsons
    plot_different_layers_diagonal_cka(
        jsons,
        ["None", "1", "3", "5", "7", "9", "11", "13", "15"],
        Path("/home/tassilowald/Code/FeatureCompV3/ke/manual_introspection/scis23/plots/diff_layers_diagonal.pdf"),
    )
