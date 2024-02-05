from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from simbench.manual_introspection.scis23.scripts.plot_cka_sim import load_json_from_scis23_scripts_file


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
    json = "expvar_5_models_layer_8.json"
    ensemble_res, _ = load_json_from_scis23_scripts_file(json)
    wanted_group_id = 3

    rem_res = [r for r in ensemble_res if r["g_id_a"] == wanted_group_id]

    all_ckas = [[None for _ in range(5)] for _ in range(5)]
    for row in range(5):
        for col in range(5):
            comp = [r for r in rem_res if (r["m_id_a"] == row) and (r["m_id_b"] == col)]
            if len(comp) == 1:
                all_ckas[row][col] = comp[0]["cka_off_diagonal"]
    for row in range(5):
        for col in range(5):
            cka = all_ckas[row][col]
            if cka is None:
                all_ckas[row][col] = np.transpose(deepcopy(all_ckas[col][row]))

    fig: plt.Figure
    ax: plt.Axes

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))

    for row in range(5):
        for col in range(5):
            ax = axes[row][col]
            sns.heatmap(all_ckas[row][col], ax=ax, vmin=0, vmax=1, cmap="magma", cbar=False)
            ax.invert_yaxis()
    plt.savefig(
        f"/home/tassilowald/Code/FeatureCompV3/ke/manual_introspection/scis23/plots/examplary_5_models_layer_8_gid_{wanted_group_id}.pdf"
    )
    plt.close()
