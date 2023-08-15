from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ke.util.data_structs import load_json


def load_json_from_scis23_scripts_file(json_name: str) -> tuple[dict, Path]:
    scis_res_dir = Path(
        "/home/tassilowald/Code/FeatureCompV3/ke/manual_introspection/scis23/representation_comp_results"
    )
    scis_out_dir = Path("/home/tassilowald/Code/FeatureCompV3/ke/manual_introspection/scis23/plots")
    path_to_comp_results = scis_res_dir / json_name
    res = load_json(str(path_to_comp_results))
    path_out = scis_out_dir / json_name
    return res, path_out


def plot_diagonal_cka(json_names: list[str], line_names: list[str], output_filepath: Path):
    all_values: list[dict] = []
    for json_name, line_name in zip(json_names, line_names):
        res, path_out = load_json_from_scis23_scripts_file(json_name)
        for r in res:
            layerwise_cka = r["layerwise_cka"]
            for cnt, cka in enumerate(layerwise_cka):
                all_values.append({"layer": cnt, "cka": cka, "Regularization": line_name})

    results = pd.DataFrame(all_values)
    sns.lineplot(data=results, x="layer", y="cka", hue="Regularization", err_style=False)
    plt.savefig(output_filepath)
    plt.close()


def plot_cka_sim(json_name: str):
    res, path_out = load_json_from_scis23_scripts_file(json_name)

    all_values = np.stack([np.array(r["cka_off_diagonal"]) for r in res])
    all_mean_values = np.nanmean(all_values, axis=0)

    ax: plt.Axes
    ax = sns.heatmap(all_mean_values, cmap="magma", square=True)
    ax.invert_yaxis()

    plt.savefig(path_out.parent / (path_out.name[:-5] + ".pdf"))
    plt.close()

    ax: plt.Axes
    ax = sns.heatmap(all_mean_values, cmap="magma", cbar=False, square=True)
    ax.invert_yaxis()

    plt.savefig(path_out.parent / (path_out.name[:-5] + "_no_cbar.pdf"))
    plt.close()
    pass
