from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from simbench.manual_introspection.scis23.scripts.plot_cka_sim import load_json_from_scis23_scripts_file
from simbench.manual_introspection.scis23.scripts.plot_cka_sim import plot_cka_sim
from simbench.manual_introspection.scripts import grouped_model_results as grm


def plot_multilayer_diagonal_cka(json_names: list[str], line_names: list[str], output_filepath: Path):
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
    colors = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65"]
    ax.axhline(y=-0.01, xmin=7.5 / 16, xmax=8.5 / 16, color="#7eb0d5", linewidth=2)
    ax.axhline(y=-0.02, xmin=6 / 16, xmax=10 / 16, color="#b2e061", linewidth=2)
    ax.axhline(y=-0.03, xmin=4 / 16, xmax=12 / 16, color="#bd7ebe", linewidth=2)
    ax.axhline(y=-0.04, xmin=2 / 16, xmax=14 / 16, color="#ffb55a", linewidth=2)
    ax.axhline(y=-0.05, xmin=0 / 16, xmax=16 / 16, color="#ffee65", linewidth=2)
    ax.set_xlim(0, 16)

    sns.lineplot(data=results, x="Layer", y="LinCKA", hue="Regularization", err_style=None, palette=colors, ax=ax)
    plt.savefig(output_filepath)
    plt.close()

    print("What what")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--json_name", type=str, required=True, help="Name of the json file located in representation_comp_results."
    )

    jsons = [f + ".json" for f in grm.layer_SCIS23_MULTI_tdepth_1_expvar_1.keys()]
    for name in jsons:
        plot_cka_sim(name)
    # Add baseline for the diagonal plot
    jsons = ["baseline_cifar_10.json"] + jsons
    plot_multilayer_diagonal_cka(
        jsons,
        ["None", "Layer 8", "Layer 6-10", "Layer 4-12", "Layer 2-14", "Layer 0-16"],
        output_filepath=Path(
            "/home/tassilowald/Code/FeatureCompV3/ke/manual_introspection/scis23/plots/multilayer_diagonal_cka.pdf"
        ),
    )
