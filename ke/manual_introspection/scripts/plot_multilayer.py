from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import seaborn as sns

from ke.util.file_io import load_json


@dataclass()
class PlotInfo:
    index_x: int
    index_y: int
    json_name: str


def is_last(max_x, max_y, plot_info: PlotInfo):
    """ Checks if the plot will be the last one. Will be used for colorbar creation."""
    return True if (plot_info.index_x == max_x) and (plot_info.index_y == max_y) else False


def plot_mutli_layer_jsons():
    all_plot_infos: list[PlotInfo] = []
    for cnt, json_name in enumerate(
            [
                'baselines.json',
                'layer_9to9_tdepth_9_expvar_1.json',
                'layer_7to11_tdepth_9_expvar_1.json',
                'layer_5to13_tdepth_9_expvar_1.json',
                'layer_3to15_tdepth_9_expvar_1.json',
                'layer_0to16_tdepth_9_expvar_1.json', ]):
        all_plot_infos.append(PlotInfo(cnt, 0, json_name))
    out_name = "multi_layer_regularization_joint_plot"

    max_x = max([api.index_x for api in all_plot_infos])

    cols = max_x + 1

    fig: plt.Figure
    fig, axes = plt.subplots(1, cols, layout="constrained")
    fig.set_size_inches(cols * 3, 3)
    cur_file = Path(__file__)

    last = len(all_plot_infos) - 1
    for cnt, plot_info in enumerate(all_plot_infos):
        path_to_comp_results = cur_file.parent.parent / "representation_comp_results" / plot_info.json_name

        res = load_json(str(path_to_comp_results))

        all_values = np.stack([np.array(r['cka_off_diagonal']) for r in res])
        all_mean_values = np.nanmean(all_values, axis=0)
        if cnt != 0:
            yticklabels = False
        else:
            yticklabels = np.arange(17)
        g = sns.heatmap(all_mean_values, vmin=0, vmax=1, cmap='magma', square=True, ax=axes[cnt], cbar=False,
                        yticklabels=yticklabels)
        g.invert_yaxis()
        if cnt == last:
            norm = colors.Normalize(0, 1)
            pcm = cm.ScalarMappable(norm=norm, cmap='magma')
            fig.colorbar(pcm, ax=axes[cnt], shrink=1)

    pdf_path_out = cur_file.parent.parent / "plots" / (out_name + ".pdf")
    plt.savefig(pdf_path_out)
    plt.close()


def plot_positions_jsons():
    all_plot_infos = []
    for cnt, json in enumerate([
        'baselines.json',
        'layer_1_tdepth_1_expvar_1.json',
        'layer_3_tdepth_1_expvar_1.json',
        'layer_5_tdepth_1_expvar_1.json',
        'layer_7_tdepth_1_expvar_1.json',
        'layer_9_tdepth_1_expvar_1.json',
        'layer_11_tdepth_1_expvar_1.json',
        'layer_13_tdepth_1_expvar_1.json',
        'layer_15_tdepth_1_expvar_1.json',
    ]):
        all_plot_infos.append(PlotInfo(index_x=cnt % 3, index_y=cnt // 3, json_name=json))

    fig: plt.Figure
    fig, axes = plt.subplots(3, 3, layout="constrained")
    fig.set_size_inches(12, 12)
    cur_file = Path(__file__)

    for plot_info in all_plot_infos:
        path_to_comp_results = cur_file.parent.parent / "representation_comp_results" / plot_info.json_name

        res = load_json(str(path_to_comp_results))

        all_values = np.stack([np.array(r['cka_off_diagonal']) for r in res])
        all_mean_values = np.nanmean(all_values, axis=0)
        if plot_info.index_x == 0:
            yticklabels = np.arange(17)
        else:
            yticklabels = False
        if plot_info.index_y == 2:
            xticklabels = np.arange(17)
        else:
            xticklabels = False
        g = sns.heatmap(all_mean_values, vmin=0, vmax=1, cmap='magma', square=True,
                        ax=axes[plot_info.index_y,plot_info.index_x], cbar=False, xticklabels=xticklabels,
                        yticklabels=yticklabels)
        g.invert_yaxis()
    # add central bottom colorbar

    norm = colors.Normalize(0, 1)
    pcm = cm.ScalarMappable(norm=norm, cmap='magma')
    fig.colorbar(pcm, ax=axes[2, 1], shrink=1, location='bottom')

    pdf_path_out = cur_file.parent.parent / "plots" / "all_layers.pdf"
    plt.savefig(pdf_path_out)
    plt.close()


def plot_cka_sim(json_name: str):
    cur_file = Path(__file__)
    path_to_comp_results = cur_file.parent.parent / "representation_comp_results" / json_name
    path_out = cur_file.parent.parent / "plots" / json_name
    res = load_json(str(path_to_comp_results))

    all_values = np.stack([np.array(r['cka_off_diagonal']) for r in res])
    all_mean_values = np.nanmean(all_values, axis=0)

    ax: plt.Axes
    ax = sns.heatmap(all_mean_values, cmap='magma', square=True)
    ax.invert_yaxis()

    plt.savefig(path_out.parent / (path_out.name[:-5] + '.pdf'))
    plt.close()

    ax: plt.Axes
    ax = sns.heatmap(all_mean_values, cmap='magma', cbar=False, square=True)
    ax.invert_yaxis()

    plt.savefig(path_out.parent / (path_out.name[:-5] + '_no_cbar.pdf'))
    plt.close()
    pass


if __name__ == "__main__":
    plot_mutli_layer_jsons()

