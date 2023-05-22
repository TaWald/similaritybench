from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from ke.util.file_io import load_json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cka_sim(json_name: str):
    cur_file = Path(__file__)
    path_to_comp_results = cur_file.parent.parent / "representation_comp_results" / json_name
    path_out = cur_file.parent.parent / "plots" / json_name
    res = load_json(str(path_to_comp_results))

    all_values = np.stack([np.array(r['cka_off_diagonal']) for r in res])
    all_mean_values = np.mean(all_values, axis=0)

    ax: plt.Axes
    ax = sns.heatmap(all_mean_values, cmap='magma', square=True)
    ax.invert_yaxis()

    plt.savefig(path_out.parent / (path_out.name[:-4]+'.pdf'))
    plt.close()

    ax: plt.Axes
    ax = sns.heatmap(all_mean_values, cmap='magma', cbar=False, square=True)
    ax.invert_yaxis()

    plt.savefig(path_out.parent / (path_out.name[:-5] + '_no_cbar.pdf'))
    plt.close()

    print(0)
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--json_name', type=str, required=True, help="Name of the json file located in representation_comp_results.")

    for name in [
        'non_reg_2_non_reg__layer_9_tdepth_1_expvar_1.json',
        'in_seed_non_reg_to_reg__layer_9_tdepth_1_expvar_1.json',
        'cross_seed_reg_2_reg__layer_9_tdepth_1_expvar_1.json',
        'cross_seed_reg_2_non_reg__layer_9_tdepth_1_expvar_1.json']:
        plot_cka_sim(name)

