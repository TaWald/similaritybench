from pathlib import Path

import numpy as np

from ke.manual_introspection.scripts.plot_multilayer import load_json_from_scrips_file
from matplotlib import pyplot as plt
import seaborn as sns

from ke.util.file_io import save_json


def extract_mean_cka_array_from_json(results: dict, row_idx: int, col_idx: int) -> np.ndarray:
    """ Averages the CKA of the different sequences of the same row and col idx location.
    Returns the mean array that is to be plotted in a heatmap.
    """
    all_results = []

    for res in results:
        if not res['m_id_a'] == row_idx:
            continue
        if not res['m_id_b'] == col_idx:
            continue
        cka_matrix = np.array(res['cka_off_diagonal'])
        all_results.append(cka_matrix)

    all_values = np.stack(all_results)
    all_mean_values = np.nanmean(all_values, axis=0)
    return all_mean_values


def plot_nxn_cka_matrix(results: dict, filepath: Path) -> None:
    n = 5
    fig: plt.Figure
    fig, axes = plt.subplots(n, n, )
    fig.set_size_inches(3 * n, 3 * n)

    for i in range(n):
        for j in range(n):
            cka_matrix = extract_mean_cka_array_from_json(results, i, j)
            if i == 4:
                xticklabels = np.arange(17)
            else:
                xticklabels = False
            if j == 0:
                yticklabels = np.arange(17)
            else:
                yticklabels = False

            g = sns.heatmap(cka_matrix, vmin=0, vmax=1, cmap='magma', square=True,
                            ax=axes[i, j], cbar=False, xticklabels=xticklabels,
                            yticklabels=yticklabels)
            g.invert_yaxis()

    plt.savefig(filepath)
    return


def load_wanted_metric_matrix(result_dict: dict, wanted_metric_key: str) -> np.ndarray:
    """ Loads scalar values of all 1to1 model compaisons and returns them in a nxn Matrix

    """
    max_idx = max([max(r['m_id_a'], r['m_id_b']) for r in result_dict])
    results_array: list[list[list[float]]] = [[[] for _ in range(max_idx)] for __ in range(max_idx)]
    metric_values = np.zeros((max_idx, max_idx), dtype=float)

    for r in result_dict:
        metric = r[wanted_metric_key]
        results_array[r['m_id_a']][r['m_id_b']].append(metric)

    for i in range(max_idx):
        for j in range(max_idx):
            metric_values[i, j] = np.nanmean(results_array[i][j])

    return metric_values


def calc_rel_ens_acc_matrix(result_dict: dict) -> np.ndarray:
    """ Loads scalar values of all 1to1 model compaisons and returns them in a nxn Matrix

    """
    max_idx = max([max(r['m_id_a'], r['m_id_b']) for r in result_dict])
    rel_ens_acc: list[list[list[float]]] = [[[] for _ in range(max_idx)] for __ in range(max_idx)]

    rel_ens = np.zeros((max_idx, max_idx), dtype=float)

    for r in result_dict:
        orig_acc_metric = r['accuracy_orig']
        reg_acc_metric = r['accuracy_reg']
        ens_acc_metric = r['ensemble_acc']
        rel_ens[r['m_id_a']][r['m_id_b']].append(float(((orig_acc_metric + reg_acc_metric) / 2) / ens_acc_metric))

    for i in range(max_idx):
        for j in range(max_idx):
            rel_ens[i, j] = np.nanmean(rel_ens_acc[i][j])

    return rel_ens


def average_non_diagonal_entries(arr):
    """ Expects non-negative np arrays and averages all non-diagonal elements. """
    # Get the diagonal indices
    diag_indices = np.diag_indices_from(arr)
    # Set the diagonal elements to zero
    arr[diag_indices] = -1
    # Calculate the average of the non-diagonal elements
    avg = np.mean(arr[arr != -1])
    return avg


def calculate_mean_metric_for_some_n(metric_mat: np.ndarray, n: int) -> float:
    """ Calculates the mean values of off-diagonal matrix values.
    Intended to average metric values (which are not negative)."""
    rem_mat = metric_mat[:n, :n]
    return average_non_diagonal_entries(rem_mat)


def main(json_filename: str):
    results, out_filepath = load_json_from_scrips_file(json_filename)
    iterative_json_path = out_filepath.parent.parent / 'pairwise_metric_jsons' / (out_filepath.name + '.json')

    # Plot the 5x5 Matrix of the ensemble showing that they are different to each other.
    head = out_filepath.parent
    tail = out_filepath.name + '.pdf'
    plot_nxn_cka_matrix(results, head / tail)

    ensemble_metric_results = {"arrays": {}}

    jsd = load_wanted_metric_matrix(results, "jensen_shannon_div")
    cohens_kappa = load_wanted_metric_matrix(results, 'cohens_kappa')
    ensemble_acc = load_wanted_metric_matrix(results, 'ensemble_acc')
    rel_ens_acc = calc_rel_ens_acc_matrix(results)

    ensemble_metric_results['arrays']['jsd'] = jsd.tolist()
    ensemble_metric_results['arrays']['cohens_kappa'] = cohens_kappa.tolist()
    ensemble_metric_results['arrays']['ensemble_acc'] = ensemble_acc.tolist()
    ensemble_metric_results['arrays']['rel_ens_acc'] = rel_ens_acc.tolist()

    # Load the similarity values and create mean values for the
    for i in range(2, 6):
        n_result = dict()
        n_result['cohens_kappa'] = calculate_mean_metric_for_some_n(cohens_kappa, i)
        n_result['jensen_shannon_div'] = calculate_mean_metric_for_some_n(jsd, i)
        n_result['ensemble_acc'] = calculate_mean_metric_for_some_n(ensemble_acc, i)
        n_result['relative_ensemble_acc'] = calculate_mean_metric_for_some_n(rel_ens_acc, i)
        ensemble_metric_results[i] = n_result

    save_json(ensemble_metric_results, iterative_json_path)


if __name__ == "__main__":
    main("exp_var_5_models.json")
