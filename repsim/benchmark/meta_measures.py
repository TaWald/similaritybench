import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr


def layerwise_forward_sim(sim: npt.NDArray) -> float:
    """Calculate the spearman rank correlation of the similarity to the layers"""
    aranged_1 = np.arange(sim.shape[0])[:, None]
    aranged_2 = np.arange(sim.shape[0])[None, :]
    dist = np.abs(aranged_1 - aranged_2)

    forward_corrs = []
    backward_corrs = []
    for i in range(sim.shape[0]):
        current_line_sim = sim[i]
        current_line_dist = dist[i]
        forward_sims = current_line_sim[i:]
        backward_sims = current_line_sim[:i]
        forward_dists = current_line_dist[i:]
        backward_dists = current_line_dist[:i]
        if len(forward_sims) > 1:
            corr, _ = spearmanr(forward_sims, forward_dists)
            forward_corrs.append(corr)
        if len(backward_sims) > 1:
            corr, _ = spearmanr(backward_sims, backward_dists)
            backward_corrs.append(corr)

    return np.nanmean(forward_corrs + backward_corrs)


def intra_group_meta_accuracy(sim: np.ndarray, higher_value_more_similar: bool = True) -> float:
    """Calculate the spearman rank correlation of the similarity to the layers"""

    def _x_more_similar_than_y(x, y):
        if higher_value_more_similar:
            return x > y
        else:
            return x < y

    n_rows, n_cols = sim.shape

    n_violations = 0
    n_comb_count = 0

    for i in range(n_rows):
        for j in range(i + 1, n_cols):

            for k in range(i, j):
                for l in range(k + 1, j + 1):
                    n_comb_count += 1
                    # if sim[i, j] < sim[k, l]:
                    if _x_more_similar_than_y(sim[i, j], sim[k, l]):
                        n_violations += 1

    return 1 - n_violations / n_comb_count
