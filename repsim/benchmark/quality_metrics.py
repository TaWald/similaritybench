from itertools import product

import numpy as np
from sklearn.metrics import average_precision_score


def violation_rate(intra_group: list[float], cross_group: list[float]) -> float:
    """
    Calculate the rate of violations, that the intra-group similarity is less than the cross-group similarity.
    Args:
        intra_group: List of intra-group similarities.
        cross_group: List of cross-group similarities.
    """
    violations = sum([in_sim <= cross_sim for in_sim, cross_sim in product(intra_group, cross_group)])
    adherence = sum([in_sim > cross_sim for in_sim, cross_sim in product(intra_group, cross_group)])
    violation_rate = violations / (violations + adherence)
    return violation_rate


def auprc(intra_group: list[float], cross_group: list[float]) -> float:
    """
    Calculate the area under the precision-recall curve.
    Args:
        intra_group: List of intra-group similarities.
        cross_group: List of cross-group similarities.
    """
    in_group_sims = np.array(intra_group)
    cross_group_sims = np.array(cross_group)
    y_true = np.concatenate([np.ones_like(in_group_sims), np.zeros_like(cross_group_sims)])
    y_score = np.concatenate([in_group_sims, cross_group_sims])
    auprc = average_precision_score(
        y_true, y_score
    )  # 1 for perfect separation, 0.5 for random, 0 for inverse separation (inverted metric)
    return auprc
