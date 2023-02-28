import torch


def error_ratios(y_hats_1: torch.Tensor, y_hats_2: torch.Tensor, groundtruth: torch.Tensor) -> float:
    """Calculates the ratio between two models making the same mistake vs them making different mistakes
    :param y_hats_1: Most probable class of model 1  # N_SAMPLES
    :param y_hats_2: Most probable class of model 2  # N_SAMPLES
    :param groundtruth: Actual true class
    """

    errors_1 = y_hats_1 != groundtruth
    errors_2 = y_hats_2 != groundtruth

    both_wrong = errors_1 & errors_2
    n_both_wrong = torch.sum(both_wrong, dtype=torch.float32)
    n_both_wrong_same_way = torch.sum((y_hats_1 == y_hats_2)[both_wrong], dtype=torch.float32)
    n_both_wrong_different = n_both_wrong - n_both_wrong_same_way

    error_ratio = float(n_both_wrong_different / n_both_wrong_same_way)

    return error_ratio
