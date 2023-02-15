import random
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rep_trans.comp.cca_core import get_cca_similarity
from rep_trans.comp.cca_pw import compute_pwcca
from scipy.stats import ortho_group
from sklearn import cross_decomposition


def calculate_cca(actis_a: np.ndarray, actis_b: np.ndarray):
    cca = cross_decomposition.CCA(n_components=min(actis_a.shape[1], actis_b.shape[1]))
    cca.fit(actis_a, actis_b)
    trans_a, trans_b = cca.transform(actis_a, actis_b)
    trans_a /= np.std(trans_a, axis=0)
    trans_b /= np.std(trans_b, axis=0)

    correlations = np.mean(trans_a * trans_b, axis=0)
    mean_correlation = np.mean(correlations)
    return mean_correlation, correlations


def calculate_svcca(actis_a: np.ndarray, actis_b: np.ndarray, threshold: float = 0.98):
    res = get_cca_similarity(actis_a.T, actis_b.T, threshold=threshold)
    return res["mean"][0]


def calculate_pwcca(actis_a: np.ndarray, actis_b: np.ndarray):
    sim, weights, coef = compute_pwcca(actis_a.T, actis_b.T)
    return sim


def calculate_lincka(actis_a: np.ndarray, actis_b: np.ndarray):
    actis_a = actis_a - np.mean(actis_a, axis=0, keepdims=True)
    actis_b = actis_b - np.mean(actis_b, axis=0, keepdims=True)

    numerator = np.linalg.norm(np.matmul(actis_b.T, actis_a), ord="fro") ** 2
    denominator = np.linalg.norm(np.matmul(actis_a.T, actis_a), ord="fro") * np.linalg.norm(
        np.matmul(actis_b.T, actis_b), ord="fro"
    )
    return numerator / denominator


def signal_noise_channel_ratio_toy_exp(output_path: Path):
    n_channels = 200
    n_samples = 2000
    signal_std = 1.0  # std of the signal
    noise_std = 0.25
    monte_carlo_runs = 15
    results = []
    for signal_to_noise_channel_ratio in np.linspace(0, 1, 20):
        n_signal = int(signal_to_noise_channel_ratio * n_channels)
        n_noise = n_channels - n_signal

        signal = np.random.normal(loc=0, scale=signal_std, size=[n_samples, n_signal])
        noise_1 = np.random.normal(loc=0, scale=noise_std, size=[n_samples, n_noise])
        noise_2 = np.random.normal(loc=0, scale=noise_std, size=[n_samples, n_noise])

        all_channels_1 = np.concatenate([signal, noise_1], axis=1)
        all_channels_2 = np.concatenate([signal, noise_2], axis=1)

        for j in range(monte_carlo_runs):
            trans1 = ortho_group.rvs(dim=int(all_channels_1.shape[1]))
            trans2 = ortho_group.rvs(dim=int(all_channels_2.shape[1]))

            reps_1 = all_channels_1.dot(trans1)
            reps_2 = all_channels_2.dot(trans2)

            unweighted_cca_results = calculate_svcca(reps_1, reps_2, threshold=1.0)
            svcca_results = calculate_svcca(reps_1, reps_2, threshold=0.99)
            pwcca_similarity = calculate_pwcca(reps_1, reps_2)
            lincka_similarity = calculate_lincka(reps_1, reps_2)
            results.extend(
                [
                    {
                        "SN Channel ratio": signal_to_noise_channel_ratio,
                        "similarity": unweighted_cca_results,
                        "method": "unfiltered CCA",
                    },
                    {
                        "SN Channel ratio": signal_to_noise_channel_ratio,
                        "similarity": svcca_results,
                        "method": "SVCCA",
                    },
                    {
                        "SN Channel ratio": signal_to_noise_channel_ratio,
                        "similarity": pwcca_similarity,
                        "method": "PWCCA",
                    },
                    {
                        "SN Channel ratio": signal_to_noise_channel_ratio,
                        "similarity": lincka_similarity,
                        "method": "LinCKA",
                    },
                ]
            )
    res = pd.DataFrame(results)
    sns.lineplot(data=res, x="SN Channel ratio", y="similarity", hue="method")
    plt.show()
    plt.savefig(output_path / "SN_channel_ratio_exp.png")
    return


def main():
    # Different Cases to be considered:
    #   - Pruning channels removes channels (which are not important for the downstream task)
    #   - Pruning sparsely removes additive noise (lower noise on signal channels)
    # Identical "important" features
    # Case 1: High SNR with varying amounts of Signal Channels.
    # Case 2: Low SNR with varying amounts of Signal Channels

    # Partially overlapping "important" / High SNR features
    #   Some Models have shared features to begin with, but "unlearn" to rely on them
    #   during the training process and only rely on a Subset.

    # These features remain in the representations and make the representations more similar than expected

    # Case 3: High SNR features with varying amounts of Signal Channels
    #       Partially remove the high SNR "signal" channel as well
    # Case 4: High SNR features with

    # Same Signal to Noise ratio?

    monte_carlo_runs = 15

    # --------------------------------------------------------------------------
    # Case 0: (Same important signal channels & independent noise -- noise with small magnitude)
    n_channels = 200
    n_samples = 2000
    signal_std = 1.0  # std of the signal
    noise_std = 0.1

    out_path = Path("/home/tassilowald/Data/Results/exp_nips_22/visualizations")

    signal_noise_channel_ratio_toy_exp(output_path=out_path)

    # --------------------------------------------------------------------------
    # Case 1: (Same important signal channels & independent noise -- same magnitude)
    n_channels = 200
    n_samples = 2000
    n_joint_relevant = 100
    n_joint_irrelevant = 0  # Each has 25 overlapping that remain but are unused
    n_joint = n_joint_relevant + n_joint_irrelevant
    signal_std = 1.0  # std of the signal
    noise_std = 1.0
    joint_channels = np.random.normal(
        loc=0, scale=signal_std, size=[n_samples, n_joint_relevant + n_joint_irrelevant]
    )

    # Create unique noise channels
    noise_channels_1 = np.random.normal(loc=0, scale=noise_std, size=[n_samples, n_channels - n_joint])
    noise_channels_2 = np.random.normal(loc=0, scale=noise_std, size=[n_samples, n_channels - n_joint])

    all_channels_1 = np.concatenate([joint_channels, noise_channels_1], axis=1)
    all_channels_2 = np.concatenate([joint_channels, noise_channels_2], axis=1)

    n_joint_unpruneable = list(np.arange(100))  # First 100 are signal and important
    unpruneable_signal_channels_1 = n_joint_unpruneable
    unpruneable_signal_channels_2 = n_joint_unpruneable

    n_pruneable = n_channels - len(unpruneable_signal_channels_1)
    results = []
    for j in range(monte_carlo_runs):
        for rem_pct in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]:
            remaining_channels_1 = unpruneable_signal_channels_1 + random.sample(
                list(set(np.arange(n_channels)) - set(unpruneable_signal_channels_1)),
                int(n_pruneable * (float(rem_pct) / 100)),
            )
            remaining_channels_2 = unpruneable_signal_channels_2 + random.sample(
                list(set(np.arange(n_channels)) - set(unpruneable_signal_channels_2)),
                int(n_pruneable * (float(rem_pct) / 100)),
            )

            filtered_ch_1 = np.copy(all_channels_1[:, remaining_channels_1])
            filtered_ch_2 = np.copy(all_channels_2[:, remaining_channels_2])

            trans1 = ortho_group.rvs(dim=int(filtered_ch_1.shape[1]))
            trans2 = ortho_group.rvs(dim=int(filtered_ch_2.shape[1]))

            reps_1 = filtered_ch_1.dot(trans1)
            reps_2 = filtered_ch_2.dot(trans2)

            mean_cca, _ = calculate_cca(reps_1, reps_2)
            svcca_no_cutoff = calculate_svcca(reps_1, reps_2, threshold=1.0)
            svcca_results = calculate_svcca(reps_1, reps_2)
            pwcca_similarity = calculate_pwcca(reps_1, reps_2)
            lincka_similarity = calculate_lincka(reps_1, reps_2)
            results.extend(
                [
                    {"pruned_percent": 100 - rem_pct, "similarity": svcca_no_cutoff, "method": "SVCCA no thres"},
                    {"pruned_percent": 100 - rem_pct, "similarity": svcca_results, "method": "SVCCA thres=0.98"},
                    {"pruned_percent": 100 - rem_pct, "similarity": pwcca_similarity, "method": "PWCCA"},
                    {"pruned_percent": 100 - rem_pct, "similarity": lincka_similarity, "method": "LinCKA"},
                ]
            )
    res = pd.DataFrame(results)
    sns.lineplot(data=res, x="pruned_percent", y="similarity", hue="method")
    plt.show()
    plt.close()

    # ToDo: Two cases:
    #    Case A: Reducing the overall noise of the channels (Sparse Noise transform)
    #    Case B: Reducing the number of the channels (Lower dimensional orthonormal transform)


if __name__ == "__main__":
    main()
