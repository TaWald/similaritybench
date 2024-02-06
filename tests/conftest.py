import numpy as np
import numpy.testing
import torch

SEED = 123


def get_rng(seed: int = SEED):
    return np.random.default_rng(seed)


def get_rep(N: int, D: int, seed: int) -> torch.Tensor:
    rep = get_rng(seed).random((N, D))
    return torch.from_numpy(rep)


def _test_generic_measure(func, rep1, rep2, shape, expected_outcome, *args, **kwargs):
    retval = func(rep1, rep2, shape, *args, **kwargs)
    assert isinstance(retval, float)
    np.testing.assert_allclose(retval, expected_outcome)


def get_identical_reps(N, D):
    rep1, rep2 = get_rep(N, D, seed=SEED), get_rep(N, D, seed=SEED)
    return [rep1, rep2, "nd"]


def get_distinct_reps(N, D):
    rep1, rep2 = get_rep(N, D, seed=SEED), get_rep(N, D, seed=SEED + 1)
    return [rep1, rep2, "nd"]
