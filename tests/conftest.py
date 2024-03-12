import numpy as np
import numpy.testing
import torch
from scipy.stats import ortho_group

SEED = 123


def get_rng(seed: int = SEED):
    return np.random.default_rng(seed)


def get_rep(N: int, D: int, seed: int) -> torch.Tensor:
    rep = get_rng(seed).random((N, D))
    return torch.from_numpy(rep)


def _test_generic_measure(func, rep1, rep2, shape, expected_outcome, atol=None, *args, **kwargs):
    retval = func(rep1, rep2, shape, *args, **kwargs)
    assert isinstance(retval, float)
    if expected_outcome == 0:
        atol = 1e-7
    if atol:
        np.testing.assert_allclose(retval, expected_outcome, atol=atol)
    else:
        np.testing.assert_allclose(retval, expected_outcome)


def get_identical_reps(N, D):
    rep1, rep2 = get_rep(N, D, seed=SEED), get_rep(N, D, seed=SEED)
    return [rep1, rep2, "nd"]


def get_distinct_reps(N, D):
    rep1, rep2 = get_rep(N, D, seed=SEED), get_rep(N, D, seed=SEED + 1)
    return [rep1, rep2, "nd"]


def get_orthogonal_reps(N, D):
    rep1, rep2 = get_rep(N, D, seed=SEED), get_rep(N, D, seed=SEED)
    orth_trafo = ortho_group.rvs(D, random_state=SEED)
    rep2 = rep2 @ orth_trafo
    return [rep1, rep2, "nd"]
