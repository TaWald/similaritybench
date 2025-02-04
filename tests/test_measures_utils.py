import numpy as np
import pytest
from repsim.measures import centered_kernel_alignment
from repsim.measures.utils import adjust_dimensionality
from repsim.measures.utils import center_columns
from repsim.measures.utils import normalize_matrix_norm
from repsim.measures.utils import normalize_row_norm
from repsim.measures.utils import Pipeline
from repsim.measures.utils import to_numpy_if_needed
from tests.conftest import get_identical_reps
from tests.conftest import get_rep
from tests.conftest import SEED


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_shape",
    [
        (*get_identical_reps(5, 2), (5, 2)),
        (get_rep(5, 2, SEED), get_rep(5, 100, SEED), "nd", (5, 100)),
        (get_rep(5, 100, SEED), get_rep(5, 2, SEED), "nd", (5, 100)),
    ],
)
def test_adjust_dimensionality(rep1, rep2, shape, expected_shape):
    rep1, rep2 = to_numpy_if_needed(rep1, rep2)
    rep1, rep2 = adjust_dimensionality(rep1, rep2)
    assert rep1.shape == rep2.shape
    assert rep1.shape == expected_shape


def test_adjust_dimensionality_pad_zeros():
    rep1, rep2 = get_rep(5, 2, SEED), get_rep(5, 100, SEED)
    rep1, rep2 = to_numpy_if_needed(rep1, rep2)
    rep1, rep2 = adjust_dimensionality(rep1, rep2)
    assert rep1[0, -1] == 0


def test_to_numpy_if_needed():
    rep = to_numpy_if_needed(get_rep(5, 2, SEED))[0]
    assert isinstance(rep, np.ndarray)


def test_center_columns():
    rep = to_numpy_if_needed(get_rep(5, 20, SEED))[0]
    centered_rep = center_columns(rep)
    np.testing.assert_allclose(centered_rep.mean(axis=0), np.zeros_like(centered_rep.mean(axis=0)), atol=1e-5)


def test_normalize_matrix_norm():
    rep = to_numpy_if_needed(get_rep(5, 20, SEED))[0]
    normed_rep = normalize_matrix_norm(rep)
    np.testing.assert_almost_equal(np.linalg.norm(normed_rep, ord="fro"), 1)


def test_normalize_row_norm():
    rep = to_numpy_if_needed(get_rep(5, 20, SEED))[0]
    normed_rep = normalize_row_norm(rep)
    np.testing.assert_allclose(
        np.linalg.norm(normed_rep, ord=2, axis=1),
        np.ones_like(normed_rep.shape[0]),
        atol=1e-5,
    )


def test_Pipeline():
    rep1, rep2, shape = get_identical_reps(5, 2)
    rep1, rep2 = to_numpy_if_needed(rep1, rep2)

    pipeline = Pipeline([center_columns], centered_kernel_alignment)
    result = pipeline(rep1, rep2, shape)
    np.testing.assert_approx_equal(result, 1)
