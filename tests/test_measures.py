import numpy as np
import pytest
from repsim.measures import aligned_cossim
from repsim.measures import centered_kernel_alignment
from repsim.measures import concentricity_difference
from repsim.measures import concentricity_nrmse
from repsim.measures import distance_correlation
from repsim.measures import eigenspace_overlap_score
from repsim.measures import geometry_score
from repsim.measures import gulp
from repsim.measures import hard_correlation_match
from repsim.measures import imd_score
from repsim.measures import jaccard_similarity
from repsim.measures import joint_rank_jaccard_similarity
from repsim.measures import linear_reg
from repsim.measures import magnitude_difference
from repsim.measures import magnitude_nrmse
from repsim.measures import orthogonal_angular_shape_metric
from repsim.measures import orthogonal_procrustes
from repsim.measures import orthogonal_procrustes_centered_and_normalized
from repsim.measures import permutation_procrustes
from repsim.measures import procrustes_size_and_shape_distance
from repsim.measures import pwcca
from repsim.measures import rank_similarity
from repsim.measures import representational_similarity_analysis
from repsim.measures import rsm_norm_diff
from repsim.measures import second_order_cosine_similarity
from repsim.measures import soft_correlation_match
from repsim.measures import svcca
from repsim.measures import uniformity_difference
from tests.conftest import _test_generic_measure
from tests.conftest import get_distinct_reps
from tests.conftest import get_identical_reps
from tests.conftest import get_rep

N_ROWS = 5
N_DIM = 2


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_aligned_cossim(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(aligned_cossim, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_centered_kernel_alignment(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(centered_kernel_alignment, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_eigenspace_overlap_score(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(eigenspace_overlap_score, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_linear_reg(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(linear_reg, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_orthogonal_procrustes(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(orthogonal_procrustes, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
        [
            get_rep(N_ROWS, N_DIM, seed=123),
            get_rep(N_ROWS, N_DIM, seed=123) @ np.array([[0, 1], [1, 0]]),
            "nd",
            0,
        ],
    ],
)
def test_permutation_procrustes(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(permutation_procrustes, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome,kwargs",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1] + [dict(inner="correlation", outer="spearman")],
        pytest.param(
            *get_distinct_reps(N_ROWS, N_DIM),
            [1],
            [dict(inner="correlation", outer="spearman")],
            marks=pytest.mark.xfail
        ),
        get_identical_reps(N_ROWS, N_DIM) + [0] + [dict(inner="euclidean", outer="euclidean")],
        pytest.param(
            *get_distinct_reps(N_ROWS, N_DIM),
            [0],
            [dict(inner="euclidean", outer="euclidean")],
            marks=pytest.mark.xfail
        ),
    ],
)
def test_representational_similarity_analysis(rep1, rep2, shape, expected_outcome, kwargs):
    _test_generic_measure(representational_similarity_analysis, rep1, rep2, shape, expected_outcome, **kwargs)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_rsm_norm_diff(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(rsm_norm_diff, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_magnitude_difference(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(magnitude_difference, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_magnitude_nrmse(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(magnitude_nrmse, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_uniformity_difference(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(uniformity_difference, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_distance_correlation(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(distance_correlation, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_orthogonal_angular_shape_metric(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(orthogonal_angular_shape_metric, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_concentricity_difference(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(concentricity_difference, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_concentricity_nrmse(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(concentricity_nrmse, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_jaccard_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(jaccard_similarity, rep1, rep2, shape, expected_outcome, k=2)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_second_order_cosine_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(second_order_cosine_similarity, rep1, rep2, shape, expected_outcome, k=2)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_rank_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(rank_similarity, rep1, rep2, shape, expected_outcome, k=2)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_joint_rank_jaccard_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(joint_rank_jaccard_similarity, rep1, rep2, shape, expected_outcome, k=2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_geometry_score(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(geometry_score, rep1, rep2, shape, expected_outcome)


def test_imd_score():
    n_dim = 10
    n_rows = 100
    rep1, rep2, shape = get_identical_reps(n_rows, n_dim)
    imd_score(rep1, rep2, shape, approximation_steps=10)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome,lmbda",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0] + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], [0], marks=pytest.mark.xfail),
        get_identical_reps(N_ROWS, N_DIM) + [0] + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], [1], marks=pytest.mark.xfail),
    ],
)
def test_gulp(rep1, rep2, shape, expected_outcome, lmbda):
    retval = gulp(rep1, rep2, shape, lmbda=lmbda)
    np.testing.assert_allclose(retval, expected_outcome, atol=1e-7)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_svcca(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(svcca, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_pwcca(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(pwcca, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_hard_correlation_match(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(hard_correlation_match, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_soft_correlation_match(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(soft_correlation_match, rep1, rep2, shape, expected_outcome)


@pytest.mark.special
@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_orthogonal_procrustes_centered_and_normalized(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(orthogonal_procrustes_centered_and_normalized, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_procrustes_size_and_shape_distance(rep1, rep2, shape, expected_outcome):
    retval = procrustes_size_and_shape_distance(rep1, rep2, shape)
    assert isinstance(retval, float)
    np.testing.assert_allclose(retval, expected_outcome, atol=1e-7)
