import numpy as np
import pytest

from llmcomp.measures import (
    aligned_cossim,
    centered_kernel_alignment,
    concentricity_difference,
    concentricity_nrmse,
    correlation_match,
    distance_correlation,
    eigenspace_overlap_score,
    geometry_score,
    gulp,
    imd_score,
    jaccard_similarity,
    joint_rank_jaccard_similarity,
    linear_reg,
    magnitude_difference,
    magnitude_nrmse,
    orthogonal_angular_shape_metric,
    orthogonal_procrustes,
    permutation_procrustes,
    pwcca,
    rank_similarity,
    representational_similarity_analysis,
    rsm_norm_diff,
    second_order_cosine_similarity,
    svcca,
    uniformity_difference,
)
from tests.conftest import (
    _test_generic_measure,
    get_distinct_reps,
    get_identical_reps,
    get_rep,
)

N_ROWS = 5
N_DIM = 2


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_aligned_cossim(rep1, rep2, expected_outcome):
    _test_generic_measure(aligned_cossim, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_centered_kernel_alignment(rep1, rep2, expected_outcome):
    _test_generic_measure(centered_kernel_alignment, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_eigenspace_overlap_score(rep1, rep2, expected_outcome):
    _test_generic_measure(eigenspace_overlap_score, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_linear_reg(rep1, rep2, expected_outcome):
    _test_generic_measure(linear_reg, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_orthogonal_procrustes(rep1, rep2, expected_outcome):
    _test_generic_measure(orthogonal_procrustes, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
        [
            get_rep(N_ROWS, N_DIM, seed=123),
            get_rep(N_ROWS, N_DIM, seed=123) @ np.array([[0, 1], [1, 0]]),
            0,
        ],
    ],
)
def test_permutation_procrustes(rep1, rep2, expected_outcome):
    _test_generic_measure(permutation_procrustes, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome,kwargs",
    [
        get_identical_reps(N_ROWS, N_DIM)
        + [1]
        + [dict(inner="correlation", outer="spearman")],
        pytest.param(
            *get_distinct_reps(N_ROWS, N_DIM),
            [1],
            [dict(inner="correlation", outer="spearman")],
            marks=pytest.mark.xfail
        ),
        get_identical_reps(N_ROWS, N_DIM)
        + [0]
        + [dict(inner="euclidean", outer="euclidean")],
        pytest.param(
            *get_distinct_reps(N_ROWS, N_DIM),
            [0],
            [dict(inner="euclidean", outer="euclidean")],
            marks=pytest.mark.xfail
        ),
    ],
)
def test_representational_similarity_analysis(rep1, rep2, expected_outcome, kwargs):
    _test_generic_measure(
        representational_similarity_analysis, rep1, rep2, expected_outcome, **kwargs
    )


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_rsm_norm_diff(rep1, rep2, expected_outcome):
    _test_generic_measure(rsm_norm_diff, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome,mode",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1] + ["soft"],
        pytest.param(
            *get_distinct_reps(N_ROWS, N_DIM), [1], ["soft"], marks=pytest.mark.xfail
        ),
        get_identical_reps(N_ROWS, N_DIM) + [1] + ["hard"],
        pytest.param(
            *get_distinct_reps(N_ROWS, N_DIM), [1], ["hard"], marks=pytest.mark.xfail
        ),
    ],
)
def test_correlation_match(rep1, rep2, expected_outcome, mode):
    _test_generic_measure(correlation_match, rep1, rep2, expected_outcome, mode=mode)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_magnitude_difference(rep1, rep2, expected_outcome):
    _test_generic_measure(magnitude_difference, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_magnitude_nrmse(rep1, rep2, expected_outcome):
    _test_generic_measure(magnitude_nrmse, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_uniformity_difference(rep1, rep2, expected_outcome):
    _test_generic_measure(uniformity_difference, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_distance_correlation(rep1, rep2, expected_outcome):
    _test_generic_measure(distance_correlation, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_orthogonal_angular_shape_metric(rep1, rep2, expected_outcome):
    _test_generic_measure(orthogonal_angular_shape_metric, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_concentricity_difference(rep1, rep2, expected_outcome):
    _test_generic_measure(concentricity_difference, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_concentricity_nrmse(rep1, rep2, expected_outcome):
    _test_generic_measure(concentricity_nrmse, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_jaccard_similarity(rep1, rep2, expected_outcome):
    _test_generic_measure(jaccard_similarity, rep1, rep2, expected_outcome, k=2)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_second_order_cosine_similarity(rep1, rep2, expected_outcome):
    _test_generic_measure(
        second_order_cosine_similarity, rep1, rep2, expected_outcome, k=2
    )


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_rank_similarity(rep1, rep2, expected_outcome):
    _test_generic_measure(rank_similarity, rep1, rep2, expected_outcome, k=2)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_joint_rank_jaccard_similarity(rep1, rep2, expected_outcome):
    _test_generic_measure(
        joint_rank_jaccard_similarity, rep1, rep2, expected_outcome, k=2
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_geometry_score(rep1, rep2, expected_outcome):
    _test_generic_measure(geometry_score, rep1, rep2, expected_outcome)


def test_imd_score():
    n_dim = 10
    n_rows = 100
    rep1, rep2 = get_identical_reps(n_rows, n_dim)
    imd_score(rep1, rep2, approximation_steps=10)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome,lmbda",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0] + [0],
        pytest.param(
            *get_distinct_reps(N_ROWS, N_DIM), [0], [0], marks=pytest.mark.xfail
        ),
        get_identical_reps(N_ROWS, N_DIM) + [0] + [1],
        pytest.param(
            *get_distinct_reps(N_ROWS, N_DIM), [0], [1], marks=pytest.mark.xfail
        ),
    ],
)
def test_gulp(rep1, rep2, expected_outcome, lmbda):
    retval = gulp(rep1, rep2, lmbda=lmbda)
    np.testing.assert_allclose(retval, expected_outcome, atol=1e-7)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_svcca(rep1, rep2, expected_outcome):
    _test_generic_measure(svcca, rep1, rep2, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_pwcca(rep1, rep2, expected_outcome):
    _test_generic_measure(pwcca, rep1, rep2, expected_outcome)
