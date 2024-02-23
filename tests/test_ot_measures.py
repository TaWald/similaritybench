import numpy as np
import pytest
from repsim.measures import aligned_cossim
from repsim.measures import centered_kernel_alignment
from repsim.measures import concentricity_difference
from repsim.measures import correlation_match
from repsim.measures import distance_correlation
from repsim.measures import eigenspace_overlap_score
from repsim.measures import geometry_score
from repsim.measures import gulp
from repsim.measures import imd_score
from repsim.measures import jaccard_similarity
from repsim.measures import joint_rank_jaccard_similarity
from repsim.measures import linear_reg
from repsim.measures import magnitude_difference
from repsim.measures import magnitude_nrmse
from repsim.measures import orthogonal_angular_shape_metric
from repsim.measures import orthogonal_procrustes
from repsim.measures import permutation_procrustes
from repsim.measures import rank_similarity
from repsim.measures import representational_similarity_analysis
from repsim.measures import rsm_norm_diff
from repsim.measures import second_order_cosine_similarity
from repsim.measures import svcca
from repsim.measures import uniformity_difference
from tests.conftest import _test_generic_measure
from tests.conftest import get_distinct_reps
from tests.conftest import get_identical_reps
from tests.conftest import get_orthogonal_reps

N_ROWS = 20
N_DIM = 2


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_svcca(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(svcca, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_cka(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(centered_kernel_alignment, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_distance_correlation(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(distance_correlation, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_linear_reg(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(linear_reg, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_jaccard_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(jaccard_similarity, rep1, rep2, shape, expected_outcome, k=2)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome,lmbda",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [0] + [0],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [0], [0], marks=pytest.mark.xfail),
        get_identical_reps(N_ROWS, N_DIM) + [0] + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_gulp(rep1, rep2, shape, expected_outcome, lmbda):
    retval = gulp(rep1, rep2, shape, lmbda=lmbda)
    np.testing.assert_allclose(retval, expected_outcome, atol=1e-7)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_aligned_cossim(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(aligned_cossim, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_second_order_cosine_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(second_order_cosine_similarity, rep1, rep2, shape, expected_outcome, k=2)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_eigenspace_overlap_score(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(eigenspace_overlap_score, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_geometry_score(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(geometry_score, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(20, 3) + [0],
        pytest.param(*get_orthogonal_reps(20, 3), [0], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_rsm_norm_diff(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(rsm_norm_diff, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_rank_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(rank_similarity, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_joint_rank_jaccard_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(joint_rank_jaccard_similarity, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_uniformity_difference(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(uniformity_difference, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_magnitude_difference(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(magnitude_difference, rep1, rep2, shape, expected_outcome)


def test_imd_score():
    n_dim = 10
    n_rows = 100
    rep1, rep2, shape = get_orthogonal_reps(n_rows, n_dim)
    imd_score(rep1, rep2, shape, approximation_steps=10)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_orthogonal_procrustes(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(orthogonal_procrustes, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_orthogonal_angular_shape_metric(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(orthogonal_angular_shape_metric, rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_ortho_inv_concentricity_difference(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(concentricity_difference, rep1, rep2, shape, expected_outcome)


# @pytest.mark.parametrize(
#     "rep1,rep2,shape",
#     [
#         get_orthogonal_reps(100, 5),
#         pytest.param(*get_orthogonal_reps(100, 5), marks=pytest.mark.xfail),
#     ],
# )
# def test_not_ortho_inv_pwcca(rep1, rep2, shape):
#     sim = pwcca(rep1, rep2, shape)
#     assert not np.isclose(sim, 1)  # Shouldn't be inv to orthogonal trafos


@pytest.mark.parametrize(
    "rep1,rep2,shape",
    [
        get_orthogonal_reps(N_ROWS, N_DIM),
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), marks=pytest.mark.xfail),
    ],
)
def test_not_ortho_inv_rsa(rep1, rep2, shape):
    sim = representational_similarity_analysis(rep1, rep2, shape)
    assert not np.isclose(sim, 1)  # Shouldn't be inv to orthogonal trafos


@pytest.mark.parametrize(
    "rep1,rep2,shape,mode",
    [
        get_orthogonal_reps(N_ROWS, N_DIM) + ["soft"],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), ["soft"], marks=pytest.mark.xfail),
        get_orthogonal_reps(N_ROWS, N_DIM) + ["hard"],
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), ["hard"], marks=pytest.mark.xfail),
    ],
)
def test_not_ortho_correlation_match(rep1, rep2, shape, mode):
    sim = correlation_match(rep1, rep2, shape, mode)
    assert not np.isclose(sim, 1)  # Shouldn't be inv to orthogonal trafos


@pytest.mark.parametrize(
    "rep1,rep2,shape",
    [
        get_orthogonal_reps(N_ROWS, N_DIM),
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), marks=pytest.mark.xfail),
    ],
)
def test_not_ortho_permutation_procrustes(rep1, rep2, shape):
    sim = permutation_procrustes(rep1, rep2, shape)
    assert not np.isclose(sim, 1)  # Shouldn't be inv to orthogonal trafos


@pytest.mark.parametrize(
    "rep1,rep2,shape",
    [
        get_orthogonal_reps(N_ROWS, N_DIM),
        pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), marks=pytest.mark.xfail),
    ],
)
def test_not_ortho_magnitude_nrmse(rep1, rep2, shape):
    sim = magnitude_nrmse(rep1, rep2, shape)
    assert not np.isclose(sim, 1)  # Shouldn't be inv to orthogonal trafos


# @pytest.mark.parametrize(
#     "rep1,rep2,shape,expected_outcome",
#     [
#         get_orthogonal_reps(N_ROWS, N_DIM) + [1],
#         pytest.param(*get_orthogonal_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
#     ],
# )
# def test_ortho_inv_concentricity_nrmse(rep1, rep2, shape, expected_outcome):
#     _test_generic_measure(concentricity_nrmse, rep1, rep2, shape, expected_outcome)
