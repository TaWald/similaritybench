import numpy as np
import pytest
from repsim.measures import AlignedCosineSimilarity
from repsim.measures import CKA
from repsim.measures import ConcentricityDifference
from repsim.measures import DistanceCorrelation
from repsim.measures import EigenspaceOverlapScore
from repsim.measures import GeometryScore
from repsim.measures import Gulp
from repsim.measures import HardCorrelationMatch
from repsim.measures import IMDScore
from repsim.measures import JaccardSimilarity
from repsim.measures import LinearRegression
from repsim.measures import MagnitudeDifference
from repsim.measures import OrthogonalAngularShapeMetricCentered
from repsim.measures import OrthogonalProcrustesCenteredAndNormalized
from repsim.measures import PermutationProcrustes
from repsim.measures import ProcrustesSizeAndShapeDistance
from repsim.measures import PWCCA
from repsim.measures import RankSimilarity
from repsim.measures import RSA
from repsim.measures import RSMNormDifference
from repsim.measures import SecondOrderCosineSimilarity
from repsim.measures import SoftCorrelationMatch
from repsim.measures import SVCCA
from repsim.measures import UniformityDifference
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
    _test_generic_measure(AlignedCosineSimilarity(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_centered_kernel_alignment(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(CKA(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_eigenspace_overlap_score(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(EigenspaceOverlapScore(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_linear_reg(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(LinearRegression(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_orthogonal_procrustes_centered_and_normalized(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(OrthogonalProcrustesCenteredAndNormalized(), rep1, rep2, shape, expected_outcome)


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
    _test_generic_measure(PermutationProcrustes(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_representational_similarity_analysis(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(RSA(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_rsm_norm_diff(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(RSMNormDifference(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_magnitude_difference(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(MagnitudeDifference(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_uniformity_difference(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(UniformityDifference(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_distance_correlation(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(DistanceCorrelation(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_orthogonal_angular_shape_metric_centered(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(OrthogonalAngularShapeMetricCentered(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_concentricity_difference(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(ConcentricityDifference(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_jaccard_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(JaccardSimilarity(), rep1, rep2, shape, expected_outcome, k=2)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_second_order_cosine_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(SecondOrderCosineSimilarity(), rep1, rep2, shape, expected_outcome, k=2)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_rank_similarity(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(RankSimilarity(), rep1, rep2, shape, expected_outcome, k=2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_geometry_score(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(GeometryScore(), rep1, rep2, shape, expected_outcome)


def test_imd_score():
    n_dim = 10
    n_rows = 100
    rep1, rep2, shape = get_identical_reps(n_rows, n_dim)
    IMDScore()(rep1, rep2, shape, approximation_steps=10)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_gulp(rep1, rep2, shape, expected_outcome):
    retval = Gulp()(rep1, rep2, shape)
    np.testing.assert_allclose(retval, expected_outcome, atol=1e-7)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_svcca(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(SVCCA(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_pwcca(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(PWCCA(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_hard_correlation_match(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(HardCorrelationMatch(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [1],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [1], marks=pytest.mark.xfail),
    ],
)
def test_soft_correlation_match(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(SoftCorrelationMatch(), rep1, rep2, shape, expected_outcome)


@pytest.mark.parametrize(
    "rep1,rep2,shape,expected_outcome",
    [
        get_identical_reps(N_ROWS, N_DIM) + [0],
        pytest.param(*get_distinct_reps(N_ROWS, N_DIM), [0], marks=pytest.mark.xfail),
    ],
)
def test_procrustes_size_and_shape_distance(rep1, rep2, shape, expected_outcome):
    _test_generic_measure(ProcrustesSizeAndShapeDistance(), rep1, rep2, shape, expected_outcome)
