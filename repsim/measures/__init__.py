from .cca import PWCCA
from .cca import pwcca
from .cca import SVCCA
from .cca import svcca
from .cka import centered_kernel_alignment
from .cka import CKA
from .correlation_match import hard_correlation_match
from .correlation_match import HardCorrelationMatch
from .correlation_match import soft_correlation_match
from .correlation_match import SoftCorrelationMatch
from .distance_correlation import distance_correlation
from .distance_correlation import DistanceCorrelation
from .eigenspace_overlap import eigenspace_overlap_score
from .eigenspace_overlap import EigenspaceOverlapScore
from .geometry_score import geometry_score
from .geometry_score import GeometryScore
from .gulp import Gulp
from .gulp import gulp
from .linear_regression import linear_reg
from .linear_regression import LinearRegression
from .multiscale_intrinsic_distance import imd_score
from .multiscale_intrinsic_distance import IMDScore
from .nearest_neighbor import jaccard_similarity
from .nearest_neighbor import JaccardSimilarity
from .nearest_neighbor import joint_rank_jaccard_similarity  # noqa: F401
from .nearest_neighbor import rank_similarity
from .nearest_neighbor import RankSimilarity
from .nearest_neighbor import second_order_cosine_similarity
from .nearest_neighbor import SecondOrderCosineSimilarity
from .output_similarity import Disagreement
from .output_similarity import JSD
from .procrustes import aligned_cossim
from .procrustes import AlignedCosineSimilarity
from .procrustes import orthogonal_angular_shape_metric  # noqa: F401
from .procrustes import orthogonal_angular_shape_metric_centered
from .procrustes import orthogonal_procrustes  # noqa: F401
from .procrustes import orthogonal_procrustes_centered_and_normalized
from .procrustes import OrthogonalAngularShapeMetricCentered
from .procrustes import OrthogonalProcrustesCenteredAndNormalized
from .procrustes import permutation_aligned_cossim  # noqa: F401
from .procrustes import permutation_angular_shape_metric  # noqa: F401
from .procrustes import permutation_procrustes
from .procrustes import PermutationProcrustes
from .procrustes import procrustes_size_and_shape_distance
from .procrustes import ProcrustesSizeAndShapeDistance
from .rsa import representational_similarity_analysis
from .rsa import RSA
from .rsm_norm_difference import rsm_norm_diff
from .rsm_norm_difference import RSMNormDifference
from .statistics import concentricity_difference
from .statistics import concentricity_nrmse  # noqa: F401
from .statistics import ConcentricityDifference
from .statistics import magnitude_difference
from .statistics import magnitude_nrmse  # noqa: F401
from .statistics import MagnitudeDifference
from .statistics import uniformity_difference
from .statistics import UniformityDifference

# Some measures are not listed via any of the following constants: joint_rank_jaccard_similarity, nrmse of statistics

# Measures in this list follow the preprocessing as proposed in the original papers
# Some measures that are symmetric, but currently not implemented: RTD, Soft-permutation alignment (Alex Williams und co, Unireps 2023), the CNN similarity measure from Neurips 2023
SYMMETRIC_MEASURES = [
    magnitude_difference,
    concentricity_difference,
    uniformity_difference,
    rsm_norm_diff,
    eigenspace_overlap_score,
    aligned_cossim,
    procrustes_size_and_shape_distance,  # williams version of orthogonal procrustes
    orthogonal_procrustes_centered_and_normalized,  # ding version of orthogonal procrustes
    permutation_procrustes,
    representational_similarity_analysis,
    centered_kernel_alignment,
    hard_correlation_match,
    soft_correlation_match,
    distance_correlation,
    orthogonal_angular_shape_metric_centered,  # williams version of ortho angular shape metric
    jaccard_similarity,  # gwilliam 2022 and hyrniowski 2022 have the essentially the same setup (euclidean)
    second_order_cosine_similarity,
    rank_similarity,
    geometry_score,
    imd_score,
    gulp,  # different lambdas?
    svcca,
]

ASYMMETRIC_MEASURES = [pwcca, linear_reg]

CLASSES = [
    PWCCA,
    SVCCA,
    HardCorrelationMatch,
    SoftCorrelationMatch,
    DistanceCorrelation,
    EigenspaceOverlapScore,
    GeometryScore,
    IMDScore,
    Gulp,
    LinearRegression,
    JaccardSimilarity,
    RankSimilarity,
    SecondOrderCosineSimilarity,
    AlignedCosineSimilarity,
    OrthogonalAngularShapeMetricCentered,
    OrthogonalProcrustesCenteredAndNormalized,
    PermutationProcrustes,
    ProcrustesSizeAndShapeDistance,
    RSA,
    RSMNormDifference,
    ConcentricityDifference,
    MagnitudeDifference,
    UniformityDifference,
    CKA,
]

ALL_MEASURES = {m().name: m() for m in CLASSES}

FUNCTIONAL_SIMILARITY_MEASURES = {m().name: m() for m in [JSD, Disagreement]}
