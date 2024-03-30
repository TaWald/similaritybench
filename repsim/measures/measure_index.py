from repsim.measures.cca import PWCCA
from repsim.measures.cca import SVCCA
from repsim.measures.cka import CKA
from repsim.measures.correlation_match import HardCorrelationMatch
from repsim.measures.correlation_match import SoftCorrelationMatch
from repsim.measures.distance_correlation import DistanceCorrelation
from repsim.measures.eigenspace_overlap import EigenspaceOverlapScore
from repsim.measures.geometry_score import GeometryScore
from repsim.measures.gulp import Gulp
from repsim.measures.linear_regression import LinearRegression
from repsim.measures.multiscale_intrinsic_distance import IMDScore
from repsim.measures.nearest_neighbor import JaccardSimilarity
from repsim.measures.nearest_neighbor import NearestNeighborSimilarityFunction
from repsim.measures.nearest_neighbor import RankSimilarity
from repsim.measures.nearest_neighbor import SecondOrderCosineSimilarity
from repsim.measures.procrustes import AlignedCosineSimilarity
from repsim.measures.procrustes import OrthogonalAngularShapeMetricCentered
from repsim.measures.procrustes import OrthogonalProcrustesCenteredAndNormalized
from repsim.measures.procrustes import PermutationProcrustes
from repsim.measures.procrustes import ProcrustesSizeAndShapeDistance
from repsim.measures.rsa import RSA
from repsim.measures.rsm_norm_difference import RSMNormDifference
from repsim.measures.statistics import ConcentricityDifference
from repsim.measures.statistics import MagnitudeDifference
from repsim.measures.statistics import UniformityDifference
from repsim.measures.utils import SimilarityMeasure

# Potentially rework this registry.
#   Naming of metrics should be the same as the class, not the same as the function, for better readability.
ALL_SIMILARITY_MEASURES = [
    # cka,
    CKA,
    # correlation_match,
    HardCorrelationMatch,
    SoftCorrelationMatch,
    # distance_correlation
    DistanceCorrelation,
    # eigenspace_overlap
    EigenspaceOverlapScore,
    # geometry_score
    GeometryScore,
    # gulp
    Gulp,
    # linear regression
    LinearRegression,
    # multiscale_intrinsic_distance
    IMDScore,
    # nearest_neighbor
    JaccardSimilarity,
    SecondOrderCosineSimilarity,
    RankSimilarity,
    # Procrustes
    # procrustes_size_and_shape_distance,
    ProcrustesSizeAndShapeDistance,
    # orthogonal_procrustes_centered_and_normalized,
    OrthogonalProcrustesCenteredAndNormalized,
    # permutation_procrustes,
    PermutationProcrustes,
    #  permutation_angular_shape_metric, <-- No Class?
    # orthogonal_angular_shape_metric_centered,
    OrthogonalAngularShapeMetricCentered,
    # aligned_cossim,
    AlignedCosineSimilarity,
    # permutation_aligned_cossim, <-- No Class?
    # rsa
    RSA,
    # representational_similarity_analysis,
    # rsm_norm_diff
    RSMNormDifference,
    # statistics
    MagnitudeDifference,
    # magnitude_nrmse,
    UniformityDifference,
    ConcentricityDifference,
    # concentricity_nrmse,
    # cca
    SVCCA,
    PWCCA,
]

ALL_MEASURES: dict[str, SimilarityMeasure] = {}
for m in ALL_SIMILARITY_MEASURES:
    ALL_MEASURES[m().name] = m()
