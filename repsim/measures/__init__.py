from .cca import pwcca
from .cca import svcca
from .cka import centered_kernel_alignment
from .correlation_match import correlation_match
from .distance_correlation import distance_correlation
from .eigenspace_overlap import eigenspace_overlap_score
from .geometry_score import geometry_score
from .gulp import gulp
from .linear_regression import linear_reg
from .multiscale_intrinsic_distance import imd_score
from .nearest_neighbor import jaccard_similarity
from .nearest_neighbor import joint_rank_jaccard_similarity
from .nearest_neighbor import rank_similarity
from .nearest_neighbor import second_order_cosine_similarity
from .procrustes import aligned_cossim
from .procrustes import orthogonal_angular_shape_metric
from .procrustes import orthogonal_procrustes
from .procrustes import permutation_aligned_cossim
from .procrustes import permutation_angular_shape_metric
from .procrustes import permutation_procrustes
from .rsa import representational_similarity_analysis
from .rsm_norm_difference import rsm_norm_diff
from .statistics import concentricity_difference
from .statistics import concentricity_nrmse
from .statistics import magnitude_difference
from .statistics import magnitude_nrmse
from .statistics import uniformity_difference

MEASURES = [
    rsm_norm_diff,
    eigenspace_overlap_score,
    linear_reg,
    aligned_cossim,
    orthogonal_procrustes,
    permutation_procrustes,
    permutation_angular_shape_metric,
    permutation_aligned_cossim,
    representational_similarity_analysis,
    centered_kernel_alignment,
    correlation_match,
    magnitude_difference,
    magnitude_nrmse,
    concentricity_difference,
    concentricity_nrmse,
    distance_correlation,
    orthogonal_angular_shape_metric,
    uniformity_difference,
    jaccard_similarity,
    second_order_cosine_similarity,
    rank_similarity,
    joint_rank_jaccard_similarity,
    geometry_score,
    imd_score,
    gulp,
    pwcca,
    svcca,
]
