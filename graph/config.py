import multiprocessing
import os

from repsim.measures import aligned_cossim
from repsim.measures import centered_kernel_alignment
from repsim.measures import correlation_match
from repsim.measures import distance_correlation
from repsim.measures import eigenspace_overlap_score
from repsim.measures import geometry_score
from repsim.measures import gulp
from repsim.measures import imd_score
from repsim.measures import jaccard_similarity
from repsim.measures import joint_rank_jaccard_similarity
from repsim.measures import linear_reg
from repsim.measures import orthogonal_angular_shape_metric
from repsim.measures import orthogonal_procrustes
from repsim.measures import pwcca
from repsim.measures import rank_similarity
from repsim.measures import representational_similarity_analysis
from repsim.measures import rsm_norm_diff
from repsim.measures import second_order_cosine_similarity
from repsim.measures import svcca
from repsim.measures.utils import center_columns
from repsim.measures.utils import normalize_matrix_norm
from repsim.measures.utils import normalize_row_norm
from torch_geometric.nn.models import GAT
from torch_geometric.nn.models import GCN
from torch_geometric.nn.models import GraphSAGE

# ----------------------------------------------------------------------------------------------------------------------
# GENERAL PATH-RELATED VARIABLES
# ----------------------------------------------------------------------------------------------------------------------

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(MAIN_DIR, "data")

# output-related paths - make sure they exist here
OUTPUT_DIR = os.path.join(MAIN_DIR, "output")
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

RES_DIR = os.path.join(OUTPUT_DIR, "results")
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

SIMILARITIES_FILE_NAME = "similarities.pkl"
TEST_RESULTS_JSON_NAME = "results.json"


def TORCH_STATE_DICT_FILE_NAME_AT_SEED(s):
    return f"model_s{s}.pt"


NUM_CORES = multiprocessing.cpu_count()


MEASURE_DICT_FUNC_KEY = "func"
MEASURE_DICT_PREP_KEY = "prep"

MEASURE_DICT = {
    "rsm-norm": {MEASURE_DICT_FUNC_KEY: rsm_norm_diff, MEASURE_DICT_PREP_KEY: []},
    "eos": {MEASURE_DICT_FUNC_KEY: eigenspace_overlap_score, MEASURE_DICT_PREP_KEY: []},
    "lin-reg": {MEASURE_DICT_FUNC_KEY: linear_reg, MEASURE_DICT_PREP_KEY: []},
    "cos-sim": {MEASURE_DICT_FUNC_KEY: aligned_cossim, MEASURE_DICT_PREP_KEY: []},
    "procrustes": {MEASURE_DICT_FUNC_KEY: orthogonal_procrustes, MEASURE_DICT_PREP_KEY: []},
    "ang-shape": {
        MEASURE_DICT_FUNC_KEY: orthogonal_angular_shape_metric,
        MEASURE_DICT_PREP_KEY: [normalize_matrix_norm],
    },
    "rsa": {MEASURE_DICT_FUNC_KEY: representational_similarity_analysis, MEASURE_DICT_PREP_KEY: []},
    "cka": {MEASURE_DICT_FUNC_KEY: centered_kernel_alignment, MEASURE_DICT_PREP_KEY: [center_columns]},
    "corr-match": {MEASURE_DICT_FUNC_KEY: correlation_match, MEASURE_DICT_PREP_KEY: [center_columns]},
    "cdor": {MEASURE_DICT_FUNC_KEY: distance_correlation, MEASURE_DICT_PREP_KEY: []},
    "jac": {MEASURE_DICT_FUNC_KEY: jaccard_similarity, MEASURE_DICT_PREP_KEY: []},
    "2nd-cos": {MEASURE_DICT_FUNC_KEY: second_order_cosine_similarity, MEASURE_DICT_PREP_KEY: []},
    "rank-sim": {MEASURE_DICT_FUNC_KEY: rank_similarity, MEASURE_DICT_PREP_KEY: []},
    "jac-rank": {MEASURE_DICT_FUNC_KEY: joint_rank_jaccard_similarity, MEASURE_DICT_PREP_KEY: []},
    "gs": {MEASURE_DICT_FUNC_KEY: geometry_score, MEASURE_DICT_PREP_KEY: []},
    "imd": {MEASURE_DICT_FUNC_KEY: imd_score, MEASURE_DICT_PREP_KEY: []},
    "gulp": {MEASURE_DICT_FUNC_KEY: gulp, MEASURE_DICT_PREP_KEY: [center_columns, normalize_row_norm]},
    "pwcca": {MEASURE_DICT_FUNC_KEY: pwcca, MEASURE_DICT_PREP_KEY: [center_columns]},
    "svcca": {MEASURE_DICT_FUNC_KEY: svcca, MEASURE_DICT_PREP_KEY: [center_columns]},
}

MEASURE_LIST = MEASURE_DICT.keys()

# DEFAULT_SEEDS = [1]
DEFAULT_SEEDS = list(range(1, 11))

GNN_PARAMS_DEFAULT_DIMENSION = 128
GNN_PARAMS_DEFAULT_N_LAYERS = 2
GNN_PARAMS_DEFAULT_DROPOUT = 0.5
GNN_PARAMS_DEFAULT_LR = 0.01
GNN_PARAMS_DEFAULT_N_EPOCHS = 500

GNN_DICT = {"gcn": GCN, "sage": GraphSAGE, "gat": GAT}

GNN_LIST = list(GNN_DICT.keys())

DATASET_LIST = {"ogbn-arxiv"}

LAYER_TEST_NAME = "layer_test"
LAYER_TEST_N_LAYERS = 6
