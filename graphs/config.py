import multiprocessing
import os

from graphs.gnn import GCN
from graphs.gnn import SAGE
from repsim.benchmark.types_globals import ARXIV_DATASET
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_SEED
from repsim.benchmark.types_globals import LABEL_TEST_NAME
from repsim.benchmark.types_globals import LAYER_TEST_NAME
from repsim.benchmark.types_globals import SETTING_IDENTIFIER

# from torch_geometric.nn.models import GAT
# from torch_geometric.nn.models import GCN
# from torch_geometric.nn.models import GraphSAGE

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


def TORCH_STATE_DICT_FILE_NAME_SEED(sd: GRAPH_EXPERIMENT_SEED):
    return f"model_s{sd}.pt"


def TORCH_STATE_DICT_FILE_NAME_SETTING_SEED(st: SETTING_IDENTIFIER, sd: GRAPH_EXPERIMENT_SEED):
    return f"model_{st}_s{sd}.pt"


def TRAIN_LOG_FILE_NAME_SEED(sd: GRAPH_EXPERIMENT_SEED):
    return f"train_results_s{sd}.pt"


def TRAIN_LOG_FILE_NAME_SETTING_SEED(st: SETTING_IDENTIFIER, sd: GRAPH_EXPERIMENT_SEED):
    return f"train_results_{st}_s{sd}.pt"


NUM_CORES = multiprocessing.cpu_count()

MEASURE_DICT_FUNC_KEY = "func"
MEASURE_DICT_PREP_KEY = "prep"


# DEFAULT_SEEDS = [1]

GNN_PARAMS_DEFAULT_DIMENSION = 128
GNN_PARAMS_DEFAULT_N_LAYERS = 3
GNN_PARAMS_DEFAULT_DROPOUT = 0.5
GNN_PARAMS_DEFAULT_LR = 0.01
GNN_PARAMS_DEFAULT_N_EPOCHS = 500
GNN_PARAMS_DEFAULT_NORM = "BatchNorm"

# GNN_DICT = {"gcn": GCN, "sage": GraphSAGE, "gat": GAT}
GNN_DICT = {"GCN": GCN, "GraphSAGE": SAGE}

GNN_LIST = list(GNN_DICT.keys())

DATASET_LIST = [ARXIV_DATASET]

LAYER_TEST_N_LAYERS = 10

NN_TESTS_LIST = [LAYER_TEST_NAME, LABEL_TEST_NAME]
