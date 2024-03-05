from typing import Dict
from typing import get_args
from typing import List
from typing import Literal

# -------------------- All categories  trained model that can -------------------- #
DOMAIN_TYPE = Literal["VISION", "NLP", "GRAPHS"]
VISION_DOMAIN, NLP_DOMAIN, GRAPH_DOMAIN = (
    get_args(DOMAIN_TYPE)[0],
    get_args(DOMAIN_TYPE)[1],
    get_args(DOMAIN_TYPE)[2],
)
# ----------------------------- All Architectures ---------------------------- #
VISION_ARCHITECTURE_TYPE = Literal["ResNet18", "ResNet34", "ResNet101", "VGG11", "VGG19", "ViT-b19"]
NLP_ARCHITECTURE_TYPE = Literal["BERT", "LSTM", "GRU"]
GRAPH_ARCHITECTURE_TYPE = Literal["GCN", "GAT", "GraphSAGE"]

NN_ARCHITECTURE_TYPE = VISION_ARCHITECTURE_TYPE | NLP_ARCHITECTURE_TYPE | GRAPH_ARCHITECTURE_TYPE

# ----------------------------- Datasets trained on ---------------------------- #
VISION_DATASET_TRAINED_ON = Literal["CIFAR10", "CIFAR100", "ImageNet"]
NLP_DATASET_TRAINED_ON = Literal["IMDB"]
GRAPH_DATASET_TRAINED_ON = Literal["obgn-arxiv"]

# ---------------------------- Identifier_settings --------------------------- #
# These are shared across domains and datasets
SETTING_IDENTIFIER = Literal[
    "Standard",
    "RandomInit",
    "RandomLabels_25",
    "RandomLabels_50",
    "RandomLabels_75",
    "RandomLabels_100",
    "Shortcut_25",
    "Shortcut_50",
    "Shortcut_75",
]

STANDARD_SETTING = get_args(SETTING_IDENTIFIER)[0]
RANDOM_LABEL_25_SETTING, RANDOM_LABEL_50_SETTING, RANDOM_LABEL_75_SETTING, RANDOM_LABEL_100_SETTING = (
    get_args(SETTING_IDENTIFIER)[2],
    get_args(SETTING_IDENTIFIER)[3],
    get_args(SETTING_IDENTIFIER)[4],
    get_args(SETTING_IDENTIFIER)[5],
)

SHORTCUT_25_SETTING, SHORTCUT_50_SETTING, SHORTCUT_75_SETTING = (
    get_args(SETTING_IDENTIFIER)[6],
    get_args(SETTING_IDENTIFIER)[7],
    get_args(SETTING_IDENTIFIER)[8],
)

EXPERIMENT_IDENTIFIER = Literal["layer_test", "label_test", "shortcut_test"]

LAYER_TEST_NAME, LABEL_TEST_NAME, SHORTCUT_TEST_NAME = (
    get_args(EXPERIMENT_IDENTIFIER)[0],
    get_args(EXPERIMENT_IDENTIFIER)[1],
    get_args(EXPERIMENT_IDENTIFIER)[2],
)

EXPERIMENT_DICT: Dict[EXPERIMENT_IDENTIFIER, List[SETTING_IDENTIFIER]] = dict(
    {
        LAYER_TEST_NAME: [STANDARD_SETTING],
        LABEL_TEST_NAME: [
            STANDARD_SETTING,
            RANDOM_LABEL_25_SETTING,
            RANDOM_LABEL_50_SETTING,
            RANDOM_LABEL_75_SETTING,
            RANDOM_LABEL_100_SETTING,
        ],
        SHORTCUT_TEST_NAME: [SHORTCUT_25_SETTING, SHORTCUT_50_SETTING, SHORTCUT_75_SETTING],
    }
)

EXPERIMENT_SEED = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
