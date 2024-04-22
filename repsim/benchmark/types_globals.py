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
NLP_ARCHITECTURE_TYPE = Literal["BERT-L"]
GRAPH_ARCHITECTURE_TYPE = Literal["GCN", "GAT", "GraphSAGE"]

NN_ARCHITECTURE_TYPE = VISION_ARCHITECTURE_TYPE | NLP_ARCHITECTURE_TYPE | GRAPH_ARCHITECTURE_TYPE

BENCHMARK_NN_ARCHITECTURES = (
    list(get_args(VISION_ARCHITECTURE_TYPE))
    + list(get_args(NLP_ARCHITECTURE_TYPE))
    + list(get_args(GRAPH_ARCHITECTURE_TYPE))
)

# ----------------------------- Datasets trained on ---------------------------- #
VISION_DATASET_TRAINED_ON = Literal["CIFAR10", "CIFAR100", "ImageNet"]
NLP_DATASET_TRAINED_ON = Literal["IMDB"]
GRAPH_DATASET_TRAINED_ON = Literal["ogbn-arxiv", "cora", "reddit", "flickr"]

ARXIV_DATASET_NAME = get_args(GRAPH_DATASET_TRAINED_ON)[0]
CORA_DATASET_NAME = get_args(GRAPH_DATASET_TRAINED_ON)[1]
REDDIT_DATASET_NAME = get_args(GRAPH_DATASET_TRAINED_ON)[2]
FLICKR_DATASET_NAME = get_args(GRAPH_DATASET_TRAINED_ON)[3]

BENCHMARK_DATASET = VISION_DATASET_TRAINED_ON | NLP_DATASET_TRAINED_ON | GRAPH_DATASET_TRAINED_ON
BENCHMARK_DATASETS_LIST = (
    list(get_args(VISION_DATASET_TRAINED_ON))
    + list(get_args(NLP_DATASET_TRAINED_ON))
    + list(get_args(GRAPH_DATASET_TRAINED_ON))
)


# ---------------------------- Identifier_settings --------------------------- #
# These are shared across domains and datasets
SETTING_IDENTIFIER = Literal[
    "Normal",
    "RandomInit",
    "RandomLabels_25",
    "RandomLabels_50",
    "RandomLabels_75",
    "RandomLabels_100",
    "Shortcut_0",
    "Shortcut_25",
    "Shortcut_50",
    "Shortcut_75",
    "Shortcut_100",
    "MultiLayer",
    "Augmentation_25",
    "Augmentation_50",
    "Augmentation_75",
    "Augmentation_100",
]

STANDARD_SETTING = get_args(SETTING_IDENTIFIER)[0]
RANDOM_LABEL_25_SETTING, RANDOM_LABEL_50_SETTING, RANDOM_LABEL_75_SETTING, RANDOM_LABEL_100_SETTING = (
    get_args(SETTING_IDENTIFIER)[2],
    get_args(SETTING_IDENTIFIER)[3],
    get_args(SETTING_IDENTIFIER)[4],
    get_args(SETTING_IDENTIFIER)[5],
)

SHORTCUT_0_SETTING, SHORTCUT_25_SETTING, SHORTCUT_50_SETTING, SHORTCUT_75_SETTING, SHORTCUT_100_SETTING = (
    get_args(SETTING_IDENTIFIER)[6],
    get_args(SETTING_IDENTIFIER)[7],
    get_args(SETTING_IDENTIFIER)[8],
    get_args(SETTING_IDENTIFIER)[9],
    get_args(SETTING_IDENTIFIER)[10],
)

MULTI_LAYER_SETTING = get_args(SETTING_IDENTIFIER)[11]

# TODO: These numbers are brittle. Make a small change in the above list, and everything here is different. Build list from these constants?
AUGMENTATION_25_SETTING, AUGMENTATION_50_SETTING, AUGMENTATION_75_SETTING, AUGMENTATION_100_SETTING = (
    get_args(SETTING_IDENTIFIER)[12],
    get_args(SETTING_IDENTIFIER)[13],
    get_args(SETTING_IDENTIFIER)[14],
    get_args(SETTING_IDENTIFIER)[15],
)

SINGLE_MODEL_EXPERIMENT_IDENTIFIER = Literal["layer_test"]
MULTIMODEL_EXPERIMENT_IDENTIFIER = Literal["label_test", "shortcut_test"]

EXPERIMENT_IDENTIFIER = SINGLE_MODEL_EXPERIMENT_IDENTIFIER | MULTIMODEL_EXPERIMENT_IDENTIFIER
BENCHMARK_EXPERIMENTS_LIST = list(get_args(SINGLE_MODEL_EXPERIMENT_IDENTIFIER)) + list(
    get_args(MULTIMODEL_EXPERIMENT_IDENTIFIER)
)


LAYER_EXPERIMENT_NAME, LABEL_EXPERIMENT_NAME, SHORTCUT_EXPERIMENT_NAME = (
    get_args(SINGLE_MODEL_EXPERIMENT_IDENTIFIER)[0],
    get_args(MULTIMODEL_EXPERIMENT_IDENTIFIER)[0],
    get_args(MULTIMODEL_EXPERIMENT_IDENTIFIER)[1],
)

EXPERIMENT_DICT: Dict[EXPERIMENT_IDENTIFIER, List[SETTING_IDENTIFIER]] = dict(
    {
        LAYER_EXPERIMENT_NAME: [MULTI_LAYER_SETTING],
        LABEL_EXPERIMENT_NAME: [
            STANDARD_SETTING,
            RANDOM_LABEL_25_SETTING,
            RANDOM_LABEL_50_SETTING,
            RANDOM_LABEL_75_SETTING,
            RANDOM_LABEL_100_SETTING,
        ],
        SHORTCUT_EXPERIMENT_NAME: [
            SHORTCUT_0_SETTING,
            SHORTCUT_25_SETTING,
            SHORTCUT_50_SETTING,
            SHORTCUT_75_SETTING,
            SHORTCUT_100_SETTING,
        ],
    }
)

EXPERIMENT_SEED = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
GRAPH_EXPERIMENT_SEED = EXPERIMENT_SEED
SHORTCUT_EXPERIMENT_SEED = 2024
