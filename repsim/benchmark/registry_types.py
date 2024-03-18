# -------------------- All categories  trained model that can -------------------- #
from typing import Literal


DOMAIN_TYPE = Literal["VISION", "NLP", "GRAPHS"]
# ----------------------------- All Architectures ---------------------------- #
VISION_ARCHITECTURE_TYPE = Literal["ResNet18", "ResNet34", "ResNet101", "VGG11", "VGG19", "ViT-b19"]
NLP_ARCHITECTURE_TYPE = Literal["BERT"]
GRAPH_ARCHITECTURE_TYPE = Literal["GCN", "GAT", "GraphSAGE"]

# ----------------------------- Datasets trained on ---------------------------- #
VISION_DATASET_TRAINED_ON = Literal["CIFAR10", "CIFAR100", "ImageNet"]
NLP_DATASET_TRAINED_ON = Literal["MNLI", "SST2"]
GRAPH_DATASET_TRAINED_ON = Literal["Cora", "CiteSeer", "PubMed"]

# ---------------------------- Identifier_settings --------------------------- #
# These are shared across domains and datasets
EXPERIMENT_IDENTIFIER = Literal[
    "Normal",
    "RandomInit",
    "RandomLabels_25",
    "RandomLabels_50",
    "RandomLabels_75",
    "RandomLabels_100",
    "Shortcut_25",
    "Shortcut_50",
    "Shortcut_75",
]
