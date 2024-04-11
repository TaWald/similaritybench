from collections.abc import Sequence
from typing import get_args

import repsim.benchmark.paths
import repsim.nlp
import repsim.utils
from repsim.benchmark.types_globals import EXPERIMENT_DICT
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_SEED
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import LAYER_EXPERIMENT_NAME
from repsim.benchmark.types_globals import STANDARD_SETTING
from repsim.utils import NLPDataset
from repsim.utils import NLPModel
from repsim.utils import TrainedModel

NLP_TRAIN_DATASETS = {
    "sst2": NLPDataset("sst2", "sst2"),
    "sst2_sc_rate0558": NLPDataset("sst2_sc_rate0558", path="sst2", shortcut_rate=0.558, shortcut_seed=0),
    "sst2_sc_rate0668": NLPDataset("sst2_sc_rate0668", path="sst2", shortcut_rate=0.668, shortcut_seed=0),
    "sst2_sc_rate0779": NLPDataset("sst2_sc_rate0779", path="sst2", shortcut_rate=0.779, shortcut_seed=0),
    "sst2_sc_rate0889": NLPDataset("sst2_sc_rate0889", path="sst2", shortcut_rate=0.889, shortcut_seed=0),
    "sst2_sc_rate10": NLPDataset("sst2_sc_rate10", path="sst2", shortcut_rate=1.0, shortcut_seed=0),
}
NLP_REPRESENTATION_DATASETS = {
    "sst2": NLPDataset("sst2", path="sst2", split="validation"),
    "sst2_sc_rate0": NLPDataset(
        name="sst2_sc_rate0",
        path="sst2",
        # The local version would be useful if the modified tokenizer is saved with the trained models. But it's not,
        # so the shortcuts are added on the fly.
        # local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "shortcut" / "sst2_sc_rate0"),
        split="validation",
        feature_column="sentence",
        shortcut_rate=0,
        shortcut_seed=0,
    ),
    "sst2_sc_rate0558": NLPDataset(
        name="sst2_sc_rate00558",
        path="sst2",
        split="validation",
        feature_column="sentence",
        shortcut_rate=0.558,
        shortcut_seed=0,
    ),
    "sst2_mem_rate0": NLPDataset("sst2", "sst2", split="validation"),
    "sst2_aug_rate0": NLPDataset("sst2", "sst2", split="validation"),
}


def all_trained_vision_models() -> list[TrainedModel]:
    all_trained_vision_models = []
    for i in range(5):
        for arch in ["ResNet18"]:
            for dataset in ["CIFAR10", "CIFAR100"]:
                for identifier in [STANDARD_SETTING]:
                    all_trained_vision_models.append(
                        TrainedModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=identifier,
                            seed=i,
                            additional_kwargs={},
                        )
                    )
    for i in range(5):
        for arch in ["ResNet18", "ResNet34", "VGG11"]:
            for dataset in [
                "ColorDot_100_CIFAR10DataModule",
                "ColorDot_75_CIFAR10DataModule",
                "ColorDot_50_CIFAR10DataModule",
                "ColorDot_25_CIFAR10DataModule",
                "ColorDot_0_CIFAR10DataModule",
            ]:
                for identifier in ["Shortcut_ColorDot"]:
                    all_trained_vision_models.append(
                        TrainedModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=identifier,
                            seed=i,
                            additional_kwargs={},
                        )
                    )
    for i in range(2):
        for arch in ["ResNet18"]:
            for dataset in [
                "Gauss_Max_CIFAR10DataModule",
                "Gauss_L_CIFAR10DataModule",
                "Gauss_M_CIFAR10DataModule",
                "Gauss_S_CIFAR10DataModule",
                "ColorDot_Off_CIFAR10DataModule",  # N
            ]:
                for identifier in ["GaussNoise"]:
                    all_trained_vision_models.append(
                        TrainedModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=identifier,
                            seed=i,
                            additional_kwargs={"setting_identifier": identifier},
                        )
                    )

    return all_trained_vision_models


def all_trained_nlp_models() -> Sequence[TrainedModel]:
    base_sst2_models = [
        NLPModel(
            train_dataset="sst2",
            identifier="Normal",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
        )
        for i in range(10)
    ]

    shortcut_sst2_models = []
    for seed in range(10):
        for rate in ["0558", "0668", "0779", "0889", "10"]:
            shortcut_sst2_models.append(
                NLPModel(
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH / "shortcut" / f"sst2_pre{seed}_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )

    return base_sst2_models + shortcut_sst2_models


def all_trained_graph_models() -> list[TrainedModel]:
    all_trained_models = []

    for i in get_args(GRAPH_EXPERIMENT_SEED):
        for arch in get_args(GRAPH_ARCHITECTURE_TYPE):
            for dataset in get_args(GRAPH_DATASET_TRAINED_ON):
                for experiment in [LAYER_EXPERIMENT_NAME, LABEL_EXPERIMENT_NAME]:
                    for setting in EXPERIMENT_DICT[experiment]:
                        all_trained_models.append(
                            TrainedModel(
                                domain="GRAPHS",
                                architecture=arch,
                                train_dataset=dataset,
                                identifier=setting,
                                seed=i,
                                additional_kwargs={},
                            )
                        )
    return all_trained_models


ALL_TRAINED_MODELS: list[TrainedModel | NLPModel] = []
ALL_TRAINED_MODELS.extend(all_trained_vision_models())
ALL_TRAINED_MODELS.extend(all_trained_nlp_models())
ALL_TRAINED_MODELS.extend(all_trained_graph_models())
