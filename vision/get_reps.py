import os
from dataclasses import dataclass

from loguru import logger
from repsim.benchmark.paths import get_experiments_path
from repsim.utils import ModelRepresentations
from repsim.utils import SingleLayerRepresentation
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.arch.arch_loading import load_model_from_info_file
from vision.toy_examples.rel_rep_to_jsd import extract_representations
from vision.util import data_structs as ds
from vision.util import default_params as dp
from vision.util import find_architectures as fa
from vision.util import find_datamodules as fd
from vision.util.download import maybe_download_all_models
from vision.util.file_io import get_vision_model_info


@dataclass
class VisionModelInfo:
    architecture_name: str
    train_dataset: str
    seed_id: int
    setting_identifier: str


def _format_reps_appropriately(all_outs) -> list[SingleLayerRepresentation]:
    all_single_layer_reps = []
    for layer_id, reps in all_outs["reps"].items():
        if len(reps.shape) == 4:
            shape = "nchw"
        elif len(reps.shape) == 3:
            shape = "ntd"
        elif len(reps.shape) == 2:
            shape = "nc"
        else:
            raise ValueError(f"Unknown shape of representations: {reps.shape}")
        all_single_layer_reps.append(
            SingleLayerRepresentation(layer_id=int(layer_id), representation=reps, shape=shape)
        )
    return all_single_layer_reps


def get_vision_representations(
    architecture_name: str,
    train_dataset: str,
    seed_id: int,
    setting_identifier: str | None,
    representation_dataset: str,
) -> ModelRepresentations:
    """
    Finds the representations for a given model and dataset.
    :param architecture_name: The name of the architecture.
    :param seed_id: The id of the model.
    :param dataset: The name of the dataset.
    """

    if setting_identifier == "Normal":
        model_info: ds.ModelInfo = get_vision_model_info(
            architecture_name=architecture_name,
            dataset=train_dataset,
            seed_id=seed_id,
        )
    else:
        model_info: ds.ModelInfo = get_vision_model_info(
            architecture_name=architecture_name,
            dataset=train_dataset,
            seed_id=seed_id,
            setting_identifier=setting_identifier,
        )

    loaded_model: AbsActiExtrArch = load_model_from_info_file(model_info, load_ckpt=True)
    datamodule = fd.get_datamodule(dataset=representation_dataset)
    test_dataloader = datamodule.test_dataloader(batch_size=100)

    # Optionally there shoudl be a saving step in here.
    logger.info("Extracting representations from the model.")
    all_outs = extract_representations(
        loaded_model, test_dataloader, rel_reps=None, meta_info=True, remain_spatial=True
    )
    all_single_layer_reps = _format_reps_appropriately(all_outs)
    model_reps = ModelRepresentations(
        setting_identifier=model_info.setting_identifier,
        architecture_name=model_info.architecture,
        seed=model_info.seed_id,
        train_dataset=model_info.dataset,
        representation_dataset=representation_dataset,
        representations=tuple(all_single_layer_reps),
    )
    return model_reps


if __name__ == "__main__":
    maybe_download_all_models()
    model_info = {"architecture_name": "ResNet18", "train_dataset": "CIFAR10", "seed_id": 1}
    reps_c10 = get_vision_representations(model_info, "CIFAR10")
    reps_c100 = get_vision_representations(model_info, "CIFAR100")
    print(reps_c10)
    print(reps_c100)
