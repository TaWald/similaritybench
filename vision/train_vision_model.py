from argparse import ArgumentParser
from functools import partial

from loguru import logger
from vision.arch.arch_loading import load_model_from_info_file
from vision.data.random_labels.rl_c10_dm import RandomLabel_CIFAR10DataModule
from vision.losses.dummy_loss import DummyLoss
from vision.training.ke_train_modules.base_training_module import BaseLightningModule
from vision.training.ke_train_modules.shortcut_lightning_module import ShortcutLightningModule
from vision.training.trainers.base_trainer import BaseTrainer
from vision.training.trainers.shortcut_trainer import ShortcutTrainer
from vision.util import data_structs as ds
from vision.util import default_params as dp
from vision.util import find_architectures as fa
from vision.util import find_datamodules as fd
from vision.util.default_parser_args import add_vision_training_params
from vision.util.file_io import get_vision_model_info

STANDARD_DATAMODULES = [
    ds.Dataset.TinyIMAGENET,
    ds.Dataset.CIFAR10,
    ds.Dataset.CIFAR100,
]

SHORTCUT_DATAMODULES = [
    ds.Dataset.CDOT100,
    ds.Dataset.CDOT75,
    ds.Dataset.CDOT50,
    ds.Dataset.CDOT25,
    ds.Dataset.CDOT0,
]

AUGMENTATION_DATAMODULES = [
    ds.Dataset.GaussMAX,
    ds.Dataset.GaussL,
    ds.Dataset.GaussM,
    ds.Dataset.GaussS,
    ds.Dataset.GaussOff,
]


def load_model_and_datamodule(model_info: ds.ModelInfo):
    """Load instances of the model and the datamodule from the infos of the info_file."""
    datamodule = fd.get_datamodule(dataset=model_info.dataset)
    params = dp.get_default_parameters(model_info.architecture, model_info.dataset)
    arch_kwargs = dp.get_default_arch_params(model_info.dataset)
    if model_info.info_file_exists():
        loaded_model = load_model_from_info_file(model_info)
    else:
        architecture = fa.get_base_arch(model_info.architecture)
        loaded_model = architecture(**arch_kwargs)
    return loaded_model, datamodule, params, arch_kwargs


def train_vision_model(
    architecture_name: str, train_dataset: str, seed_id: int, setting_identifier: str, overwrite: bool = False
):
    model_info: ds.ModelInfo = get_vision_model_info(
        architecture_name=architecture_name,
        dataset=train_dataset,
        seed_id=seed_id,
        setting_identifier=setting_identifier,
    )

    if model_info.finished_training() and not overwrite:
        logger.info("Model already trained, skipping.")
        return  # No need to train the model again if it exists

    loaded_model, datamodule, params, arch_params = load_model_and_datamodule(model_info)
    if ds.Dataset(train_dataset) in SHORTCUT_DATAMODULES:
        lnm_cls = ShortcutLightningModule
        no_sc_dm, full_sc_dm = fd.get_min_max_shortcut_datamodules(train_dataset)
        trainer_cls = partial(ShortcutTrainer, no_sc_datamodule=no_sc_dm, full_sc_datamodule=full_sc_dm)
    else:
        lnm_cls = BaseLightningModule
        trainer_cls = BaseTrainer

    if isinstance(datamodule, RandomLabel_CIFAR10DataModule):
        datamodule.rng_seed = (seed_id + 1) * 123  # Different rng seeds for different seeds.

    lightning_mod = lnm_cls(
        model_info=model_info,
        network=loaded_model,
        save_checkpoints=True,
        params=params,
        hparams=arch_params,
        loss=DummyLoss(),
        log=True,
    )

    trainer = trainer_cls(
        model=lightning_mod,
        datamodule=datamodule,
        model_info=model_info,
        arch_params=arch_params,
    )
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    add_vision_training_params(parser)
    args = parser.parse_args()
    train_vision_model(args.architecture, args.dataset, args.seed, args.setting_identifier, args.overwrite)
