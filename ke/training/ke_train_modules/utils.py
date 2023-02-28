from ke.arch.ke_architectures.single_model import SingleModel
from ke.losses.dummy_loss import DummyLoss
from ke.training.ke_train_modules.single_lightning_module import SingleLightningModule
from ke.training.trainers.base_trainer import BaseTrainer
from ke.util import data_structs as ds
from ke.util import find_datamodules as fd
from ke.util.data_structs import ArchitectureInfo
from ke.util.data_structs import BaseArchitecture
from ke.util.data_structs import BasicTrainingInfo
from ke.util.data_structs import Dataset
from ke.util.find_architectures import get_base_arch


def get_first_model_base_trainer(
    first_model_info: BasicTrainingInfo,
    arch_params: dict,
    hparams: dict,
    dataset: Dataset,
    base_info: ds.BasicTrainingInfo,
    params: ds.Params,
) -> BaseTrainer:
    datamodule = fd.get_datamodule(dataset=dataset)
    tbt_arch_info = ArchitectureInfo(base_info.architecture, arch_params, base_info.path_ckpt, None)
    module = get_base_arch(BaseArchitecture(tbt_arch_info.arch_type_str))(**arch_params)
    arch = SingleModel(module)

    loss = DummyLoss(ce_weight=1.0)
    slm = SingleLightningModule(
        model_info=base_info,
        network=arch,
        save_checkpoints=True,
        params=params,
        hparams=hparams,
        loss=loss,
        skip_n_epochs=None,
        log=True,
    )

    hparams.update({"model_id": 0, "is_regularized": False})
    trainer = BaseTrainer(
        model=slm, datamodule=datamodule, params=params, basic_training_info=first_model_info, arch_params=arch_params
    )
    return trainer
