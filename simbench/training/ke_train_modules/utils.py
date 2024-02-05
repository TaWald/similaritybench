from simbench.arch.ke_architectures.single_model import SingleModel
from simbench.losses.dummy_loss import DummyLoss
from simbench.training.ke_train_modules.single_lightning_module import SingleLightningModule
from simbench.training.trainers.base_trainer import BaseTrainer
from simbench.util import data_structs as ds
from simbench.util import find_datamodules as fd
from simbench.util.data_structs import ArchitectureInfo
from simbench.util.data_structs import BaseArchitecture
from simbench.util.data_structs import Dataset
from simbench.util.find_architectures import get_base_arch


def get_first_model_base_trainer(
    first_model_info: ds.FirstModelInfo,
    arch_params: dict,
    hparams: dict,
    dataset: Dataset,
    base_info: ds.FirstModelInfo,
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
