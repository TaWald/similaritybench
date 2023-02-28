import argparse
import sys
from pathlib import Path

from ke.arch.ke_architectures.single_model import SingleModel
from ke.losses.dummy_loss import DummyLoss
from ke.training.ke_train_modules.single_lightning_module import WarmStartSingleLightningModule
from ke.training.trainers.base_trainer import BaseTrainer
from ke.util import data_structs as ds
from ke.util import default_parser_args as dpa
from ke.util import file_io
from ke.util import find_datamodules as fd
from ke.util import name_conventions as nc
from ke.util.default_params import get_default_arch_params
from ke.util.default_params import get_default_parameters
from ke.util.find_architectures import get_tv_arch


def main():
    """
    This files is made to train Pretrained ResNets only!
    # Its supposed to train them pretrained or not pretrained and compare the various configurations and the
    impact it has on output predictive behavior.
    """
    parser = argparse.ArgumentParser(description="Specify model hyperparams.")
    dpa.pretrained_parser_arguments(parser)
    args = parser.parse_args()

    experiment_description: str = args.experiment_name
    warmup_pretrained: bool = args.warmup_pretrained
    linear_probe_only: bool = args.linear_probe_only
    pretrain: bool = args.pretrained

    dataset: ds.Dataset = ds.Dataset(args.dataset)
    arch_params = get_default_arch_params(dataset)
    architecture = ds.BasicPretrainableArchitectures = ds.BasicPretrainableArchitectures(args.architecture)

    p: ds.Params = get_default_parameters(architecture.value, dataset)
    p = dpa.overwrite_params(p, args)

    datamodule = fd.get_datamodule(dataset=dataset)
    if args.split >= datamodule.max_splits:
        raise ValueError(
            f"Can't use splits greater than max splits of Datamodule! (currently: {datamodule.max_splits})!"
        )
    del datamodule  # Only to check that the splits is in Range!

    # KE specific values.
    group_id = args.group_id

    exp_name = nc.PretrainedNameEncoder.encode(
        experiment_description=experiment_description,
        dataset=dataset.value,
        architecture=architecture.value,
        group_id=group_id,
        pretrained=1 if pretrain else 0,
        warmup_pretrained=1 if warmup_pretrained else 0,
        linear_probe_only=1 if linear_probe_only else 0,
    )

    base_data_path = Path(file_io.get_experiments_data_root_path())
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())

    ke_data_path = base_data_path / nc.PRETRAINED_TEST_DIRNAME
    ke_ckpt_path = base_ckpt_path / nc.PRETRAINED_TEST_DIRNAME

    ke_data_root_dir = ke_data_path / exp_name
    ke_ckpt_root_dir = ke_ckpt_path / exp_name

    # Do the baseline model creation if it not already exists!
    training_info = ds.PretrainedTrainingInfo(
        experiment_name=exp_name,
        experiment_description=experiment_description,
        dir_name="model_0000",
        model_id=0,
        group_id=group_id,
        architecture=str(architecture.value),
        dataset=str(dataset.value),
        learning_rate=p.learning_rate,
        split=p.split,
        weight_decay=p.weight_decay,
        batch_size=p.batch_size,
        path_data_root=ke_data_root_dir,
        path_ckpt_root=ke_ckpt_root_dir,
        pretrained=pretrain,
        warmup_pretrained=warmup_pretrained,
        linear_probe_only=linear_probe_only,
    )
    if training_info.path_output_json.exists():
        raise FileExistsError("Final output.json exists already.")

    if linear_probe_only:
        p.num_epochs = 200

    hparams = {
        "crossentropy_loss_weight": 1.0,
        "group_id": group_id,
        "model_id": 0,
        "pretrained": pretrain,
        "warmup_pretrained": warmup_pretrained,
        "linear_probe_only": linear_probe_only,
        **vars(p),
    }

    datamodule = fd.get_datamodule(dataset=dataset)
    # Archinfo = ArchitectureInfo(training_info.architecture, arch_params, training_info.path_ckpt, None)
    module = get_tv_arch(architecture, pretrained=pretrain, n_cls=arch_params["n_cls"])
    arch = SingleModel(module)
    loss = DummyLoss(ce_weight=1.0)
    slm = WarmStartSingleLightningModule(
        model_info=training_info,
        network=arch,
        save_checkpoints=True,
        params=p,
        hparams=hparams,
        loss=loss,
        skip_n_epochs=None,
        log=True,
        warmup_pretrained=warmup_pretrained,
        linear_probe_only=linear_probe_only,
    )

    hparams.update({"model_id": 0, "is_regularized": False})
    trainer = BaseTrainer(
        model=slm, datamodule=datamodule, params=p, basic_training_info=training_info, arch_params=arch_params
    )

    trainer.train()
    trainer.save_outputs("test")
    # trainer.measure_generalization()

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
