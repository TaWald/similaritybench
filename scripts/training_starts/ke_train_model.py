import argparse
import sys
from pathlib import Path
from warnings import warn


from simbench.arch.ke_architectures.single_model import SingleModel
from simbench.losses.dummy_loss import DummyLoss
from simbench.training.ke_train_modules.calibrate import calibrate_model

from simbench.training.ke_train_modules.single_lightning_module import SingleLightningModule
from simbench.training.trainers.base_trainer import BaseTrainer
from simbench.util import data_structs as ds
from simbench.util import default_parser_args as dpa
from simbench.util import file_io
from simbench.util import find_architectures as fa
from simbench.util import find_datamodules as fd
from simbench.util import name_conventions as nc
from simbench.util.data_structs import ArchitectureInfo
from simbench.util.default_params import get_default_arch_params
from simbench.util.default_params import get_default_parameters


def main():
    print("Getting started!")
    parser = argparse.ArgumentParser(description="Specify model hyperparams.")
    dpa.ke_default_parser_arguments(parser)
    args = parser.parse_args()
    print("Parsing done.")

    experiment_description: str = args.experiment_name
    architecture: ds.BaseArchitecture = ds.BaseArchitecture(args.architecture)
    dataset: ds.Dataset = ds.Dataset(args.dataset)
    no_activations = args.no_activations
    aggregate_reps = args.aggregate_reps

    train_till_n_models = args.xz

    arch_params = get_default_arch_params(dataset)
    p: ds.Params = get_default_parameters(architecture.value, dataset)
    p = dpa.overwrite_params(p, args)
    tbt_arch = fa.get_base_arch(architecture)
    tmp_arch = tbt_arch(**arch_params)
    max_hooks = len(tmp_arch.hooks)

    # KE specific values.
    transfer_positions: list[int] = args.transfer_positions

    tbt_hook: tuple[ds.Hook] = tuple([tmp_arch.hooks[h] for h in transfer_positions])

    if any([trp > max_hooks for trp in transfer_positions]):
        raise ValueError(f"Got {transfer_positions} but max hook id is {max_hooks}")

    trans_depth = args.transfer_depth
    trans_kernel = args.transfer_kernel
    group_id = args.group_id

    softmax_mets = args.softmax_metrics
    celu_alpha = args.celu_alpha
    sim_loss = args.sim_loss
    sim_loss_weight = args.sim_loss_weight
    dis_loss = args.dis_loss
    dis_loss_weight = args.dis_loss_weight
    ce_weight = args.ce_loss_weight
    epochs_before_regularization = args.epochs_before_regularization

    if dis_loss_weight == 0.0:
        dis_loss = "None"

    # Create the directory name that will get used for experiment.
    exp_name = nc.KENameEncoder.encode(
        experiment_description=experiment_description,
        dataset=dataset.value,
        architecture=architecture.value,
        hook_positions=transfer_positions,
        transfer_depth=trans_depth,
        kernel_width=trans_kernel,
        group_id=group_id,
        sim_loss=sim_loss,
        sim_loss_weight=sim_loss_weight,
        dis_loss=dis_loss,
        dis_loss_weight=dis_loss_weight,
        ce_loss_weight=ce_weight,
        aggregate_reps=aggregate_reps,
        softmax_metrics=softmax_mets,
        epochs_before_regularization=epochs_before_regularization,
    )

    # Create paths
    base_data_path = Path(file_io.get_experiments_data_root_path())
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())

    ke_data_path = base_data_path / nc.KNOWLEDGE_EXTENSION_DIRNAME
    ke_ckpt_path = base_ckpt_path / nc.KNOWLEDGE_EXTENSION_DIRNAME

    ke_data_root_dir = ke_data_path / exp_name
    ke_ckpt_root_dir = ke_ckpt_path / exp_name

    # Do the baseline model creation if it not already exists!
    first_model_info = file_io.get_first_model(
        ke_data_path=ke_data_path,
        ke_ckpt_path=ke_ckpt_path,
        params=p,
        group_id=group_id,
    )

    hparams = {
        "aggregate_source_reps": aggregate_reps,
        "crossentropy_loss_weight": ce_weight,
        "dissimilarity_loss_weight": dis_loss_weight,
        "softmax_metrics": softmax_mets,
        "epochs_before_regularization": epochs_before_regularization,
        "similarity_loss": sim_loss,
        "dissimilarity_loss": dis_loss,
        "celu_alpha": celu_alpha,
        "group_id": group_id,
        "model_id": None,
        "is_regularized": None,
        "n_cls": fd.get_datamodule(dataset).n_classes,
        **vars(p),
    }

    arch_params = get_default_arch_params(dataset)
    print(f"First model: {'Finished' if first_model_info.model_is_finished else 'Missing'}")
    if not first_model_info.model_is_finished():
        print("Creating first model!")
        tbt_arch_info = ArchitectureInfo(first_model_info.architecture, arch_params, first_model_info.path_ckpt, None)
        base_arch = fa.get_base_arch(ds.BaseArchitecture(tbt_arch_info.arch_type_str))(**arch_params)
        base_arch_wrapper = SingleModel(base_arch)

        loss = DummyLoss(ce_weight=1.0)
        lightning_mod = SingleLightningModule(first_model_info, base_arch_wrapper, True, p, hparams, loss, None, True)
        hparams.update({"model_id": 0, "is_regularized": False})
        training_info = first_model_info
    else:
        return  # Already exists. Nothing to be done.

    datamodule = fd.get_datamodule(dataset=dataset)
    if args.split >= datamodule.max_splits:
        raise ValueError(
            f"Can't use splits greater than max splits of Datamodule! (currently: {datamodule.max_splits})!"
        )
    trainer = BaseTrainer(
        model=lightning_mod,
        datamodule=datamodule,
        params=p,
        basic_training_info=training_info,
        arch_params=arch_params,
    )
    trainer.train()
    trainer.save_outputs("test")
    calibrate_model(training_info)
    if not no_activations:
        trainer.save_activations()

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
