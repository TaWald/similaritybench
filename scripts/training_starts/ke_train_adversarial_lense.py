import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch
from ke.arch.ke_architectures.adversarial_lense_new_model_training_arch import (
    AdversarialLenseNewModelTrainingArch,
)
from ke.arch.ke_architectures.adversarial_lense_training_arch import AdversarialLenseTrainingArch
from ke.arch.lense_architecture import UNetLense
from ke.losses.ke_lense_output_loss import KELenseOutputTrainLoss
from ke.losses.static_lense_losses.negative_cross_entropy import NegativeCrossEntropyLenseLoss
from ke.training.ke_train_modules.AdversarialLenseLightningModule import AdversarialLenseLightningModule
from ke.training.ke_train_modules.AdversarialLenseModelTrainingLightningModule import (
    AdversarialLenseModelTrainingLightningModule,
)
from ke.training.ke_train_modules.utils import get_first_model_base_trainer
from ke.training.trainers.base_trainer import BaseTrainer
from ke.training.trainers.lense_trainer import LenseTrainer
from ke.util import data_structs as ds
from ke.util import default_parser_args as dpa
from ke.util import file_io
from ke.util import find_datamodules as fd
from ke.util import name_conventions as nc
from ke.util.data_structs import ArchitectureInfo
from ke.util.default_params import get_default_arch_params
from ke.util.default_params import get_default_parameters
from torch import nn


def main():
    parser = argparse.ArgumentParser(description="Specify model hyperparams.")
    dpa.ke_adversarial_lense_parser_arguments(parser)
    args = parser.parse_args()

    experiment_description: str = args.experiment_name
    no_activations = args.no_activations
    train_until_n_models = args.train_till_n_models
    lense_reconstruction_weight: float = args.lense_reco_weight
    lense_adversarial_weight: float = args.lense_adversarial_weight
    lense_setting: str = args.lense_setting
    architecture: ds.BaseArchitecture | ds.BasicPretrainableArchitectures

    dataset: ds.Dataset = ds.Dataset(args.dataset)
    arch_params = get_default_arch_params(dataset)
    try:
        architecture = ds.BaseArchitecture(args.architecture)
    except ValueError:
        try:
            architecture = ds.BasicPretrainableArchitectures(args.architecture)
        except ValueError as e:
            raise e

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

    softmax_mets = args.softmax_metrics
    dis_loss = args.dis_loss
    dis_loss_weight = args.dis_loss_weight
    ce_weight = args.ce_loss_weight
    epochs_before_regularization = args.epochs_before_regularization

    if dis_loss_weight == 0.0:
        raise NotImplementedError("Makes not sense without any adversarial loss")

    exp_name = nc.KEAdversarialLenseOutputNameEncoder.encode(
        experiment_description=experiment_description,
        dataset=dataset.value,
        architecture=architecture.value,
        group_id=group_id,
        adv_loss=dis_loss,
        lense_adversarial_weight=lense_adversarial_weight,
        ce_loss_weight=ce_weight,
        lense_reconstruction_weight=lense_reconstruction_weight,
        lense_setting=lense_setting,
    )

    base_data_path = Path(file_io.get_experiments_data_root_path())
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())

    ke_data_path = base_data_path / nc.KE_ADVERSARIAL_LENSE_DIRNAME
    ke_ckpt_path = base_ckpt_path / nc.KE_ADVERSARIAL_LENSE_DIRNAME

    ke_data_root_dir = ke_data_path / exp_name
    ke_ckpt_root_dir = ke_ckpt_path / exp_name

    # Do the baseline model creation if it not already exists!
    first_model = file_io.get_first_model(
        experiment_description=exp_name,
        ke_data_path=ke_data_path,
        ke_ckpt_path=ke_ckpt_path,
        architecture=args.architecture,
        dataset=p.dataset,
        learning_rate=p.learning_rate,
        split=p.split,
        weight_decay=p.weight_decay,
        batch_size=p.batch_size,
        group_id=group_id,
    )

    hparams = {
        "crossentropy_loss_weight": ce_weight,
        "dissimilarity_loss_weight": dis_loss_weight,
        "softmax_metrics": softmax_mets,
        "epochs_before_regularization": epochs_before_regularization,
        "dissimilarity_loss": dis_loss,
        "group_id": group_id,
        "model_id": None,
        "is_regularized": None,
        **vars(p),
    }

    if not file_io.first_model_trained(first_model):
        trainer = get_first_model_base_trainer(
            first_model_info=first_model,
            arch_params=arch_params,
            hparams=hparams,
            dataset=dataset,
            base_info=first_model,
            params=p,
        )
        trainer.train()
        trainer.save_outputs("test")
        if not no_activations:
            trainer.save_activations()
    else:
        n_trained_models: int = len(file_io.get_trained_adversarial_lense_models(ke_data_root_dir, ke_ckpt_root_dir))
        if (n_trained_models + 1) >= train_until_n_models:
            return
        else:
            hparams.update({"model_id": n_trained_models + 1, "is_regularized": True})
            prev_training_infos: list[
                ds.KEAdversarialLenseOutputTrainingInfo
            ] = file_io.get_trained_adversarial_lense_models(ke_data_root_dir, ke_ckpt_root_dir)
            model_id = len(prev_training_infos) + 1
            new_model = nc.MODEL_NAME_TMPLT.format(model_id)

            tbt_model_data_dir = ke_data_root_dir / new_model
            tbt_model_ckpt_dir = ke_ckpt_root_dir / new_model

            training_info = ds.KEAdversarialLenseOutputTrainingInfo(
                experiment_name=exp_name,
                experiment_description=experiment_description,
                dir_name=new_model,
                model_id=model_id,
                group_id=group_id,
                architecture=str(architecture.value),
                dataset=str(dataset.value),
                lense_reconstruction_weight=lense_reconstruction_weight,
                lense_adversarial_weight=lense_adversarial_weight,
                lense_setting=lense_setting,
                learning_rate=p.learning_rate,
                split=p.split,
                weight_decay=p.weight_decay,
                batch_size=p.batch_size,
                path_data_root=tbt_model_data_dir,
                path_ckpt_root=tbt_model_ckpt_dir,
                adversarial_loss="NegCELoss",
            )

            all_prev_infos = [ArchitectureInfo(first_model.architecture, arch_params, first_model.path_ckpt, None)]
            for tkem in prev_training_infos:
                all_prev_infos.append(ArchitectureInfo(tkem.architecture, arch_params, tkem.path_ckpt, None))

            datamodule = fd.get_datamodule(dataset=dataset)
            lense = UNetLense(lense_setting, arch_params["in_ch"])
            # Train Lense architecture if it doesn't exist yet.
            if training_info.path_lense_checkpoint.exists():
                lense.load_state_dict(torch.load(training_info.path_lense_checkpoint))
            else:
                lense_loss = KELenseOutputTrainLoss(
                    adversarial_loss=NegativeCrossEntropyLenseLoss(),
                    reconstruction_loss=nn.MSELoss(),
                    lense_adversarial_weight=lense_adversarial_weight,
                    lense_reconstruction_weight=lense_reconstruction_weight,
                )
                lense_net = AdversarialLenseTrainingArch(old_models=all_prev_infos, lense=lense)
                lense_params = deepcopy(p)
                lense_params.num_epochs = 50
                allm = AdversarialLenseLightningModule(
                    model_info=training_info,
                    network=lense_net,
                    save_checkpoints=True,
                    params=lense_params,
                    hparams=hparams,
                    loss=lense_loss,
                    skip_n_epochs=None,
                    log=True,
                )

                lense_trainer = LenseTrainer(
                    model=allm,
                    datamodule=datamodule,
                    params=lense_params,
                    basic_training_info=training_info,
                    arch_params=arch_params,
                )
                lense_trainer.train()

            # Train new model with the lense.
            new_arch_info = ArchitectureInfo(training_info.architecture, arch_params, training_info.path_ckpt, None)
            new_model_lense_net = AdversarialLenseNewModelTrainingArch(
                new_model=new_arch_info, old_models=all_prev_infos, lense=lense
            )
            almtlm = AdversarialLenseModelTrainingLightningModule(
                training_info,
                new_model_lense_net,
                save_checkpoints=True,
                params=p,
                hparams=hparams,
                skip_n_epochs=None,
                log=True,
            )

            new_model_trainer = BaseTrainer(
                model=almtlm,
                datamodule=datamodule,
                params=p,
                basic_training_info=training_info,
                arch_params=arch_params,
            )
            new_model_trainer.train()
            new_model_trainer.save_outputs("test")
            new_model_trainer.measure_generalization()

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
