import argparse
import sys
from pathlib import Path

from ke.arch.ke_architectures.feature_approximation import FAArch
from ke.losses.ke_loss import KETrainLoss
from ke.training.ke_train_modules.IntermediateRepresentationLightningModule import (
    IntermediateRepresentationLightningModule,
)
from ke.training.ke_train_modules.utils import get_first_model_base_trainer
from ke.training.trainers.base_trainer import BaseTrainer
from ke.util import data_structs as ds
from ke.util import default_parser_args as dpa
from ke.util import file_io
from ke.util import find_architectures as fa
from ke.util import find_datamodules as fd
from ke.util import name_conventions as nc
from ke.util.data_structs import ArchitectureInfo
from ke.util.default_params import get_default_arch_params
from ke.util.default_params import get_default_parameters
from ke.util.find_ke_loss import find_ke_loss


def main():
    parser = argparse.ArgumentParser(description="Specify model hyperparams.")
    dpa.ke_default_parser_arguments(parser)
    args = parser.parse_args()

    experiment_description: str = args.experiment_name
    architecture: ds.BaseArchitecture = ds.BaseArchitecture(args.architecture)
    dataset: ds.Dataset = ds.Dataset(args.dataset)
    no_activations = args.no_activations
    aggregate_reps = args.aggregate_reps

    train_till_n_models = args.train_till_n_models

    arch_params = get_default_arch_params(dataset)
    p: ds.Params = get_default_parameters(architecture.value, dataset)
    p = dpa.overwrite_params(p, args)
    tbt_arch = fa.get_base_arch(architecture)
    tmp_arch = tbt_arch(**arch_params)
    max_hooks = len(tmp_arch.hooks)

    datamodule = fd.get_datamodule(dataset=dataset)
    if args.split >= datamodule.max_splits:
        raise ValueError(
            f"Can't use splits greater than max splits of Datamodule! (currently: {datamodule.max_splits})!"
        )
    del datamodule  # Only to check that the splits is in Range!

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

    base_data_path = Path(file_io.get_experiments_data_root_path())
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())

    ke_data_path = base_data_path / nc.KNOWLEDGE_EXTENSION_DIRNAME
    ke_ckpt_path = base_ckpt_path / nc.KNOWLEDGE_EXTENSION_DIRNAME

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

    sim_l = find_ke_loss(sim_loss, softmax_mets, celu_alpha)
    dis_l = find_ke_loss(dis_loss, softmax_mets, celu_alpha)

    hparams = {
        "trans_depth": trans_depth,
        "trans_kernel": trans_kernel,
        "trans_hooks": str(transfer_positions),
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
        **vars(p),
    }

    arch_params = get_default_arch_params(dataset)

    if not file_io.first_model_trained(first_model):
        trainer = get_first_model_base_trainer(
            first_model_info=first_model,
            arch_params=arch_params,
            hparams=hparams,
            dataset=dataset,
            base_info=first_model,
            params=p,
        )

    else:
        n_trained_models: int = len(file_io.get_trained_ke_models(ke_data_root_dir, ke_ckpt_root_dir))
        if (n_trained_models + 1) >= train_till_n_models:
            return
        else:
            hparams.update({"model_id": n_trained_models + 1, "is_regularized": True})
            prev_training_infos: list[ds.KETrainingInfo] = file_io.get_trained_ke_models(
                ke_data_root_dir, ke_ckpt_root_dir
            )
            model_id = len(prev_training_infos) + 1
            new_model = nc.MODEL_NAME_TMPLT.format(model_id)

            tbt_model_data_dir = ke_data_root_dir / new_model
            tbt_model_ckpt_dir = ke_ckpt_root_dir / new_model

            training_info = ds.KETrainingInfo(
                experiment_name=exp_name,
                experiment_description=experiment_description,
                dir_name=new_model,
                model_id=model_id,
                group_id=group_id,
                architecture=str(architecture.value),
                dataset=str(dataset.value),
                aggregate_source_reps=aggregate_reps,
                softmax_metrics=bool(softmax_mets),
                similarity_loss=sim_loss,
                similarity_loss_weight=sim_loss_weight,
                dissimilarity_loss=dis_loss,
                crossentropy_loss_weight=ce_weight,
                dissimilarity_loss_weight=dis_loss_weight,
                epochs_before_regularization=epochs_before_regularization,
                learning_rate=p.learning_rate,
                split=p.split,
                weight_decay=p.weight_decay,
                batch_size=p.batch_size,
                trans_hooks=transfer_positions,
                trans_depth=trans_depth,
                trans_kernel=trans_kernel,
                path_data_root=tbt_model_data_dir,
                path_ckpt_root=tbt_model_ckpt_dir,
            )

            all_src_arch_infos = [
                ArchitectureInfo(first_model.architecture, arch_params, first_model.path_ckpt, tbt_hook)
            ]
            for pti in prev_training_infos:
                all_src_arch_infos.append(ArchitectureInfo(pti.architecture, arch_params, pti.path_ckpt, tbt_hook))
            tbt_arch_info = ArchitectureInfo(
                training_info.architecture, arch_params, training_info.path_ckpt, tbt_hook
            )
            module = FAArch(
                old_model_info=all_src_arch_infos,
                new_model_info=tbt_arch_info,
                aggregate_old_reps=aggregate_reps,
                transfer_depth=trans_depth,
                transfer_kernel_width=trans_kernel,
            )
            loss = KETrainLoss(
                similar_loss=sim_l,
                dissimilar_loss=dis_l,
                ce_weight=ce_weight,
                dissim_weight=dis_loss_weight,
                sim_weight=sim_loss_weight,
                regularization_epoch_start=epochs_before_regularization,
                n_classes=arch_params["n_cls"],
            )
            kelm = IntermediateRepresentationLightningModule(
                model_info=training_info,
                network=module,
                save_checkpoints=True,
                params=p,
                hparams=hparams,
                loss=loss,
                skip_n_epochs=None,
                log=True,
                save_approx=args.save_approximation_layers,
            )
            datamodule = fd.get_datamodule(dataset=dataset)
            trainer = BaseTrainer(
                model=kelm,
                datamodule=datamodule,
                params=p,
                basic_training_info=training_info,
                arch_params=arch_params,
            )
    trainer.train()
    trainer.save_outputs("test")
    trainer.measure_generalization()
    if not no_activations:
        trainer.save_activations()

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
