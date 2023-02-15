import argparse
import sys
from pathlib import Path

from rep_trans.arch.ke_architectures.output_regularization import OutputRegularizerArch
from rep_trans.losses.ke_output_loss import KEOutputTrainLoss
from rep_trans.training.ke_train_modules.KEOutputTrainingModule import KEOutputLightningModule
from rep_trans.training.ke_train_modules.KEOutputTrainingModule import KEOutputPCGrad
from rep_trans.training.ke_train_modules.utils import get_first_model_base_trainer
from rep_trans.training.trainers.base_trainer import BaseTrainer
from rep_trans.util import data_structs as ds
from rep_trans.util import default_parser_args as dpa
from rep_trans.util import file_io
from rep_trans.util import find_datamodules as fd
from rep_trans.util import name_conventions as nc
from rep_trans.util.data_structs import ArchitectureInfo
from rep_trans.util.default_params import get_default_arch_params
from rep_trans.util.default_params import get_default_parameters
from rep_trans.util.find_ke_loss import find_output_ke_loss


def main():
    parser = argparse.ArgumentParser(description="Specify model hyperparams.")
    dpa.keo_default_parser_arguments(parser)
    args = parser.parse_args()

    experiment_description: str = args.experiment_name
    no_activations = args.no_activations
    train_until_n_models = args.train_till_n_models
    freeze_pretrained: bool = args.freeze_pretrained
    pretrain: bool = args.pretrained
    architecture: ds.BaseArchitecture | ds.BasicPretrainableArchitectures
    pcgrad: bool = args.pc_grad

    # ToDo: Rename existing directories so they don't throw errors.
    # ToDo: Rework that the paths of existing models is hardwired.
    #   When moving the directories their contents are not correct.

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
    dis_loss_weights = dis_loss_weight.split("-")
    dis_loss_args = None
    if len(dis_loss_weights) == 1:
        dis_loss_weight = float(dis_loss_weight)
    else:
        dis_loss_args = [float(dlw) for dlw in dis_loss_weights]
        dis_loss_weight = 1.0  # Scaling has to happen internally

    ce_weight = args.ce_loss_weight
    epochs_before_regularization = args.epochs_before_regularization

    if dis_loss_weight == 0.0:
        dis_loss = "None"
    if dis_loss == "None":
        dis_loss_weight = 0.0

    exp_name = nc.KEOutputNameEncoder.encode(
        experiment_description=experiment_description,
        dataset=dataset.value,
        architecture=architecture.value,
        group_id=group_id,
        dis_loss=dis_loss,
        dis_loss_weight=dis_loss_weights,
        ce_loss_weight=ce_weight,
        softmax_metrics=softmax_mets,
        epochs_before_regularization=epochs_before_regularization,
        pc_grad=1 if pcgrad else 0,
    )

    base_data_path = Path(file_io.get_experiments_data_root_path())
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())

    ke_data_path = base_data_path / nc.KE_OUTPUT_REGULARIZATION_DIRNAME
    ke_ckpt_path = base_ckpt_path / nc.KE_OUTPUT_REGULARIZATION_DIRNAME

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
        "pretrained": pretrain,
        "freeze_model": freeze_pretrained,
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
    else:
        n_trained_models: int = len(file_io.get_trained_keo_models(ke_data_root_dir, ke_ckpt_root_dir))
        if (n_trained_models + 1) >= train_until_n_models:
            return
        else:
            hparams.update({"model_id": n_trained_models + 1, "is_regularized": True})
            prev_training_infos: list[ds.KEOutputTrainingInfo] = file_io.get_trained_keo_models(
                ke_data_root_dir, ke_ckpt_root_dir
            )
            model_id = len(prev_training_infos) + 1
            new_model = nc.MODEL_NAME_TMPLT.format(model_id)

            tbt_model_data_dir = ke_data_root_dir / new_model
            tbt_model_ckpt_dir = ke_ckpt_root_dir / new_model
            training_info = ds.KEOutputTrainingInfo(
                experiment_name=exp_name,
                experiment_description=experiment_description,
                dir_name=new_model,
                model_id=model_id,
                group_id=group_id,
                architecture=str(architecture.value),
                dataset=str(dataset.value),
                softmax_metrics=bool(softmax_mets),
                dissimilarity_loss=dis_loss,
                crossentropy_loss_weight=ce_weight,
                dissimilarity_loss_weight=dis_loss_weight,
                epochs_before_regularization=epochs_before_regularization,
                learning_rate=p.learning_rate,
                split=p.split,
                weight_decay=p.weight_decay,
                batch_size=p.batch_size,
                path_data_root=tbt_model_data_dir,
                path_ckpt_root=tbt_model_ckpt_dir,
                pc_grad=pcgrad,
            )

            all_src_arch_infos = [
                ArchitectureInfo(first_model.architecture, arch_params, first_model.path_ckpt, None)
            ]
            for tkem in prev_training_infos:
                all_src_arch_infos.append(ArchitectureInfo(tkem.architecture, arch_params, tkem.path_ckpt, None))
            tbt_arch_info = ArchitectureInfo(training_info.architecture, arch_params, training_info.path_ckpt, None)
            knowledge_extension_module = OutputRegularizerArch(
                sources=all_src_arch_infos,
                tbt_model=tbt_arch_info,
            )
            if dis_loss_args is None:
                dis_l = find_output_ke_loss(dis_loss, arch_params["n_cls"])
            else:
                dis_l = find_output_ke_loss(dis_loss, arch_params["n_cls"], *dis_loss_args)
            loss = KEOutputTrainLoss(
                dissimilar_loss=dis_l,
                ce_weight=ce_weight,
                dissim_weight=dis_loss_weight,
                regularization_epoch_start=epochs_before_regularization,
            )
            if pcgrad:
                kelm = KEOutputPCGrad(
                    model_info=training_info,
                    network=knowledge_extension_module,
                    save_checkpoints=True,
                    params=p,
                    hparams=hparams,
                    loss=loss,
                    skip_n_epochs=None,
                    log=True,
                )
            else:
                kelm = KEOutputLightningModule(
                    model_info=training_info,
                    network=knowledge_extension_module,
                    save_checkpoints=True,
                    params=p,
                    hparams=hparams,
                    loss=loss,
                    skip_n_epochs=None,
                    log=True,
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
