import os
import sys
from pathlib import Path

from ke.arch.ke_architectures.feature_approximation import FAArch
from ke.losses.ke_loss import KETrainLoss
from ke.training.ke_train_modules.calibrate import calibrate_model
from ke.training.ke_train_modules.IntermediateRepresentationLightningModule import (
    IntermediateRepresentationLightningModule,
)
from ke.training.trainers.base_trainer import BaseTrainer
from ke.util import data_structs as ds
from ke.util import file_io
from ke.util import find_architectures as fa
from ke.util import find_datamodules as fd
from ke.util import name_conventions as nc
from ke.util.data_structs import ArchitectureInfo
from ke.util.default_params import get_default_arch_params
from ke.util.default_params import get_default_parameters
from ke.util.find_ke_loss import find_ke_loss


def main():
    # --------------- Parsing

    if "data" in os.environ:  # on cluster
        on_cluster = True
        path_to_ckpt_dir = Path("/dkfz/cluster/gpu/checkpoints/OE0441/t006d/results/knowledge_extension")
    else:  # local
        on_cluster = False
        path_to_ckpt_dir = Path("/mnt/cluster-checkpoint-all/t006d/results/knowledge_extension")

    for exp in path_to_ckpt_dir.iterdir():
        if "FIRST_MODELS" in exp.name:
            continue

        exp_name = exp.name
        hparams = nc.KENameEncoder.decode(dirname=exp.name)
        experiment_description = hparams[0]
        dataset = ds.Dataset(hparams[1])
        architecture = ds.BaseArchitecture(hparams[2])
        transfer_positions = hparams[3]
        trans_depth = hparams[4]  # tdepth_i,
        trans_kernel = hparams[5]  # kwidth_i,
        group_id = hparams[6]  # group_id_i,
        sim_loss = hparams[7]  # sim_loss,
        sim_loss_weight = hparams[8]  # sim_loss_weight,
        dis_loss = hparams[9]  # dis_loss,
        dis_loss_weight = hparams[10]  # dis_loss_weight,
        ce_weight = hparams[11]  # ce_loss_weight,
        aggregate_reps = hparams[12]  # agg,
        softmax_mets = hparams[13]  # sm,
        epochs_before_regularization = hparams[14]  # ebr,

        # Create paths
        if on_cluster:
            base_data_path = Path(file_io.get_experiments_data_root_path())
            base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())

            ke_data_path = base_data_path / nc.KNOWLEDGE_EXTENSION_DIRNAME
            ke_ckpt_path = base_ckpt_path / nc.KNOWLEDGE_EXTENSION_DIRNAME
        else:
            ke_data_path = Path("/mnt/cluster-data/results/knowledge_extension")
            ke_ckpt_path = path_to_ckpt_dir

        ke_data_root_dir = ke_data_path / exp_name
        ke_ckpt_root_dir = ke_ckpt_path / exp_name

        p: ds.Params = get_default_parameters(architecture.value, dataset)

        # Do the baseline model creation if it not already exists!
        first_model = file_io.get_first_model(
            ke_data_path=ke_data_path,
            ke_ckpt_path=ke_ckpt_path,
            params=p,
            group_id=group_id,
        )
        celu_alpha = 1  # Default value from kwargs (doens't matter since no training done)
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
        tbt_arch = fa.get_base_arch(architecture)
        tmp_arch = tbt_arch(**arch_params)
        tbt_hook: tuple[ds.Hook] = tuple([tmp_arch.hooks[h] for h in transfer_positions])

        if not first_model.model_is_finished():
            continue  # Just for eval of models that have trained base models.
        else:
            n_trained_models: int = len(file_io.get_trained_ke_models(ke_data_root_dir, ke_ckpt_root_dir))
            hparams.update({"model_id": n_trained_models + 1, "is_regularized": True})
            prev_training_infos: list[ds.KETrainingInfo] = file_io.get_trained_ke_models(
                ke_data_root_dir, ke_ckpt_root_dir
            )
            model_id = len(prev_training_infos) + 1
            new_model = nc.MODEL_NAME_TMPLT.format(model_id)

            tbt_model_data_dir = ke_data_root_dir / new_model
            tbt_model_ckpt_dir = ke_ckpt_root_dir / new_model

            training_info: ds.FirstModelInfo = ds.KETrainingInfo(
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
            if (not training_info.has_checkpoint()) or (training_info.model_is_finished()):
                continue

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
                save_approx=True,
            )
            datamodule = fd.get_datamodule(dataset=dataset)
            trainer = BaseTrainer(
                model=kelm,
                datamodule=datamodule,
                params=p,
                basic_training_info=training_info,
                arch_params=arch_params,
            )
            trainer.post_train_eval()
            trainer.save_outputs("test")
            calibrate_model(training_info)

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
