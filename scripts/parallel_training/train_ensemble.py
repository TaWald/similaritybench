import argparse
import sys
from pathlib import Path

from ke.arch.ke_architectures.parallel_model_arch import ParallelModelArch
from ke.losses.output_losses.adaptive_diversity_promoting_regularization import AdaptiveDiversityPromotionV2
from ke.losses.parallel_output_loss import IndependentLoss
from ke.training.ke_train_modules.parallel_ensemble import ParallelEnsembleLM
from ke.training.trainers.base_trainer_v2 import BaseTrainerV2
from ke.util import data_structs as ds
from ke.util import default_parser_args as dpa
from ke.util import file_io
from ke.util import find_datamodules as fd
from ke.util import name_conventions as nc
from ke.util.data_structs import ArchitectureInfo
from ke.util.default_params import get_default_arch_params
from ke.util.default_params import get_default_parameters


def main():
    print("Getting started!")
    parser = argparse.ArgumentParser(description="Specify model hyperparams.")
    dpa.ke_parallel_parser_arguments(parser)
    args = parser.parse_args()
    print("Parsing done.")

    experiment_description: str = args.experiment_name
    architecture: ds.BaseArchitecture = ds.BaseArchitecture(args.architecture)
    dataset: ds.Dataset = ds.Dataset(args.dataset)

    n_models = args.n_models

    arch_params = get_default_arch_params(dataset)

    p: ds.Params = get_default_parameters(architecture.value, dataset)
    p = dpa.overwrite_params(p, args)
    p.learning_rate = 1e-3  # As demanded in Pang
    p.batch_size = 64
    p.num_epochs = 180
    n_models = 3

    group_id = args.group_id

    dis_loss = args.dis_loss
    dis_loss_weight = args.dis_loss_weight
    ce_weight = args.ce_loss_weight

    if dis_loss_weight == 0.0:
        dis_loss = "None"
    elif dis_loss_weight is None:
        dis_loss_weight = ""
    else:
        dis_loss_weight = f"{dis_loss_weight:.2f}"

    # Create the directory name that will get used for experiment.
    exp_name = nc.KEParallel.encode(
        experiment_description=experiment_description,
        dataset=dataset.value,
        n_models=n_models,
        architecture=architecture.value,
        group_id=group_id,
        dis_loss=dis_loss,
        dis_loss_weight=dis_loss_weight,
        ce_loss_weight=ce_weight,
    )

    # Create paths
    base_ckpt_path = Path(file_io.get_experiments_checkpoints_root_path())
    # Added dataset back in because its so annoying without
    ke_ckpt_path = base_ckpt_path / nc.KNOWLEDGE_EXTENSION_DIRNAME / dataset.value
    loss = AdaptiveDiversityPromotionV2()
    loss = IndependentLoss()
    # loss = SaliencyLoss()

    hparams = {
        "crossentropy_loss_weight": ce_weight,
        "dissimilarity_loss_weight": dis_loss_weight,
        "dissimilarity_loss": dis_loss,
        "group_id": group_id,
        "model_id": None,
        "is_regularized": None,
        "n_cls": fd.get_datamodule(dataset).n_classes,
        **vars(p),
    }

    arch_params = get_default_arch_params(dataset)
    parallel_info: ds.ParallelModelInfo = ds.ParallelModelInfo(
        dir_name=exp_name,
        n_models=n_models,
        group_id=group_id,
        architecture=architecture,
        dataset=dataset,
        path_root=ke_ckpt_path,
    )
    arch_infos = [ArchitectureInfo(architecture, arch_params, None, None) for _ in range(n_models)]
    net = ParallelModelArch(model_infos=arch_infos)  # Done
    lightning_mod = ParallelEnsembleLM(
        model_infos=[parallel_info for _ in range(n_models)],
        network=net,
        params=p,
        hparams=hparams,
        loss=loss,
        log=True,
    )
    datamodule = fd.get_datamodule(dataset=dataset)
    trainer = BaseTrainerV2(
        model=lightning_mod,
        datamodule=datamodule,
        params=p,
        root_dir=ke_ckpt_path,
        arch_params=arch_params,
    )
    trainer.train()

    return


if __name__ == "__main__":
    main()
    sys.exit(0)
