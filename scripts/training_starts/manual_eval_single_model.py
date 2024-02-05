from dataclasses import asdict
from pathlib import Path

import torch
from simbench.arch.arch_loading import strip_state_dict_of_keys
from simbench.metrics.ke_metrics import single_output_metrics
from simbench.metrics.ke_metrics import SingleOutMetrics
from simbench.util import data_structs as ds
from simbench.util import find_architectures
from simbench.util import find_datamodules
from simbench.util import name_conventions as nc
from simbench.util.file_io import load_json


def compare_single_model(path_to_dir: Path) -> SingleOutMetrics:
    """
    Loads the models and calculates the performance of the growing ensemble of models.
    So returns a list of results of len(models) - 1 (one for each possible stopping point 2, 3, 4, ..., n models)
    """
    hparams = load_json(path_to_dir / nc.KE_INFO_FILE)
    output_json = load_json(path_to_dir / nc.OUTPUT_TMPLT)
    ckpt_dir = path_to_dir / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME
    arch_kwargs = {
        "n_cls": output_json["n_cls"],
        "in_ch": output_json["in_ch"],
        "input_resolution": output_json["input_resolution"],
        "early_downsampling": output_json["early_downsampling"],
        "global_average_pooling": output_json["global_average_pooling"],
    }

    arch = find_architectures.get_base_arch(ds.BaseArchitecture(hparams["architecture"]))(**arch_kwargs)
    state_dict: dict = torch.load(str(ckpt_dir))

    try:
        arch.load_state_dict(state_dict)
    except RuntimeError as _:  # noqa
        try:
            stripped = strip_state_dict_of_keys(state_dict)
            arch.load_state_dict(stripped)
        except RuntimeError as e:
            raise e

    datamodule = find_datamodules.get_datamodule(ds.Dataset(hparams["dataset"]))
    val_dataloader = datamodule.val_dataloader(
        0,
        transform=ds.Augmentation.VAL,
        **{
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "batch_size": 250,
            "num_workers": 0,
            "persistent_workers": False,
        },
    )

    arch.cuda()
    arch.eval()

    gt = []
    logits = []

    # create 2d array of combinations of all_activations_a and all_activations_b
    with torch.no_grad():
        for batch in val_dataloader:
            x, y = batch
            x = x.cuda()
            gt.append(y.detach().cpu())
            logits.append(arch(x).detach().cpu())

    gt = torch.cat(gt, axis=0)
    predictions = torch.cat(logits, axis=0)

    metrics = single_output_metrics(predictions, gt, datamodule.n_classes)

    print(asdict(metrics))

    return 0


if __name__ == "__main__":
    single_1 = compare_single_model(
        Path("/mnt/cluster-checkpoint-all/t006d/results")
        / nc.KNOWLEDGE_EXTENSION_DIRNAME
        / "FIRST_MODELS__CIFAR10__ResNet34"
        / "groupid_1"
    )
    single_0 = compare_single_model(
        Path("/mnt/cluster-checkpoint-all/t006d/results")
        / nc.KNOWLEDGE_EXTENSION_DIRNAME
        / "FIRST_MODELS__CIFAR10__ResNet34"
        / "groupid_0"
    )
