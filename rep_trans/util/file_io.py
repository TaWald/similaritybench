from __future__ import annotations

import collections
import json
import logging
import os
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Iterable
from warnings import warn

import numpy as np

from rep_trans.data.base_datamodule import BaseDataModule
from rep_trans.util import data_structs as ds
from rep_trans.util import name_conventions as nc
from rep_trans.util.find_datamodules import get_datamodule
from rep_trans.util.name_conventions import KEAdversarialLenseOutputNameEncoder
from rep_trans.util.name_conventions import KEAdversarialNameEncoder
from rep_trans.util.name_conventions import KENameEncoder
from rep_trans.util.name_conventions import KEOutputNameEncoder
from rep_trans.util.name_conventions import KESubNameEncoder
from rep_trans.util.name_conventions import KEUnusableDownstreamNameEncoder

logger = logging.getLogger(__name__)


def all_paths_exists(*args: Path):
    for a in args:
        if not a.exists():
            return False
    return True


def get_experiments_data_root_path():
    try:
        EXPERIMENTS_ROOT_PATH = os.environ["DATA_RESULTS_FOLDER"]
    except KeyError:
        raise KeyError("Could not find 'DATA_RESULTS_FOLDER'")
    return EXPERIMENTS_ROOT_PATH


def get_experiments_checkpoints_root_path() -> str:
    try:
        CHECKPOINT_ROOT_PATH = os.environ["CHECKPOINTS_FOLDER"]
    except KeyError:
        raise KeyError("Could not find 'INPUT_DATA_RESULTS_FOLDER'")
    return CHECKPOINT_ROOT_PATH


def save(
    obj: Any,
    path: str | Path,
    filename: str | None = None,
    overwrite: bool = True,
    make_dirs: bool = True,
) -> None:
    """Saves an data to disk. If a filename is given the path is considered to an directory.
    If the filename is not given the path has to have an extension that is supported and gets detected automatically.


    :param obj: Data to be saved
    :param path: Path to a file to save the data to or to a directory.
    :param filename: Optional. can be given should path lead to a directory
     and specifies the filename specifically.
    :param overwrite: Flag if overwriting should take place if a file
     should already exist.
    :param make_dirs: Flag if not existing directories should be created
    :return:
    :raises RuntimeError: If the file should exist and overwrite is not set to true.
    :raises ValueError: If the path leads to a file
    """
    p = str(path)
    _, ext = os.path.splitext(p)

    if (filename is None) and (ext == ""):
        raise ValueError("Expected either a filename in the path or a filename with extension. Can't have neither.")
    elif (filename is None) and (ext != ""):
        dirpath, filename = os.path.split(p)
        if not make_dirs:
            assert os.path.exists(dirpath), f"Given directory does not exist! Path: {dirpath}"
        else:
            os.makedirs(dirpath, exist_ok=True)
    else:
        dirpath = p
        if os.path.isfile(dirpath):
            raise ValueError(
                "Path to a file was given AND a filename is provided." " Filename should be None in this case though!"
            )
        os.makedirs(dirpath, exist_ok=True)

    full_path: str = os.path.join(dirpath, filename)
    extension = os.path.splitext(filename)[1]

    if not overwrite and os.path.isfile(full_path):
        raise FileExistsError(
            "Expected not existing file, found file with identical name." " Set overwrite to true to ignore this."
        )
    else:
        if extension == ".json":
            save_json(obj, full_path)
        elif extension == ".npz":
            save_npz(obj, full_path)
        elif extension == ".pkl":
            save_pickle(obj, full_path)
        elif extension == ".csv":
            save_csv(obj, full_path)
        elif extension == ".npy":
            save_np(obj, full_path)
        else:
            supported_extensions = "json npz pkl csv".split(" ")
            raise NotImplementedError(
                "The given extensions is supported. Supported are: {}".format(supported_extensions)
            )


def load(path: str | Path, filename: str | None = None, mmap=None) -> Any:
    """Basic loading method of the comp Manager.
    Retrieves and loads the file from the specified directory, depending on the
     file extension.
    """
    p = str(path)
    if filename is None:
        path, filename = os.path.split(p)
    else:
        if os.path.isfile(p):
            raise ValueError("Path to a file was given. Filename should be None in this case!")
        else:
            p = os.path.join(p, filename)

    extension = os.path.splitext(filename)[-1]
    if not os.path.exists(p):
        raise ValueError(f"Given path does not exists: {p}")
    else:

        if extension == ".npz":
            return load_npz(p)
        elif extension == ".json":
            return load_json(p)
        elif extension == ".pkl":
            return load_pickle(p)
        elif extension == ".csv":
            return load_csv(p)
        elif extension == ".npy":
            return load_np(p, mmap)
        else:
            supported_extensions = "json npz pkl csv".split(" ")
            raise NotImplementedError(
                f"Loading given extension is not supported!" f" Given: {extension}, Supported:{supported_extensions}"
            )


def strip_state_dict_of_keys(state_dict: dict) -> OrderedDict:
    """Removes the `net` value from the keys in the state_dict

    Example: original contains: "net.features.0.weight"
        current model expects: "features.0.weight"

    :return:
    """
    new_dict = collections.OrderedDict()
    for key, val in state_dict.items():
        new_dict[".".join(key.split(".")[1:])] = val

    return new_dict


def get_first_model(
    experiment_description: str,
    ke_data_path: str | Path,
    ke_ckpt_path: str | Path,
    architecture: str,
    dataset: str,
    learning_rate: float,
    split: int,
    weight_decay: float,
    batch_size: int,
    group_id: int,
) -> ds.BasicTrainingInfo:
    """Return the checkpoint of the group id if it already exists!"""
    edp: Path = (
        Path(ke_data_path)
        / nc.KE_FIRST_MODEL_DIR.format(dataset, architecture)
        / nc.KE_FIRST_MODEL_GROUP_ID_DIR.format(group_id)
    )
    ecp: Path = (
        Path(ke_ckpt_path)
        / nc.KE_FIRST_MODEL_DIR.format(dataset, architecture)
        / nc.KE_FIRST_MODEL_GROUP_ID_DIR.format(group_id)
    )

    first_model = ds.BasicTrainingInfo(
        experiment_description=experiment_description,
        experiment_name=experiment_description,
        dir_name=nc.KE_FIRST_MODEL_GROUP_ID_DIR.format(group_id),
        architecture=architecture,
        dataset=dataset,
        learning_rate=learning_rate,
        split=split,
        weight_decay=weight_decay,
        batch_size=batch_size,
        path_data_root=edp,
        path_ckpt_root=ecp,
        group_id=group_id,
        model_id=0,
    )

    return first_model


def first_model_trained(first_model: ds.BasicTrainingInfo) -> bool:
    """Return true if the info file and checkpoint exists."""

    return first_model.path_train_info_json.exists() and first_model.path_ckpt.exists()


def get_trained_keo_models(exp_data_dir: str | Path, exp_ckpt_dir: str | Path) -> list[ds.KEOutputTrainingInfo]:
    edp: Path = Path(exp_data_dir)
    ecp: Path = Path(exp_ckpt_dir)
    if not all_paths_exists(edp, ecp):
        return []
    else:
        all_models = []
        for m in edp.iterdir():
            model_name = m.name
            if re.match(nc.MODEL_NAME_RE, model_name):
                model_id = int(model_name.split("_")[-1])
                if not ((m / nc.OUTPUT_TMPLT).exists() and (m / nc.KE_INFO_FILE).exists()):
                    continue
                (
                    exp_desc,
                    dataset,
                    architecture,
                    group_id,
                    dis_loss,
                    dis_loss_weight,
                    ce_loss_weight,
                    softmax,
                    epochs_before_regularization,
                    pc_grad,
                ) = KEOutputNameEncoder.decode(ecp.name)

                info = load_json(m / nc.KE_INFO_FILE)
                try:
                    trained_KE_model = ds.KEOutputTrainingInfo(
                        experiment_name=ecp.name,
                        experiment_description=exp_desc,
                        dir_name=model_name,
                        model_id=model_id,
                        group_id=group_id,
                        # HParams for continuation
                        architecture=architecture,
                        dataset=dataset,
                        dissimilarity_loss=dis_loss,
                        softmax_metrics=softmax,
                        epochs_before_regularization=epochs_before_regularization,
                        # HParams
                        learning_rate=info["learning_rate"],
                        split=info["split"],
                        weight_decay=info["weight_decay"],
                        batch_size=info["batch_size"],
                        # KE Trans stuff
                        crossentropy_loss_weight=info["crossentropy_loss_weight"],
                        dissimilarity_loss_weight=info["dissimilarity_loss_weight"],
                        # Relevant paths
                        path_data_root=edp / model_name,
                        path_ckpt_root=ecp / model_name,
                        pc_grad=pc_grad,
                    )
                except KeyError as k:
                    warn(f"Unexpected key in dict {edp / model_name / nc.KE_INFO_FILE}")
                    raise k
                all_models.append(trained_KE_model)
        return all_models


def get_trained_adversarial_lense_models(
    exp_data_dir: str | Path, exp_ckpt_dir: str | Path
) -> list[ds.KEAdversarialLenseOutputTrainingInfo]:
    edp: Path = Path(exp_data_dir)
    ecp: Path = Path(exp_ckpt_dir)
    if not all_paths_exists(edp, ecp):
        return []
    else:
        all_models = []
        for m in edp.iterdir():
            model_name = m.name
            if re.match(nc.MODEL_NAME_RE, model_name):
                model_id = int(model_name.split("_")[-1])
                if not ((m / nc.OUTPUT_TMPLT).exists() and (m / nc.KE_INFO_FILE).exists()):
                    continue
                (
                    exp_desc,
                    dataset,
                    architecture,
                    group_id,
                    lense_setting,
                    adv_loss,
                    adv_loss_weight,
                    lense_rc_weight,
                    ce_loss_weight,
                ) = KEAdversarialLenseOutputNameEncoder.decode(ecp.name)

                info = load_json(m / nc.KE_INFO_FILE)
                try:
                    trained_KE_model = ds.KEAdversarialLenseOutputTrainingInfo(
                        experiment_name=ecp.name,
                        experiment_description=exp_desc,
                        dir_name=model_name,
                        model_id=model_id,
                        group_id=group_id,
                        # HParams for continuation
                        architecture=architecture,
                        dataset=dataset,
                        learning_rate=info["learning_rate"],
                        split=info["split"],
                        weight_decay=info["weight_decay"],
                        batch_size=info["batch_size"],
                        path_data_root=edp / model_name,
                        path_ckpt_root=ecp / model_name,
                        lense_setting=lense_setting,
                        adversarial_loss=adv_loss,
                        lense_reconstruction_weight=info["lense_reconstruction_weight"],
                        lense_adversarial_weight=info["lense_adversarial_weight"],
                        # Relevant paths
                    )
                except KeyError as k:
                    warn(f"Unexpected key in dict {edp / model_name / nc.KE_INFO_FILE}")
                    raise k
                all_models.append(trained_KE_model)
        return all_models


def load_datamodule(source_path) -> BaseDataModule:
    """
    Returns an instance of the datamodule that was used in training of the trained model from the path.
    """
    oj = load_json(source_path / nc.OUTPUT_TMPLT)
    dataset = ds.Dataset(oj["dataset"])
    return get_datamodule(dataset)


def get_trained_ke_models(exp_data_dir: str | Path, exp_ckpt_dir: str | Path) -> list[ds.KETrainingInfo]:
    """
    Iterates through the knowledge_extension results directory.
     CHecks each folder for **sequentially** trained models and whether they finished training.
     Finished models are determined by the existence of the checkpoint and the output.json file being present.
    """

    edp: Path = Path(exp_data_dir)
    ecp: Path = Path(exp_ckpt_dir)
    if not all_paths_exists(edp, ecp):
        return []
    else:
        all_models = []
        for m in edp.iterdir():
            model_name = m.name
            if re.match(nc.MODEL_NAME_RE, model_name):
                model_id = int(model_name.split("_")[-1])
                if not ((m / nc.OUTPUT_TMPLT).exists() and (m / nc.KE_INFO_FILE).exists()):
                    continue
                (
                    exp_desc,
                    dataset_name,
                    architecture_name,
                    hook_positions,
                    tdepth,
                    kwidth,
                    group_id,
                    sim_loss,
                    sim_loss_weight,
                    dis_loss,
                    dis_loss_weight,
                    ce_loss_weight,
                    aggregate_reps,
                    softmax,
                    epochs_before_regularization,
                ) = KENameEncoder.decode(ecp.name)

                info = load_json(m / nc.KE_INFO_FILE)
                try:
                    trained_KE_model = ds.KETrainingInfo(
                        experiment_name=ecp.name,
                        experiment_description=exp_desc,
                        dir_name=model_name,
                        model_id=model_id,
                        group_id=group_id,
                        # HParams for continuation
                        architecture=architecture_name,
                        dataset=dataset_name,
                        similarity_loss=sim_loss,
                        dissimilarity_loss=dis_loss,
                        aggregate_source_reps=aggregate_reps,
                        softmax_metrics=softmax,
                        epochs_before_regularization=epochs_before_regularization,
                        # HParams
                        learning_rate=info["learning_rate"],
                        split=info["split"],
                        weight_decay=info["weight_decay"],
                        batch_size=info["batch_size"],
                        # KE Trans stuff
                        trans_hooks=hook_positions,
                        trans_depth=tdepth,
                        trans_kernel=kwidth,
                        crossentropy_loss_weight=ce_loss_weight,
                        dissimilarity_loss_weight=dis_loss_weight,
                        similarity_loss_weight=sim_loss_weight,
                        # Relevant paths
                        path_data_root=edp / model_name,
                        path_ckpt_root=ecp / model_name,
                    )
                except KeyError as k:
                    warn(f"Unexpected key in dict {edp / model_name / nc.KE_INFO_FILE}")
                    raise k
                all_models.append(trained_KE_model)
        return all_models


def get_trained_ke_unuseable_models(
    exp_data_dir: str | Path, exp_ckpt_dir: str | Path
) -> list[ds.KEUnuseableDownstreamTrainingInfo]:
    """
    Iterates through the knowledge_extension results directory.
     CHecks each folder for **sequentially** trained models and whether they finished training.
     Finished models are determined by the existence of the checkpoint and the output.json file being present.
    """

    edp: Path = Path(exp_data_dir)
    ecp: Path = Path(exp_ckpt_dir)
    if not all_paths_exists(ecp, edp):
        return []
    else:
        all_models = []
        for m in edp.iterdir():
            model_name = m.name
            if re.match(nc.MODEL_NAME_RE, model_name):
                model_id = int(model_name.split("_")[-1])
                if not ((m / nc.OUTPUT_TMPLT).exists() and (m / nc.KE_INFO_FILE).exists()):
                    continue
                (
                    exp_desc,
                    dataset_name,
                    architecture_name,
                    hook_positions,
                    tdepth,
                    kwidth,
                    group_id,
                    transfer_loss_weight,
                    gradient_reversal_scale,
                    ce_loss_weight,
                    epochs_before_regularization,
                ) = KEUnusableDownstreamNameEncoder.decode(ecp.name)

                info = load_json(m / nc.KE_INFO_FILE)
                try:
                    trained_KE_model = ds.KEUnuseableDownstreamTrainingInfo(
                        experiment_name=ecp.name,
                        experiment_description=exp_desc,
                        dir_name=model_name,
                        model_id=model_id,
                        group_id=group_id,
                        # HParams for continuation
                        architecture=architecture_name,
                        dataset=dataset_name,
                        epochs_before_regularization=epochs_before_regularization,
                        # HParams
                        learning_rate=info["learning_rate"],
                        split=info["split"],
                        weight_decay=info["weight_decay"],
                        batch_size=info["batch_size"],
                        # KE Trans stuff
                        trans_hooks=hook_positions,
                        trans_depth=tdepth,
                        trans_kernel=kwidth,
                        crossentropy_loss_weight=ce_loss_weight,
                        transfer_loss_weight=transfer_loss_weight,
                        gradient_reversal_scale=gradient_reversal_scale,
                        # Relevant paths
                        path_data_root=edp / model_name,
                        path_ckpt_root=ecp / model_name,
                    )
                except KeyError as k:
                    warn(f"Unexpected key in dict {edp / model_name / nc.KE_INFO_FILE}")
                    raise k
                all_models.append(trained_KE_model)
        return all_models


def get_trained_ke_adv_models(
    exp_data_dir: str | Path, exp_ckpt_dir: str | Path
) -> list[ds.KEAdversarialTrainingInfo]:
    """
    Iterates through the knowledge_extension results directory.
     CHecks each folder for **sequentially** trained models and whether they finished training.
     Finished models are determined by the existence of the checkpoint and the output.json file being present.
    """

    edp: Path = Path(exp_data_dir)
    ecp: Path = Path(exp_ckpt_dir)
    if not all_paths_exists(ecp, edp):
        return []
    else:
        all_models = []
        for m in edp.iterdir():
            model_name = m.name
            if re.match(nc.MODEL_NAME_RE, model_name):
                model_id = int(model_name.split("_")[-1])
                if not ((m / nc.OUTPUT_TMPLT).exists() and (m / nc.KE_INFO_FILE).exists()):
                    continue
                (
                    exp_desc,
                    dataset_name,
                    architecture_name,
                    hook_positions,
                    tdepth,
                    kwidth,
                    group_id,
                    adv_loss,
                    adv_loss_weight,
                    gradient_reversal_scale,
                    ce_loss_weight,
                    epochs_before_regularization,
                ) = KEAdversarialNameEncoder.decode(ecp.name)

                info = load_json(m / nc.KE_INFO_FILE)
                try:
                    trained_KE_model = ds.KEAdversarialTrainingInfo(
                        experiment_name=ecp.name,
                        experiment_description=exp_desc,
                        dir_name=model_name,
                        model_id=model_id,
                        group_id=group_id,
                        # HParams for continuation
                        architecture=architecture_name,
                        dataset=dataset_name,
                        adversarial_loss=adv_loss,
                        adversarial_loss_weight=adv_loss_weight,
                        gradient_reversal_scale=gradient_reversal_scale,
                        crossentropy_loss_weight=ce_loss_weight,
                        epochs_before_regularization=epochs_before_regularization,
                        # HParams
                        learning_rate=info["learning_rate"],
                        split=info["split"],
                        weight_decay=info["weight_decay"],
                        batch_size=info["batch_size"],
                        # KE Trans stuff
                        trans_hooks=hook_positions,
                        trans_depth=tdepth,
                        trans_kernel=kwidth,
                        # Relevant paths
                        path_data_root=edp / model_name,
                        path_ckpt_root=ecp / model_name,
                    )
                except KeyError as k:
                    warn(f"Unexpected key in dict {edp / model_name / nc.KE_INFO_FILE}")
                    raise k
                all_models.append(trained_KE_model)
        return all_models


def get_trained_kesub_models(exp_data_dir: str | Path, exp_ckpt_dir: str | Path) -> list[ds.KESubTrainingInfo]:
    """
    Iterates through the knowledge_extension results directory.
     CHecks each folder for **sequentially** trained models and whether they finished training.
     Finished models are determined by the existence of the checkpoint and the output.json file being present.
    """

    edp: Path = Path(exp_data_dir)
    ecp: Path = Path(exp_ckpt_dir)
    if not all_paths_exists(ecp, edp):
        return []
    else:
        all_models = []
        for m in edp.iterdir():
            model_name = m.name
            if re.match(nc.MODEL_NAME_RE, model_name):
                model_id = int(model_name.split("_")[-1])
                if not ((m / nc.OUTPUT_TMPLT).exists() and (m / nc.KE_INFO_FILE).exists()):
                    continue
                (
                    exp_desc,
                    dataset_name,
                    architecture_name,
                    hook_positions,
                    tdepth,
                    kwidth,
                    group_id,
                    sim_loss,
                    sim_loss_weight,
                    ce_loss_weight,
                    epochs_before_regularization,
                ) = KESubNameEncoder.decode(ecp.name)

                info = load_json(m / nc.KE_INFO_FILE)
                try:
                    trained_KE_model = ds.KESubTrainingInfo(
                        experiment_name=ecp.name,
                        experiment_description=exp_desc,
                        dir_name=model_name,
                        model_id=model_id,
                        group_id=group_id,
                        # HParams for continuation
                        architecture=architecture_name,
                        dataset=dataset_name,
                        similarity_loss=sim_loss,
                        epochs_before_regularization=epochs_before_regularization,
                        # HParams
                        learning_rate=info["learning_rate"],
                        split=info["split"],
                        weight_decay=info["weight_decay"],
                        batch_size=info["batch_size"],
                        # KE Trans stuff
                        trans_hooks=hook_positions,
                        trans_depth=tdepth,
                        trans_kernel=kwidth,
                        crossentropy_loss_weight=ce_loss_weight,
                        similarity_loss_weight=sim_loss_weight,
                        # Relevant paths
                        path_data_root=edp / model_name,
                        path_ckpt_root=ecp / model_name,
                    )
                except KeyError as k:
                    warn(f"Unexpected key in dict {edp / model_name / nc.KE_INFO_FILE}")
                    raise k
                all_models.append(trained_KE_model)
        return all_models


def save_pickle(data: Any, filepath: str) -> None:
    """Save Python object to pickle.

    :param data: Data to be saved
    :param filepath: Path to save the object to
    :return: None
    """
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    return


def load_pickle(filepath: str) -> Any:
    """Loads the pickled file from the filepath.

    :param filepath: Path to the file to load
    :return: loaded python object.
    """
    with open(filepath, "rb") as f:
        ret = pickle.load(f)
    return ret


def save_json(data: Any, filepath: Path | str) -> None:
    """

    :param data:
    :param filepath:
    :return:
    """
    with open(str(filepath), "w") as f:
        json.dump(data, f, indent=4)
    return


def load_json(filepath: str | Path) -> Any:
    """Load the json again

    :param filepath:
    :return:
    """
    with open(str(filepath)) as f:
        ret = json.load(f)
    return ret


def save_npz(data: dict, filepath: str | Path) -> None:
    # np.savez_compressed(filepath, **data)
    np.savez(str(filepath), **data)
    return


def save_np(data: np.ndarray, filepath: str | Path) -> None:
    # np.savez_compressed(filepath, **data)
    np.save(str(filepath), data)
    return


def load_np(filepath: str | Path, mmap: str = None):
    data = np.load(str(filepath), mmap_mode=mmap)
    return data


def load_npz(filepath: str, mmap: str = None) -> np.ndarray | Iterable | int | float | tuple | dict | np.memmap:
    data = np.load(filepath, mmap_mode=mmap)
    return data


def save_csv(data: np.ndarray, filepath: str) -> None:
    """Saves np.ndarray into csv file."""

    np.savetxt(filepath, data)  # noqa: type
    return


def load_csv(filepath: str) -> np.ndarray:
    """Loads the csv file into a np.ndarray

    :param filepath:
    :return:
    """
    return np.loadtxt(filepath)

