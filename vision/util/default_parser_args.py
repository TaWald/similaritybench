from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Iterable

import git
from vision.util import data_structs as ds
from vision.util import name_conventions as nc


def get_git_hash_of_repo() -> str:
    if "data" in os.environ:
        raise NotImplementedError()
        # repo_path = "/home/t006d/Code/"
        # repo = git.Repo(repo_path, search_parent_directories=True)
    else:
        repo = git.Repo(search_parent_directories=True)

    sha = repo.head.object.hexsha
    return sha


def dir_parser_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-ke",
        "--ke_dir_name",
        type=str,
        choices=[
            nc.KNOWLEDGE_EXTENSION_DIRNAME,
            nc.KNOWLEDGE_UNUSEABLE_DIRNAME,
            nc.KNOWLEDGE_ADVERSARIAL_DIRNAME,
            nc.KE_ADVERSARIAL_LENSE_DIRNAME,
            nc.KE_OUTPUT_REGULARIZATION_DIRNAME,
        ],
        nargs="?",
    )
    parser.add_argument(
        "-n_parallel",
        "--n_parallel",
        type=int,
        default=20,
    )
    parser.add_argument("-id", "--id", type=int, required=True)


def ke_default_parser_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-exp", "--experiment_name", default="cifar10", nargs="?")
    parser.add_argument(
        "-d",
        "--dataset",
        default=ds.Dataset.CIFAR10.value,
        nargs="?",
        choices=[c.value for c in list(ds.Dataset)],
        type=str,
        help="Dataset name to be trained on.",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        choices=[c.value for c in list(ds.BaseArchitecture)],
        default=ds.BaseArchitecture.RESNET50.value,
        type=str,
        nargs="?",
        help="Name of the architecture to train.",
    )
    parser.add_argument(
        "-gid",
        "--group_id",
        type=int,
        required=True,
        help="To differentiate between different groups with same config (for MC runs)",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=int,
        required=False,
        default=0,
        help="Split of the Dataset to train on",
    )
    parser.add_argument(
        "-na",
        "--no_activations",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="Skips creating the activations and only creates outputs preds.",
    )


def overwrite_params(params: ds.Params, args: argparse.Namespace, keys_to_keep: Iterable[str] = tuple()) -> ds.Params:
    """Replaces default values in ds.Params with values passed as args."""
    values_to_overwrite = {}
    p_dict = vars(params)
    for field in p_dict.keys():
        if field in keys_to_keep:
            continue
        if hasattr(args, field) and getattr(args, field) is not None:
            values_to_overwrite[field] = getattr(args, field)

    p = replace(params, **values_to_overwrite)
    return p


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
