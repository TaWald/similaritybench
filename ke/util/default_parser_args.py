from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Iterable

import git
from ke.util import data_structs as ds
from ke.util import name_conventions as nc


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
        "-ar",
        "--aggregate_reps",
        type=str2bool,
        required=True,
        help="Indicate whether to aggregate representations of all source models before approximating or not."
        "Setting to false leads to multiple approximations which independently try to approximate the target.",
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
        "-ebr",
        "--epochs_before_regularization",
        type=int,
        required=False,
        default=-1,
        help="Epochs till the regularization of representations starts",
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
    parser.add_argument(
        "-td",
        "--transfer_depth",
        default=1,
        type=int,
        nargs="?",
        help="How many convs are between the layers." "1 = 1 conv and 0 ReLus. (n_relu = transfer_depth - 1)",
    )
    parser.add_argument(
        "-tp",
        "--transfer_positions",
        type=int,
        nargs="+",
        required=False,
        help="List of Hook IDs where regularization is supposed to take place!",
    )
    parser.add_argument(
        "-tk",
        "--transfer_kernel",
        default=1,
        type=int,
        choices=[1, 3, 5],
        help="How large is the convolutional kernel for the approximation of the architecture?",
    )
    parser.add_argument(
        "-sm",
        "--softmax_metrics",
        default=False,
        type=str2bool,
        help="Indicator if the r2s of the layers should be softmaxed before averaging.",
    )
    parser.add_argument(
        "-celu_a",
        "--celu_alpha",
        default=1,
        type=float,
        help="Alpha value for the celu activation function.",
    )
    parser.add_argument(
        "-tr_n_models",
        "--train_till_n_models",
        default=1,
        type=int,
        help="The total number of models that should be trained.",
    )
    parser.add_argument(
        "-siml",
        "--sim_loss",
        default="L2Corr",
        type=str,
        help="Indicator which loss will be used for training.",
    )
    parser.add_argument(
        "-simw",
        "--sim_loss_weight",
        default=1,
        type=float,
        help="Weight of the Dissimilarity in the CELU loss",
    )
    parser.add_argument(
        "-disl",
        "--dis_loss",
        default="L2Corr",
        type=str,
        help="Indicator which loss will be used for training.",
    )
    parser.add_argument(
        "-ce",
        "--ce_loss_weight",
        default=1,
        type=float,
        help="Weight of the Cross Entropy in the CELU loss",
    )
    parser.add_argument(
        "-disw",
        "--dis_loss_weight",
        default=1,
        type=float,
        help="Weight of the Dissimilarity in the CELU loss",
    )
    parser.add_argument(
        "-save_apx",
        "--save_approximation_layers",
        default=True,
        type=str2bool,
        required=False,
        help="Flag if approximation layers should be saved.",
    )


def ke_unuseable_downstream_parser_arguments(parser: argparse.ArgumentParser):
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
        "-ar",
        "--aggregate_reps",
        type=str2bool,
        required=True,
        help="Indicate whether to aggregate representations of all source models before approximating or not."
        "Setting to false leads to multiple approximations which independently try to approximate the target.",
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
        "-ebr",
        "--epochs_before_regularization",
        type=int,
        required=False,
        default=-1,
        help="Epochs till the regularization of representations starts",
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
    parser.add_argument(
        "-td",
        "--transfer_depth",
        default=1,
        type=int,
        nargs="?",
        help="How many convs are between the layers." "1 = 1 conv and 0 ReLus. (n_relu = transfer_depth - 1)",
    )
    parser.add_argument(
        "-tp",
        "--transfer_positions",
        type=int,
        nargs="+",
        required=False,
        help="List of Hook IDs where regularization is supposed to take place!",
    )
    parser.add_argument(
        "-tk",
        "--transfer_kernel",
        default=1,
        type=int,
        choices=[1, 3, 5],
        help="How large is the convolutional kernel for the approximation of the architecture?",
    )
    parser.add_argument(
        "-tr_n_models",
        "--train_till_n_models",
        default=1,
        type=int,
        help="The total number of models that should be trained.",
    )
    parser.add_argument(
        "-ce",
        "--ce_loss_weight",
        default=1,
        type=float,
        help="Weight of the Cross Entropy in the CELU loss",
    )
    parser.add_argument(
        "-transw",
        "--transfer_loss_weight",
        default=1,
        type=float,
        help="Weight of the Dissimilarity in the CELU loss",
    )
    parser.add_argument(
        "-grs",
        "--gradient_reversal_scaling",
        default=1,
        type=float,
        help="Gradient reversal scaling factor. Can reduce the factor of the inversed gradient",
    )
    parser.add_argument(
        "-save_apx",
        "--save_approximation_layers",
        default=False,
        type=str2bool,
        required=False,
        help="Flag if approximation layers should be saved.",
    )


def ke_advesarial_default_parser_arguments(parser: argparse.ArgumentParser):
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
        "-ar",
        "--aggregate_reps",
        type=str2bool,
        required=True,
        help="Indicate whether to aggregate representations of all source models before approximating or not."
        "Setting to false leads to multiple approximations which independently try to approximate the target.",
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
        "-ebr",
        "--epochs_before_regularization",
        type=int,
        required=False,
        default=-1,
        help="Epochs till the regularization of representations starts",
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
    parser.add_argument(
        "-td",
        "--transfer_depth",
        default=1,
        type=int,
        nargs="?",
        help="How many convs are between the layers." "1 = 1 conv and 0 ReLus. (n_relu = transfer_depth - 1)",
    )
    parser.add_argument(
        "-tp",
        "--transfer_positions",
        type=int,
        nargs="+",
        required=False,
        help="List of Hook IDs where regularization is supposed to take place!",
    )
    parser.add_argument(
        "-tk",
        "--transfer_kernel",
        default=1,
        type=int,
        choices=[1, 3, 5],
        help="How large is the convolutional kernel for the approximation of the architecture?",
    )
    parser.add_argument(
        "-tr_n_models",
        "--train_till_n_models",
        default=1,
        type=int,
        help="The total number of models that should be trained.",
    )
    parser.add_argument(
        "-advl",
        "--adv_loss",
        default="L2CorrAdversarial",
        type=str,
        help="Indicator which loss will be used for training.",
    )
    parser.add_argument(
        "-advw",
        "--adversarial_loss_weight",
        default=1,
        type=float,
        help="Weight of the Dissimilarity in the CELU loss",
    )
    parser.add_argument(
        "-grs",
        "--gradient_reversal_scaling",
        default=1,
        type=float,
        help="Gradient reversal scaling factor. Can reduce the factor of the inversed gradient",
    )
    parser.add_argument(
        "-ce",
        "--ce_loss_weight",
        default=1,
        type=float,
        help="Weight of the Cross Entropy in the CELU loss",
    )
    parser.add_argument(
        "-save_apx",
        "--save_approximation_layers",
        default=False,
        type=str2bool,
        required=False,
        help="Flag if approximation layers should be saved.",
    )


def keo_default_parser_arguments(parser: argparse.ArgumentParser):
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
        choices=[c.value for c in list(ds.BaseArchitecture)]
        + [c.value for c in list(ds.BasicPretrainableArchitectures)],
        default=ds.BaseArchitecture.RESNET50.value,
        type=str,
        nargs="?",
        help="Name of the architecture to train.",
    )
    parser.add_argument(
        "-pt",
        "--pretrained",
        default=False,
        type=str2bool,
        nargs="?",
        help="Should the architecture be pretrained.",
    )
    parser.add_argument(
        "-fz",
        "--freeze_pretrained",
        default=False,
        type=str2bool,
        nargs="?",
        help="Should the architecture be pretrained.",
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
        "-ebr",
        "--epochs_before_regularization",
        type=int,
        required=False,
        default=-1,
        help="Epochs till the regularization of representations starts",
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
    parser.add_argument(
        "-sm",
        "--softmax_metrics",
        default=False,
        type=str2bool,
        help="Indicator if the r2s of the layers should be softmaxed before averaging.",
    )
    parser.add_argument(
        "-disl",
        "--dis_loss",
        default="NegativeClassCrossEntropy",
        type=str,
        help="Indicator which loss will be used for training.",
    )
    parser.add_argument(
        "-ce",
        "--ce_loss_weight",
        default=1,
        type=float,
        help="Weight of the Cross Entropy in the CELU loss",
    )
    parser.add_argument(
        "-disw",
        "--dis_loss_weight",
        type=str,
        help="Weight of the Dissimilarity in the CELU loss",
    )
    parser.add_argument(
        "-tr_n_models",
        "--train_till_n_models",
        default=1,
        type=int,
        help="The total number of models that should be trained.",
    )
    parser.add_argument(
        "-pcg",
        "--pc_grad",
        default=False,
        type=str2bool,
        help="Flag if PCGrad should be used.",
    )


def pretrained_parser_arguments(parser: argparse.ArgumentParser):
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
        choices=[c.value for c in list(ds.BasicPretrainableArchitectures)],
        type=str,
        required=True,
        help="Name of the architecture to train.",
    )
    parser.add_argument(
        "-pt",
        "--pretrained",
        default=False,
        type=str2bool,
        nargs="?",
        help="Should the architecture be pretrained.",
    )
    parser.add_argument(
        "-warm_pt",
        "--warmup_pretrained",
        default=0,
        type=str2bool,
        nargs="?",
        help="Should the architecture be pretrained.",
    )
    parser.add_argument(
        "-lpo",
        "--linear_probe_only",
        default=0,
        type=str2bool,
        nargs="?",
        help="Should the architecture be pretrained.",
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


def ke_adversarial_lense_parser_arguments(parser: argparse.ArgumentParser):
    keo_default_parser_arguments(parser)
    parser.add_argument(
        "-lrw",
        "--lense_reco_weight",
        type=float,
        required=True,
        help="Reconstruction weight that makes the lense keep the image intact",
    )
    parser.add_argument(
        "-law",
        "--lense_adversarial_weight",
        type=float,
        required=True,
        help="Weight of the thing that makes the lense augment the image",
    )
    parser.add_argument(
        "-ls",
        "--lense_setting",
        type=str,
        required=True,
        help="Architecture String indicating how the Lense is defined.",
    )


def keo_alternating_default_parser_arguments(parser: argparse.ArgumentParser):
    keo_default_parser_arguments(parser)
    parser.add_argument(
        "-tp",
        "--transfer_positions",
        type=int,
        required=True,
        help="Hook ID where regularization is supposed to take place!",
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
