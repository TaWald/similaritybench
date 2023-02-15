import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.utils.data
from manual_introspection.introspection_tools.histogram_introspection import *
from manual_introspection.introspection_tools.scatter_introspection import all_layer_scatterplot_histogram
from manual_introspection.utils.activation_results import ActivationResult
from rep_trans.arch.abstract_acti_extr import AbsActiExtrArch
from rep_trans.data.base_datamodule import BaseDataModule
from rep_trans.util import data_structs as ds
from rep_trans.util import name_conventions as nc
from rep_trans.util.file_io import load_np
from torch.utils.data import DataLoader


def load_activation(activation_path: Path) -> (np.ndarray, np.ndarray):
    """
    Expects Model of Interest/Reference paths and loads the models and the corresponding data module.
    The paths are expected to point to (data, ckpt) directory.
    """
    test_preds = activation_path / nc.MODEL_TEST_PD_TMPLT
    gt_preds = activation_path / nc.MODEL_TEST_GT_TMPLT

    predictions: np.ndarray = load_np(test_preds)
    groundtruths: np.ndarray = load_np(gt_preds)

    return predictions, groundtruths


def interesting_exp_var_examples_paths() -> dict:
    examples = {
        "baseline_2": Path(
            "/mnt/cluster-checkpoint/results/knowledge_extension_subtraction/FIRST_MODELS__CIFAR100__ResNet18/groupid_2/activations"
        ),
        "moi_2": Path(
            "/mnt/cluster-data/results/knowledge_extension_subtraction/first_subtraction_test__CIFAR100__ResNet18__GroupID_2__Hooks_8__TDepth_9__KWidth_3__Sim_ExpVar_1.00_1.00__ebr_0/model_0001/activations"
        ),
        "baseline_0": Path(
            "/mnt/cluster-checkpoint/results/knowledge_extension_subtraction/FIRST_MODELS__CIFAR100__ResNet18/groupid_0/activations"
        ),
        "moi_0": Path(
            "/mnt/cluster-data/results/knowledge_extension_subtraction/first_subtraction_test__CIFAR100__ResNet18__GroupID_2__Hooks_8__TDepth_9__KWidth_3__Sim_ExpVar_1.00_1.00__ebr_0/model_0001/activations"
        ),
    }
    return examples


def check_weird_first_results():
    examples = interesting_exp_var_examples_paths()
    baseline_pred, baseline_gt = load_activation(examples["baseline_0"])
    moi_pred, moi_gt = load_activation(examples["moi_0"])

    baseline_yhat = np.argmax(baseline_pred, axis=-1)
    baseline_yhat_gt = np.argmax(baseline_gt, axis=-1)
    moi_yhat = np.argmax(moi_pred, axis=-1)
    moi_yhat_gt = np.argmax(moi_gt, axis=-1)

    assert np.all(moi_yhat_gt == baseline_yhat_gt)

    baseline_acc = np.mean(baseline_yhat == baseline_yhat_gt)
    moi_acc = np.mean(moi_yhat == moi_yhat_gt)

    correct_baseline_pred = baseline_yhat == baseline_yhat_gt
    correct_moi_pred = moi_yhat == moi_yhat_gt

    observed_align = np.mean(correct_baseline_pred == correct_moi_pred)
    expected_align = baseline_acc * moi_acc + (1 - baseline_acc) * (1 - moi_acc)

    cohens_kappa = (observed_align - expected_align) / (1 - expected_align)

    return 0

    # Choose position where to inspect


def main():
    moi: AbsActiExtrArch
    im: AbsActiExtrArch
    datamodule: BaseDataModule

    output_path = Path("/home/tassilowald/Data/Results/knolwedge_extension_pics/introspection_output_pics")
    check_weird_first_results()
    # Model of interest is a regularized model.
    #   Model of refernece is a "normal" == unregularized model!


if __name__ == "__main__":
    main()
    sys.exit(0)
