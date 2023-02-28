from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ke.arch.abstract_acti_extr import AbsActiExtrArch
from ke.data.base_datamodule import BaseDataModule
from ke.deprecated_files.prediction_evaluation import binary_cohens_kappa
from ke.deprecated_files.prediction_evaluation import error_inconsistency
from ke.util import file_io as io
from ke.util import name_conventions as nc


@dataclass
class Output:
    prediction: np.ndarray
    groundtruth: np.ndarray


def load_pretrained_val(model_path: Path) -> Output | None:
    test_pd = model_path / nc.ACTI_DIR_NAME / nc.MODEL_TEST_PD_TMPLT
    test_gt = model_path / nc.ACTI_DIR_NAME / nc.MODEL_TEST_GT_TMPLT

    if test_pd.exists() and test_gt.exists():
        preds = io.load(test_pd)
        gt = io.load(test_gt)
        if len(gt.shape) == 1:
            old_gt = gt
            gt = np.zeros((gt.shape[0], int(np.max(old_gt)) + 1))
            gt[np.arange(gt.shape[0]), old_gt] = 1.0
        return Output(prediction=preds, groundtruth=gt)
    else:
        return None


def calculate_cohens_kappa(out1: Output, out2: Output) -> float:
    pred1 = np.argmax(out1.prediction, axis=-1)
    pred2 = np.argmax(out2.prediction, axis=-1)
    gt = np.argmax(out1.groundtruth, axis=-1)
    cohens_kappa = binary_cohens_kappa(pred1, pred2, gt)
    return cohens_kappa


def calculate_error_inconsitency(out1: Output, out2: Output) -> float:
    pred1 = np.argmax(out1.prediction, axis=-1)
    pred2 = np.argmax(out2.prediction, axis=-1)
    gt = np.argmax(out1.groundtruth, axis=-1)
    cohens_kappa = error_inconsistency(pred1, pred2, gt)
    return cohens_kappa


def calculate_error_iou(out1: Output, out2: Output) -> float:
    pred1 = np.argmax(out1.prediction, axis=-1)
    pred2 = np.argmax(out2.prediction, axis=-1)
    gt = np.argmax(out1.groundtruth, axis=-1)
    union = np.sum(np.logical_or((pred1 != gt), (pred2 != gt), dtype=int), dtype=float)
    intersect = np.sum(np.logical_and((pred1 != gt), (pred2 != gt), dtype=int), dtype=float)
    return intersect / union


def filter_none(some_list: list[Any | None]) -> list:
    return [o for o in some_list if o is not None]


def main():
    moi: AbsActiExtrArch
    im: AbsActiExtrArch
    datamodule: BaseDataModule

    # Pretrain init no warm up
    pretrain_init_path = Path("/home/tassilowald/Data/Results/knowledge_extension_output_reg/unfrozen_pretrained")
    pretrained_models = ["groupid_0", "groupid_1", "groupid_2", "groupid_3", "groupid_4"]
    # Scratch
    scratch_model_path = Path("/home/tassilowald/Data/Results/knowledge_extension_output_reg/unfrozen_scratch")
    scratch_trained_models = ["groupid_10", "groupid_11", "groupid_12", "groupid_13"]

    linear_probing_with_warmup = Path("/home/tassilowald/Data/Results/test_pretraining")
    lp_trained_models = [
        "test_pretraining__CIFAR10__TvResNet34__GroupID_0__Pretrained_1__WarmUp_1__LinearProbeOnly_1",
        "test_pretraining__CIFAR10__TvResNet34__GroupID_1__Pretrained_1__WarmUp_1__LinearProbeOnly_1",
        "test_pretraining__CIFAR10__TvResNet34__GroupID_2__Pretrained_1__WarmUp_1__LinearProbeOnly_1",
        "test_pretraining__CIFAR10__TvResNet34__GroupID_3__Pretrained_1__WarmUp_1__LinearProbeOnly_1",
        "test_pretraining__CIFAR10__TvResNet34__GroupID_4__Pretrained_1__WarmUp_1__LinearProbeOnly_1",
    ]

    pretrained_outputs = filter_none([load_pretrained_val(pretrain_init_path / ptm) for ptm in pretrained_models])
    scratch_outputs = filter_none([load_pretrained_val(scratch_model_path / ptm) for ptm in scratch_trained_models])
    lp_outputs = filter_none([load_pretrained_val(linear_probing_with_warmup / lptm) for lptm in lp_trained_models])

    combis = itertools.combinations(pretrained_outputs + scratch_outputs + lp_outputs, r=2)
    all_close = []
    for out1, out2 in combis:
        all_close.append(np.allclose(out1.groundtruth, out2.groundtruth))
    assert np.all(all_close), "Should be same gt order!"

    pretrained_combis = list(itertools.combinations(pretrained_outputs, r=2))
    scratch_combis = list(itertools.combinations(scratch_outputs, r=2))
    lp_combis = list(itertools.combinations(lp_outputs, r=2))
    pt_scr_combis = list(itertools.product(pretrained_outputs, scratch_outputs))
    pt_lp_combis = list(itertools.product(pretrained_outputs, lp_outputs))
    scr_lp_combis = list(itertools.product(scratch_outputs, lp_outputs))

    pt_cohens_kappas = [calculate_cohens_kappa(out1, out2) for out1, out2 in pretrained_combis]
    sc_cohens_kappas = [calculate_cohens_kappa(out1, out2) for out1, out2 in scratch_combis]
    lp_cohens_kappas = [calculate_cohens_kappa(out1, out2) for out1, out2 in lp_combis]
    pt_sc_cross_cohens_kappas = [calculate_cohens_kappa(out1, out2) for out1, out2 in pt_scr_combis]
    pt_lp_cross_cohens_kappas = [calculate_cohens_kappa(out1, out2) for out1, out2 in pt_lp_combis]
    scr_lp_cross_cohens_kappas = [calculate_cohens_kappa(out1, out2) for out1, out2 in scr_lp_combis]

    mean_pt_cohens_kappa = np.mean(pt_cohens_kappas)  # noqa
    mean_sc_cohens_kappa = np.mean(sc_cohens_kappas)  # noqa
    maen_lp_cohens_kappa = np.mean(lp_cohens_kappas)  # noqa
    mean_pt_sc_cohens_kappa = np.mean(pt_sc_cross_cohens_kappas)  # noqa
    mean_pt_lp_cohens_kappa = np.mean(pt_lp_cross_cohens_kappas)  # noqa
    mean_scr_lp_cohens_kappa = np.mean(scr_lp_cross_cohens_kappas)  # noqa

    pt_iou = [calculate_error_iou(out1, out2) for out1, out2 in pretrained_combis]
    sc_iou = [calculate_error_iou(out1, out2) for out1, out2 in scratch_combis]
    lp_iou = [calculate_error_iou(out1, out2) for out1, out2 in lp_combis]

    mean_pt_iou = np.mean(pt_iou)  # noqa
    mean_sc_iou = np.mean(sc_iou)  # noqa
    mean_lp_iou = np.mean(lp_iou)  # noqa

    pt_error_inc = [calculate_error_inconsitency(out1, out2) for out1, out2 in pretrained_combis]
    sc_error_inc = [calculate_error_inconsitency(out1, out2) for out1, out2 in scratch_combis]
    lp_error_inc = [calculate_error_inconsitency(out1, out2) for out1, out2 in lp_combis]

    mean_pt_error_inc = np.mean(pt_error_inc)  # noqa
    mean_sc_error_inc = np.mean(sc_error_inc)  # noqa
    mean_lp_error_inc = np.mean(lp_error_inc)  # noqa

    return 0


if __name__ == "__main__":
    main()
    sys.exit(0)
