from __future__ import annotations

from pathlib import Path
from rep_trans.util import name_conventions as nc


def is_calibrated(trained_model_data_root_path: Path) -> bool:
    """Returns true if model finished calibration already.
    Checks if the calib.json exists in the data_root_directory.

    :param trained_model_data_root_path: Data root directory of the model
    """
    calib_file = trained_model_data_root_path / nc.CALIB_TMPLT
    return calib_file.exist()


def is_trained(trained_model_data_root_path: Path) -> bool:
    """Returns true if model finished calibration already.
    Checks if the output.json exists in the data_root_directory.

    :param trained_model_data_root_path: Data root directory of the model
    """
    output_file = trained_model_data_root_path / nc.OUTPUT_TMPLT
    return output_file.exists()


def has_checkpoint(trained_model_ckpt_root_path: Path) -> bool:
    """Returns true if model has the final checkpoint already.
    Checks if the final.ckpt exists in the ckpt directory.

    :param trained_model_ckpt_root_path: CKPT root directory of the model
    """
    ckpt_file = trained_model_ckpt_root_path / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME
    return ckpt_file.exists()


def measured_generalization(trained_model_ckpt_root_path: Path) -> bool:
    """Returns true if model evaluated generalization performance.
    Checks if the output.json exists in the data_root_directory.

    :param trained_model_ckpt_root_path: Data root directory of the model
    """
    gen_file = trained_model_ckpt_root_path / nc.GENERALIZATION_TMPLT
    return gen_file.exists()


def has_predictions(trained_model_data_root_path: Path) -> bool:
    """
    Returns true if model has prediction logits and groundtruths of the test set already.
    Checks if the output.json exists in the data_root_directory.

    :param trained_model_data_root_path: Data root directory of the model
    """
    preds = trained_model_data_root_path / nc.ACTI_DIR_NAME / nc.MODEL_TEST_PD_TMPLT
    gts = trained_model_data_root_path / nc.ACTI_DIR_NAME / nc.MODEL_TEST_GT_TMPLT
    return preds.exists() and gts.exists()


def model_is_finished(data_path: Path, ckpt_path) -> bool:
    """
    Verifies that the path provded contains a finished trained model.
    """
    ckpt = ckpt_path / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME
    output_json = data_path / nc.OUTPUT_TMPLT

    return ckpt.exists() and output_json.exists()
