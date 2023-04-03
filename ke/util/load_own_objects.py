from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ke.data.base_datamodule import BaseDataModule
from ke.util import data_structs as ds
from ke.util import name_conventions as nc
from ke.util.find_datamodules import get_datamodule


def load_json(filepath: str | Path) -> Any:
    """Load the json again

    :param filepath:
    :return:
    """
    with open(str(filepath)) as f:
        ret = json.load(f)
    return ret


def load_datamodule(source_path) -> BaseDataModule:
    """
    Returns an instance of the datamodule that was used in training of the trained model from the path.
    """
    oj = load_json(source_path / nc.OUTPUT_TMPLT)
    dataset = ds.Dataset(oj["dataset"])
    return get_datamodule(dataset)


def load_datamodule_from_info(model_info: ds.FirstModelInfo) -> BaseDataModule:
    """
    Returns an instance of the datamodule that was used in training of the trained model from the path.
    """
    oj = load_json(model_info.path_output_json)
    dataset = ds.Dataset(oj["dataset"])
    return get_datamodule(dataset)


def load_temperature_from_info(model_info: ds.FirstModelInfo):
    if model_info.is_calibrated():
        return load_json(model_info.path_calib_json)["val"]["temperature"]
    else:
        return np.NAN
