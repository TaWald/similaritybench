from __future__ import annotations

from ke.data.base_datamodule import BaseDataModule
from ke.util import data_structs as ds
from ke.util import name_conventions as nc
from ke.util.file_io import load_json
from ke.util.find_datamodules import get_datamodule


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