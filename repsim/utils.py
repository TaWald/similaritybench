import logging
from typing import Optional

import datasets

log = logging.getLogger(__name__)


def get_dataset(dataset_name: str, config: Optional[str] = None) -> datasets.dataset_dict.DatasetDict:
    ds = datasets.load_dataset(dataset_name, config)
    assert isinstance(ds, datasets.dataset_dict.DatasetDict)
    return ds


def convert_to_path_compatible(s: str) -> str:
    return s.replace("\\", "-").replace("/", "-")
