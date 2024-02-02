import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets

log = logging.getLogger(__name__)


def get_dataset(
    dataset_name: str, config: Optional[str] = None
) -> datasets.dataset_dict.DatasetDict:
    ds = datasets.load_dataset(dataset_name, config)
    assert isinstance(ds, datasets.dataset_dict.DatasetDict)
    return ds


def convert_to_path_compatible(s: str) -> str:
    return s.replace("\\", "-").replace("/", "-")


def _parse_llmeval_metadata(arrow_directory: Union[str, Path]) -> Dict[str, Any]:
    with open(Path(arrow_directory).joinpath("dataset_info.json"), "r") as f:
        info = json.load(f)

    regex = (
        r"Logged outputs \(hidden representations\) of the language model "
        r"(?P<model>.*) on the (?P<dataset>.*) dataset.\n    "
        r"Model information: (?P<model_info>.*)"
    )
    match = re.search(regex, info["description"], re.MULTILINE)
    if match:
        info = match.groupdict()
        if "validation" in str(arrow_directory):
            split = "validation"
            info["split"] = split
    info["model"] = convert_to_path_compatible(info["model"]).replace("_", "-")
    info["dataset"] = convert_to_path_compatible(info["dataset"]).replace("_", "-")

    return info


def _parse_hfpipe_metadata(directory: Union[str, Path]) -> Dict[str, Any]:
    tensors_fname = next(Path(directory).glob("*.safetensors")).name
    _, model, dataset, split = tensors_fname.split("__")
    split = split.split(".")[0]
    return {"split": split, "model": model, "dataset": dataset}


def extract_info(directory: Union[str, Path]) -> Dict[str, Any]:
    # Representations may come from two different sources in different formats
    log.debug("Looking for metadata in %s", directory)
    if Path(directory, "dataset_info.json").exists():  # comes from LLMEval
        return _parse_llmeval_metadata(directory)
    elif list(Path(directory).glob("*.safetensors")):  # comes from this project
        return _parse_hfpipe_metadata(directory)
    else:
        raise ValueError(f"No metadata found at {str(directory)}")


def filter_combinations(
    combinations: List[Tuple[Path, Path]],
    must_contain_all: List[str],
    must_contain_any: List[str],
    must_not_contain: List[str],
    one_must_contain: List[str],
) -> List[Tuple[Path, Path]]:
    if must_contain_all:
        combinations = [
            c
            for c in combinations
            if all(s in str(c[0]) and s in str(c[1]) for s in must_contain_all)
        ]

    if must_contain_any:
        combinations = [
            c
            for c in combinations
            if any(s in str(c[0]) for s in must_contain_any)
            and any(s in str(c[1]) for s in must_contain_any)
        ]

    if must_not_contain:
        combinations = [
            c
            for c in combinations
            if not any(s in str(c[0]) for s in must_not_contain)
            and not any(s in str(c[1]) for s in must_not_contain)
        ]

    if one_must_contain:
        combinations = [
            c
            for c in combinations
            if all(s in str(c[0]) or s in str(c[1]) for s in one_must_contain)
        ]

    return combinations


def pair_should_be_compared(
    info1: Dict[str, str],
    info2: Dict[str, str],
    pair_results_path: Path,
    recompute: bool,
) -> bool:
    if (
        info1["dataset"] != info2["dataset"]
        or info1["split"] != info2["split"]
        or info1["model"] == info2["model"]
    ):
        return False

    if pair_results_path.exists() and not recompute:
        log.info(f"Skipping comparison due to existing results at {pair_results_path}")
        return False

    return True


def extend_dict_of_lists(
    to_extend: Dict[Any, List[Any]], to_add: Dict[Any, List[Any]]
) -> None:
    for key, value in to_add.items():
        if key in to_extend:
            to_extend[key].extend(value)
        else:
            to_extend[key] = value
