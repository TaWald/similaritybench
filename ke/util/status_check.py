from __future__ import annotations

from pathlib import Path

from ke.util import name_conventions as nc


def model_is_finished(data_path: Path, ckpt_path) -> bool:
    """
    Verifies that the path provded contains a finished trained model.
    """
    ckpt = ckpt_path / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME
    output_json = data_path / nc.OUTPUT_TMPLT

    return ckpt.exists() and output_json.exists()
