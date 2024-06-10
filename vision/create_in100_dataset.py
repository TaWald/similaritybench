import os
from argparse import ArgumentParser
from pathlib import Path

from repsim.benchmark.paths import VISION_DATA_PATH
from vision.data.imagenet100_ds import create_IN100_datset_from_IN1k


def main(in1k_path: str):
    if "RAW_DATA" in os.environ:
        dataset_path = Path(os.environ["RAW_DATA"])
    elif "data" in os.environ:
        dataset_path = Path(os.environ["data"])
    else:
        dataset_path = Path(VISION_DATA_PATH)
    create_IN100_datset_from_IN1k(IN100_outpath=(dataset_path / "Imagenet100"), path_to_IN1k=in1k_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--in1k_path",
        "-ip",
        help="Path to a dir, containing `ILSVRC` directory.",
        type=str,
        required=True,
    )

    in1k_path = parser.parse_args().in1k_path
    main(in1k_path)
