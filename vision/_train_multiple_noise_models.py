import shutil
import subprocess

from loguru import logger
from vision.train_vision_model import AUGMENTATION_DATAMODULES
from vision.train_vision_model import SHORTCUT_DATAMODULES
from vision.train_vision_model import STANDARD_DATAMODULES


def main():
    src_path = "/home/tassilowald/Code/similaritybench"
    architecture = ["VGG11", "ResNet18", "ResNet34"]

    datasets = AUGMENTATION_DATAMODULES
    seeds = [0, 1, 2, 3, 4]
    setting_identifier = "GaussNoise"
    overwrite = False

    for arch in architecture:
        for dataset in datasets:
            for seed in seeds:
                logger.info(f"Training {arch} on {dataset} with seed {seed} and setting {setting_identifier}")
                subprocess.run(
                    [
                        "python",
                        "/home/tassilowald/Code/similaritybench/vision/train_vision_model.py",
                        "-a",
                        arch,
                        "-d",
                        str(dataset.value),
                        "-s",
                        str(seed),
                        "-sid",
                        setting_identifier,
                        "-o",
                        str(overwrite),
                    ]
                )
                print(f"Trained {arch} on {dataset} with seed {seed} and setting {setting_identifier}")
    del datasets, setting_identifier, seeds


if __name__ == "__main__":
    main()
