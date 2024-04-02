import shutil
import subprocess

from loguru import logger
from vision.train_vision_model import AUGMENTATION_DATAMODULES
from vision.train_vision_model import SHORTCUT_DATAMODULES
from vision.train_vision_model import train_vision_model


def main():
    src_path = "/home/tassilowald/Code/similaritybench"
    architecture = ["ResNet18", "ResNet34"]
    datasets = SHORTCUT_DATAMODULES
    seeds = [0, 1, 2, 3, 4]
    setting_identifier = "Shortcut_ColorDot"
    overwrite = False

    for arch in architecture:
        for dataset in datasets:
            for seed in seeds:
                logger.info(f"Training {arch} on {dataset} with seed {seed} and setting {setting_identifier}")
                train_vision_model(arch, str(dataset.value), seed, setting_identifier, overwrite)
                print(f"Trained {arch} on {dataset} with seed {seed} and setting {setting_identifier}")


if __name__ == "__main__":
    main()
