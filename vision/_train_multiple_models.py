import shutil
import subprocess

from vision.train_vision_model import AUGMENTATION_DATAMODULES
from vision.train_vision_model import train_vision_model


def main():
    src_path = "/home/tassilowald/Code/similaritybench"
    architecture = ["ResNet18"]
    datasets = AUGMENTATION_DATAMODULES
    seeds = [0, 1]
    setting_identifier = "GaussNoise"
    overwrite = False

    for arch in architecture:
        for dataset in datasets:
            for seed in seeds:
                train_vision_model(arch, str(dataset.value), seed, setting_identifier, overwrite)
                print(f"Trained {arch} on {dataset} with seed {seed} and setting {setting_identifier}")


if __name__ == "__main__":
    main()
