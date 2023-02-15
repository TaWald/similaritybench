from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import contrast_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import gaussian_amplitude_noise_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import gaussian_noise_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import gaussian_phase_noise_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import grayscale_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import high_pass_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import impulse_noise_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import low_pass_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import power_equalization_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import salt_pepper_noise_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import speckle_amplitude_noise_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import speckle_noise_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import uniform_noise_dataset
from augmented_datasets.augmentations.cifar10_augmentation_hyperparams import uniform_phase_noise_dataset
from augmented_datasets.data.augmented_cifar10.base_datamodule import BasicDataModule
from augmented_datasets.utils import AugmentedDataLoader


def get_augmented_cifar10_test_dataloader(torch_dataloader_kwargs: dict) -> list[AugmentedDataLoader]:
    """Returns Testset dataloaders of the created CIFAR10 dataset.
    The dataloaders contain the name of the augmentation, its value in case the augmentation has one
    and the (torch) DataLoader instance that can be used to extract
    """
    datasets = [
        # Basics
        grayscale_dataset,
        contrast_dataset,
        # Noise
        gaussian_noise_dataset,
        impulse_noise_dataset,
        salt_pepper_noise_dataset,
        speckle_noise_dataset,
        uniform_noise_dataset,
        # Fourier Space
        power_equalization_dataset,
        gaussian_amplitude_noise_dataset,
        speckle_amplitude_noise_dataset,
        gaussian_phase_noise_dataset,
        uniform_phase_noise_dataset,
        low_pass_dataset,
        high_pass_dataset,
    ]

    all_dataloaders: list[AugmentedDataLoader] = []
    for ds in datasets:
        if ds.values is not None:
            for v in ds.values:
                dm = BasicDataModule(ds, v)
                all_dataloaders.append(
                    AugmentedDataLoader(v, dm.augmentation_dirname, dm.test_dataloader(**torch_dataloader_kwargs))
                )
        else:
            dm = BasicDataModule(ds, None)
            all_dataloaders.append(
                AugmentedDataLoader(None, dm.augmentation_dirname, dm.test_dataloader(**torch_dataloader_kwargs))
            )
    return all_dataloaders
