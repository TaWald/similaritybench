import functools
import os
from multiprocessing import Pool
from typing import List
from typing import Tuple

import cv2
import numpy as np
from augmented_datasets.augmentations import contrast_reduction
from augmented_datasets.augmentations import grayscale
from augmented_datasets.augmentations import power_equalization
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
from augmented_datasets.utils import InputImageTypes
from tqdm import tqdm


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        res = pickle.load(fo, encoding="latin1")
    return res


def get_cifar_images() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Returns all test images and their labels of CIFAR10 as lists of np.ndarrays"""
    test_batch = "test_batch"

    if "RAW_DATA" in os.environ:
        dataset_path = os.path.join(os.environ["RAW_DATA"], "CIFAR10", "cifar-10-batches-py")
    elif "data" in os.environ:
        dataset_path = os.path.join(os.environ["data"], "cifar10", "cifar-10-batches-py")
    else:
        raise FileNotFoundError("Could not find dataset paths in os.environ. Provide either 'RAW_DATA' or 'data'.")

    test = unpickle(os.path.join(dataset_path, test_batch))
    test_images = np.reshape(test["data"], (-1, 3, 32, 32))
    test_samples = test_images.transpose((0, 2, 3, 1))
    test_labels = test["labels"]

    test_images_l = [test_samples[i] for i in range(test_samples.shape[0])]
    test_labels_l = [test_labels[i] for i in range(test_samples.shape[0])]
    return test_images_l, test_labels_l


def save_array_to_image_folder(image: np.ndarray, filename_of_id_label: str, path: str):  # noqa: E501
    """Saves the images as float values.  They should be normalized beforehand though?
    Creates the path if it should not exist beforehand!

    :param image: Image of the
    :param filename_of_id_label:
    :param path:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, filename_of_id_label + ".png")

    image_uint8 = (image * 255).astype(np.uint8)
    image_uint8 = np.expand_dims(image_uint8, axis=-1)
    cv2.imwrite(output_path, image_uint8)
    return


def save_array_to_numpy_folder(image: np.ndarray, filename_of_id_label: str, path: str):  # noqa: E501
    """Saves the images as float values.  They should be normalized beforehand though?
    Creates the path if it should not exist beforehand!

    :param image: Image of the
    :param filename_of_id_label:
    :param path:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    image = image.astype(np.float32)
    np.save(os.path.join(path, filename_of_id_label + ".npy"), image)
    return


def augment_single_image(
    image,
    filename,
    prefix,
    output_path: str,
    mean_power_spectrum: np.ndarray,
    save_as_numpy: bool,
):
    datasets_of_interest = [
        contrast_dataset,
        low_pass_dataset,
        high_pass_dataset,
        uniform_phase_noise_dataset,
        gaussian_phase_noise_dataset,
        gaussian_amplitude_noise_dataset,
        speckle_amplitude_noise_dataset,
        gaussian_noise_dataset,
        speckle_noise_dataset,
        uniform_noise_dataset,
        impulse_noise_dataset,
        salt_pepper_noise_dataset,
    ]

    if save_as_numpy:
        saving_func = save_array_to_numpy_folder
    else:
        saving_func = save_array_to_image_folder

    color_image = image / 255.0
    # Start with manual Datasets!
    # Grayscaling and keeping as basis for other methods
    gray_scale_path = os.path.join(output_path, grayscale_dataset.augmentation_dirname, prefix)
    os.makedirs(gray_scale_path, exist_ok=True)
    gray_image = grayscale_dataset.augmentation(color_image)
    saving_func(gray_image, filename, path=gray_scale_path)

    low_contrast_gray_image = contrast_reduction.reduce_contrast(gray_image, 30)

    # Power equalization
    power_equalized_image = power_equalization_dataset.augmentation(gray_image, mean_power_spectrum)
    power_equalization_path = os.path.join(output_path, power_equalization_dataset.augmentation_dirname, prefix)
    os.makedirs(power_equalization_path, exist_ok=True)
    saving_func(power_equalized_image, filename, power_equalization_path)

    # All other datasets that are no exceptions. reduced image for later
    for dataset in datasets_of_interest:
        write_path = os.path.join(output_path, dataset.augmentation_dirname, prefix)
        os.makedirs(write_path, exist_ok=True)
        if dataset.input_image_type == InputImageTypes.GRAY:
            image = gray_image
        elif dataset.input_image_type == InputImageTypes.LOWCONTRASTGRAY:
            image = low_contrast_gray_image
        elif dataset.input_image_type == InputImageTypes.COLOR:
            image = color_image
        else:
            raise ValueError(f"Unexpected Enum value received! {dataset.input_image_type}")

        if dataset.values is None:  # Has no multiple values and expects no parameters.
            augmented_image = dataset.augmentation(image)
            saving_func(augmented_image, filename, write_path)
        else:
            for val in dataset.values:
                val_write_path = os.path.join(write_path, str(val))
                os.makedirs(val_write_path, exist_ok=True)
                augmented_image = dataset.augmentation(image, val)
                saving_func(augmented_image, filename, val_write_path)
    return


def main():
    # Settings:
    do_parallel = False  # Set this to False for reproducible craetion
    n_workers: int = 24
    prefix = "test"
    n_example_images: int = 20
    only_example: bool = False
    np.random.seed(12345)

    # Check paths & Set output paths:
    if "RAW_DATA" in os.environ:
        dataset_path = os.path.join(os.environ["RAW_DATA"])
    elif "data" in os.environ:
        dataset_path = os.environ["data"]
    else:
        raise FileNotFoundError("Couldn't find the image data path")

    output_path = os.path.join(dataset_path, "AugmentedCIFAR10")
    png_example_path = os.path.join(output_path, "examples")
    np_path = os.path.join(output_path, "data")

    # Extract Test Images
    test_images, test_labels = get_cifar_images()
    gray_test_images = [grayscale.grayscale_image(train_image) for train_image in test_images]
    mean_test_power_spectrum = power_equalization.calculate_mean_power_spectrum(gray_test_images)

    all_images = []
    all_filenames = []

    # Define outputs
    for image_id, (image, label) in enumerate(zip(test_images, test_labels)):
        filename = f"{image_id:06d}_{int(label):02d}"
        all_images.append(image)
        all_filenames.append(filename)

    # Start Creation
    augment_example_img = functools.partial(
        augment_single_image,
        save_as_numpy=False,
        output_path=png_example_path,
        prefix=prefix,
        mean_power_spectrum=mean_test_power_spectrum,
    )
    augment_numpy_img = functools.partial(
        augment_single_image,
        save_as_numpy=True,
        output_path=np_path,
        prefix=prefix,
        mean_power_spectrum=mean_test_power_spectrum,
    )

    n_all_images = len(all_images)

    if do_parallel:
        # Create Examples
        p = Pool(n_workers)
        p.starmap(augment_example_img, zip(all_images[:n_example_images], all_filenames[:n_example_images]))
        if not only_example:
            p.starmap(augment_numpy_img, zip(all_images, all_filenames))
        p.close()
        p.join()
    else:
        for image_id, (image, filename) in tqdm(
            enumerate(zip(all_images, all_filenames)), desc="Images augmented", total=n_all_images
        ):
            if image_id < n_example_images:
                augment_example_img(image=image, filename=filename)
            if not only_example:
                augment_numpy_img(image=image, filename=filename)
    pass


if __name__ == "__main__":
    main()
