import os
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
from augmented_datasets.augmentations import grayscale


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        res = pickle.load(fo, encoding="latin1")
    return res


def get_cifar_images() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    data_batches = ["data_batch_" + str(i + 1) for i in range(5)]
    test_batch = "test_batch"

    if "RAW_DATA" in os.environ:
        dataset_path_p = Path(os.environ["RAW_DATA"]) / "CIFAR10" / "cifar-10-batches-py"
    elif "data" in os.environ:
        dataset_path_p = Path(os.environ["data"])
    else:
        raise KeyError("Couldn't find environ variable 'RAW_DATA' or 'data'.")

    all_train_images = []
    all_train_labels = []

    for data_batch in data_batches:
        tmp = unpickle(str(dataset_path_p / data_batch))
        all_train_images.append(np.reshape(tmp["data"], (-1, 3, 32, 32)))
        all_train_labels.append(tmp["labels"])

    training_samples = np.concatenate(all_train_images, axis=0)
    training_samples = training_samples.transpose((0, 2, 3, 1))
    train_labels = np.concatenate(all_train_labels, axis=0)
    training_images_l = [training_samples[i] for i in range(training_samples.shape[0])]
    training_labels_l = [train_labels[i] for i in range(train_labels.shape[0])]

    test = unpickle(str(dataset_path_p / test_batch))
    test_images = np.reshape(test["data"], (-1, 3, 32, 32))
    test_samples = test_images.transpose((0, 2, 3, 1))
    test_labels = test["labels"]

    test_images_l = [test_samples[i] for i in range(test_samples.shape[0])]
    test_labels_l = [test_labels[i] for i in range(test_samples.shape[0])]
    return training_images_l, training_labels_l, test_images_l, test_labels_l


def main():
    """Calculate the normalization parameters for the grayscaled image.
    One could only preprocess correcly should the color channels be independent of
    each other.
    If they are not then the variance will vary from the calculated one by meaning them.

    :return:
    """
    train_images, train_labels, test_images, test_labels = get_cifar_images()

    if "RAW_DATA" in os.environ:
        dataset_path_p = Path(os.environ["RAW_DATA"])
    elif "data" in os.environ:
        dataset_path_p = Path(os.environ["data"])
    else:
        raise KeyError("Couldn't find environ variable 'RAW_DATA' or 'data'.")

    output_path = dataset_path_p / "AugmentedCIFAR10"  # noqa
    # png_example_path = os.path.join(output_path, "examples")
    # np_path = os.path.join(output_path, "data")

    # do_parallel = True

    gray_train_images = [grayscale.grayscale_image(train_image) / 255.0 for train_image in train_images]
    mean = np.mean(gray_train_images)
    std = np.std(gray_train_images)
    print("Mean:", mean, "\n", "Std:", std)


if __name__ == "__main__":
    main()
