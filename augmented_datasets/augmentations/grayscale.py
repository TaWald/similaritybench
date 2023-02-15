import numpy as np
from skimage.color import rgb2gray


def grayscale_image(image: np.ndarray) -> np.ndarray:
    """Calculates the mean value along the last dimension (which has to be the color
    channels).
    Then expands it so the channels are the same again.

    :param image: Image with vlaues between [0, and 1]
    :return:
    """
    aug_image_single_channel = np.mean(image, axis=-1)
    return aug_image_single_channel


def scikit_grayscale_image(image: np.ndarray) -> np.ndarray:
    """Calculates the greyscale images optiimized for CRT images.

    :param image: Image with channels last
    :return: augmented image, without channel dimension.
    """
    return rgb2gray(image)
