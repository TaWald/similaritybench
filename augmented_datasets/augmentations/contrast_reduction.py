import numpy as np


def reduce_contrast(image: np.ndarray, contrast_percentage: float) -> np.ndarray:
    """Reduces the range of values by scaling it lower and moving it to the middle of
    the

    :param image: RGB or Grayscale image with values between 0 and 1
    :param contrast_percentage: Perecentage of the original contrast -- IN PERCENT!
    :return:
    """
    aug_image = (contrast_percentage / 100) * image  # Change Intensity values
    aug_image = aug_image + (1 - (contrast_percentage / 100)) / 2  # Shift it to the middle of the value range
    return aug_image
