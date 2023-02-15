import numpy as np


def gaussian_noise(contrast_reduced_image: np.ndarray, sigma: float) -> np.ndarray:
    """Takes a contrast reduced image and adds gaussian noise to it. Values above 1
    and below 0 are clipped to the correspoding values.
    The reduction in contrast is necessary to not get too many values out of [0,
    1) range of valid values.


    :param contrast_reduced_image: Contrast reduced image to 30% contrast.
    :param sigma: Standard deviation of the gaussian.
    :return:
    """

    gaussian = np.random.normal(0.0, scale=sigma, size=contrast_reduced_image.shape)
    noised_image = contrast_reduced_image + gaussian
    clipped_image = np.clip(gaussian + noised_image, a_min=0.0, a_max=1.0)

    return clipped_image
