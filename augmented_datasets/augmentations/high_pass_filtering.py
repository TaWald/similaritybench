import numpy as np

from augmented_datasets.augmentations import low_pass_filtering

"""
The high-pass experimentalso had eight conditions.
 Standard deviations were 0.4, 0.45, 0.55, 0.7, 1, 1.5, 3 pixels and inf (original
 image) (Figure 11).
  The high-pass filtered images were produced by subtracting a low-pass filtered
  image as described above from the original image.
    However, many of the high-pass filtered images’pixels fell outside the [0, 1] range.
     To resolve this, we calculated the difference between the mean pixel valueover
     all test images (0.4423) and the mean pixel value of the high-pass filtered image.
      That difference was addedback to the image. This had the effect that images
      approached a uniform mean grey image of value 0.4423 forlow standard deviations.
       For both experiments pixel values were clipped to the [0, 1] range, if lying
       outside afterthe filtering.
        This only happened for <0.001% of pixels with a mean clipped away value of
        <0.001 for the bothfiltering experiments
"""


def high_pass_filtering(image: np.ndarray, sigma: float) -> np.ndarray:
    """Calculates the high pass filtered image by subtracting the low_pass_filtered
    image from the normal image.
    To keep the values in [0, 1] range we follow comment above, calculate the mean of
    normal images and the mean of the low_pass_filtered images, and add difference to
    the
    normal image.

    :param image: Original image with channel dimension last.
    :param sigma: Standard deviation of the gaussian used for low-pass filtering
    :return:
    """

    current_low_passed_image = low_pass_filtering.low_pass_filtering(image, sigma)
    low_passed_image_mean = np.mean(current_low_passed_image)
    current_image_mean = np.mean(image)
    high_passed_image = (
        image + (low_passed_image_mean - current_image_mean)
    ) - current_low_passed_image

    if np.max(high_passed_image) > 1 and np.min(high_passed_image) < 0:
        print("Warning: Values out of range 0 and 1.")

    high_passed_clipped_image = np.clip(high_passed_image, a_min=0, a_max=1)

    return high_passed_clipped_image
