import numpy as np
from scipy.ndimage import filters

"""
For the low-pass and high-pass experiments we used
thescipy.ndimage.filters.gaussian_filter()function.
Thelow-pass experiment’s eight conditions differed in the standard deviation of the
Gaussian filter.
Standard deviations were 0 (original image), 1, 3, 7, 10, 15 and 40 pixels
(Figure 11. We used constant paddingwith the mean pixel value over the testing images
(0.4423) and truncation at four standard deviations.
"""


def low_pass_filtering(image: np.ndarray, sigma: float) -> np.ndarray:
    aug_image = filters.gaussian_filter(input=image, sigma=sigma)
    return aug_image
