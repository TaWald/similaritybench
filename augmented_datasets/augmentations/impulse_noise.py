import numpy as np


def impulse_noise(
    image: np.ndarray, noise_probability: float, lower_bound=0, upper_bound=1.0
):
    """Takes an (gray) image and applies Salt and Pepper noise to it with the
    probability given by `noise_probability` which is in PERCENT!

    :param image: Grayscale image
    :param noise_probability: Probability for impulse to apply [0, 100] (%)
    :param lower_bound: lower bound for values default:0 in range [0, upper_bound]
    :param upper_bound: upper bound for values default:1 in range [lower_bound, 1]
    :return:
    """

    noise_probability = noise_probability / 100
    uniform__noise = np.random.random(size=image.shape)
    random_impulse_values = np.random.uniform(
        low=lower_bound, high=upper_bound, size=image.shape
    )

    assert (
        upper_bound >= lower_bound >= 0
    ), "Values have to be smaller or equal to 1 and greater or equal 0!"
    assert (
        1 >= upper_bound >= lower_bound
    ), "Values have to be smaller or equal to 1 and greater or equal 0!"

    indices = np.where(
        uniform__noise >= noise_probability, image, random_impulse_values
    )
    return indices
