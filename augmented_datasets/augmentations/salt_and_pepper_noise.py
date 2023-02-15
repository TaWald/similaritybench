import numpy as np

"""
For thesalt and pepper noise experiment, used in DNNtraining experiments,
 we also scaled the greyscale image to a contrast level of 30% prior to adding noise
 in orderto ensure maximal comparability with the uniform noise experiment.
 Salt and pepper noise, i.e. setting pixels toeither black or white, was drawn
 pixelwise with a certain probabilityp, p ∈ {0,10,20,35,50,65,80,95}%.
 See Figure 14 for example salt-and-pepper stimuli at all conditions
"""


def salt_and_pepper_noise(image: np.ndarray, noise_probability: float) -> np.ndarray:
    """Takes an (gray) image and applies Salt and Pepper noise to it with the
    probability given by `noise_probability` which is in PERCENT!


    :param image: Grayscale image
    :param noise_probability: Probability for salt or pepper to apply [0, 100] (%)
    :param val_a: 'salt' value [0, 1]
    :param val_b: 'pepper' value [0, 1]
    :return:
    """
    val_a = 0.0
    val_b = 1.0

    noise_probability = noise_probability / 100
    uniform__noise = np.random.random(size=image.shape)
    random_black_white_values = np.random.random(size=image.shape)

    assert 1 >= val_a >= 0, "Values have to be smaller or equal to 1 and greater or equal 0!"
    assert 1 >= val_b >= 0, "Values have to be smaller or equal to 1 and greater or equal 0!"

    black_white_values = np.where(
        random_black_white_values >= 0.5,
        np.full_like(image, fill_value=val_a),
        np.full_like(image, val_b),
    )  # Random Black/Whitevalues
    indices = np.where(uniform__noise >= noise_probability, image, black_white_values)
    return indices
