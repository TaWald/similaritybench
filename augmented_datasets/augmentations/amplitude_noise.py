import copy

import numpy as np
from augmented_datasets.augmentations import power_equalization


def point_symmetric_noise(noise: np.ndarray):
    """ Make the noise of noise[i+1,j+1] == noise[-(i+1), -(j+1)].
    Also makes constant part (middle =0) so no bias is introduced.

    :param noise:
    :return:
    """ ""
    sym_noise = copy.deepcopy(noise)
    assert noise.shape[0] == noise.shape[1], "Expecting a square image right now."
    if noise.shape[0] % 2 != 0:
        mid_id = int(noise.shape[0] / 2) + 1  # Will be actual middle of image now
    else:
        mid_id = int(noise.shape[0] / 2)

    for i in range(mid_id + 1):
        for j in range(mid_id + 1):
            if i == mid_id and j == mid_id:
                sym_noise[i, j] = 0.0
            sym_noise[i + 1, j + 1] = sym_noise[-(i + 1), -(j + 1)]
    return sym_noise


def gaussian_amplitude_noise(image: np.ndarray, noise_sigma: float) -> np.ndarray:
    amplitude, phase = power_equalization.image_to_amplitude_phases(image)

    noise = np.random.normal(loc=0, scale=noise_sigma, size=amplitude.shape)
    sym_noise = point_symmetric_noise(noise)

    amplitude_noised = np.clip(amplitude + sym_noise, a_min=0, a_max=None)  # Negative amplitudes do not exist!
    noised_image = power_equalization.amplitude_phase_to_image(amplitude_noised, phase)
    final_im = np.clip(noised_image, a_min=0, a_max=1)

    return final_im


def speckle_amplitude_noise(image: np.ndarray, noise_sigma: float) -> np.ndarray:
    amplitude, phase = power_equalization.image_to_amplitude_phases(image)

    noise = np.random.normal(loc=0, scale=noise_sigma, size=amplitude.shape)
    sym_noise = point_symmetric_noise(noise)

    amplitude_noised = amplitude * (1 + sym_noise)
    noised_image = power_equalization.amplitude_phase_to_image(amplitude_noised, phase)
    final_im = np.clip(noised_image, a_min=0, a_max=1)

    return final_im
