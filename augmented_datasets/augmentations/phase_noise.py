import copy

import numpy as np
from augmented_datasets.augmentations import power_equalization

"""
There were seven conditions in thephasenoise experiment.  These were 0, 30, 60, 90,
120, 150 and 180 degrees noise widthw(Figure 13).
  To eachfrequency’s phase a phase shift randomly drawn from a continuous uniform
  distribution over the interval[−w,w]was added.
 To ensure that the imaginary parts would later cancel out again, we added the same
 phase noise toboth frequencies of each symmetric pair.
  After performing the respective manipulations, aFnewwas calculatedby recombining
  the new phases and amplitudes.
  Then we did an inverse Fourier transform usingifftshift()and thenifft2(). Finally
  we clipped all pixel values to the [0, 1] range.
 This was the case for 0.038% of pixelswith a mean clipped value of about 0.003 for
 the phase noise experiment and for 0.013% of pixels with a meanclipped value of
 0.005 for the power-equalisation experiment.
"""


def point_asymmetric_noise(noise: np.ndarray):
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
            sym_noise[i + 1, j + 1] = -sym_noise[-(i + 1), -(j + 1)]
    return sym_noise


def uniform_phase_noise(image: np.ndarray, phase_noise_dgr) -> np.ndarray:
    amplitude, phase = power_equalization.image_to_amplitude_phases(image)

    noise = np.random.random(phase.shape) * (phase_noise_dgr / 180) * np.pi
    sign = np.where(np.random.random(phase.shape) > 0.5, np.ones(phase.shape), -np.ones(phase.shape))
    signed_noise = noise * sign

    symmetric_noise = point_asymmetric_noise(signed_noise)

    new_phase = phase + symmetric_noise
    phase_noised_image = power_equalization.amplitude_phase_to_image(amplitude, new_phase)
    phase_noised_image = np.clip(phase_noised_image, a_min=0, a_max=1)
    return phase_noised_image


def gaussian_phase_noise(image: np.ndarray, phase_noise_sigma) -> np.ndarray:
    """Gaussian noise as value of Pi

    :param image:
    :param phase_noise_sigma:
    :return:
    """
    amplitude, phase = power_equalization.image_to_amplitude_phases(image)

    noise = np.random.normal(loc=0.0, scale=phase_noise_sigma, size=phase.shape)

    assert phase.shape[0] == phase.shape[1], "Expecting a square image right now."
    if phase.shape[0] % 2 != 0:
        mid_id = int(phase.shape[0] / 2) + 1  # Will be actual middle of image now
    else:
        mid_id = int(phase.shape[0] / 2)

    for i in range(mid_id + 1):
        for j in range(mid_id + 1):
            if i == mid_id and j == mid_id:
                noise[i, j] = 0.0
            noise[i + 1, j + 1] = -noise[-(i + 1), -(j + 1)]
    symmetric_noise = noise

    new_phase = phase + symmetric_noise
    phase_noised_image = power_equalization.amplitude_phase_to_image(amplitude, new_phase)
    phase_noised_image = np.clip(phase_noised_image, a_min=0, a_max=1)
    return phase_noised_image
