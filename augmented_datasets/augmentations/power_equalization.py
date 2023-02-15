from multiprocessing import Pool
from typing import List

import numpy as np
from scipy import fftpack

"""
 We implemented the equalisation of the power spectra and phase noise in the Fourier
 domain.
 Conversionto frequency domain was accomplished by a fast Fourier transform through
 the application of the fft2() and then fftshift() functions of the Python package
 scipy.fftpack.
 This results in a matrix of complexnumbers F, which represents both the phases and
 amplitudes of the individual frequencies in one complexnumber.
 F is organised in symmetric pairs of complex numbers with just their imaginary part
 differing inits sign and cancelling each other out when reversing the Fourier
 transform again.
 When transforming F to polar coordinates, the angle represents the respective
 frequency’s phase and the distance from the origin represents its amplitude.
 Hence, we extracted the phases and amplitudes of the individual frequencies with the
 functions numpy.angle(F) and numpy.abs(F), respectively.
 Thepower equalisation experiment had two conditions: original and power-equalised (
 Figure 13).
 For the power-equalised images, we first calculated the mean amplitude spectrum over
 all test images, which showed the typical 1f shape [e.g.77,78].
 Thereafter,we set all images amplitudes to the mean amplitude spectrum. Since the
 power spectrum is the square of theamplitude spectrum, the images were essentially
 power-equalised
"""


def forward_fft(image: np.ndarray):
    """Calculates the fft of the image and shifts the frequencies so 0 is at the
    middle of the matrix.

    :param image:
    :return:
    """
    fourier = fftpack.fft2(image)
    shifted = fftpack.fftshift(fourier)
    return shifted


def inverse_fft(shifted_fft):
    """Shifts the 0 frequency back to top left and then calculates the reverse fft to
    go back to x,y space

    :param shifted_fft:
    :return:
    """
    reverse_shift = fftpack.ifftshift(shifted_fft)
    inverse_fft = np.real(fftpack.ifft2(reverse_shift))  # np.real to get rid of small imaginary values.
    return inverse_fft


def fft_to_amplitude_phase(fft: np.ndarray):
    """Returns the mean power spectrum of the gray scale image.
    Transforms it into fourier space via dfft, and calculates the power spectrum.

    :param fft:
    :return:
    """
    amplitudes = np.abs(fft)
    phases = np.angle(fft)

    return amplitudes, phases


def image_to_amplitude_phases(image: np.ndarray):
    shifted_fft = forward_fft(image)
    return fft_to_amplitude_phase(shifted_fft)


def amplitude_phase_to_image(amplitude, phase):
    shifted_fft = amplitude_phase_to_fft(amplitude, phase)
    return np.real(inverse_fft(shifted_fft))


def amplitude_phase_to_fft(amplitude, phase):
    shifted = amplitude * np.exp(1j * phase)
    return shifted


def power_equalization(gray_image: np.ndarray, mean_power_spectrum: np.ndarray) -> np.ndarray:
    """Power Equalization for an gray image.

    :param gray_image: Gray valued image
    :param mean_power_spectrum: Mean power spectrum. (mean amplitudes)
    :return:
    """
    amplitude, phase = image_to_amplitude_phases(gray_image)
    power_equalized_image = amplitude_phase_to_image(mean_power_spectrum, phase)
    return power_equalized_image


# Calculate the mean power across all images before meaning it here
def calculate_mean_power_spectrum(images: List[np.ndarray]) -> np.ndarray:
    p = Pool(24)
    res = p.map(image_to_amplitude_phases, images)
    p.close()
    p.join()
    amplitudes = np.array([amplitude for amplitude, phase in res])
    mean_power_spectrum = np.mean(amplitudes, axis=0)

    return mean_power_spectrum
