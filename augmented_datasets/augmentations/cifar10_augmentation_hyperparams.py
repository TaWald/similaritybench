from augmented_datasets.augmentations.amplitude_noise import gaussian_amplitude_noise
from augmented_datasets.augmentations.amplitude_noise import speckle_amplitude_noise
from augmented_datasets.augmentations.contrast_reduction import reduce_contrast
from augmented_datasets.augmentations.gaussian_noise import gaussian_noise
from augmented_datasets.augmentations.grayscale import grayscale_image
from augmented_datasets.augmentations.high_pass_filtering import high_pass_filtering
from augmented_datasets.augmentations.impulse_noise import impulse_noise
from augmented_datasets.augmentations.low_pass_filtering import low_pass_filtering
from augmented_datasets.augmentations.phase_noise import gaussian_phase_noise
from augmented_datasets.augmentations.phase_noise import uniform_phase_noise
from augmented_datasets.augmentations.power_equalization import power_equalization
from augmented_datasets.augmentations.salt_and_pepper_noise import salt_and_pepper_noise
from augmented_datasets.augmentations.speckle_noise import speckle_noise
from augmented_datasets.augmentations.uniform_noise import uniform_noise
from augmented_datasets.utils import AugmentationDatasetInfo
from augmented_datasets.utils import InputImageTypes

"""
#### Basic shit
contrast_reduction_values = [100, 50, 30, 15, 10, 5, 3, 1]
# Grayscaling

#### FREQUENCY STUFF
low_pass_values = [0, 0.5, 1, 1.5, 3, 5]
high_pass_values = [1000, 3, 1.5, 1, 0.7, 0.55, 0.45, 0.4]
# Power equalization
uniform_phase_noise_ranges = [0, 30, 60, 90, 120, 150, 180]
gaussian_phase_noise_sigmas = [0.0, 0.25, 0.5, 1, 2, 4]
gaussian_amplitude_noise_sigmas = [0.0, 0.3, 0.7, 1.5, 3, 6]
speckle_amplitude_noise_sigmas = [0.0, 0.1, 0.3, 0.8, 2, 5]

#### NOISES
gaussian_noise_sigmas = [0.0, 0.005, 0.015, 0.03, 0.06, 0.12]
speckle_noise_sigmas = [0.0, 0.005, 0.015, 0.03, 0.06, 0.12]
uniform_noise_sigmas = [0.0, 0.005, 0.015, 0.03, 0.06, 0.12]
impulse_noise_probs = [0, 5, 10, 15, 20, 40, 70, 95]
salt_pepper_noise_probs = [0, 5, 10, 15, 20, 40, 70, 95]
multiplicative_amplitude_noise = [0.0, 0.005, 0.015, 0.03, 0.06, 0.12]
"""
# Had to reduce number of different noise ranges to speed up generalization measurement


# Basic shit
contrast_dataset = AugmentationDatasetInfo(
    augmentation_dirname="contrast",
    values=[100, 50, 30, 5, 3, 1],
    augmentation=reduce_contrast,
    input_image_type=InputImageTypes.GRAY,
)
# Grayscaling
grayscale_dataset = AugmentationDatasetInfo(
    augmentation_dirname="grayscale",
    values=None,
    augmentation=grayscale_image,
    input_image_type=InputImageTypes.COLOR,
)
# FREQUENCY STUFF
low_pass_dataset = AugmentationDatasetInfo(
    augmentation_dirname="low_pass",
    values=[0, 0.5, 1, 1.5, 5],
    augmentation=low_pass_filtering,
    input_image_type=InputImageTypes.GRAY,
)
high_pass_dataset = AugmentationDatasetInfo(
    augmentation_dirname="high_pass",
    values=[1000, 1.5, 0.7, 0.45],
    augmentation=high_pass_filtering,
    input_image_type=InputImageTypes.GRAY,
)
# Power equalization

power_equalization_dataset = AugmentationDatasetInfo(
    augmentation_dirname="power_equalization",
    values=None,
    augmentation=power_equalization,
    input_image_type=InputImageTypes.GRAY,
)

uniform_phase_noise_dataset = AugmentationDatasetInfo(
    augmentation_dirname="uniform_phase_noise",
    values=[0, 60, 120, 180],
    augmentation=uniform_phase_noise,
    input_image_type=InputImageTypes.GRAY,
)
gaussian_phase_noise_dataset = AugmentationDatasetInfo(
    augmentation_dirname="gaussian_phase_noise",
    values=[0.0, 0.25, 0.5, 1, 4],
    augmentation=gaussian_phase_noise,
    input_image_type=InputImageTypes.GRAY,
)
gaussian_amplitude_noise_dataset = AugmentationDatasetInfo(
    augmentation_dirname="gaussian_amplitude_noise",
    values=[0.0, 0.3, 1.5, 6],
    augmentation=gaussian_amplitude_noise,
    input_image_type=InputImageTypes.GRAY,
)
speckle_amplitude_noise_dataset = AugmentationDatasetInfo(
    augmentation_dirname="speckle_amplitude_noise",
    values=[0.0, 0.3, 0.8, 5],
    augmentation=speckle_amplitude_noise,
    input_image_type=InputImageTypes.GRAY,
)

# NOISES
gaussian_noise_dataset = AugmentationDatasetInfo(
    augmentation_dirname="gaussian_noise",
    values=[0.0, 0.005, 0.015, 0.03],
    augmentation=gaussian_noise,
    input_image_type=InputImageTypes.LOWCONTRASTGRAY,
)
speckle_noise_dataset = AugmentationDatasetInfo(
    augmentation_dirname="speckle_noise",
    values=[0.0, 0.005, 0.015, 0.03],
    augmentation=speckle_noise,
    input_image_type=InputImageTypes.LOWCONTRASTGRAY,
)
uniform_noise_dataset = AugmentationDatasetInfo(
    augmentation_dirname="uniform_noise",
    values=[0.0, 0.015, 0.03, 0.12],
    augmentation=uniform_noise,
    input_image_type=InputImageTypes.LOWCONTRASTGRAY,
)
impulse_noise_dataset = AugmentationDatasetInfo(
    augmentation_dirname="impulse_noise",
    values=[0, 5, 15, 40],
    augmentation=impulse_noise,
    input_image_type=InputImageTypes.GRAY,
)
salt_pepper_noise_dataset = AugmentationDatasetInfo(
    augmentation_dirname="salt_pepper_noise",
    values=[0, 5, 10, 20],
    augmentation=salt_and_pepper_noise,
    input_image_type=InputImageTypes.GRAY,
)

# multiplicative_amplitude_noise = [0.0, 0.015, 0.06, 0.12]
