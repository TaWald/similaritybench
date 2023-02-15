import numpy as np

"""
For  theuniform  noise  experiment,
we  first  scaled  all  images  to  a  contrast  levelofc=  30%.Subsequently,
   white  uniform  noise  of  range[−w,w] was  added  pixelwise, w ∈ {0.0,0.03,0.05,
   0.1,0.2,0.35,0.6,0.9}.
     In case this resulted in a value out of the [0, 1] range, this value was clipped
     to either 0 or 1.
      By design, this never occurred for a noise range less or equal to 0.35 due to
      the reduced contrast (see above).
       For w= 0.6, clipping occurred in 17.2%of all pixels and for w= 0.9 in 44.4% of
       all pixels.
"""


def uniform_noise(image: np.ndarray, noise_max_val: float) -> np.ndarray:
    """

    :param image: Image with contrast scaling to 30% of original, to minimize the
    clipped pixels.
    :param noise_max_val: indicates the interval the uniform noise can take [
    -noise_max_val, +noise_max_val)
    :return:
    """

    noise = np.random.uniform(low=-noise_max_val, high=noise_max_val, size=image.shape)
    noised_image = image + noise
    clipped_image = np.clip(noised_image, a_min=0, a_max=1)
    return clipped_image
