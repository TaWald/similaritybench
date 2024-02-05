import numpy as np
import pandas as pd
from simbench.comp import cca_core
from simbench.comp.cca_dft import fft_resize
from simbench.comp.cca_pw import compute_pwcca


def pwcca_from_raw_activations(conv_acts1, conv_acts2):  # noqa
    """Computes cca similarity between two conv layers with DFT.

    This function takes in two sets of convolutional activations, conv_acts1,
    conv_acts2 After resizing the spatial dimensions to be the same, applies fft
    and then computes the ccas.

    Finally, it applies the inverse fourier transform to get the CCA directions
    and neuron coefficients.

    Args:
              conv_acts1: numpy array with shape
                          [batch_size, height1, width1, num_channels1]
              conv_acts2: numpy array with shape
                          [batch_size, height2, width2, num_channels2]
              compute_dirns: boolean, used to determine whether results also
                             contain actual cca directions.

    Returns:
              all_results: a pandas dataframe, with cca results for every spatial
                           location. Columns are neuron coefficients (combinations
                           of neurons that correspond to cca directions), the cca
                           correlation coefficients (how well aligned directions
                           correlate) x and y idxs (for computing cca directions
                           on the fly if compute_dirns=False), and summary
                           statistics. If compute_dirns=True, the cca directions
                           are also computed.
    """

    height1, width1 = conv_acts1.shape[1], conv_acts1.shape[2]
    height2, width2 = conv_acts2.shape[1], conv_acts2.shape[2]
    if height1 != height2 or width1 != width2:
        height = min(height1, height2)
        width = min(width1, width2)
        new_size = [height, width]
        resize = True
    else:
        height = height1
        width = width1
        new_size = None
        resize = False

    # resize and preprocess with fft
    fft_acts1 = fft_resize(conv_acts1, resize=resize, new_size=new_size)
    fft_acts2 = fft_resize(conv_acts2, resize=resize, new_size=new_size)

    # loop over spatial dimensions and get cca coefficients
    all_results = pd.DataFrame()
    for i in range(height):
        for j in range(width):
            results_dict = compute_pwcca(
                fft_acts1[:, i, j, :].T,
                fft_acts2[:, i, j, :].T,
            )

            # apply inverse FFT to get coefficients and directions if specified
            # if return_coefs:
            #     results_dict["neuron_coeffs1"] = np.fft.ifft2(
            #         results_dict["neuron_coeffs1"]
            #     )
            #     results_dict["neuron_coeffs2"] = np.fft.ifft2(
            #         results_dict["neuron_coeffs2"]
            #     )
            # else:
            #     del results_dict["neuron_coeffs1"]
            #     del results_dict["neuron_coeffs2"]

            # accumulate results
            results_dict["location"] = (i, j)
            results_dict["mean_cca_coef1"] = np.mean(results_dict["cca_coef1"])
            results_dict["mean_cca_coef2"] = np.mean(results_dict["cca_coef2"])
            all_results = all_results.append(results_dict, ignore_index=True)

    return all_results
