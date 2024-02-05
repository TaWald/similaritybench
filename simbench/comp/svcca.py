"""
This is a wrapper around the CCA similarity of Raghu et al.
"""


import torch

from simbench.comp.cca_dft import fourier_ccas


def svcca_from_raw_activations(activations_a: torch.Tensor, activations_b: torch.Tensor, threshold: float = 0.98):
    """
    Compare two representations using Singular Vector Canonical Correlation Analysis (SVCCA).

    This function calculates the SVCCA similarity between two sets of activations, activations_a and activations_b,
    irrespective of their shape. SVCCA measures the similarity between two sets of activations by computing the
    correlation between their singular vectors.

    Parameters:
        activations_a (torch.Tensor): The first set of activations [Batch (data-points) x .
        activations_b (torch.Tensor): The second set of activations.
        threshold (float, optional): The threshold value for considering two activations as similar. Defaults to 0.98.

    Returns:
        float: The SVCCA similarity score between the two sets of activations.
    """

    shape_a = activations_a.shape
    shape_b = activations_b.shape



    if len(shape_a) == 4 and len(shape_b) == 4:
        return fourier_ccas(activations_a.cpu().numpy(), activations_b.cpu().numpy())  # Threshold is 0.98
    else:
        raise NotImplementedError("The SVCCA metric is only implemented for 4D tensors (Batch x Spatial x Channels).")
