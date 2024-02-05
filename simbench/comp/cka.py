import numpy as np
import torch


def calc_rsm(activations: torch.Tensor) -> torch.Tensor:
    """Computes the RSM of the activations.
    Expects the activations to be in format: B x p with p being the number of features."""

    activations = activations.cuda()
    zero_mean_acti = activations - torch.mean(activations, dim=0, keepdim=True)

    return zero_mean_acti @ zero_mean_acti.T


def biased_hsic(K: torch.Tensor, L: torch.Tensor):
    """Calculates the unbiased HSIC estimate between two variables X and Y.
    Shape of the input should be (N, N) (already calculated)
    """
    return torch.trace(K @ L) / ((K.shape[0] - 1) ** 2)


def cka_from_rsms(K: torch.Tensor, L: torch.Tensor) -> float:
    """Compares the activations of both networks and outputs.
    Expects the activations to be in format: B x p with p being the number of neurons."""

    K = K.cuda()
    L = L.cuda()

    kl = float(biased_hsic(K, L).cpu().numpy())
    kk = float(biased_hsic(K, K).cpu().numpy())
    ll = float(biased_hsic(L, L).cpu().numpy())

    cka = float(kl / (np.sqrt(kk * ll)))
    return cka


def cka_from_activations(acti_a: torch.Tensor, acti_b: torch.Tensor, batch_size=None):
    """
    Calculates the CKA between two activations.
    Expects the activations to be in format: B x p with p being the number of neurons.
    """
    if batch_size is None:
        batch_size = acti_a.shape[0]

    K = calc_rsm(acti_a)
    L = calc_rsm(acti_b)
    cka = cka_from_rsms(K, L)
    return cka


def cka_from_raw_activation(acti_a: torch.Tensor, acti_b: torch.Tensor):
    """
    Calculates the CKA between two activations.
    Activations need to be [Batch x Spatial [x ...] x Channels]."""
    acti_a = acti_a.reshape(acti_a.shape[0], -1)
    acti_b = acti_b.reshape(acti_b.shape[0], -1)
    return cka_from_activations(acti_a, acti_b)
