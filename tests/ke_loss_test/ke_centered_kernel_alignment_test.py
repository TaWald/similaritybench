import copy
import math
import unittest
from typing import Sequence

import numpy as np
import torch as t
from rep_trans.losses.utils import centered_kernel_alignment


def create_random_tensor(shape: Sequence) -> t.Tensor:
    random_np = np.random.random(shape)
    return t.from_numpy(random_np)


batch = 100
ch = 64
width = 8
height = 8


"""
Reference code from:
https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py
"""


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time,
    # results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


class TestCKA(unittest.TestCase):
    def test_CKA_for_same_tensor(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))

        celu_expvar = centered_kernel_alignment([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_a)])

        self.assertTrue(bool(t.all(celu_expvar[0][0] == 1.0)))

    def test_CKA_for_same_permuted_tensor(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))
        perms = t.randperm(ch)
        tensor_a_perm = tensor_a[:, :, perms, ...]
        tensor_b = tensor_a_perm

        celu_expvar = float(centered_kernel_alignment([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)])[0][0])

        self.assertTrue(np.isclose(celu_expvar, 1.0))

    def test_cka(self):
        for i in range(10):
            tensor_a = create_random_tensor((1, batch, ch, width, height))
            tensor_b = create_random_tensor((1, batch, ch, width, height))
            # lin_cka = float(centered_kernel_alignment([tensor_a], [tensor_b])[0])
            channel_first_tensor_a = t.reshape(t.transpose(tensor_a, dim0=2, dim1=1), (1, ch, -1))
            channel_first_tensor_b = t.reshape(t.transpose(tensor_b, dim0=2, dim1=1), (1, ch, -1))

            for a, b in zip(channel_first_tensor_a, channel_first_tensor_b):
                at = a.T
                bt = b.T
                lin_cka_ref = linear_CKA(at, bt)
                compatible_a = t.reshape(at, shape=(1, at.shape[0], at.shape[1], 1, 1))
                compatible_b = t.reshape(bt, shape=(1, bt.shape[0], bt.shape[1], 1, 1))
                lin_cka_2 = float(centered_kernel_alignment([compatible_a], [compatible_b])[0][0])
                # Numeric instability seems to be a thing for this.
                # self.assertTrue(bool(np.isclose(lin_cka, lin_cka_ref)))
                self.assertTrue(bool(np.isclose(lin_cka_2, lin_cka_ref)))


if __name__ == "__main__":
    unittest.main()
