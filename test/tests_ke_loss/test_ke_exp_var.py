import copy
import unittest
from typing import Sequence

import numpy as np
import torch as t
import torch.nn.functional as F
from ke.losses.utils import celu_explained_variance
from torchmetrics.functional import explained_variance


def create_random_tensor(shape: Sequence) -> t.Tensor:
    random_np = np.random.random(shape)
    return t.from_numpy(random_np)


batch = 100
ch = 64
width = 8
height = 8


class TestCeluExpVar(unittest.TestCase):
    def test_celu_exp_var_one_for_same_tensor(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))
        tensor_b = tensor_a

        celu_expvar = celu_explained_variance([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)])

        self.assertTrue(bool(t.all(celu_expvar[0] == 1.0)))

    def test_celu_exp_var_of_random_tensors_identical_to_torchmetrics(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))
        tensor_b = create_random_tensor((1, batch, ch, width, height))

        cev = celu_explained_variance([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)])

        tensor_a_flat = t.reshape(t.transpose(copy.deepcopy(t.squeeze(tensor_a, 0)), dim0=1, dim1=0), shape=(ch, -1))
        tensor_b_flat = t.reshape(t.transpose(copy.deepcopy(t.squeeze(tensor_b, 0)), dim0=1, dim1=0), shape=(ch, -1))

        # tm_corr = np.corrcoef(tensor_a_flat, tensor_b_flat, rowvar=True)
        tm_cev = []
        for a, b in zip(tensor_a_flat, tensor_b_flat):
            tm_cev.append(F.celu(explained_variance(b, a), inplace=True))

        tm_cev = t.stack(tm_cev)

        all_close = bool(t.allclose(cev[0], tm_cev, rtol=1e-2, atol=1e-4))
        self.assertTrue(all_close)  # add assertion here


if __name__ == "__main__":
    unittest.main()
