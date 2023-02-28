import copy
import unittest
from typing import Sequence

import numpy as np
import torch
import torch as t
import torch.nn.functional as F
from ke.losses.utils import topk_celu_explained_variance


def create_random_tensor(shape: Sequence) -> t.Tensor:
    random_np = np.random.random(shape)
    return t.from_numpy(random_np)


batch = 100
ch = 64
width = 8
height = 8


class TestTopkCeluExpVar(unittest.TestCase):
    def test_topk_celu_exp_var_one_for_same_tensor(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))
        tensor_b = tensor_a

        celu_expvar = topk_celu_explained_variance([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)])

        self.assertTrue(bool(t.all(celu_expvar[0][0] == 1.0)))

    def test_topk_celu_exp_var_of_random_tensors_identical_to_torchmetrics(self):
        for k in [10, 100, 1000]:
            tensor_a = create_random_tensor((1, batch, ch, width, height))
            tensor_b = create_random_tensor((1, batch, ch, width, height))

            topk_cev = topk_celu_explained_variance([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)], k)

            tensor_a = torch.squeeze(tensor_a, dim=0)
            tensor_b = torch.squeeze(tensor_b, dim=0)
            a_centered = tensor_a - t.mean(tensor_a, dim=(0, 2, 3), keepdim=True)
            b_centered = tensor_b - t.mean(tensor_b, dim=(0, 2, 3), keepdim=True)

            a = t.reshape(a_centered, shape=(batch, -1)).numpy()
            b = t.reshape(b_centered, shape=(batch, -1)).numpy()

            # tm_corr = np.corrcoef(tensor_a_flat, tensor_b_flat, rowvar=True)
            k_th_val_a = np.sort(np.abs(a), axis=1)[:, -k]
            k_th_val_b = np.sort(np.abs(b), axis=1)[:, -k]
            mask_a = np.abs(a) >= np.expand_dims(k_th_val_a, axis=-1)
            mask_b = np.abs(b) >= np.expand_dims(k_th_val_b, axis=-1)

            a_flat = np.reshape(a, -1)
            b_flat = np.reshape(b, -1)
            joint_mask = np.reshape(np.logical_or(mask_a, mask_b), (-1,))

            a_topk_vals = a_flat[joint_mask]
            b_topk_vals = b_flat[joint_mask]

            exp_var = 1 - ((np.sum((a_topk_vals - b_topk_vals) ** 2)) / (np.sum(a_topk_vals**2) + 1e-4))
            np_cev = F.celu(torch.from_numpy(np.reshape(exp_var, (1,))))

            self.assertTrue(bool(t.allclose(topk_cev[0][0], np_cev)))


if __name__ == "__main__":
    unittest.main()
