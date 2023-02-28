import copy
import unittest
from typing import Sequence

import numpy as np
import torch
from ke.losses.utils import topk_correlation
from torchmetrics.functional import pearson_corrcoef


def create_random_tensor(shape: Sequence) -> torch.Tensor:
    random_np = np.random.random(shape)
    return torch.from_numpy(random_np)


batch = 100
ch = 64
width = 8
height = 8


class TestTopkCorrelation(unittest.TestCase):
    def test_topk_correlation_one_for_same_tensor(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))
        tensor_b = tensor_a

        corr = topk_correlation([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)])
        c = corr[0][0]  # List of correlations for each hook (dimension
        all_equal = bool(torch.all(torch.isclose(c, torch.full_like(c, 1.0))))

        self.assertTrue(all_equal)

    def test_correlation_of_random_tensors_identical_to_torchmetrics(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))
        tensor_b = create_random_tensor((1, batch, ch, width, height))

        for k in [10, 100, 1000]:
            corr = topk_correlation([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)], k)
            c = corr[0][0]  # List of correlations for each hook (dimension
            a = torch.reshape(tensor_a, shape=(batch, -1)).numpy()
            b = torch.reshape(tensor_b, shape=(batch, -1)).numpy()

            # tm_corr = np.corrcoef(tensor_a_flat, tensor_b_flat, rowvar=True)
            k_th_val_a = np.sort(a, axis=1)[:, -k]
            k_th_val_b = np.sort(b, axis=1)[:, -k]
            mask_a = a >= np.expand_dims(k_th_val_a, axis=-1)
            mask_b = b >= np.expand_dims(k_th_val_b, axis=-1)
            a_flat = np.reshape(a, -1)
            b_flat = np.reshape(b, -1)
            joint_mask = np.reshape(np.logical_or(mask_a, mask_b), (-1,))
            a_topk_vals = torch.from_numpy(a_flat[joint_mask])
            b_topk_vals = torch.from_numpy(b_flat[joint_mask])
            tm_corr = torch.abs(pearson_corrcoef(a_topk_vals, b_topk_vals))

            # a_topk_vals = a_flat[joint_mask]
            # b_topk_vals = b_flat[joint_mask]
            # centered_a = a_topk_vals - np.mean(a_topk_vals)
            # centered_b = b_topk_vals - np.mean(b_topk_vals)
            # np_corr = np.abs(
            #     np.sum(
            #         (centered_a * centered_b)
            #         / (np.std(centered_a) * np.std(centered_b) * (centered_a.shape[0] - 1) + 1e-9)
            #     )
            # )

            self.assertTrue(bool(torch.allclose(c, tm_corr)))  # Own vs torch metrics
            # Somehow the numpy version is not close enough ...
            # self.assertTrue(bool(np.isclose(c.numpy(), np_corr)))  # Own vs own numpy


if __name__ == "__main__":
    unittest.main()
