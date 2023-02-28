import copy
import unittest
from typing import Sequence

import numpy as np
import torch
from ke.losses.utils import correlation
from torchmetrics.functional import pearson_corrcoef


def create_random_tensor(shape: Sequence) -> torch.Tensor:
    random_np = np.random.random(shape)
    return torch.from_numpy(random_np)


batch = 100
ch = 64
width = 8
height = 8


class TestCorrelation(unittest.TestCase):
    def test_correlation_one_for_same_tensor(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))

        corr, _ = correlation([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_a)])
        c = corr[0]  # List of correlations for each hook (dimension
        all_equal = bool(torch.all(torch.isclose(c, torch.full_like(c, 1.0))))

        self.assertTrue(all_equal)

    def test_correlation_for_same_tensor_multiple_models(self):
        tensor_a = create_random_tensor((2, batch, ch, width, height))
        tensor_b = tensor_a

        corr, _ = correlation([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)])
        c = corr[0]  # List of correlations for each hook (dimension
        all_equal = bool(torch.all(torch.isclose(c, torch.full_like(c, 1.0))))

        self.assertTrue(all_equal)

    def test_correlation_for_same_tensor_multiple_models_broadcast(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))
        tensor_b = torch.repeat_interleave(tensor_a, repeats=2, dim=0)

        corr, _ = correlation([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)])
        c = corr[0]  # List of correlations for each hook (dimension
        all_equal = bool(torch.all(torch.isclose(c, torch.full_like(c, 1.0))))

        self.assertTrue(all_equal)

    def test_correlation_of_random_tensors_identical_to_torchmetrics(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))
        tensor_b = create_random_tensor((1, batch, ch, width, height))

        corr, _ = correlation([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)])
        c = corr[0]  # List of correlations for each hook (dimension
        tensor_a_flat = torch.reshape(
            torch.transpose(copy.deepcopy(torch.squeeze(tensor_a, 0)), dim0=1, dim1=0), shape=(ch, -1)
        )
        tensor_b_flat = torch.reshape(
            torch.transpose(copy.deepcopy(torch.squeeze(tensor_b, 0)), dim0=1, dim1=0), shape=(ch, -1)
        )

        # tm_corr = np.corrcoef(tensor_a_flat, tensor_b_flat, rowvar=True)
        tm_corrs = []
        for a, b in zip(tensor_a_flat, tensor_b_flat):
            tm_corrs.append(pearson_corrcoef(a, b))
        tm_corrs = torch.stack(tm_corrs)

        all_close = bool(torch.allclose(c, tm_corrs))
        self.assertTrue(all_close)  # add assertion here

    def test_correlation_of_random_tensors_identical_to_torchmetrics_multi_model(self):
        tensor_a = create_random_tensor((1, batch, ch, width, height))
        tensor_b = create_random_tensor((2, batch, ch, width, height))

        corr, _ = correlation([copy.deepcopy(tensor_a)], [copy.deepcopy(tensor_b)])
        c = corr[0]  # List of correlations for each hook (dimension
        tensor_a_flat = torch.repeat_interleave(
            torch.reshape(torch.transpose(tensor_a, dim0=2, dim1=1), shape=(1, ch, -1)), repeats=2, dim=0
        )
        tensor_b_flat = torch.reshape(torch.transpose(tensor_b, dim0=2, dim1=1), shape=(2, ch, -1))

        # tm_corr = np.corrcoef(tensor_a_flat, tensor_b_flat, rowvar=True)
        tm_corrs = []
        for m_a, m_b in zip(tensor_a_flat, tensor_b_flat):
            layerwise_corrs = []
            for a, b in zip(m_a, m_b):
                layerwise_corrs.append(pearson_corrcoef(a, b))
            tm_corrs.append(torch.stack(layerwise_corrs, dim=0))
        tm_corrs = torch.stack(tm_corrs)

        all_close = bool(torch.allclose(c, tm_corrs))
        self.assertTrue(all_close)  # add assertion here


if __name__ == "__main__":
    unittest.main()
