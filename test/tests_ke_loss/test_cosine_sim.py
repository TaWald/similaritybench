import unittest

import numpy as np
import torch
from ke.losses.utils import cosine_similarity


class TestCosineSim(unittest.TestCase):
    def test_similarity(self):
        for i in range(10):
            a = np.random.rand(128, 10000)
            a_t = torch.from_numpy(a)
            a_t = torch.unsqueeze(a_t, 0)  # Somehow also necessary. Dunno remember why right now

            sim = np.ndarray((a.shape[0], a.shape[0]))
            for i in range(a.shape[0]):
                for j in range(a.shape[0]):
                    sim[i, j] = np.dot(a[i], a[j]) / (np.linalg.norm(a[i]) * np.linalg.norm(a[j]))
            # Written for list of tensors as input
            res = cosine_similarity([a_t])[0][0]
            res = res.numpy()
            is_same = np.all(np.isclose(res, sim))

            self.assertTrue(bool(is_same))  # add assertion here


if __name__ == "__main__":
    unittest.main()
