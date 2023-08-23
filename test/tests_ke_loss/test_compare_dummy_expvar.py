import unittest
from typing import Sequence

import numpy as np
import torch as t
import torch.nn.functional as F
from ke.losses.dummy_loss import DummyLoss
from ke.losses.ke_loss import KETrainLoss
from ke.losses.representation_similarity_losses.ke_exp_var import ExpVarLoss
from ke.losses.representation_similarity_losses.ke_none_loss import NoneLoss


def create_random_logits(shape: Sequence) -> t.Tensor:
    """Create random tensor values between -1 and 1."""
    random_np = np.random.random(shape) * 2 - 1
    return t.from_numpy(random_np)


batch = 100
classes = 10
width = 8
height = 8


class CompareKELoss(unittest.TestCase):
    def test_none_loss_for_same_tensor(self):
        for i in range(20):
            with self.subTest(i=i):
                pseudo_prediction = create_random_logits((batch, classes))
                pseudo_probs = F.softmax(pseudo_prediction, dim=1)
                pseudo_label = t.randint(0, 10, (batch,))
                a = NoneLoss(softmax_channel_metrics=True)
                b = NoneLoss(softmax_channel_metrics=True)
                ke_train_loss = KETrainLoss(
                    a,
                    b,
                    ce_weight=1.0,
                    dissim_weight=1.0,
                    sim_weight=1.0,
                    regularization_epoch_start=-1,
                    n_classes=classes,
                )
                self.baseline_loss = DummyLoss(ce_weight=1.0)
                ke_train_loss_val = (
                    ke_train_loss.forward(
                        pseudo_label, pseudo_probs, [t.zeros(1)], t.zeros(1), epoch_num=10, global_step=10000
                    )["loss"]
                    .cpu()
                    .numpy()
                )
                dummy_loss_val = self.baseline_loss(pseudo_label, pseudo_probs)["loss"].cpu().numpy()

                self.assertTrue(bool(np.all(np.isclose(ke_train_loss_val, dummy_loss_val))))

    def test_zero_weighted_ce_loss_vs_dummy(self):
        for i in range(20):
            with self.subTest(i=i):
                pseudo_prediction = create_random_logits((batch, classes))

                pseudo_reps = [create_random_logits((batch, 1, batch, 2, 2)) for _ in range(3)]
                pseudo_probs = F.softmax(pseudo_prediction, dim=1)
                pseudo_label = t.randint(0, 10, (batch,))
                a = ExpVarLoss(softmax_channel_metrics=True)
                b = ExpVarLoss(softmax_channel_metrics=True)
                ke_train_loss = KETrainLoss(
                    a,
                    b,
                    ce_weight=1.0,
                    dissim_weight=0.0,
                    sim_weight=0.0,
                    regularization_epoch_start=-1,
                    n_classes=classes,
                )
                self.baseline_loss = DummyLoss(ce_weight=1.0)
                ke_train_loss_val = (
                    ke_train_loss.forward(
                        pseudo_label, pseudo_probs, pseudo_reps, pseudo_reps, epoch_num=10, global_step=10000
                    )["loss"]
                    .cpu()
                    .numpy()
                )
                dummy_loss_val = self.baseline_loss(pseudo_label, pseudo_probs)["loss"].cpu().numpy()

                self.assertTrue(bool(np.all(np.isclose(ke_train_loss_val, dummy_loss_val))))


if __name__ == "__main__":
    unittest.main()
