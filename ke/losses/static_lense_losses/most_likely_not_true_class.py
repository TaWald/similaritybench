import torch
from ke.losses.representation_similarity_losses.ke_abs_loss import AbstractStaticLenseLoss
from torch import nn


class MostLikelyNotTrueClassLenseLoss(AbstractStaticLenseLoss):
    def __init__(self):
        """CE Loss that is minimial when the Most Likely Not True (ensemble) class is predicted."""
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        old_outputs: list[torch.Tensor],
        groundtruth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Abstract method that is used for the forward pass
        """
        gt = torch.zeros((groundtruth.shape[0], self.n_classes), device=old_outputs[0].device)
        gt[torch.arange(groundtruth.shape[0]), groundtruth] = 1.0

        probs = torch.softmax(torch.stack(old_outputs, dim=0), dim=2)
        most_likely_not_true_class = torch.argmax(torch.mean(probs - gt[None, ...]))

        ce_losses = []
        for o in old_outputs:
            ce_losses.append(self.ce(o, most_likely_not_true_class))
        return torch.mean(torch.stack(ce_losses))
