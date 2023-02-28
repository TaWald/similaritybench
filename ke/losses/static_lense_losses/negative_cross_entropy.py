import torch
from ke.losses.representation_similarity_losses.ke_abs_loss import AbstractStaticLenseLoss
from torch import nn


class NegativeCrossEntropyLenseLoss(AbstractStaticLenseLoss):
    def __init__(self):
        """CE Loss that is minimial when the Most Likely Not True (ensemble) class is predicted."""
        super().__init__()
        self.nlll = nn.NLLLoss()

    def forward(
        self,
        old_outputs: list[torch.Tensor],
        groundtruth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Abstract method that is used for the forward pass
        """
        ensemble_probs = torch.log(torch.softmax(torch.stack(old_outputs), dim=-1).mean(dim=0))

        ens_ce_loss = self.nlll(ensemble_probs, groundtruth)
        return -torch.mean(ens_ce_loss)
