import torch
from ke.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from ke.losses.utils import celu_explained_variance
from torch import nn


class ExpVarLoss(AbstractRepresentationLoss):
    def __init__(self, softmax_channel_metrics: bool, celu_alpha: float = 3.0):
        super(ExpVarLoss, self).__init__(softmax_channel_metrics)
        self.softmax_channel_metrics = softmax_channel_metrics
        self.celu = nn.CELU(alpha=celu_alpha)

    def forward(
        self, tbt_inter: list[torch.Tensor], approx_inter: list[torch.Tensor], make_dissimilar: bool
    ) -> torch.Tensor:

        # 1 When similar -inf when shitty
        # 1 when similar -alpha when shitty (after celu)
        celu_loss = celu_explained_variance(tbt_inter, approx_inter)
        loss = torch.mean(torch.stack([torch.mean(c) for c in celu_loss]))

        if make_dissimilar:
            # If we optimize the new model it has to make it dissimilar --> minimize r2
            loss = loss
        else:
            # If we optimize the approximation branch we want ot make it similar --> minimize -r2
            loss = -loss

        return loss
