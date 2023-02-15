import torch
from rep_trans.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from rep_trans.losses.utils import topk_celu_explained_variance
from torch import nn


class TopKExpVarLoss(AbstractRepresentationLoss):
    def __init__(self, softmax_channel_metrics: bool, celu_alpha: float = 3.0):
        super(TopKExpVarLoss, self).__init__(softmax_channel_metrics)
        self.softmax_channel_metrics = softmax_channel_metrics
        self.celu = nn.CELU(alpha=celu_alpha)

    def forward(
        self, tbt_inter: list[torch.Tensor], approx_inter: list[torch.Tensor], make_dissimilar: bool
    ) -> torch.Tensor:

        # 1 When similar -inf when shitty
        # 1 when similar -alpha when shitty (after celu)
        exp_vars: list[list[torch.Tensor]] = topk_celu_explained_variance(tbt_inter, approx_inter)
        # Number of Channels is identical for entries so averaging can be done later
        celu_loss = torch.mean(torch.stack([torch.stack(cev) for cev in exp_vars]))

        if make_dissimilar:
            # If we optimize the new model it has to make it dissimilar --> minimize r2
            loss = celu_loss
        else:
            # If we optimize the approximation branch we want ot make it similar --> minimize -r2
            loss = -celu_loss

        return loss
