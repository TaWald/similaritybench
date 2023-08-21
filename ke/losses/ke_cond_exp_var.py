import torch
from ke.losses.representation_similarity_losses.ke_abs_loss import KEAbstractConditionalLoss
from ke.losses.utils import celu_explained_variance
from torch import nn


class CondExpVarLoss(KEAbstractConditionalLoss):
    def __init__(self, softmax_channel_metrics: bool, celu_alpha: float = 3.0):
        super(CondExpVarLoss, self).__init__(softmax_channel_metrics)
        self.softmax_channel_metrics = softmax_channel_metrics
        self.celu = nn.CELU(alpha=celu_alpha)

    def forward(
        self,
        new_inter: list[torch.Tensor],
        old_inter: list[torch.Tensor],
        new_out: torch.Tensor,
        old_outs: list[torch.Tensor],
        gt: torch.Tensor,
        make_dissimilar: bool,
    ) -> torch.Tensor:
        # 1 When similar -inf when shitty
        # 1 when similar -alpha when shitty (after celu)

        correct_trues = torch.argmax(new_out, dim=-1) == gt
        corret_outs = [torch.argmax(o, dim=-1) == gt for o in old_outs]

        both_true = [(correct_trues == co) for co in corret_outs]  # noqa

        # Goal is to not penalize where both are correct as that is okay
        #   Instead try to penalize

        celu_loss = celu_explained_variance(tbt_inter, approx_inter)  # noqa
        loss = torch.mean(torch.stack([torch.mean(c) for c in celu_loss]))

        if make_dissimilar:
            # If we optimize the new model it has to make it dissimilar --> minimize r2
            loss = loss
        else:
            # If we optimize the approximation branch we want ot make it similar --> minimize -r2
            loss = -loss

        return loss
