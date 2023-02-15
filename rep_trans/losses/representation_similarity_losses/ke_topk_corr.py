import torch
from rep_trans.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from rep_trans.losses.utils import topk_correlation


class TopKL2CorrLoss(AbstractRepresentationLoss):
    def forward(
        self, tbt_inter: list[torch.Tensor], approx_inter: list[torch.Tensor], make_dissimilar: bool
    ) -> torch.Tensor:
        # L_Similar: Maximize correlation --> Loss sollte hoch sei wenn corr = 0
        #  detach the true representations because the approx. network is supposed to do the work.
        topk_corrs: list[list[torch.Tensor]] = topk_correlation(tbt_inter, approx_inter)
        # Number of channels is independent for topk correlation --> Averaging can be done later
        corr = torch.mean(torch.stack([torch.stack([c**2 for c in corrs]) for corrs in topk_corrs]))

        if make_dissimilar:
            # Optimize for dissimilarity --> Minimize Correlation
            loss = corr
        else:
            # Optimize for similarity --> high correlation
            # 1- to make the minimum 0
            loss = 1 - corr
        return loss
