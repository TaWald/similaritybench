import torch
from rep_trans.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from rep_trans.losses.utils import correlation


class WeightedL1CorrLoss(AbstractRepresentationLoss):
    def forward(
        self, tbt_inter: list[torch.Tensor], approx_inter: list[torch.Tensor], make_dissimilar: bool
    ) -> torch.Tensor:
        # L_Similar: Maximize correlation --> Loss sollte hoch sei wenn corr = 0
        #  detach the true representations because the approx. network is supposed to do the work.

        corr, stds = correlation(tbt_inter, approx_inter)
        l1_corrs = [torch.abs(c) for c in corr]
        tt_weights = [std / torch.sum(std) for std in stds]  # Weight high STD channels more than low channels!

        if make_dissimilar:
            # Optimize for dissimilarity --> Minimize Correlation
            loss = torch.mean(torch.stack([torch.sum(l1c * ttw) for l1c, ttw in zip(l1_corrs, tt_weights)]))
        else:
            # Optimize for similarity --> high correlation
            # 1- to make the minimum 0
            loss = torch.mean(torch.stack([torch.sum((1 - l1c) * ttw) for l1c, ttw in zip(l1_corrs, tt_weights)]))
        return loss
