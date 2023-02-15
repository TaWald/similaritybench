import torch
from rep_trans.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from rep_trans.losses.utils import correlation


class L2CorrLoss(AbstractRepresentationLoss):
    def forward(
        self, tbt_inter: list[torch.Tensor], approx_inter: list[torch.Tensor], make_dissimilar: bool
    ) -> torch.Tensor:
        # L_Similar: Maximize correlation --> Loss sollte hoch sei wenn corr = 0
        #  detach the true representations because the approx. network is supposed to do the work.

        corr, _ = correlation(tbt_inter, approx_inter)

        if make_dissimilar:
            # Optimize for dissimilarity --> Minimize Correlation
            loss = torch.mean(torch.stack([torch.mean(c**2) for c in corr]))
        else:
            # Optimize for similarity --> high correlation
            # 1- to make the minimum 0
            loss = 1 - torch.mean(torch.stack([torch.mean(c**2) for c in corr]))
        return loss


class L1CorrLoss(L2CorrLoss):
    def forward(
        self, tbt_inter: list[torch.Tensor], approx_inter: list[torch.Tensor], make_dissimilar: bool
    ) -> torch.Tensor:
        # L_Similar: Maximize correlation --> Loss sollte hoch sei wenn corr = 0
        corr: list[torch.Tensor]
        corr, _ = correlation(tbt_inter, approx_inter)

        if make_dissimilar:
            # Optimizing the currently trained model only for dissimilarity --> Minimize Correlation
            loss = torch.mean(torch.stack([torch.mean(torch.abs(c)) for c in corr]))
        else:
            # Optimize for similarity --> want high correlation
            # 1- to make the minimum 0
            loss = 1 - torch.mean(torch.stack([torch.mean(torch.abs(c)) for c in corr]))
        return loss
