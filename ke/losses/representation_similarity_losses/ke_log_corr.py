import torch
from ke.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from ke.losses.utils import correlation


class LogCorrLoss(AbstractRepresentationLoss):
    def forward(
        self, tbt_inter: list[torch.Tensor], approx_inter: list[torch.Tensor], make_dissimilar: bool
    ) -> torch.Tensor:
        # L_Similar: Maximize correlation --> Loss sollte hoch sei wenn corr = 0
        #  detach the true representations because the approx. network is supposed to do the work.

        corr, _ = correlation(tbt_inter, approx_inter)

        if make_dissimilar:
            # Optimize for dissimilarity --> Minimize Correlation (0) --> log(0) == -inf
            loss = torch.mean(
                torch.stack([torch.mean(torch.log(torch.clip(c, min=1e-7, max=1 - 1e-7))) for c in corr])
            )
        else:
            # Optimize for similarity --> high correlation
            loss = torch.mean(
                torch.stack([torch.mean(torch.log(torch.clip(1 - c, min=1e-7, max=1 - 1e-7))) for c in corr])
            )

        return loss
