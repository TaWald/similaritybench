import torch
from rep_trans.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from rep_trans.losses.utils import centered_kernel_alignment


class CKALoss(AbstractRepresentationLoss):
    def forward(
        self, tbt_inter: list[torch.Tensor], approx_inter: list[torch.Tensor], make_dissimilar: bool
    ) -> torch.Tensor:
        # L_Similar: Maximize correlation --> Loss sollte hoch sei wenn corr = 0
        #  detach the true representations because the approx. network is supposed to do the work.

        ckas: list[list[torch.Tensor]] = centered_kernel_alignment(tbt_inter, approx_inter)
        # CKA is scalar for entire layer so averaging can be done hook independent
        cka_loss = torch.mean(torch.stack([torch.stack(c) for c in ckas]))

        if make_dissimilar:
            # Optimize for dissimilarity --> Minimize Correlation
            loss = cka_loss
        else:
            # Optimize for similarity --> high correlation
            # 1- to make the minimum 0
            loss = 1 - cka_loss
        return loss
