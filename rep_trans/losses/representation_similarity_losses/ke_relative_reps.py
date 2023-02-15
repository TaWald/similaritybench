import torch
from rep_trans.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from rep_trans.losses.utils import cosine_similarity
from rep_trans.losses.utils import euclidean_distance_csim


class EuclideanRelativeRepresentationLoss(AbstractRepresentationLoss):
    def forward(
        self, tbt_inter: list[torch.Tensor], approx_inter: list[torch.Tensor], make_dissimilar: bool
    ) -> torch.Tensor:

        if make_dissimilar:
            cos_sim_tr = cosine_similarity(tbt_inter)
            with torch.no_grad():
                cos_sim_apx = cosine_similarity(approx_inter)
        else:
            with torch.no_grad():
                cos_sim_tr = cosine_similarity(tbt_inter)
            cos_sim_apx = cosine_similarity(approx_inter)
        euclidean_dist = euclidean_distance_csim(zip(cos_sim_tr, cos_sim_apx))
        # High == dissimilar; low == similar

        if make_dissimilar:
            # If we optimize the new model it has to make it dissimilar --> maximize euclidean distance of cos.sim
            loss = -euclidean_dist
        else:
            # If we optimize the approximation branch we want ot make it similar --> minimize -r2
            loss = euclidean_dist

        return loss
