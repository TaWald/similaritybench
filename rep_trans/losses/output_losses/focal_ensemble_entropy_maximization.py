import torch
from rep_trans.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss
from torch.nn import functional as F


class FocalEnsembleEntropyMaximization(KEAbstractOutputLoss):
    num_tasks = 1

    def __init__(self, n_classes: int, dis_loss_weight: float, focal_weight: float):
        super(FocalEnsembleEntropyMaximization, self).__init__(n_classes)
        self.focal_weight: float = focal_weight
        self.dis_loss_weight: float = dis_loss_weight

    def forward(
        self, logit_prediction: torch.Tensor, groundtruth: torch.Tensor, logit_other_predictions: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Abstract method that is used for the forward pass

        :param logit_other_predictions: list[batch x Class Probs]
        :param groundtruth: Groundtruth (Batch) -- Value encodes class! --> One hot this.
        :param logit_prediction: Prediction of the to be trained model.
        """

        gt = torch.zeros((groundtruth.shape[0], self.n_classes), device=logit_prediction.device)
        gt[torch.arange(groundtruth.shape[0]), groundtruth] = 1.0
        gt = gt.type(torch.bool)

        ens_prob = torch.mean(
            F.softmax(torch.stack(logit_other_predictions + [logit_prediction], dim=0), dim=-1), dim=0
        )
        p = ens_prob[gt]

        entropy = torch.sum(-ens_prob * torch.log(ens_prob + 1e-20), dim=-1)
        sample_wise_focal_entropy = (1 - p) ** self.focal_weight * entropy * self.dis_loss_weight
        focal_entropy = torch.mean(sample_wise_focal_entropy)
        return [focal_entropy]
