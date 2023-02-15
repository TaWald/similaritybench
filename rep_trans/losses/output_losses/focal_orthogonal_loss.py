import torch
from rep_trans.losses.output_losses.adaptive_diversity_promoting_regularization import (
    cos_sim_ensemble_diversity,
)
from rep_trans.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss


class FocalCosineSimProbability(KEAbstractOutputLoss):
    num_tasks = 1

    def __init__(self, n_classes: int, dis_loss_weight: float, focal_weight: float):
        super().__init__(n_classes)
        self.focal_weight = focal_weight
        self.dis_loss_weight = dis_loss_weight

    def forward(
        self, logit_prediction: torch.Tensor, groundtruth: torch.Tensor, logit_other_predictions: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Abstract method that is used for the forward pass

        :param logit_other_predictions: list[batch x Class Probs]
        :param groundtruth: Groundtruth (Batch) -- Value encodes class! --> One hot this.
        :param logit_prediction: Prediction of the to be trained model.
        """
        # Calculate softmax Probabilties
        gt = torch.zeros((groundtruth.shape[0], self.n_classes), device=logit_prediction.device)
        gt[torch.arange(groundtruth.shape[0]), groundtruth] = 1.0
        gt = gt.type(torch.bool)

        softmax_other_predictions = [torch.softmax(lop, dim=-1) for lop in logit_other_predictions]
        mean_ensemble = torch.stack(softmax_other_predictions).mean(dim=0)

        true_class_prob = mean_ensemble[gt]
        softmax_prediction = torch.softmax(logit_prediction, dim=-1)

        cos_sim = cos_sim_ensemble_diversity(
            softmax_prediction, torch.stack(softmax_other_predictions, dim=0), groundtruth
        )

        focal_cos_sim_loss = ((1 - true_class_prob) ** self.focal_weight) * cos_sim
        return [self.dis_loss_weight * focal_cos_sim_loss.mean()]
