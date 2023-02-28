import torch
from ke.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss
from torch.nn import functional as F


class EnsembleEntropyMaximization(KEAbstractOutputLoss):
    num_tasks = 1

    def __init__(self, n_classes: int):
        super(EnsembleEntropyMaximization, self).__init__(n_classes)
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, logit_prediction: torch.Tensor, groundtruth: torch.Tensor, logit_other_predictions: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Abstract method that is used for the forward pass

        :param logit_other_predictions: list[batch x Class Probs]
        :param groundtruth: Groundtruth (Batch) -- Value encodes class! --> One hot this.
        :param logit_prediction: Prediction of the to be trained model.
        """

        pred_prob = F.softmax(logit_prediction, dim=-1)
        pred_classes = torch.argmax(pred_prob, dim=-1)

        other_predictions = [F.softmax(l, dim=-1) for l in logit_other_predictions]
        ensemble_without_new_prediction = torch.mean(torch.stack(other_predictions, dim=0), dim=0)
        ensemble_with_new_prediction = torch.mean(torch.stack(other_predictions + [pred_prob], dim=0), dim=0)
        ensemble_class_pred = torch.argmax(ensemble_without_new_prediction, dim=1)

        ens_corr = ensemble_class_pred == groundtruth
        pred_corr = pred_classes == groundtruth

        both_false = ((~pred_corr) & (~ens_corr)).to(dtype=logit_prediction.dtype)
        n_both_false = torch.sum(both_false)

        entropy = torch.sum(
            ensemble_with_new_prediction
            * torch.log(ensemble_with_new_prediction)
            * torch.unsqueeze(both_false, dim=-1)
        )
        normalized_entropy = entropy / (n_both_false + 1)

        return [normalized_entropy]
