import torch
from rep_trans.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss
from torch.nn import functional as F


class FalsePredictedCrossEntropy(KEAbstractOutputLoss):
    num_tasks = 1

    def __init__(self, n_classes: int):
        super(FalsePredictedCrossEntropy, self).__init__(n_classes)
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

        pred_classes = torch.argmax(logit_prediction, dim=-1)
        old_pred_classes = [torch.argmax(lop, dim=-1) for lop in logit_other_predictions]

        pred_corr = pred_classes == groundtruth
        other_pred_corr = [torch.eq(opc, groundtruth) for opc in old_pred_classes]

        both_false = [((~pred_corr) & (~opc)).to(dtype=logit_prediction.dtype) for opc in other_pred_corr]
        n_both_false = [torch.sum(bf) for bf in both_false]

        other_pred_probs = [F.softmax(l, dim=-1) for l in logit_other_predictions]

        ce_losses = [
            torch.sum(F.celu(self.ce(logit_prediction, opp)) * bf / (nbf + 1))
            for opp, bf, nbf in zip(other_pred_probs, both_false, n_both_false)
        ]

        dis_loss = -torch.mean(torch.stack(ce_losses, dim=0))

        return [dis_loss]
