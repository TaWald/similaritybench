import torch
from rep_trans.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss
from torch.nn import functional as F


class EntropyWeightedBoosting(KEAbstractOutputLoss):
    num_tasks = 1

    def __init__(self, n_classes: int):
        super(EntropyWeightedBoosting, self).__init__(n_classes)
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
        self.nlll = torch.nn.NLLLoss(reduction="none")

    def forward(
        self, logit_prediction: torch.Tensor, groundtruth: torch.Tensor, logit_other_predictions: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Abstract method that is used for the forward pass

        :param logit_other_predictions: list[batch x Class Probs]
        :param groundtruth: Groundtruth (Batch) -- Value encodes class! --> One hot this.
        :param logit_prediction: Prediction of the to be trained model.
        """

        # Only based on the actually bad examples
        with torch.no_grad():
            other_predictions = [F.softmax(l, dim=-1) for l in logit_other_predictions]
            ensemble_without_new_prediction = torch.mean(torch.stack(other_predictions, dim=0), dim=0)
            # Either put more weight on these examples based on entropy independent on correctly predicted class!
            entropy = torch.sum(-ensemble_without_new_prediction * torch.log(ensemble_without_new_prediction), dim=-1)
            normalized_entropy = entropy / (torch.mean(entropy))

        # Redistribute weight to the uncertain examples!
        boosted_ce_loss = torch.mean(self.ce(logit_prediction, groundtruth) * normalized_entropy.detach())

        return [boosted_ce_loss]
