import torch
from ke.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss
from ke.losses.utils import pseudo_inversion
from torch.nn import functional as F


class NegativeClassCrossEntropy(KEAbstractOutputLoss):
    num_tasks = 1

    def __init__(self, n_classes: int):
        super(NegativeClassCrossEntropy, self).__init__(n_classes)
        self.ce = torch.nn.CrossEntropyLoss()

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

        positive_mask = gt
        negative_mask = 1.0 - gt

        pos_class_value = positive_mask * torch.tensor(float(-1e6), device=positive_mask.device)

        rescaled_logit_predictions = (negative_mask * logit_prediction) + pos_class_value
        rescaled_other_logit_predictions = [(negative_mask * l) + pos_class_value for l in logit_other_predictions]

        rescaled_other_prediction_probs = [F.softmax(l, dim=-1) for l in rescaled_other_logit_predictions]

        # CE expectes RAW unnormalized values AKA Logits not probabilities!
        #
        dis_loss = torch.mean(
            torch.stack(
                [self.ce(rescaled_logit_predictions, pseudo_inversion(op)) for op in rescaled_other_prediction_probs]
            )
        )

        return [dis_loss]
