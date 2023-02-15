import torch
from rep_trans.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss


class OrthogonalLogitLoss(KEAbstractOutputLoss):
    def __init__(self, n_classes: int):
        super(OrthogonalLogitLoss, self).__init__(n_classes)
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
        mean_inner_product_losses = []
        normed_logits = logit_prediction / torch.linalg.vector_norm(logit_prediction, ord=2, dim=-1, keepdim=True)
        for logits in logit_other_predictions:
            normed_other_logits = logits / torch.linalg.vector_norm(logits, ord=2, dim=-1, keepdim=True)
            inner_prod = torch.sum(normed_logits * normed_other_logits, dim=-1)
            mean_inner_product_losses.append(inner_prod)

        return [torch.mean(torch.stack(mean_inner_product_losses))]
