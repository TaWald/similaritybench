import torch
from ke.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss


class OrthogonalProbabilityNegativeClasses(KEAbstractOutputLoss):
    num_tasks = 1

    def __init__(self, n_classes: int):
        super(OrthogonalProbabilityNegativeClasses, self).__init__(n_classes)
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

        logit_prediction = logit_prediction.type(torch.float32)
        logit_other_predictions = [lop.type(torch.float32) for lop in logit_other_predictions]
        groundtruth = groundtruth

        gt = torch.zeros((groundtruth.shape[0], self.n_classes), device=logit_prediction.device, dtype=torch.float32)
        gt[torch.arange(groundtruth.shape[0]), groundtruth] = 1.0

        preds = torch.softmax(logit_prediction, dim=-1, dtype=torch.float32)
        other_preds = [torch.softmax(lop, dim=-1, dtype=torch.float32) for lop in logit_other_predictions]

        negative_mask = 1.0 - gt

        filtered_preds = negative_mask * preds
        rescaled_filtered_preds = filtered_preds / torch.linalg.vector_norm(filtered_preds, dim=-1, keepdim=True)
        rfp = rescaled_filtered_preds

        filtered_other_preds = [(negative_mask * op) for op in other_preds]
        rescaled_filtered_other_preds = [
            fop / torch.linalg.vector_norm(filtered_preds, dim=-1, keepdim=True) for fop in filtered_other_preds
        ]
        rfop = rescaled_filtered_other_preds

        dis_loss = torch.mean(torch.stack([torch.mean(torch.sum(rfp * r, dim=-1)) for r in rfop]))

        return [dis_loss]
