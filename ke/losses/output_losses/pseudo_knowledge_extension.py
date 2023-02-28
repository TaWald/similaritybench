import torch
from ke.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss
from torch.nn import functional as F


class PseudoKnowledgeExtension1(KEAbstractOutputLoss):
    num_tasks = 1

    def __init__(self, n_classes: int):
        super(PseudoKnowledgeExtension1, self).__init__(n_classes)
        self.ce = torch.nn.CrossEntropyLoss()
        self.temperature_scaling_value = 1.0

    def forward(
        self, logit_prediction: torch.Tensor, groundtruth: torch.Tensor, logit_other_predictions: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Calculation of the PseudoKnowledgeExtension in which the previously trained models have their
        output probabilities reshuffled so classes with higher probability are less likely

        :param logit_other_predictions: list[batch x Class Probs]
        :param groundtruth: Groundtruth (Batch) -- Value encodes class! --> One hot this.
        :param logit_prediction: Prediction of the to be trained model.
        """

        # One hot creation from the labels
        gt = torch.zeros((groundtruth.shape[0], self.n_classes), device=logit_prediction.device)
        gt[torch.arange(groundtruth.shape[0]), groundtruth] = 1.0

        # Positive and negative mask for output classes
        positive_mask = gt
        negative_mask = 1.0 - gt

        # Softmax the other predictions to have probabilities to work with
        other_probs = [F.softmax(lop / self.temperature_scaling_value, dim=-1) for lop in logit_other_predictions]
        # Filter all the negative probabilities out and set to 0
        pos_class_probs = [op * positive_mask for op in other_probs]
        # Calculate how much probabilities the correct class contributes when reshuffling probs.
        pos_class_prob_influence = [
            torch.sum(pcp, dim=-1, keepdim=True) / (self.n_classes - 2) for pcp in pos_class_probs
        ]

        # Calculate a reshuffle matrix that moves probs from the current class to the other classes equally.
        #   It has a factor of 1/n-2 because false classes can't move to themselves and not to the positive class!
        reshuffle_matrix = (
            1.0
            - torch.eye(self.n_classes, self.n_classes, device=logit_prediction.device, dtype=logit_prediction.dtype)
        ) * ((self.n_classes - 2) ** (-1))
        # Calculate the reshuffled probs
        reshuffled_other_probs = [op @ reshuffle_matrix for op in other_probs]

        # Remove the influence of the positive class reshuffling
        negative_class_probs_with_pos_class = [
            rop - pcpi for rop, pcpi in zip(reshuffled_other_probs, pos_class_prob_influence)
        ]
        # Set the prob value of the positive class to 0.
        negative_class_probs = [ncppc * negative_mask for ncppc in negative_class_probs_with_pos_class]

        # Calculate the final probabilities one will try to approximate.
        final_other_class_probs = [pcp + ncp for pcp, ncp in zip(pos_class_probs, negative_class_probs)]

        # Calculate the CE loss with the final_other_class_probs that were modded.
        # As a sidenote no
        pke_loss = torch.mean(torch.stack([self.ce(logit_prediction, focp) for focp in final_other_class_probs]))

        return [pke_loss]


class PseudoKnowledgeExtension2(PseudoKnowledgeExtension1):
    def __init__(self, n_classes: int):
        super(PseudoKnowledgeExtension2, self).__init__(n_classes)
        self.temperature_scaling_value = 2.0


class PseudoKnowledgeExtension4(PseudoKnowledgeExtension1):
    def __init__(self, n_classes: int):
        super(PseudoKnowledgeExtension4, self).__init__(n_classes)
        self.temperature_scaling_value = 4.0


class PseudoKnowledgeExtension5(PseudoKnowledgeExtension1):
    def __init__(self, n_classes: int):
        super(PseudoKnowledgeExtension5, self).__init__(n_classes)
        self.temperature_scaling_value = 5.0


class PseudoKnowledgeExtension8(PseudoKnowledgeExtension1):
    def __init__(self, n_classes: int):
        super(PseudoKnowledgeExtension8, self).__init__(n_classes)
        self.temperature_scaling_value = 8.0
