from abc import ABC
from abc import abstractmethod

import torch
from torch import nn


class KEAbstractOutputLoss(nn.Module, ABC):
    num_tasks: int

    def __init__(self, n_classes: int):
        super(KEAbstractOutputLoss, self).__init__()
        self.n_classes = n_classes

    @abstractmethod
    def forward(
        self, logit_prediction: torch.Tensor, groundtruth: torch.Tensor, logit_other_predictions: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Abstract method that is used for the forward pass

        :param logit_other_predictions: src_models X batch x Class Probs
        :param groundtruth: Groundtruth
        :param logit_prediction: Prediction of the to be trained model.
        """
        pass
