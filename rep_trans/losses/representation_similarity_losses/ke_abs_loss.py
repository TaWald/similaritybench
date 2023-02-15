from abc import ABC
from abc import abstractmethod
from typing import Protocol

import torch
from torch import nn


class AbstractRepresentationLoss(nn.Module, ABC):
    def __init__(self, softmax_channel_metrics: bool):
        super(AbstractRepresentationLoss, self).__init__()
        self.softmax_channel_metrics = softmax_channel_metrics

    @abstractmethod
    def forward(
        self,
        tbt_inter: list[torch.Tensor],
        approx_inter: list[torch.Tensor],
        make_dissimilar: bool,
    ) -> torch.Tensor:
        """
        Abstract method that is used for the forward pass
        """
        pass


class KEAbstractConditionalLoss(nn.Module, ABC):
    def __init__(self, softmax_channel_metrics: bool):
        super(KEAbstractConditionalLoss, self).__init__()
        self.softmax_channel_metrics = softmax_channel_metrics

    @abstractmethod
    def forward(
        self,
        new_inter: list[torch.Tensor],
        old_inter: list[torch.Tensor],
        new_out: torch.Tensor,
        old_outs: torch.Tensor,
        make_dissimilar: bool,
    ) -> torch.Tensor:
        """
        Abstract method that is used for the forward pass
        """
        pass


class AbstractStaticLenseLoss(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        old_outputs: list[torch.Tensor],
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Abstract method that is used for the forward pass
        """
        pass


class ReconstructionLossProto(Protocol):
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Abstract method that is used for the forward pass
        """
        pass
