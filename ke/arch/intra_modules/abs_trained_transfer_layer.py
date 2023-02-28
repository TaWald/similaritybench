from abc import abstractmethod

import torch
from torch import nn


class AbstractTrainedApproximation(nn.Module):
    def __init__(self, n_source_channels: list[int], n_target_channels: int):
        super(AbstractTrainedApproximation, self).__init__()
        rebias_shape = (1, n_target_channels, 1, 1)
        self.debias: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(torch.zeros((1, n_ch, 1, 1), dtype=torch.float32)))
                for n_ch in n_source_channels
            ]
        )
        self.rebias: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(torch.Tensor(torch.zeros(rebias_shape, dtype=torch.float32))) for _ in n_source_channels]
        )
        return

    @abstractmethod
    def forward(self, sources: list[torch.Tensor]) -> list[torch.Tensor]:
        pass
