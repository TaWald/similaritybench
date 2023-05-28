import torch
from ke.arch.intra_modules.abs_trained_transfer_layer import AbstractTrainedApproximation
from torch import nn


class TrainedApproximation(AbstractTrainedApproximation):
    def __init__(self, n_source_channels: list[int], n_target_channels: int, n_layers: int, kernel_size: int):
        """
        Approximates intermediate representations from many sources to a single target channel.
        """
        assert n_layers >= 1, "Number of layers should be positive signed integer."
        super().__init__(n_source_channels, n_target_channels)
        padding = (int((kernel_size - 1) / 2), int((kernel_size - 1) / 2))
        # convs contains the number of convolutions for each parent architecture.
        #   This means we finish one approximation and continue to the next.
        self.convs: nn.ModuleList[nn.ModuleList[nn.Conv2d]] = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            n_src,
                            n_src,
                            kernel_size=kernel_size,
                            stride=(1, 1),
                            padding=padding,
                            padding_mode="zeros",
                            bias=True,
                        )
                        for _ in range(n_layers - 1)
                    ]
                )
                for n_src in n_source_channels
            ]
        )
        self.bns: nn.ModuleList[nn.ModuleList[nn.Conv2d]] = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(n_src) for _ in range(n_layers - 1)]) for n_src in n_source_channels]
        )
        self.relu = nn.functional.relu
        self.out_conv: nn.ModuleList[nn.Conv2d] = nn.ModuleList(
            [
                nn.Conv2d(
                    n_src,
                    n_target_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=padding,
                    padding_mode="zeros",
                    bias=False,
                )
                for n_src in n_source_channels
            ]
        )

    def forward(self, sources: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Debiases (learnable), linearly combines to same channel dimension as target
        and rebiases (learnable) the representations.
        :param sources: Sources contains the outputs of the already trained models.
        They can either be aggregated along the channel dimension, effectively leading to a single

        """
        approximations: list[torch.Tensor] = []
        for src, debias, rebias, conv, out_conv, bns in zip(
            sources, self.debias, self.rebias, self.convs, self.out_conv, self.bns
        ):
            x = src + debias
            for c, bn in zip(conv, bns):
                x = c(x)
                x = bn(x)
                x = self.relu(x)
            approximations.append(out_conv(x) + rebias)
        return approximations


class NotAligned(nn.Module):
    def __init__(self):
        """
        Empty module that concats the sources along channel dimension for Sub-Space metrics.
        """
        super().__init__()
        # convs contains the number of convolutions for each parent architecture.
        #   This means we finish one approximation and continue to the next.

    def forward(self, sources: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Debiases (learnable), linearly combines to same channel dimension as target
        and rebiases (learnable) the representations.
        :param sources: Sources contain the outputs of the already trained models.
        They are aggregated along the channel dimension, effectively leading to a single representation

        """

        return sources
