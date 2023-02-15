import torch
from torch import nn
from torch.autograd import Function

# From https://github.com/tadeephuy/GradientReversal/blob/master/gradient_reversal/functional.py
"""
@misc{ganin2015unsupervised,
      title={Unsupervised Domain Adaptation by Backpropagation},
      author={Yaroslav Ganin and Victor Lempitsky},
      year={2015},
      eprint={1409.7495},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
"""


# class GradReverse(Function):
#     "Implementation of GRL from DANN (Domain Adaptation Neural Network) paper"
#
#     @staticmethod
#     def forward(ctx, x):
#         return x.contiguous().view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.neg()
#
#
# def grad_reverse(x):
#     """
#     GRL must be placed between the feature extractor and the domain classifier
#     """
#     return GradReverse.apply(x)
#


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


revgrad = GradientReversal.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)


class TrainedAdversarialApproximation(nn.Module):
    def __init__(
        self,
        n_source_channels: int,
        n_target_channels: list[int],
        n_layers: int,
        kernel_size: int,
        gradient_reversal_scale: float,
    ):
        assert n_layers >= 1, "Number of layers should be positive signed integer."
        super().__init__()
        padding = (int((kernel_size - 1) / 2), int((kernel_size - 1) / 2))

        debias_shape = (1, n_source_channels, 1, 1)
        rebias_shape = (1, sum(n_target_channels), 1, 1)
        self.debias: nn.Parameter = nn.Parameter(torch.Tensor(torch.zeros(debias_shape, dtype=torch.float32)))
        self.n_target_channels: list[int] = n_target_channels
        self.rebias: nn.Parameter = nn.Parameter(torch.Tensor(torch.zeros(rebias_shape, dtype=torch.float32)))

        self.reverse_gradient = GradientReversal(alpha=gradient_reversal_scale)
        # convs contains the number of convolutions for each parent architecture.
        #   This means we finish one approximation and continue to the next.
        self.convs: nn.ModuleList[nn.ModuleList[nn.Conv2d]] = nn.ModuleList(
            [
                nn.Conv2d(
                    sum(n_target_channels),
                    sum(n_target_channels),
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=padding,
                    padding_mode="zeros",
                    bias=False,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.bns: nn.ModuleList[nn.ModuleList[nn.Conv2d]] = nn.ModuleList(
            [nn.BatchNorm2d(sum(n_target_channels)) for _ in range(n_layers - 1)]
        )

        self.relu = nn.functional.relu
        self.in_conv: nn.Conv2d = nn.Conv2d(
            n_source_channels,
            sum(n_target_channels),
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            padding_mode="zeros",
            bias=False,
        )

    def forward(self, source: torch.Tensor) -> list[torch.Tensor]:
        """
        This is very similar to the Trained Approximation.
        It creates a learnable layer that transfers between architectures.
        IMPORTANT: On the backward pass the gradients are reversed in order to make the
        getting trained model as incapable as possible in approximating the other layer.

        Debiases (learnable) the original feature maps, expands their channels
        to the number of the channels of all other architectures, rebiases them
        and splits them into architecture specific models to enable various
         dimension as target
        and rebiases (learnable) the representations.
        :param source: Sources contains the outputs of the training model trained models.
        They can either be aggregated along the channel dimension, effectively leading to a single

        """
        x = self.reverse_gradient(source)
        debiased_source = x - self.debias
        x = self.in_conv(debiased_source)
        for conv, bn in zip(self.convs, self.bns):
            x = self.relu(x)
            x = conv(x)
            x = bn(x)
        x = x + self.rebias
        x = torch.split(x, split_size_or_sections=self.n_target_channels, dim=1)  # Batch x Channels x Width x Height
        return x
