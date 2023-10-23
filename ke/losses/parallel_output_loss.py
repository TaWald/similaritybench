import torch
from torch import nn


class AbstractParallelOutputLoss(nn.Module):
    def forward(
        self,
        target: torch.Tensor,
        logits: list[torch.Tensor],
    ) -> dict:
        raise NotImplementedError("Saliency loss not implemented yet.")


class IndependentLoss(AbstractParallelOutputLoss):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Loss regularization affecting the outputs.
        Can receive dissim_weight. Should multiple loss_values get passed passes them to the
        loss directly and just integrates them with factor 1.0 in here.
        """
        super(IndependentLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        target: torch.Tensor,
        logits: torch.Tensor,
    ) -> dict:
        losses = [self.ce.forward(logit, target) for logit in logits]
        return {"loss": torch.mean(torch.stack(losses))}
