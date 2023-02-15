import torch
from torch import nn


class UncertainLossWeighter(nn.Module):
    def __init__(self, num_tasks):
        super(UncertainLossWeighter, self).__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros((num_tasks)))

    def forward(self, mt_losses):
        assert len(mt_losses) == self.num_tasks
        weighted_losses = []
        for i, loss in enumerate(mt_losses):
            precision0 = torch.exp(-self.log_vars[i])
            wloss = precision0 * loss + self.log_vars[i]
            weighted_losses.append(wloss)
        return weighted_losses
