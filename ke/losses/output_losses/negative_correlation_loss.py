import torch as t
from torch import nn


def negative_correlation(all_probs: t.Tensor):
    """
    Calculates the negative correlation as define in the paper
    https://www.sciencedirect.com/science/article/pii/S0893608099000738
    param all_probs: K x Batch x N_Class
    """
    mean_probs = t.mean(all_probs, dim=0, keepdim=True)  # 1 x Batch x N_Class
    diff_probs = all_probs - mean_probs  # K x Batch x N_Class
    diff_probs_exp = t.repeat_interleave(diff_probs, repeats=diff_probs.shape[0], dim=1)  # K x K x Batch x N_Class

    n_models = diff_probs.shape[0]
    non_diag = t.ones((n_models, n_models) - t.eye(n_models, n_models), device=all_probs.device)[:, :, None, None]

    non_diag_diff_probs = t.sum(diff_probs_exp * non_diag, dim=1)  # K x Batch x N_Class
    neg_corr = diff_probs * non_diag_diff_probs  # K x Batch x N_Class
    return t.sum(neg_corr, dim=(1, 2))  # K


class NegativeCorrelationLoss(nn.Module):
    def __init__(self, weight_nc: float = 0.1, weight_ce: float = 1.0):
        super(NegativeCorrelationLoss, self).__init__()
        self.weight_nc = weight_nc
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss()

    def forward(self, target: t.Tensor, logits: t.Tensor) -> t.Tensor:
        """
        Calculates the loss for the adaptive diversity promotion loss.

        :param outputs: list of model outputs
        :param target: ground truth
        :return: loss
        """
        all_logits = logits
        all_probs = t.softmax(all_logits, dim=-1)  # K x Batch x N_Class
        nc = negative_correlation(all_probs)

        ce = t.mean(t.stack([self.weight_ce * self.ce(logit, target) for logit in logits]))

        nc_loss = nc * self.weight_nc
        ce_loss = ce * self.weight_ce

        return ce_loss + nc_loss
