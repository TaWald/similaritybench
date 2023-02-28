import torch as t
from ke.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss
from ke.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss


class NoneLoss(AbstractRepresentationLoss):
    def __init__(self, softmax_channel_metrics):
        super(NoneLoss, self).__init__(softmax_channel_metrics)
        self.zero_return = t.tensor((0.0,), device="cuda")

    def forward(self, tbt_inter: list[t.Tensor], approx_inter: list[t.Tensor], make_dissimilar: bool) -> t.Tensor:
        # L_Similar: Maximize correlation --> Loss sollte hoch sei wenn corr = 0
        #  detach the true representations because the approx. network is supposed to do the work.
        return self.zero_return


class NoneOutputLoss(KEAbstractOutputLoss):
    num_tasks = 1

    def __init__(self, n_classes: int):
        super(NoneOutputLoss, self).__init__(n_classes)
        self.zero_return = t.tensor((0.0,), device="cuda")

    def forward(
        self, logit_prediction: t.Tensor, groundtruth: t.Tensor, logit_other_predictions: list[t.Tensor]
    ) -> t.Tensor:
        return t.tensor((0.0,), dtype=logit_prediction.dtype, device=logit_prediction.device)
