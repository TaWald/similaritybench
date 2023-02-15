import numpy as np
import torch
import torchmetrics as tm
from rep_trans.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from torch import nn


class KESubTrainLoss(nn.Module):
    def __init__(
        self,
        similar_loss: AbstractRepresentationLoss,
        ce_weight: float = 1.0,
        sim_weight: float = 1.0,
        regularization_epoch_start=10,
        n_classes: int = 10,
    ):
        super(KESubTrainLoss, self).__init__()
        self.acc = tm.Accuracy()
        self.softmax = nn.Softmax(dim=1)
        self.similar_loss: AbstractRepresentationLoss = similar_loss

        self.num_classes = n_classes

        self.ce_weight = ce_weight
        self.sim_weight_target = sim_weight
        self.current_sim_weight = 0.0

        self.ce_loss = nn.CrossEntropyLoss()

        self.start_warmup_epoch = regularization_epoch_start
        self.initial_warmup_epoch_global_step = 0
        self.already_hit_warmup_epoch = False
        self.steps_till_fully_up = 10000  # == Pi change   # ~313 per epoch --> 3000?
        self.warm_up_done = False

    def warmup_weight(self, global_step: int, target_weight: float):
        """
        Does cosine annealing after the first 10 epochs passed to slowly increase the weight to the target.
        :param global_step: Current global step.
        :param target_weight: Target weight for the current weight to converge to.
        """
        current_weight = (
            target_weight
            * (
                1
                + np.cos(
                    (((global_step - self.initial_warmup_epoch_global_step) / self.steps_till_fully_up) - 1) * np.pi
                )
            )
            / 2
        )
        return current_weight

    def forward(
        self,
        label: torch.Tensor,
        new_out: torch.Tensor,
        new_inter: list[torch.Tensor],
        old_inters: list[torch.Tensor],
        epoch_num: int,
        global_step: int,
    ) -> dict:
        l_ce = self.ce_loss(new_out, label)

        tbt_inter_detached = [tbt.detach() for tbt in new_inter]
        l_sim = self.similar_loss.forward(tbt_inter_detached, old_inters, make_dissimilar=False)
        del tbt_inter_detached

        if self.start_warmup_epoch != -1:
            if epoch_num > self.start_warmup_epoch:
                if not self.already_hit_warmup_epoch:
                    self.initial_warmup_epoch_global_step = global_step
                    self.already_hit_warmup_epoch = True
                else:
                    if not self.warm_up_done:
                        if (self.initial_warmup_epoch_global_step + self.steps_till_fully_up) >= global_step:
                            self.current_sim_weight = self.warmup_weight(global_step, self.sim_weight_target)
                        if (global_step - self.initial_warmup_epoch_global_step) == self.steps_till_fully_up:
                            self.warm_up_done = True
                    else:
                        self.current_sim_weight = self.sim_weight_target
        else:
            self.current_sim_weight = self.sim_weight_target

        l_ce_weighted = l_ce * self.ce_weight
        l_sim_weighted = l_sim * self.current_sim_weight
        total_loss = l_sim_weighted + l_ce_weighted

        return {
            "loss": total_loss,
            "loss_info": {
                "similar_raw": l_sim.detach(),
                "similar_weighted": l_sim_weighted.detach(),
                "classification_raw": l_ce.detach(),
                "classification_weighted": l_ce_weighted.detach(),
                "similar_weight": self.current_sim_weight,
                "total_loss": total_loss.detach(),
            },
        }

    def on_epoch_end(self, ke_forward_dict: list[dict]) -> dict:
        """
        Creates better averages of returns for ke_foward.
        In the training loop it averages only losses since no metrics are calculated for that.
        In the validation loop it averages losses and the metrics
        """
        # Always calculate the loss stuff
        similar_raw = torch.stack([o["loss_info"]["similar_raw"] for o in ke_forward_dict]).mean()
        similar_weighted = torch.stack([o["loss_info"]["similar_weighted"] for o in ke_forward_dict]).mean()
        class_raw = torch.stack([o["loss_info"]["classification_raw"] for o in ke_forward_dict]).mean()
        class_weighted = torch.stack([o["loss_info"]["classification_weighted"] for o in ke_forward_dict]).mean()
        similar_weight = np.stack([o["loss_info"]["similar_weight"] for o in ke_forward_dict]).mean()
        total_loss = torch.stack([o["loss_info"]["total_loss"] for o in ke_forward_dict]).mean()

        loss_values = {
            "loss/total": total_loss,
            "loss_raw/similar": similar_raw,
            "loss/similar": similar_weighted,
            "loss_raw/classification": class_raw,
            "loss/classification": class_weighted,
            "weight/similar_weight": similar_weight,
        }
        return loss_values
