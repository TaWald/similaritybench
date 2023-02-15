from dataclasses import dataclass

import numpy as np
import torch
import torchmetrics as tm
from rep_trans.losses.representation_similarity_losses.ke_abs_loss import AbstractRepresentationLoss
from torch import nn


@dataclass
class EpochEndMessage:
    log_msg: str
    prog_bar_dict: dict
    tensorboard_dict: dict


class KEAdversarialTrainLoss(nn.Module):
    def __init__(
        self,
        adversarial_loss: AbstractRepresentationLoss,
        adversarial_loss_weight: float,
        cross_entropy_weight: float,
        regularization_epoch_start: int = -1,
        n_classes: int = 10,
    ):
        super(KEAdversarialTrainLoss, self).__init__()
        self.acc = tm.Accuracy()
        self.softmax = nn.Softmax(dim=1)
        self.adversarial_loss: AbstractRepresentationLoss = adversarial_loss
        self.adversarial_loss_weight_target: float = adversarial_loss_weight
        self.ce_weight: float = cross_entropy_weight

        self.num_classes = n_classes
        self.current_adv_weight = 0.0

        self.ce_loss = nn.CrossEntropyLoss()

        self.start_warmup_epoch = regularization_epoch_start
        self.initial_warmup_epoch_global_step = 0
        self.already_hit_warmup_epoch = False
        self.steps_till_fully_up = 10000  # == Pi change   # ~313 per epoch --> 3000?
        self.warm_up_done = False

    def warmup_weight(self, global_step: int, target_weight: float):
        """
        Does cosine annealing after the first 10 epochs passed to slowly increase the dissimilarity loss.
        :param global_step: Current global step.
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
        l_adv = self.adversarial_loss.forward(new_inter, old_inters, make_dissimilar=False)

        if self.start_warmup_epoch != -1:
            if epoch_num > self.start_warmup_epoch:
                if not self.already_hit_warmup_epoch:
                    self.initial_warmup_epoch_global_step = global_step
                    self.already_hit_warmup_epoch = True
                else:
                    if not self.warm_up_done:
                        if (self.initial_warmup_epoch_global_step + self.steps_till_fully_up) >= global_step:
                            self.current_adv_weight = self.warmup_weight(
                                global_step, self.adversarial_loss_weight_target
                            )
                        if (global_step - self.initial_warmup_epoch_global_step) == self.steps_till_fully_up:
                            self.warm_up_done = True
                    else:
                        self.current_adv_weight = self.adversarial_loss_weight_target
        else:
            self.current_adv_weight = self.adversarial_loss_weight_target

        l_ce_weighted = l_ce * self.ce_weight
        l_adv_weighted = l_adv * self.current_adv_weight
        total_loss = l_ce_weighted + l_adv_weighted

        return {
            "loss": total_loss,
            "loss_info": {
                "total_loss": total_loss.detach(),
                "classification_raw": l_ce.detach(),
                "classification_weighted": l_ce_weighted.detach(),
                "adversarial_weighted": l_adv_weighted.detach(),
                "adversarial_raw": l_adv.detach(),
            },
        }

    def on_epoch_end(self, ke_forward_dict: list[dict]) -> dict:
        """
        Creates better averages of returns for ke_foward.
        In the training loop it averages only losses since no metrics are calculated for that.
        In the validation loop it averages losses and the metrics
        """
        # Always calculate the loss stuff
        total_loss = torch.stack([o["loss_info"]["total_loss"] for o in ke_forward_dict]).mean()
        class_raw = torch.stack([o["loss_info"]["classification_raw"] for o in ke_forward_dict]).mean()
        class_weighted = torch.stack([o["loss_info"]["classification_weighted"] for o in ke_forward_dict]).mean()
        adversarial_raw = torch.stack([o["loss_info"]["adversarial_raw"] for o in ke_forward_dict]).mean()
        adversarial_weighted = torch.stack([o["loss_info"]["adversarial_weighted"] for o in ke_forward_dict]).mean()

        loss_values = {
            "loss/total": total_loss,
            "loss_raw/classification": class_raw,
            "loss/classification": class_weighted,
            "loss/adversarial": adversarial_weighted,
            "loss_raw/adversarial": adversarial_raw,
        }
        return loss_values
