import numpy as np
import torch
from rep_trans.losses.output_losses.ke_abs_out_loss import KEAbstractOutputLoss
from torch import nn


class KEOutputTrainLoss(nn.Module):
    def __init__(
        self,
        dissimilar_loss: KEAbstractOutputLoss,
        ce_weight: float = 1.0,
        dissim_weight: float = 1.0,
        regularization_epoch_start: int = 0,
        n_classes: int = 10,
    ):
        """
        Loss regularization affecting the outputs.
        Can receive dissim_weight. Should multiple loss_values get passed passes them to the
        loss directly and just integrates them with factor 1.0 in here.
        """
        super(KEOutputTrainLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.dissimilar_loss: KEAbstractOutputLoss = dissimilar_loss
        self.num_tasks = self.dissimilar_loss.num_tasks + 1

        self.num_classes = n_classes
        self.ce_weight = ce_weight
        self.dissim_weight_target = dissim_weight
        self.current_dis_weight = 0.0

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
        :param target_weight: Target weight that the model is supposed to reach after warm up.
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
        old_outs: list[torch.Tensor],
        epoch_num: int,
        global_step: int,
    ) -> dict:
        pred_logits = new_out
        y_src_pred_logits = old_outs
        l_ce = self.ce_loss(new_out, label)
        l_dis = self.dissimilar_loss.forward(pred_logits, label, y_src_pred_logits)

        if self.start_warmup_epoch != -1:
            if epoch_num > self.start_warmup_epoch:
                if not self.already_hit_warmup_epoch:
                    self.initial_warmup_epoch_global_step = global_step
                    self.already_hit_warmup_epoch = True
                else:
                    if not self.warm_up_done:
                        if (self.initial_warmup_epoch_global_step + self.steps_till_fully_up) >= global_step:
                            self.current_dis_weight = self.warmup_weight(global_step, self.dissim_weight_target)
                        if (global_step - self.initial_warmup_epoch_global_step) == self.steps_till_fully_up:
                            self.warm_up_done = True
                    else:
                        self.current_dis_weight = self.dissim_weight_target
        else:
            self.current_dis_weight = self.dissim_weight_target

        l_ce_weighted: list[torch.Tensor] = [l_ce * self.ce_weight]
        l_dissim_weighted: list[torch.Tensor] = [l * self.current_dis_weight for l in l_dis]
        total_loss = l_ce_weighted + l_dissim_weighted

        return {
            "loss": total_loss,
            "loss_info": {
                "classification_raw": l_ce.detach(),
                "classification_weighted": torch.stack(l_ce_weighted).mean().detach(),
                "dissimilar_weighted": torch.stack(l_dissim_weighted).mean().detach(),
                "dissimilar_raw": torch.stack(l_dis).mean().detach(),
                "dissimilar_weight": self.current_dis_weight,
                "total_loss": torch.stack(total_loss).mean(),
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
        dissimilar_raw = torch.stack([o["loss_info"]["dissimilar_raw"] for o in ke_forward_dict]).mean()
        dissimilar_weighted = torch.stack([o["loss_info"]["dissimilar_weighted"] for o in ke_forward_dict]).mean()
        dissimilar_weight = np.stack([o["loss_info"]["dissimilar_weight"] for o in ke_forward_dict]).mean()

        loss_values = {
            "loss/total": total_loss,
            "loss_raw/classification": class_raw,
            "loss/classification": class_weighted,
            "loss/dissimilar": dissimilar_weighted,
            "loss_raw/dissimilar": dissimilar_raw,
            "weight/dissimilar_weight": dissimilar_weight,
        }
        return loss_values
