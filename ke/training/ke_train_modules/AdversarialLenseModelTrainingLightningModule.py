from __future__ import annotations

from dataclasses import asdict

import torch
from ke.arch.ke_architectures.adversarial_lense_new_model_training_arch import (
    AdversarialLenseNewModelTrainingArch,
)
from ke.metrics.ke_metrics import multi_output_metrics
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.util import data_structs
from ke.util.data_structs import KEAdversarialLenseOutputTrainingInfo
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class AdversarialLenseModelTrainingLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_info: KEAdversarialLenseOutputTrainingInfo,
        network: AdversarialLenseNewModelTrainingArch,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        skip_n_epochs: int | None = None,
        log: bool = True,
    ):
        super().__init__(model_info, save_checkpoints, params, hparams, skip_n_epochs, log)
        self.loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.net: AdversarialLenseNewModelTrainingArch = network

    def save_checkpoint(self):
        if self.current_epoch == 0:
            return
        if self.save_checkpoints:
            if self.current_epoch == (self.params.num_epochs - 1):
                state_dict = self.net.get_new_model_state_dict()
                torch.save(state_dict, self.checkpoint_path)
        return

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        ckpt = torch.load(self.checkpoint_path)
        self.net.get_new_model().load_state_dict(ckpt)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, old_y_hats = self(x)
        ke_fwd = self.loss.forward(y_hat, y)
        return ke_fwd

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        mean_loss = torch.stack([o["loss"] for o in outputs], dim=0).mean()

        prog_bar_log = {"tr/loss": mean_loss}
        loss_values = {"loss/total": mean_loss}

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_values, is_train=True)

        return None

    def is_not_too_large(self, t: list[torch.Tensor]) -> bool:
        num_ele = sum([torch.numel(te) for te in t])
        if self.max_data_points >= num_ele:
            return True
        else:
            return False

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ["data"], batch["label"]
        with torch.no_grad():
            y_hat, y_hats = self.net.eval_forward(x)
            self.save_validation_values(
                old_intermediate_reps=None,
                new_intermediate_reps=None,
                groundtruths=y,
                old_y_hats=torch.stack(y_hats, dim=0),
                new_y_hat=y_hat,
                transferred_y_hats=None,
            )
        return self.loss.forward(y_hat, y)

    def validation_epoch_end(self, outputs):
        self.save_checkpoint()

        mean_loss = torch.stack(outputs, dim=0).mean()

        tensor_metrics = multi_output_metrics(self.new_y_out, self.old_y_outs, self.gts)

        self.final_metrics = {k: float(v) for k, v in asdict(tensor_metrics).items()}
        tensorboard_dict = {f"metrics/{k}": v for k, v in self.final_metrics.items()}
        prog_bar_log = {
            "val/loss": mean_loss,
            "val/acc": tensor_metrics.accuracy,
            "val/CoKa": tensor_metrics.cohens_kappa,
        }
        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(tensorboard_dict, is_train=False)
        return None
