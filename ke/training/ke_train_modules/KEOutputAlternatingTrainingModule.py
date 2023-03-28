from __future__ import annotations

from dataclasses import asdict

import torch
from ke.arch.ke_architectures.output_regularization_partial_gradient import (
    OutputRegularizerPartialGradientArch,
)
from ke.losses.dummy_loss import DummyLoss
from ke.losses.ke_output_loss import KEOutputTrainLoss
from ke.metrics.ke_metrics import multi_output_metrics
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.util import data_structs
from ke.util.data_structs import KEOutputTrainingInfo
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class KEOutputAlternatingTrainingModule(BaseLightningModule):
    def __init__(
        self,
        model_info: KEOutputTrainingInfo,
        network: OutputRegularizerPartialGradientArch,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        basic_loss: DummyLoss,
        output_loss: KEOutputTrainLoss,
        skip_n_epochs: int | None = None,
        log: bool = True,
    ):
        super().__init__(model_info, save_checkpoints, params, hparams, skip_n_epochs, log)
        self.basic_loss: DummyLoss = basic_loss
        self.regularization_loss: KEOutputTrainLoss = output_loss
        self.net: OutputRegularizerPartialGradientArch = network
        self.automatic_optimization = False
        # WARNING: In this MODULE NO AUTOMATIC GRADIENT CALCULATION HAPPENS!
        #   ONLY FOR THE SECOND MODEL GETTING TRAINED!

    def save_checkpoint(self):
        if self.current_epoch == 0:
            return
        if self.save_checkpoints:
            if self.current_epoch == (self.params.num_epochs - 1):
                state_dict = self.net.get_state_dict()
                torch.save(state_dict, self.checkpoint_path)
        return

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        ckpt = torch.load(self.checkpoint_path)
        self.net.get_new_model().load_state_dict(ckpt)

    def training_step(self, batch, batch_idx):
        whole_arch_opt, first_arch_opt = self.optimizers()

        x, y = batch
        y_hat, _ = self(x)

        # Optimize for CE
        dummy_out = self.basic_loss.forward(y, y_hat)
        ce_loss: torch.Tensor = dummy_out["loss"]
        self.zero_grad()
        self.manual_backward(ce_loss)
        whole_arch_opt.step()

        # Make unequal
        y_hat, source_y_hats = self(x)
        ke_out = self.regularization_loss.forward(
            label=y, new_out=y_hat, old_outs=source_y_hats, epoch_num=self.current_epoch, global_step=self.global_step
        )
        reg_loss: torch.Tensor = ke_out["loss"]
        self.zero_grad()
        self.manual_backward(reg_loss)
        first_arch_opt.step()
        joint_loss = {
            "ce_loss": ce_loss,
            "regularization_loss": reg_loss,
        }
        return joint_loss

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        mean_dummy_loss = torch.stack([o["ce_loss"] for o in outputs]).mean()
        mean_regularization_loss = torch.stack([o["regularization_loss"] for o in outputs]).mean()
        mean_total_loss = mean_dummy_loss + mean_regularization_loss
        prog_bar_log = {"tr/total_loss": mean_total_loss}
        log_message = {
            "loss/total": mean_total_loss,
            "loss/ce": mean_dummy_loss,
            "loss/regularization": mean_regularization_loss,
        }

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(log_message, is_train=True)
        schedulers = self.lr_schedulers()
        [s.step() for s in schedulers]

        return None

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ["data"], batch["label"]
        with torch.no_grad():
            y_hat, y_hats = self(x)
            dummy_out = self.basic_loss.forward(y, y_hat)
            ce_loss: torch.Tensor = dummy_out["loss"]

            y_hat, source_y_hats = self(x)
            ke_out = self.regularization_loss.forward(
                label=y,
                new_out=y_hat,
                old_outs=source_y_hats,
                epoch_num=self.current_epoch,
                global_step=self.global_step,
            )
            reg_loss = ke_out["loss"]

            self.save_validation_values(
                new_y_hat=y_hat,
                old_y_hats=torch.stack(y_hats),
                groundtruths=y,
                old_intermediate_reps=None,
                new_intermediate_reps=None,
                transferred_y_hats=None,
            )
            joint_loss = {
                "ce_loss": ce_loss,
                "regularization_loss": reg_loss,
            }
        return joint_loss

    def validation_epoch_end(self, outputs):
        self.save_checkpoint()
        # calculate mean losses stuff
        mean_dummy_loss = torch.stack([o["ce_loss"] for o in outputs]).mean()
        mean_regularization_loss = torch.stack([o["regularization_loss"] for o in outputs]).mean()
        mean_total_loss = mean_dummy_loss + mean_regularization_loss
        prog_bar_log = {"tr/total_loss": mean_total_loss}
        tensorboard_dict = {
            "loss/total": mean_total_loss,
            "loss/ce": mean_dummy_loss,
            "loss/regularization": mean_regularization_loss,
        }

        with torch.no_grad():
            out_metrics = multi_output_metrics(
                self.new_y_out, self.old_y_outs, self.gts, self.params.dataset, self.params.architecture_name
            )
            self.final_metrics = asdict(out_metrics)
            tensorboard_dict.update({f"metrics/{k}": v for k, v in self.final_metrics.items()})
            prog_bar_log.update(
                {
                    "val/acc": out_metrics.accuracy,
                    "val/CoKa": out_metrics.cohens_kappa,
                }
            )
            self.log_dict(prog_bar_log, prog_bar=True, logger=False)
            self.log_message(tensorboard_dict, is_train=False)
        return None

    def configure_optimizers(self):
        all_parameters = self.net.get_trainable_parameters()
        only_first_parameters = self.net.get_alt_trainable_parameters()

        all_arch_optim = torch.optim.SGD(
            params=all_parameters,
            lr=self.params.learning_rate,
            momentum=self.params.momentum,
            weight_decay=self.params.weight_decay,
            nesterov=self.params.nesterov,
        )
        only_first_optim = torch.optim.SGD(
            params=only_first_parameters,
            lr=self.params.learning_rate,
            momentum=self.params.momentum,
            weight_decay=self.params.weight_decay,
            nesterov=self.params.nesterov,
        )

        total_epochs = self.params.num_epochs
        if self.skip_n_epochs is not None:
            total_epochs = total_epochs + self.skip_n_epochs
        first_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(all_arch_optim, T_max=total_epochs, eta_min=0)
        second_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(only_first_optim, T_max=total_epochs, eta_min=0)
        if self.skip_n_epochs is not None:
            for i in range(self.skip_n_epochs):
                first_scheduler.step()
                second_scheduler.step()
        return [all_arch_optim, only_first_optim], [first_scheduler, second_scheduler]
