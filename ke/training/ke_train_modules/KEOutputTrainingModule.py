from __future__ import annotations

from dataclasses import asdict

import torch
from grad_pg.pcgrad_amp import PCGradAMP
from ke.arch.ke_architectures.output_regularization import OutputRegularizerArch
from ke.losses.ke_output_loss import KEOutputTrainLoss
from ke.metrics.ke_metrics import multi_output_metrics
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.util import data_structs
from ke.util.data_structs import KEOutputTrainingInfo
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class KEOutputLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_info: KEOutputTrainingInfo,
        network: OutputRegularizerArch,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        loss: KEOutputTrainLoss,
        skip_n_epochs: int | None = None,
        log: bool = True,
    ):
        super().__init__(model_info, save_checkpoints, params, hparams, skip_n_epochs, log)
        self.loss: KEOutputTrainLoss = loss
        self.net: OutputRegularizerArch = network

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
        x, y = batch
        y_hat, source_y_hats = self(x)
        ke_fwd = self.loss.forward(
            label=y,
            new_out=y_hat,
            old_outs=source_y_hats,
            epoch_num=self.current_epoch,
            global_step=self.global_step,
        )
        # PCGrad optimization
        ke_fwd["loss"] = torch.stack(ke_fwd["loss"]).sum()
        return ke_fwd

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        loss_values = self.loss.on_epoch_end(outputs)

        prog_bar_log = {"tr/loss": loss_values["loss/total"]}

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
            y_hat, y_hats = self(x)
            self.save_validation_values(
                old_intermediate_reps=None,
                new_intermediate_reps=None,
                groundtruths=y,
                old_y_hats=torch.stack(y_hats, dim=0),
                new_y_hat=y_hat,
                transferred_y_hats=None,
            )
            return self.loss.forward(
                label=y,
                new_out=y_hat,
                old_outs=y_hats,
                epoch_num=self.current_epoch,
                global_step=self.global_step,
            )

    def validation_epoch_end(self, outputs):
        self.save_checkpoint()
        loss_dict = self.loss.on_epoch_end(outputs)
        out_metrics = multi_output_metrics(
            self.new_y_out,
            self.old_y_outs,
            self.gts,
            self.params.dataset,
            self.params.architecture_name,
            self.ke_hparams["n_cls"],
        )
        self.final_metrics = asdict(out_metrics)

        loss_dict.update({f"metrics/{k}": v for k, v in self.final_metrics.items()})
        prog_bar_log = {
            "val/acc": out_metrics.accuracy,
            "val/CoKa": out_metrics.cohens_kappa.all_to_all_mean,
        }
        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_dict, is_train=False)
        return None


class KEOutputPCGrad(KEOutputLightningModule):
    def __init__(
        self,
        model_info: KEOutputTrainingInfo,
        network: OutputRegularizerArch,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        loss: KEOutputTrainLoss,
        skip_n_epochs: int | None = None,
        log: bool = True,
    ):
        super().__init__(model_info, network, save_checkpoints, params, hparams, loss, skip_n_epochs, log)
        self.automatic_optimization = False
        self.pc_grad_optim = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, source_y_hats = self(x)
        ke_fwd = self.loss.forward(
            label=y,
            new_out=y_hat,
            old_outs=source_y_hats,
            epoch_num=self.current_epoch,
            global_step=self.global_step,
        )
        # PCGrad optimization
        self.pc_grad_optim.backward(ke_fwd["loss"])
        self.pc_grad_optim.step()

        ke_fwd["loss"] = torch.stack(ke_fwd["loss"]).sum()  # Create single float value for logging again.

        return ke_fwd

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        loss_values = self.loss.on_epoch_end(outputs)

        prog_bar_log = {"tr/loss": loss_values["loss/total"]}

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_values, is_train=True)
        scheduler = self.lr_schedulers()
        scheduler.step()
        return None

    def configure_optimizers(self):
        has_get_parameters = hasattr(self.net, "get_trainable_parameters")
        if has_get_parameters:
            parameters_to_train = self.net.get_trainable_parameters()
        else:
            parameters_to_train = self.net.parameters()
        optim = torch.optim.SGD(
            params=parameters_to_train,
            lr=self.params.learning_rate,
            momentum=self.params.momentum,
            weight_decay=self.params.weight_decay,
            nesterov=self.params.nesterov,
        )
        scaler = torch.cuda.amp.GradScaler()
        self.pc_grad_optim = PCGradAMP(self.loss.num_tasks, optim, scaler)
        if self.params.cosine_annealing:
            total_epochs = self.params.num_epochs
            # This is ugly AF, I know. But weigh rewinding with CosineLR is a pain.
            #  Simulates normal # of epochs as previously to the scheduler.
            #   Skips n epochs to adjust the LR to the value and have the slope be equal
            if self.skip_n_epochs is not None:
                total_epochs = total_epochs + self.skip_n_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_epochs, eta_min=0)
            if self.skip_n_epochs is not None:
                for i in range(self.skip_n_epochs):
                    self.scheduler.step()
            return [optim], [scheduler]
        else:
            return [optim]
