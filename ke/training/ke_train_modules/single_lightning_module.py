from __future__ import annotations

from dataclasses import asdict

import torch
import torchvision.models.resnet
from ke.arch.ke_architectures.single_model import SingleModel
from ke.losses.dummy_loss import DummyLoss
from ke.metrics.ke_metrics import single_output_metrics
from ke.training.ke_train_modules.base_training_module import BaseLightningModule
from ke.util import data_structs
from ke.util.data_structs import BasicTrainingInfo
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa


class SingleLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_info: BasicTrainingInfo,
        network: SingleModel,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        loss: DummyLoss,
        skip_n_epochs: int | None = None,
        log: bool = True,
    ):
        super().__init__(
            model_info=model_info,
            save_checkpoints=save_checkpoints,
            params=params,
            hparams=hparams,
            skip_n_epochs=skip_n_epochs,
            log=log,
        )
        self.automatic_optimization = True
        self.net: SingleModel = network
        self.loss = loss

    def save_checkpoint(self):
        state_dict = self.net.get_new_model_state_dict()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, self.checkpoint_path)
        return

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        ckpt = torch.load(self.checkpoint_path)
        self.net.load_new_model_state_dict(ckpt)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        dy_fwd = self.loss.forward(label=y, tbt_out=y_hat)
        return dy_fwd

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        loss_values = self.loss.on_epoch_end(outputs)
        prog_bar_log = {"tr/loss": loss_values["loss/total"]}

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_values, is_train=True)

        return None

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ["data"], batch["label"]
        with torch.no_grad():
            y_hat = self(x)
            self.save_validation_values(
                old_intermediate_reps=None,
                new_intermediate_reps=None,
                groundtruths=y,
                old_y_hats=None,
                new_y_hat=y_hat,
                transferred_y_hats=None,
            )
            return self.loss.forward(
                label=y,
                tbt_out=y_hat,
            )

    def validation_epoch_end(self, outputs):
        loss_dict = self.loss.on_epoch_end(outputs)
        single_metrics = single_output_metrics(self.new_y_out, self.gts)
        self.final_metrics = asdict(single_metrics)

        loss_dict.update({f"metrics/{k}": v for k, v in self.final_metrics.items()})
        prog_bar_log = {"val/acc": single_metrics.accuracy}

        if self.current_epoch != 0:
            if self.save_checkpoints:
                self.save_checkpoint()

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_dict, is_train=False)
        return None

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            params=self.net.parameters(),
            lr=self.params.learning_rate,
            momentum=self.params.momentum,
            weight_decay=self.params.weight_decay,
            nesterov=self.params.nesterov,
        )
        if self.params.cosine_annealing:
            total_epochs = self.params.num_epochs
            # This is ugly AF, I know. But weigh rewinding with CosineLR is a pain.
            #  Simulates normal # of epochs as previously to the scheduler.
            #   Skips n epochs to adjust the LR to the value and have the slope be equal
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_epochs, eta_min=0)
            return [optim], [scheduler]


class WarmStartSingleLightningModule(SingleLightningModule):
    def __init__(
        self,
        model_info: BasicTrainingInfo,
        network: SingleModel,
        save_checkpoints: bool,
        params: data_structs.Params,
        hparams: dict,
        loss: DummyLoss,
        skip_n_epochs: int | None = None,
        log: bool = True,
        warmup_pretrained: bool = False,
        linear_probe_only: bool = False,
    ):
        super().__init__(
            model_info=model_info,
            network=network,
            save_checkpoints=save_checkpoints,
            params=params,
            hparams=hparams,
            loss=loss,
            skip_n_epochs=skip_n_epochs,
            log=log,
        )
        self.automatic_optimization = False
        self.warmup_pretrained: int = warmup_pretrained
        self.linear_probe_only = linear_probe_only

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        dy_fwd = self.loss.forward(label=y, tbt_out=y_hat)
        if self.warmup_pretrained:
            warmup_optim, hot_optim = self.optimizers()
            if self.current_epoch < 10:
                warmup_optim.zero_grad()
                self.manual_backward(dy_fwd["loss"])
                warmup_optim.step()
            else:
                hot_optim.zero_grad()
                self.manual_backward(dy_fwd["loss"])
                hot_optim.step()
        else:
            (optimizer,) = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(dy_fwd["loss"])
            optimizer.step()

        return dy_fwd

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        loss_values = self.loss.on_epoch_end(outputs)
        prog_bar_log = {"tr/loss": loss_values["loss/total"]}

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_values, is_train=True)

        if self.warmup_pretrained:
            warmup_scheduler, hot_scheduler = self.lr_schedulers()
            if self.current_epoch < 5:
                warmup_scheduler.step()
            else:
                hot_scheduler.step()
        else:
            scheduler = self.lr_schedulers()
            scheduler.step()
        return None

    def configure_optimizers(self):
        linear_probe_params = []
        parameters = self.net.parameters()
        moi = self.net.get_new_model()
        if self.linear_probe_only:  # Override parameters that are to be trained.
            if isinstance(moi, torchvision.models.resnet.ResNet):
                linear_probe_params.extend(moi.conv1.parameters())
                linear_probe_params.extend(moi.bn1.parameters())
                linear_probe_params.extend(moi.fc.parameters())
            else:
                raise NotImplementedError("Warmup only intended for pretraining models.")
        if not self.warmup_pretrained:
            optim = torch.optim.SGD(
                params=linear_probe_params if self.linear_probe_only else parameters,
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay,
                nesterov=self.params.nesterov,
            )
            if self.params.cosine_annealing:
                total_epochs = self.params.num_epochs
                # This is ugly AF, I know. But weigh rewinding with CosineLR is a pain.
                #  Simulates normal # of epochs as previously to the scheduler.
                #   Skips n epochs to adjust the LR to the value and have the slope be equal
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_epochs, eta_min=0)
                return [optim], [scheduler]
        else:
            warmup_optim = torch.optim.SGD(
                params=linear_probe_params,
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay,
                nesterov=self.params.nesterov,
            )
            hot_optim = torch.optim.SGD(
                params=parameters,
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay,
                nesterov=self.params.nesterov,
            )
            if self.params.cosine_annealing:
                total_epochs = self.params.num_epochs
                # This is ugly AF, I know. But weigh rewinding with CosineLR is a pain.
                #  Simulates normal # of epochs as previously to the scheduler.
                #   Skips n epochs to adjust the LR to the value and have the slope be equal
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    warmup_optim,
                    start_factor=self.params.learning_rate * 1e-2,
                    end_factor=self.params.learning_rate,
                    total_iters=5,
                )
                hot_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    hot_optim, T_max=total_epochs - 5, eta_min=0
                )
                return [warmup_optim, hot_optim], [warmup_scheduler, hot_scheduler]
