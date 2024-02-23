from __future__ import annotations

from dataclasses import asdict

import torch
import torchvision.models.resnet
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa
from vision.arch.ke_architectures.single_model import SingleModel
from vision.losses.dummy_loss import DummyLoss
from vision.metrics.ke_metrics import single_output_metrics
from vision.training.ke_train_modules.base_training_module import BaseLightningModule
from vision.util import data_structs as ds
from vision.util.data_structs import BasicTrainingInfo


class SingleLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_info: ds.ModelInfo,
        network: SingleModel,
        save_checkpoints: bool,
        params: ds.Params,
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

        self.net: SingleModel = network
        self.loss = loss

    def save_checkpoint(self):
        state_dict = self.net.get_new_model_state_dict()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if self.current_epoch <= (self.params.num_epochs - 1):
            torch.save(state_dict, self.checkpoint_path)
        return

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        ckpt = torch.load(self.checkpoint_path)
        self.net.load_new_model_state_dict(ckpt)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        dy_fwd = self.loss.forward(label=y, new_out=y_hat)
        return dy_fwd

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        with torch.no_grad():
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
                new_out=y_hat,
            )

    def validation_epoch_end(self, outputs):
        loss_dict = self.loss.on_epoch_end(outputs)
        single_metrics = single_output_metrics(self.new_y_out, self.gts, self.ke_hparams["n_cls"])
        self.final_metrics = asdict(single_metrics)

        loss_dict.update({f"metrics/{k}": v for k, v in self.final_metrics.items()})
        prog_bar_log = {"val/acc": single_metrics.accuracy}

        if self.current_epoch != 0:
            if self.save_checkpoints:
                self.save_checkpoint()

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_dict, is_train=False)
        return None
